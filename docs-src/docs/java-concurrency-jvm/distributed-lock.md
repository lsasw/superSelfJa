---
title: 分布式锁设计与实战
icon: lock
order: 2
category:
  - Java
tag:
  - 分布式锁
  - Redis
  - ZooKeeper
---

# 分布式锁设计与实战

> 任务编号：DL-01 / DL-02 / DL-03 / DL-04

## 一、Redis 单机分布式锁（DL-01）

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.params.SetParams;
import java.util.Collections;

public class RedisLock {
    private static final String LOCK_SUCCESS = "OK";
    private static final String UNLOCK_SUCCESS = "1";
    private static final Long RELEASE_SUCCESS = 1L;
    
    private Jedis jedis;
    
    /**
     * 加锁（SET NX + EXPIRE — 原子操作）
     */
    public boolean lock(String key, String value, int expireSeconds) {
        SetParams params = new SetParams()
            .nx()              // 不存在时才设置
            .ex(expireSeconds); // 过期时间
        
        String result = jedis.set(key, value, params);
        return LOCK_SUCCESS.equals(result);
    }
    
    /**
     * 解锁（Lua 脚本保证原子性）
     * 必须判断 value 是否是自己设置的，防止误删别人的锁
     */
    public boolean unlock(String key, String value) {
        String script = 
            "if redis.call('get', KEYS[1]) == ARGV[1] then " +
            "    return redis.call('del', KEYS[1]) " +
            "else " +
            "    return 0 " +
            "end";
        
        Object result = jedis.eval(script, 
            Collections.singletonList(key),
            Collections.singletonList(value));
        
        return RELEASE_SUCCESS.equals(result);
    }
}

// === 使用 ===
RedisLock lock = new RedisLock();
String lockKey = "order:lock:12345";
String lockValue = UUID.randomUUID().toString(); // 唯一标识

try {
    if (lock.lock(lockKey, lockValue, 30)) {
        // 执行业务逻辑
        processOrder();
    }
} finally {
    lock.unlock(lockKey, lockValue);
}
```

## 二、Redis 集群锁丢失场景复现（DL-02）

```
Client A                     Redis Master           Redis Slave
    │                              │                      │
    │  ① 获取锁成功                  │                      │
    │ ────────────────────────────→│                      │
    │                              │  ② 异步复制           │
    │                              │ ────────────────────→│
    │                              │                      │
    │                              │  ③ Master 宕机      │
    │                              │     ✗                │
    │                              │                      │
    │                              │  ④ Slave 提升        │
    │                              │     → Master         │
    │                              │                      │
    │          Client B            │                      │
    │              │               │                      │
    │  ⑤ 获取锁成功（锁不存在！）    │                      │
    │ ──────────────────────────────────────────────────→│
    │                                                      │
    ⚠️ 两个客户端同时持有"同一把锁"！
```

### 问题根因

- Redis 主从复制是**异步**的
- Master 宕机时，Slave 可能还没收到 `SET NX` 命令
- Slave 提升后，新 Master 上锁不存在
- 其他客户端可以获取同一把锁

## 三、Redlock 算法（DL-03）

```java
import java.util.List;
import java.util.ArrayList;

public class Redlock {
    private List<Jedis> redisNodes; // 多个独立的 Redis 实例
    private int quorum;             // 至少成功 N/2+1 个节点
    private int lockTTL = 30000;    // 锁的 TTL（毫秒）
    
    public Redlock(List<Jedis> nodes) {
        this.redisNodes = nodes;
        this.quorum = nodes.size() / 2 + 1;
    }
    
    public String lock(String resourceKey) {
        String lockValue = UUID.randomUUID().toString();
        int successCount = 0;
        long startTime = System.currentTimeMillis();
        
        // 1. 依次向所有节点获取锁
        for (Jedis node : redisNodes) {
            try {
                SetParams params = new SetParams()
                    .nx()
                    .px(lockTTL);
                String result = node.set(resourceKey, lockValue, params);
                if ("OK".equals(result)) {
                    successCount++;
                }
            } catch (Exception e) {
                // 节点不可用，继续尝试其他节点
            }
        }
        
        long elapsed = System.currentTimeMillis() - startTime;
        
        // 2. 判断是否获取成功
        if (successCount >= quorum && elapsed < lockTTL) {
            return lockValue;
        }
        
        // 3. 失败：释放已获取的锁
        unlock(resourceKey, lockValue);
        return null;
    }
    
    public void unlock(String key, String value) {
        for (Jedis node : redisNodes) {
            try {
                String script = 
                    "if redis.call('get', KEYS[1]) == ARGV[1] then " +
                    "    return redis.call('del', KEYS[1]) " +
                    "else return 0 end";
                node.eval(script, 
                    Collections.singletonList(key),
                    Collections.singletonList(value));
            } catch (Exception ignored) {}
        }
    }
}
```

### Redlock 的局限性

| 问题 | 说明 |
|------|------|
| 时钟漂移 | 各节点时钟不一致可能导致锁提前过期 |
| GC 暂停 | 客户端 GC 导致锁过期，其他客户端获取锁 |
| 网络分区 | 分区场景下 quorum 判断可能不准 |
| 实现复杂 | 需要至少 3-5 个独立 Redis 实例 |

## 四、ZooKeeper 分布式锁（DL-04）

```java
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.recipes.locks.InterProcessMutex;
import java.util.concurrent.TimeUnit;

public class ZkDistributedLock {
    private CuratorFramework client;
    
    public boolean tryLock(String path, long timeoutMs) {
        InterProcessMutex lock = new InterProcessMutex(client, path);
        try {
            return lock.acquire(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            return false;
        }
    }
}
```

### ZooKeeper 加锁时序

```
Client A                     ZooKeeper
    │                            │
    │  ① 创建临时顺序节点          │
    │ /lock/00000001             │
    │ ──────────────────────────→│
    │                            │
    │  ② 获取 /lock 下所有子节点   │
    │ ──────────────────────────→│
    │ ← [00000001, 00000002]     │
    │                            │
    │  ③ 判断自己是否最小？        │
    │  00000001 → 最小 → 加锁成功  │
    │                            │
    │  ④ 如果不是最小，watch 前一个│
    │  → 等前一个释放后再竞争       │
    │                            │
    │  ⑤ Client 断开 → 临时节点自动删除
```

### Redis vs ZooKeeper 分布式锁对比

| 维度 | Redis | ZooKeeper |
|------|:---:|:---:|
| 性能 | ⚡⚡⚡ 高 | ⚡⚡ 中 |
| 可靠性 | 主从切换可能丢锁 | CP 强一致，不丢锁 |
| 复杂度 | 简单（Redlock 除外） | 需要 ZooKeeper 集群 |
| 自动释放 | 依赖 TTL（可能不准） | 临时节点自动删除 |
| 阻塞等待 | 需自实现 | watch 机制原生 |
| 推荐场景 | 性能优先、可容忍短暂不一致 | 一致性优先、金融交易 |
