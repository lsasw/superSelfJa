---
title: Redis 缓存三大问题与持久化
icon: bolt
order: 2
category:
  - 数据库
tag:
  - Redis
  - 缓存
  - 持久化
---

# Redis 缓存三大问题与持久化

> 任务编号：RE-01 / RE-02 / RE-03

## 一、缓存雪崩（RE-01）

```
场景：大量 key 同时过期 → 所有请求打到 DB → DB 崩溃

解决方案：
1. 过期时间加随机值
   expire = base_time + random(0, 300) 秒

2. 永不过期 + 异步更新
   缓存不设过期时间，后台线程异步刷新

3. Redis 高可用
   主从 + 哨兵/集群，避免单点故障

4. 限流降级
   对 DB 请求做限流，超量直接返回默认值/空数据
```

### 实现：随机过期时间

```java
public void setWithRandomExpire(String key, Object value, int baseSeconds) {
    int randomSeconds = ThreadLocalRandom.current().nextInt(300);
    redis.setex(key, baseSeconds + randomSeconds, value);
}
```

## 二、缓存穿透

```
场景：查询不存在的数据 → 缓存没命中 → 每次都查 DB

解决方案：
1. 布隆过滤器
   判断 key 是否存在，不存在直接返回

2. 缓存空值
   查不到也缓存（短 TTL，如 60 秒）

3. 参数校验
   对明显非法的参数直接拦截
```

```java
// 布隆过滤器（Redisson）
RBloomFilter<String> bloomFilter = redisson.getBloomFilter("user:bloom");
bloomFilter.tryInit(1000000, 0.03);

// 查询时先过布隆
if (!bloomFilter.contains(userId)) {
    return null; // 一定不存在
}
// 再查缓存/DB
```

## 三、缓存击穿

```
场景：热点 key 过期 → 瞬间大量请求打到 DB → DB 压力剧增

解决方案：
1. 互斥锁
   第一个请求去查 DB，其他请求等待

2. 永不过期
   热点 key 不设过期时间，异步更新

3. 逻辑过期
   缓存中加一个逻辑过期时间戳，判断后异步更新
```

```java
// 互斥锁方案
public String getWithMutex(String key) {
    String value = redis.get(key);
    if (value != null) return value;
    
    // 获取锁
    String lockKey = "lock:" + key;
    if (redis.setnx(lockKey, "1", 10)) {
        try {
            value = db.query(key);
            redis.setex(key, 3600, value);
        } finally {
            redis.del(lockKey);
        }
    } else {
        // 等待后重试
        Thread.sleep(50);
        return getWithMutex(key);
    }
    return value;
}
```

## 四、RDB vs AOF（RE-02）

| 特性 | RDB | AOF |
|------|:---:|:---:|
| 原理 | 定时快照全量数据 | 记录每条写命令 |
| 文件大小 | 小（压缩二进制） | 大（文本命令） |
| 恢复速度 | ⚡快 | 慢（需重放命令） |
| 数据安全 | 可能丢失最后一次快照后的数据 | 可配置 everysec（最多丢 1s） |
| 写入性能 | 不影响（fork 子进程） | 影响（取决于 fsync 策略） |
| 推荐 | 备份、灾备 | 数据安全优先 |

### 配置建议

```conf
# redis.conf — 生产环境推荐配置

# RDB：每 5 分钟有 1 次写入就快照
save 300 1
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes

# AOF：每秒刷盘
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

## 五、Redis 分布式锁 vs Redisson（RE-03）

```java
// === 手写版 ===
public class SimpleRedisLock {
    public boolean lock(String key, String value, int seconds) {
        return "OK".equals(jedis.set(key, value, 
            SetParams.setParams().nx().ex(seconds)));
    }
    // 问题：不可重入、无自动续期
}

// === Redisson 版 ===
RLock lock = redisson.getLock("order:lock:12345");

// 自动续期（watchdog）
lock.lock(); // 默认 30s，每 10s 自动续期

// 可重入
lock.lock();
lock.lock(); // 同一线程再次加锁 OK
lock.unlock(); // count -1
lock.unlock(); // count=0 → 释放

// 公平锁
RLock fairLock = redisson.getFairLock("fair:lock");
```
