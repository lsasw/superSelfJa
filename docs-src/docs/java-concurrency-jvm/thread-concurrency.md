---
title: Java 并发编程实战
icon: sync
order: 1
category:
  - Java
tag:
  - ThreadLocal
  - HashMap
  - synchronized
---

# Java 并发编程实战

> 任务编号：JC-01 / JC-02 / JC-03 / JC-04

## 一、ThreadLocal 脏数据复现与修复（JC-01）

```java
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadLocalLeakDemo {
    // 线程池复用场景下的 ThreadLocal
    private static final ThreadLocal<UserContext> CONTEXT = 
        ThreadLocal.withInitial(() -> new UserContext());
    
    private static final ExecutorService POOL = 
        Executors.newFixedThreadPool(2);
    
    public static void main(String[] args) throws Exception {
        // 任务 1：用户 A
        POOL.submit(() -> {
            CONTEXT.get().setUserId("user-A");
            CONTEXT.get().setRole("admin");
            System.out.println("任务1: " + CONTEXT.get());
        }).get();
        
        // 任务 2：没有设置 CONTEXT，但可能拿到脏数据！
        POOL.submit(() -> {
            // ⚠️ BUG：线程池复用了线程，ThreadLocal 的值还在！
            UserContext ctx = CONTEXT.get();
            System.out.println("任务2 脏数据: " + ctx);
        }).get();
        
        // ✅ 修复：每个任务结束时必须 remove()
        POOL.submit(() -> {
            try {
                CONTEXT.get().setUserId("user-C");
                System.out.println("任务3: " + CONTEXT.get());
            } finally {
                CONTEXT.remove(); // 关键！
            }
        }).get();
        
        POOL.shutdown();
    }
}

class UserContext {
    private String userId;
    private String role;
    
    public void setUserId(String id) { this.userId = id; }
    public void setRole(String r) { this.role = r; }
    
    @Override
    public String toString() {
        return "UserContext{userId='" + userId + "', role='" + role + "'}";
    }
}
```

### ThreadLocal 内存泄漏原理

```
Thread
  └── ThreadLocalMap (Entry[] table)
       └── Entry (WeakReference<ThreadLocal> → Value)
            ↑                              ↑
          key 弱引用                    value 强引用
          （GC 时可能回收）             （GC 不会回收！）

如果 ThreadLocal 的强引用断了：
  - key 被 GC → null
  - value 仍然强引用 → 无法回收 → 内存泄漏！
```

## 二、HashMap 扩容详解（JC-02）

### JDK 1.7 头插法 → 死循环

```java
// JDK 1.7 resize() 核心逻辑（简化）
void transfer(Entry[] newTable) {
    for (Entry e : table) {
        while (e != null) {
            Entry next = e.next;
            int i = indexFor(e.hash, newCapacity);
            e.next = newTable[i]; // 头插法！
            newTable[i] = e;
            e = next;
        }
    }
}

// 并发场景下可能形成循环链表：
// A → B → A → B → ... (死循环)
```

### JDK 1.8 尾插法 → 安全但不保证顺序

```java
// JDK 1.8 resize() — 尾插法
Node<K,V> loHead = null, loTail = null;
Node<K,V> hiHead = null, hiTail = null;
Node<K,V> next;
do {
    next = e.next;
    if ((e.hash & oldCap) == 0) {
        if (loTail == null)
            loHead = e;
        else
            loTail.next = e;
        loTail = e;
    }
    // ...
} while ((e = next) != null);
```

### 扩容流程图

```
JDK 1.7（头插法）                JDK 1.8（尾插法）
初始: A→B→C                    初始: A→B→C
step1: B→A                     step1: A→B
step2: C→B→A                   step2: A→B→C
结果：逆序                      结果：保持原序
并发问题：循环链表               并发问题：数据丢失
```

## 三、synchronized 锁升级（JC-03）

```java
import org.openjdk.jol.info.ClassLayout;

public class LockUpgradeDemo {
    public static void main(String[] args) throws Exception {
        Object obj = new Object();
        
        // 1. 无锁状态（可偏向）
        System.out.println("无锁:");
        System.out.println(ClassLayout.parseInstance(obj).toPrintable());
        // Mark Word: ...00000101 (biasable)
        
        // 2. 偏向锁
        synchronized (obj) {
            System.out.println("偏向锁:");
            System.out.println(ClassLayout.parseInstance(obj).toPrintable());
            // Mark Word: 包含线程ID ...00000101
        }
        
        // 3. 轻量级锁（另一个线程竞争）
        Thread t = new Thread(() -> {
            synchronized (obj) {
                System.out.println("轻量级锁:");
                System.out.println(ClassLayout.parseInstance(obj).toPrintable());
                // Mark Word: 指向栈中 Lock Record ...00000000
            }
        });
        t.start();
        t.join();
        
        // 4. 重量级锁（高竞争时）
        // 多个线程同时竞争 → 膨胀为重量级锁
    }
}
```

### 锁升级路径

```
无锁 ──→ 偏向锁 ──→ 轻量级锁 ──→ 重量级锁
         (单线程   (多线程       (高竞争
         反复获取)  交替执行)     长时间等待)
```

### 触发条件速查

| 当前状态 | 升级条件 | 下一状态 |
|---------|---------|---------|
| 无锁 | 一个线程多次获取 | 偏向锁 |
| 偏向锁 | 另一个线程竞争 | 轻量级锁 |
| 轻量级锁 | CAS 自旋失败（默认10次） | 重量级锁 |
| 重量级锁 | 不可降级 | — |

## 四、volatile vs synchronized vs Lock（JC-04）

| 特性 | volatile | synchronized | Lock |
|------|:---:|:---:|:---:|
| **可见性** | ✅ | ✅ | ✅ |
| **原子性** | ❌ | ✅ | ✅ |
| **有序性** | ✅（禁止指令重排） | ✅ | ✅ |
| **阻塞** | 否 | 是 | 可尝试非阻塞 |
| **公平性** | — | 非公平 | 可选公平 |
| **中断响应** | — | 不可中断 | 可中断 |
| **条件变量** | — | wait/notify | Condition |
| **性能** | 低开销 | JDK6+ 优化后较好 | 更灵活 |
| **适用** | 状态标志 | 简单互斥 | 复杂同步 |

```java
// volatile: 状态标志
volatile boolean running = true;

// synchronized: 简单互斥
synchronized(lock) { count++; }

// Lock: 灵活控制
lock.lock();
try { count++; }
finally { lock.unlock(); }
```
