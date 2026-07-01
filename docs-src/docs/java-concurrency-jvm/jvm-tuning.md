---
title: JVM 调优实战
icon: wrench
order: 3
category:
  - Java
tag:
  - JVM
  - GC
  - OOM
---

# JVM 调优实战

> 任务编号：JV-01 / JV-02 / JV-03

## 一、OOM 三件套（JV-01）

### 1. 堆溢出

```java
// -Xmx256m -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/heap.hprof
public class HeapOOM {
    public static void main(String[] args) {
        List<byte[]> list = new ArrayList<>();
        while (true) {
            list.add(new byte[10 * 1024 * 1024]); // 每次 10MB
        }
    }
}
// java.lang.OutOfMemoryError: Java heap space
```

### 2. 元空间溢出

```java
// -XX:MaxMetaspaceSize=10m
public class MetaspaceOOM {
    public static void main(String[] args) {
        while (true) {
            // 动态生成类，填满元空间
            Enhancer enhancer = new Enhancer();
            enhancer.setSuperclass(OOMObject.class);
            enhancer.setUseCache(false);
            enhancer.setCallback((MethodInterceptor) (obj, method, args1, proxy) ->
                proxy.invokeSuper(obj, args1));
            enhancer.create();
        }
    }
}
// java.lang.OutOfMemoryError: Metaspace
```

### 3. 直接内存溢出

```java
// -XX:MaxDirectMemorySize=10m
public class DirectOOM {
    public static void main(String[] args) {
        List<ByteBuffer> buffers = new ArrayList<>();
        while (true) {
            buffers.add(ByteBuffer.allocateDirect(1024 * 1024)); // 1MB
        }
    }
}
// java.lang.OutOfMemoryError: Direct buffer memory
```

### MAT 分析四步法

```
1. 打开 .hprof 文件
2. Leak Suspects → 查看可疑的内存泄漏报告
3. Dominator Tree → 按保留大小排序，找最大的对象
4. Path to GC Roots → 查看为什么这个对象不能被 GC
```

## 二、CMS vs G1（JV-02）

### CMS 工作阶段

```
Initial Mark (STW)     → 标记 GC Roots 直接关联对象
   ↓
Concurrent Mark        → 并发追踪对象图
   ↓
Remark (STW)           → 修正并发标记期间的变更
   ↓
Concurrent Sweep       → 并发清理
```

### G1 工作阶段

```
Young GC               → 回收 Eden + Survivor
   ↓
Initial Mark (STW)     → 伴随 Young GC
   ↓
Concurrent Marking     → 并发标记
   ↓
Remark (STW)           → 最终标记
   ↓
Cleanup (STW)          → 计算回收价值
   ↓
Mixed GC               → 回收年轻代 + 部分老年代 Region
```

### 核心区别

| 特性 | CMS | G1 |
|------|:---:|:---:|
| 内存布局 | 连续分代 | Region 化（1-32MB） |
| 回收算法 | 标记-清除 | 标记-整理（不产生碎片） |
| 停顿预测 | ❌ 不可控 | ✅ 可设 MaxGCPauseMillis |
| 并发阶段 | 占用 CPU 较高 | 更智能的并发控制 |
| 大对象 | 直接进老年代 | Humongous Region |
| 推荐场景 | JDK 8 遗留系统 | **现代应用首选** |

## 三、G1 调优参数实践（JV-03）

```bash
# G1 基础参数
-XX:+UseG1GC
-Xms4g -Xmx4g                       # 堆大小固定，避免动态调整
-XX:MaxGCPauseMillis=200            # 期望最大停顿 200ms

# G1 进阶参数
-XX:G1HeapRegionSize=4m             # Region 大小（1/2/4/8/16/32M）
-XX:InitiatingHeapOccupancyPercent=45  # 老年代占 45% 时触发 Mixed GC
-XX:G1ReservePercent=10             # 保留 10% 空间防止晋升失败

# 日志
-Xlog:gc*:file=/tmp/gc.log:time,level,tags

# 核心指标
-XX:+PrintAdaptiveSizePolicy        # 打印自适应策略详情
```

### G1 核心监控指标

```bash
# 关键 GC 指标
jstat -gcutil <pid> 1000

  S0     S1     E      O      M     CCS    YGC     YGCT    FGC    FGCT     GCT
  0.00  98.21  45.67  23.45  89.12  85.33   1234   12.345    0     0.000   12.345

  E (Eden 使用率) → 过高触发频繁 Young GC
  O (Old 使用率)   → 接近 IHOP 时触发 Mixed GC
  FGC (Full GC 次数) → >0 说明有严重性能问题
```

> 核心原则：**即使 Full GC 次数为 0 也不代表性能好**。关注的是停顿时间和吞吐量。
