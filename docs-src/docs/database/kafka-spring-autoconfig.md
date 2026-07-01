---
title: Kafka 核心机制与 Spring 自动配置
icon: stream
order: 3
category:
  - 数据库
tag:
  - Kafka
  - Spring Boot
---

# Kafka 核心机制与 Spring 自动配置

> 任务编号：KF-01 / KF-02 / SP-01

## 一、Kafka Rebalance 机制（KF-01）

```
Consumer Group 触发 Rebalance 的场景：
1. Consumer 加入/离开 Group
2. Topic 分区数变化
3. Consumer 心跳超时（session.timeout.ms）

Rebalance 过程：
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Consumer 1 │    │   Consumer 2 │    │   Consumer 3 │
│   Partition 0│    │   Partition 1│    │   Partition 2│
└──────────────┘    └──────────────┘    └──────────────┘

Consumer 2 宕机 →
┌──────────────────────┐    ┌──────────────────────┐
│     Consumer 1       │    │     Consumer 3       │
│  Partition 0 + 1     │    │     Partition 2      │
└──────────────────────┘    └──────────────────────┘
```

### 关键配置

```properties
# 心跳间隔（与 GroupCoordinator 保持心跳）
heartbeat.interval.ms=3000

# 超时时间（超时后认为 Consumer 已死，触发 Rebalance）
session.timeout.ms=45000

# 每次 poll 的最大间隔（超时触发 Rebalance）
max.poll.interval.ms=300000

# 分区分配策略
partition.assignment.strategy=org.apache.kafka.clients.consumer.RoundRobinAssignor
```

## 二、Kafka 消息可靠性（KF-02）

### 生产端

```java
Properties props = new Properties();
props.put("acks", "all");              // 所有 ISR 确认
props.put("retries", Integer.MAX_VALUE); // 无限重试
props.put("enable.idempotence", true);   // 幂等生产
props.put("transactional.id", "tx-1");   // 事务

KafkaProducer<String, String> producer = new KafkaProducer<>(props);

// 事务消息
producer.initTransactions();
try {
    producer.beginTransaction();
    producer.send(new ProducerRecord<>("topic1", "msg1"));
    producer.send(new ProducerRecord<>("topic2", "msg2"));
    producer.commitTransaction();
} catch (Exception e) {
    producer.abortTransaction();
}
```

### 消费端

```java
props.put("enable.auto.commit", false);  // 手动提交
props.put("isolation.level", "read_committed"); // 只读已提交

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        try {
            processMessage(record);
        } catch (Exception e) {
            // 发送到死信队列
            sendToDLQ(record);
        }
    }
    // 处理完一批后手动提交
    consumer.commitSync();
}
```

### 可靠性保障矩阵

| 场景 | 方案 |
|------|------|
| 消息不丢失 | acks=all + 手动提交 + retries |
| 消息不重复 | 幂等生产 + 消费端幂等（去重表） |
| 事务一致性 | 事务消息（跨 Topic 原子写入） |
| 死信处理 | 失败消息 → DLQ Topic → 监控告警 |

## 三、Spring Boot 自动配置原理（SP-01）

### 调用链

```
@SpringBootApplication
    └── @EnableAutoConfiguration
         └── @Import(AutoConfigurationImportSelector.class)
              └── selectImports()
                   └── getAutoConfigurationEntry()
                        └── getCandidateConfigurations()
                             └── SpringFactoriesLoader.loadFactoryNames()
                                  └── 读取 META-INF/spring.factories
                                       └── 或 META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
```

### 源码关键路径

```java
// 1. 入口
@SpringBootApplication  // → 包含 @EnableAutoConfiguration

// 2. 核心选择器
public class AutoConfigurationImportSelector {
    
    protected List<String> getCandidateConfigurations(...) {
        // 读取 spring.factories 中 EnableAutoConfiguration 的所有配置类
        return SpringFactoriesLoader.loadFactoryNames(
            EnableAutoConfiguration.class, 
            getBeanClassLoader()
        );
    }
}

// 3. 条件过滤
// 例如 DataSourceAutoConfiguration：
@AutoConfiguration
@ConditionalOnClass({DataSource.class, EmbeddedDatabaseType.class})
@EnableConfigurationProperties(DataSourceProperties.class)
public class DataSourceAutoConfiguration {
    // 只有 classpath 有 DataSource 且没有手动配置时才生效
}
```

### 常用条件注解

| 注解 | 作用 |
|------|------|
| `@ConditionalOnClass` | 类存在时生效 |
| `@ConditionalOnMissingBean` | Bean 不存在时生效 |
| `@ConditionalOnProperty` | 配置项为指定值时生效 |
| `@ConditionalOnBean` | 指定 Bean 存在时生效 |
| `@ConditionalOnMissingClass` | 类不存在时生效 |
