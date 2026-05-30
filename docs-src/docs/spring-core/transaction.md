---
title: Spring 事务管理
icon: leaf
order: 3
---

# Spring 事务管理深度解析

事务管理是企业级应用开发的核心，Spring 提供了强大的声明式事务支持。

---

## 一、事务基础

### 1.1 ACID 特性

| 特性 | 说明 | 示例 |
|------|------|------|
| 原子性（Atomicity） | 事务要么全部成功，要么全部失败 | 转账操作 |
| 一致性（Consistency） | 事务执行前后数据保持一致 | 总金额不变 |
| 隔离性（Isolation） | 并发事务互不干扰 | 读写隔离 |
| 持久性（Durability） | 事务提交后数据永久保存 | 断电不丢失 |

### 1.2 并发问题

| 问题 | 描述 | 解决方案 |
|------|------|----------|
| 脏读 | 读到未提交的数据 | 设置隔离级别 |
| 不可重复读 | 同一查询结果不一致 | 设置隔离级别 |
| 幻读 | 查询结果集数量变化 | 设置隔离级别 |

### 1.3 隔离级别

| 级别 | 脏读 | 不可重复读 | 幻读 |
|------|------|-----------|------|
| READ_UNCOMMITTED | √ | √ | √ |
| READ_COMMITTED | × | √ | √ |
| REPEATABLE_READ | × | × | √ |
| SERIALIZABLE | × | × | × |

---

## 二、@Transactional 详解

### 2.1 基本使用

```java
@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Transactional
    public void createUserWithOrder(User user, Order order) {
        userRepository.save(user);
        orderRepository.save(order);
        // 如果这里抛出异常，两个操作都会回滚
    }
}
```

### 2.2 属性详解

```java
@Transactional(
    // 事务传播行为
    propagation = Propagation.REQUIRED,
    
    // 隔离级别
    isolation = Isolation.DEFAULT,
    
    // 超时时间（秒）
    timeout = 30,
    
    // 只读事务
    readOnly = false,
    
    // 异常回滚
    rollbackFor = {Exception.class},
    
    // 异常不回滚
    noRollbackFor = {BusinessException.class},
    
    // 事务管理器
    transactionManager = "transactionManager"
)
public void complexMethod() {
    // 业务逻辑
}
```

### 2.3 传播行为

| 传播行为 | 说明 | 场景 |
|----------|------|------|
| REQUIRED | 默认，加入现有事务或新建 | 大多数场景 |
| REQUIRES_NEW | 新建事务，挂起现有事务 | 日志记录 |
| NESTED | 嵌套事务，保存点机制 | 部分回滚 |
| SUPPORTS | 有事务则加入，无事务则非事务 | 查询方法 |
| NOT_SUPPORTED | 非事务方式执行 | 批量操作 |
| MANDATORY | 必须在事务中执行 | 强制事务 |
| NEVER | 不能在事务中执行 | 特殊场景 |

```java
// 场景 1：主业务需要事务，日志记录独立事务
@Service
public class OrderService {
    
    @Transactional
    public void createOrder(Order order) {
        orderRepository.save(order);
        logService.saveLog("创建订单"); // 使用 REQUIRES_NEW
    }
}

@Service
public class LogService {
    
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void saveLog(String message) {
        logRepository.save(new Log(message));
    }
}

// 场景 2：嵌套事务
@Service
public class BatchService {
    
    @Transactional
    public void batchProcess(List<Item> items) {
        for (Item item : items) {
            try {
                processItem(item); // NESTED，失败不影响其他
            } catch (Exception e) {
                // 单个失败不影响整体
            }
        }
    }
    
    @Transactional(propagation = Propagation.NESTED)
    public void processItem(Item item) {
        itemRepository.save(item);
        // 如果失败，只回滚这个保存点
    }
}
```

---

## 三、事务失效场景

### 3.1 常见失效原因

```java
// ❌ 失效 1：方法不是 public
@Transactional
private void method1() { }

// ❌ 失效 2：自调用（同类方法调用）
@Service
public class UserService {
    public void methodA() {
        this.methodB(); // 事务失效，没有经过代理
    }
    
    @Transactional
    public void methodB() { }
}

// ❌ 失效 3：异常被捕获
@Transactional
public void methodC() {
    try {
        // 业务逻辑
    } catch (Exception e) {
        // 异常被捕获，事务不会回滚
    }
}

//  失效 4：异常类型不匹配
@Transactional(rollbackFor = Exception.class)
public void methodD() {
    throw new RuntimeException(); // 默认只回滚 RuntimeException
}

// ❌ 失效 5：多线程
@Transactional
public void methodE() {
    new Thread(() -> {
        // 新线程不在事务中
    }).start();
}
```

### 3.2 解决方案

```java
// ✅ 解决自调用
@Service
public class UserService {
    
    @Autowired
    private UserService self; // 注入自身代理
    
    public void methodA() {
        self.methodB(); // 通过代理调用，事务生效
    }
    
    @Transactional
    public void methodB() { }
}

// ✅ 解决异常捕获
@Transactional
public void methodC() {
    try {
        // 业务逻辑
    } catch (Exception e) {
        TransactionAspectSupport.currentTransactionStatus().setRollbackOnly();
        throw e;
    }
}

// ✅ 解决异常类型
@Transactional(rollbackFor = Exception.class)
public void methodD() {
    throw new Exception(); // 指定回滚所有异常
}
```

---

## 四、编程式事务

```java
@Service
public class ProgrammaticTransactionService {
    
    @Autowired
    private TransactionTemplate transactionTemplate;
    
    @Autowired
    private PlatformTransactionManager transactionManager;
    
    // 方式 1：TransactionTemplate（推荐）
    public void method1() {
        transactionTemplate.execute(status -> {
            // 业务逻辑
            userRepository.save(user);
            orderRepository.save(order);
            return null;
        });
    }
    
    // 方式 2：PlatformTransactionManager
    public void method2() {
        TransactionDefinition definition = new DefaultTransactionDefinition();
        TransactionStatus status = transactionManager.getTransaction(definition);
        
        try {
            // 业务逻辑
            userRepository.save(user);
            orderRepository.save(order);
            transactionManager.commit(status);
        } catch (Exception e) {
            transactionManager.rollback(status);
            throw e;
        }
    }
}
```

---

## 五、分布式事务

### 5.1 Seata 集成

```java
// 全局事务注解
@GlobalTransactional
public void createOrderWithInventory(Order order) {
    // 调用订单服务
    orderService.create(order);
    
    // 调用库存服务
    inventoryService.deduct(order.getProductId(), order.getQuantity());
    
    // 调用支付服务
    paymentService.pay(order.getUserId(), order.getAmount());
}
```

### 5.2 消息最终一致性

```java
@Service
public class OrderService {
    
    @Transactional
    public void createOrder(Order order) {
        // 1. 保存订单
        orderRepository.save(order);
        
        // 2. 发送消息
        messageProducer.send(new OrderCreatedMessage(order));
    }
}

@Service
public class InventoryService {
    
    @RabbitListener(queues = "order.created")
    @Transactional
    public void handleOrderCreated(OrderCreatedMessage message) {
        // 扣减库存
        inventoryRepository.deduct(message.getProductId(), message.getQuantity());
    }
}
```

---

## 六、实战建议

1. **合理设置超时时间**：避免长事务占用连接
2. **缩小事务范围**：只在必要的方法上加@Transactional
3. **注意传播行为**：REQUIRES_NEW 用于独立事务场景
4. **监控慢事务**：结合 AOP 记录事务执行时间
5. **分布式事务**：优先考虑最终一致性方案

---

> 💡 **提示**：事务管理是 Spring 最强大的特性之一，但滥用会导致性能问题。理解传播行为和失效场景是关键。
