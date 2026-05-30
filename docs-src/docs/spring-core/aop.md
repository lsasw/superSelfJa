---
title: Spring AOP 面向切面编程
icon: leaf
order: 2
---

# Spring AOP 面向切面编程详解

AOP（Aspect-Oriented Programming）面向切面编程，是 Spring 框架的另一大核心特性。

---

## 一、AOP 核心概念

### 1.1 术语解释

| 术语 | 说明 | 示例 |
|------|------|------|
| 切面（Aspect） | 横切关注点的模块化 | 日志切面、事务切面 |
| 连接点（JoinPoint） | 程序执行的某个点 | 方法调用、异常抛出 |
| 通知（Advice） | 在连接点执行的动作 | @Before、@After、@Around |
| 切入点（Pointcut） | 匹配连接点的表达式 | execution(* com.example.service.*.*(..)) |
| 目标对象（Target） | 被代理的对象 | UserServiceImpl |
| 代理（Proxy） | 代理对象 | JDK 动态代理/CGLIB 代理 |

### 1.2 通知类型

```java
@Aspect
@Component
public class LogAspect {
    
    // 前置通知 - 方法执行前
    @Before("execution(* com.example.service.*.*(..))")
    public void before(JoinPoint joinPoint) {
        System.out.println("方法执行前: " + joinPoint.getSignature().getName());
    }
    
    // 后置通知 - 方法执行后（无论是否异常）
    @After("execution(* com.example.service.*.*(..))")
    public void after(JoinPoint joinPoint) {
        System.out.println("方法执行后");
    }
    
    // 返回通知 - 方法正常返回后
    @AfterReturning(pointcut = "execution(* com.example.service.*.*(..))", 
                    returning = "result")
    public void afterReturning(JoinPoint joinPoint, Object result) {
        System.out.println("方法返回值: " + result);
    }
    
    // 异常通知 - 方法抛出异常后
    @AfterThrowing(pointcut = "execution(* com.example.service.*.*(..))", 
                   throwing = "ex")
    public void afterThrowing(JoinPoint joinPoint, Exception ex) {
        System.out.println("方法异常: " + ex.getMessage());
    }
    
    // 环绕通知 - 最强大，可控制方法执行
    @Around("execution(* com.example.service.*.*(..))")
    public Object around(ProceedingJoinPoint joinPoint) throws Throwable {
        long start = System.currentTimeMillis();
        try {
            // 前置处理
            System.out.println("环绕通知 - 前置");
            
            // 执行目标方法
            Object result = joinPoint.proceed();
            
            // 后置处理
            System.out.println("环绕通知 - 后置");
            return result;
        } catch (Throwable e) {
            // 异常处理
            System.out.println("环绕通知 - 异常: " + e.getMessage());
            throw e;
        } finally {
            long cost = System.currentTimeMillis() - start;
            System.out.println("方法耗时: " + cost + "ms");
        }
    }
}
```

---

## 二、切入点表达式

### 2.1 execution 表达式

```java
// 所有 public 方法
@Pointcut("execution(public * *(..))")

// 指定包下所有类的所有方法
@Pointcut("execution(* com.example.service.*.*(..))")

// 指定包及其子包
@Pointcut("execution(* com.example.service..*.*(..))")

// 指定返回类型
@Pointcut("execution(java.util.List com.example.service.*.*(..))")

// 指定参数
@Pointcut("execution(* com.example.service.*.*(String, int))")

// 指定异常类型
@Pointcut("execution(* com.example.service.*.*(..)) throws java.io.IOException")
```

### 2.2 其他切入点指示符

```java
// @annotation - 匹配特定注解
@Pointcut("@annotation(com.example.annotation.Log)")

// @within - 匹配特定注解的类
@Pointcut("@within(org.springframework.stereotype.Service)")

// this - 匹配代理类型
@Pointcut("this(com.example.service.UserService)")

// target - 匹配目标类型
@Pointcut("target(com.example.service.UserService)")

// args - 匹配参数类型
@Pointcut("args(String)")

// bean - 匹配 Bean 名称
@Pointcut("bean(userService)")
```

### 2.3 组合表达式

```java
@Pointcut("execution(* com.example.service.*.*(..)) && @annotation(Log)")
public void logPointcut() {}

@Pointcut("execution(* com.example.service.*.*(..)) || execution(* com.example.controller.*.*(..))")
public void serviceOrController() {}
```

---

## 三、实战：日志记录

### 3.1 自定义注解

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface OperationLog {
    String value() default "";
    String module() default "";
}
```

### 3.2 日志切面

```java
@Aspect
@Component
@Slf4j
public class OperationLogAspect {
    
    @Around("@annotation(operationLog)")
    public Object around(ProceedingJoinPoint joinPoint, OperationLog operationLog) throws Throwable {
        long start = System.currentTimeMillis();
        
        // 获取方法信息
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        
        // 获取参数
        Object[] args = joinPoint.getArgs();
        String params = Arrays.toString(args);
        
        log.info("=== 操作日志开始 ===");
        log.info("模块: {}", operationLog.module());
        log.info("操作: {}", operationLog.value());
        log.info("方法: {}.{}", method.getDeclaringClass().getSimpleName(), method.getName());
        log.info("参数: {}", params);
        
        try {
            // 执行目标方法
            Object result = joinPoint.proceed();
            
            log.info("结果: {}", result);
            return result;
        } catch (Exception e) {
            log.error("异常: {}", e.getMessage(), e);
            throw e;
        } finally {
            long cost = System.currentTimeMillis() - start;
            log.info("耗时: {}ms", cost);
            log.info("=== 操作日志结束 ===");
        }
    }
}
```

### 3.3 使用示例

```java
@Service
public class UserService {
    
    @OperationLog(module = "用户管理", value = "创建用户")
    public User createUser(UserDTO dto) {
        // 业务逻辑
        return user;
    }
    
    @OperationLog(module = "用户管理", value = "删除用户")
    public void deleteUser(Long id) {
        // 业务逻辑
    }
}
```

---

## 四、实战：性能监控

```java
@Aspect
@Component
@Slf4j
public class PerformanceAspect {
    
    @Around("@annotation(PerformanceMonitor)")
    public Object monitor(ProceedingJoinPoint joinPoint) throws Throwable {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        
        try {
            return joinPoint.proceed();
        } finally {
            stopWatch.stop();
            MethodSignature signature = (MethodSignature) joinPoint.getSignature();
            log.warn("性能监控 - 方法: {}, 耗时: {}ms", 
                     signature.getName(), stopWatch.getTotalTimeMillis());
            
            // 性能告警
            if (stopWatch.getTotalTimeMillis() > 1000) {
                log.error("性能告警 - 方法: {} 耗时超过 1 秒", signature.getName());
            }
        }
    }
}
```

---

## 五、实战：接口限流

```java
@Aspect
@Component
public class RateLimitAspect {
    
    private final Map<String, AtomicLong> requestCounts = new ConcurrentHashMap<>();
    private final Map<String, Long> lastResetTime = new ConcurrentHashMap<>();
    
    @Around("@annotation(RateLimit)")
    public Object limit(ProceedingJoinPoint joinPoint, RateLimit rateLimit) throws Throwable {
        String key = getKey(joinPoint);
        long now = System.currentTimeMillis();
        long window = rateLimit.timeWindow() * 1000;
        int maxRequests = rateLimit.maxRequests();
        
        // 重置计数器
        Long lastTime = lastResetTime.get(key);
        if (lastTime == null || now - lastTime > window) {
            requestCounts.put(key, new AtomicLong(0));
            lastResetTime.put(key, now);
        }
        
        // 检查限流
        long count = requestCounts.get(key).incrementAndGet();
        if (count > maxRequests) {
            throw new RateLimitExceededException("请求过于频繁，请稍后再试");
        }
        
        return joinPoint.proceed();
    }
    
    private String getKey(ProceedingJoinPoint joinPoint) {
        // 可以根据用户 ID、IP 等生成 key
        return joinPoint.getSignature().toShortString();
    }
}
```

---

## 六、代理机制

### 6.1 JDK 动态代理 vs CGLIB

| 特性 | JDK 动态代理 | CGLIB |
|------|-------------|-------|
| 原理 | 实现接口 | 继承类 |
| 性能 | 较低 | 较高 |
| 限制 | 必须有接口 | 不能代理 final 类/方法 |
| Spring Boot 2.x 默认 | 否 | 是 |

### 6.2 配置代理方式

```yaml
spring:
  aop:
    proxy-target-class: true  # 强制使用 CGLIB
```

```java
@EnableAspectJAutoProxy(proxyTargetClass = true)
@Configuration
public class AopConfig {
}
```

---

## 七、AOP 执行顺序

```java
@Aspect
@Component
@Order(1)
public class Aspect1 {
    @Around("execution(* com.example.service.*.*(..))")
    public Object around(ProceedingJoinPoint pjp) throws Throwable {
        System.out.println("Aspect1 - Before");
        try {
            Object result = pjp.proceed();
            System.out.println("Aspect1 - After Returning");
            return result;
        } catch (Throwable e) {
            System.out.println("Aspect1 - After Throwing");
            throw e;
        } finally {
            System.out.println("Aspect1 - After");
        }
    }
}

@Aspect
@Component
@Order(2)
public class Aspect2 {
    // 同上
}
```

**执行顺序**：
```
Aspect1 - Before
  Aspect2 - Before
    目标方法执行
  Aspect2 - After Returning
  Aspect2 - After
Aspect1 - After Returning
Aspect1 - After
```

---

## 八、实战建议

1. **优先使用注解方式**：比 XML 配置更简洁
2. **合理使用@Order**：控制多个切面的执行顺序
3. **避免过度使用 AOP**：只在真正需要横切关注点时使用
4. **注意性能影响**：AOP 代理会带来一定性能开销
5. **异常处理**：在@Around 中妥善处理异常

---

> 💡 **提示**：AOP 是 Spring 事务管理的基础，理解 AOP 有助于深入理解 Spring 事务机制。
