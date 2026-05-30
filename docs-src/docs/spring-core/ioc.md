---
title: Spring IoC 容器详解
icon: leaf
order: 1
---

# Spring IoC 容器深度解析

IoC（控制反转）是 Spring 框架的核心，理解 IoC 容器是掌握 Spring 的关键。

---

## 一、IoC 核心概念

### 1.1 什么是 IoC

IoC（Inversion of Control）控制反转，是一种设计思想，将对象的创建和依赖关系的管理交给框架处理。

**传统方式 vs IoC 方式**

```java
// 传统方式 - 主动创建依赖
public class UserService {
    private UserRepository repository = new UserRepositoryImpl();
}

// IoC 方式 - 依赖注入
public class UserService {
    @Autowired
    private UserRepository repository;
}
```

### 1.2 DI（依赖注入）

DI 是 IoC 的具体实现方式，主要有三种：

| 注入方式 | 优点 | 缺点 |
|----------|------|------|
| 构造器注入 | 保证不可变性，依赖明确 | 参数多时代码冗长 |
| Setter 注入 | 灵活，可选依赖 | 对象可能处于不完整状态 |
| 字段注入 | 简洁 | 不利于测试，隐藏依赖 |

```java
// 构造器注入（推荐）
@Service
public class UserService {
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
}

// Setter 注入
@Service
public class UserService {
    private UserRepository repository;
    
    @Autowired
    public void setRepository(UserRepository repository) {
        this.repository = repository;
    }
}

// 字段注入
@Service
public class UserService {
    @Autowired
    private UserRepository repository;
}
```

---

## 二、Bean 的生命周期

### 2.1 完整生命周期

```
实例化 → 属性赋值 → 初始化 → 使用 → 销毁
```

### 2.2 生命周期回调

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {
    
    // 1. 构造方法
    public MyBean() {
        System.out.println("1. 构造方法执行");
    }
    
    // 2. @Autowired 注入
    @Autowired
    private DataSource dataSource;
    
    // 3. @PostConstruct
    @PostConstruct
    public void init() {
        System.out.println("3. @PostConstruct 执行");
    }
    
    // 4. InitializingBean
    @Override
    public void afterPropertiesSet() {
        System.out.println("4. afterPropertiesSet 执行");
    }
    
    // 5. 自定义 init-method
    public void customInit() {
        System.out.println("5. customInit 执行");
    }
    
    // 6. @PreDestroy
    @PreDestroy
    public void destroy() {
        System.out.println("6. @PreDestroy 执行");
    }
    
    // 7. DisposableBean
    @Override
    public void destroy() throws Exception {
        System.out.println("7. destroy 执行");
    }
}
```

### 2.3 BeanPostProcessor

```java
@Component
public class CustomBeanPostProcessor implements BeanPostProcessor {
    
    @Override
    public Object postProcessBeforeInitialization(Object bean, String beanName) {
        System.out.println("初始化前: " + beanName);
        return bean;
    }
    
    @Override
    public Object postProcessAfterInitialization(Object bean, String beanName) {
        System.out.println("初始化后: " + beanName);
        return bean;
    }
}
```

---

## 三、Bean 的作用域

| 作用域 | 说明 | 适用场景 |
|--------|------|----------|
| singleton | 单例（默认） | 无状态服务 |
| prototype | 每次创建新实例 | 有状态对象 |
| request | 每个 HTTP 请求 | Web 应用 |
| session | 每个 HTTP Session | 用户会话数据 |
| application | ServletContext 级别 | 全局共享数据 |

```java
@Scope("prototype")
@Component
public class PrototypeBean {
    // 每次获取都会创建新实例
}

@Scope(value = "request", proxyMode = ScopedProxyMode.TARGET_CLASS)
@Component
public class RequestBean {
    // 每个请求一个实例
}
```

---

## 四、@Configuration 与@Bean

### 4.1 Java Config 方式

```java
@Configuration
public class AppConfig {
    
    @Bean
    public DataSource dataSource() {
        HikariDataSource ds = new HikariDataSource();
        ds.setJdbcUrl("jdbc:mysql://localhost:3306/test");
        ds.setUsername("root");
        ds.setPassword("123456");
        return ds;
    }
    
    @Bean
    public SqlSessionFactory sqlSessionFactory(DataSource dataSource) {
        SqlSessionFactoryBean factory = new SqlSessionFactoryBean();
        factory.setDataSource(dataSource);
        return factory.getObject();
    }
}
```

### 4.2 条件化 Bean

```java
@Configuration
public class DataSourceConfig {
    
    @Bean
    @ConditionalOnProperty(name = "app.datasource.type", havingValue = "mysql")
    public DataSource mysqlDataSource() {
        return new MysqlDataSource();
    }
    
    @Bean
    @ConditionalOnProperty(name = "app.datasource.type", havingValue = "postgresql")
    public DataSource postgresqlDataSource() {
        return new PostgresqlDataSource();
    }
}
```

---

## 五、循环依赖问题

### 5.1 三级缓存机制

```
singletonObjects          → 一级缓存（成品）
earlySingletonObjects     → 二级缓存（半成品）
singletonFactories        → 三级缓存（ObjectFactory）
```

### 5.2 解决方案

```java
// 方案 1：@Lazy 延迟加载
@Service
public class AService {
    @Lazy
    @Autowired
    private BService bService;
}

// 方案 2：Setter 注入
@Service
public class AService {
    private BService bService;
    
    @Autowired
    public void setBService(BService bService) {
        this.bService = bService;
    }
}

// 方案 3：重构设计（推荐）
@Service
public class CService {
    @Autowired
    private DService dService;
}

@Service
public class AService {
    @Autowired
    private CService cService;
}

@Service
public class BService {
    @Autowired
    private CService cService;
}
```

---

## 六、Spring Boot 自动装配原理

### 6.1 @SpringBootApplication

```java
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan
public class MyApplication {
    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }
}
```

### 6.2 自动装配流程

```
1. 读取 META-INF/spring.factories
2. 加载 EnableAutoConfiguration 配置的类
3. 根据 @Conditional 条件判断是否创建 Bean
4. 将符合条件的 Bean 注册到容器
```

```java
@Configuration
@ConditionalOnClass(DataSource.class)
@EnableConfigurationProperties(DataSourceProperties.class)
public class DataSourceAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public DataSource dataSource(DataSourceProperties properties) {
        // 创建数据源
    }
}
```

---

## 七、实战建议

1. **优先使用构造器注入**：保证依赖不可变
2. **避免循环依赖**：重构设计优于使用@Lazy
3. **合理使用作用域**：默认 singleton，需要时再改
4. **利用条件注解**：@ConditionalOnProperty 等简化配置
5. **理解生命周期**：@PostConstruct 比 InitializingBean 更简洁

---

> 💡 **提示**：IoC 容器是 Spring 的基石，理解 Bean 的生命周期和作用域，是进阶 Spring 开发的关键。
