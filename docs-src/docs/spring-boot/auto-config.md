---
title: Spring Boot 自动配置原理
icon: boot
order: 1
---

# Spring Boot 自动配置原理

Spring Boot 的核心特性就是自动配置，理解其原理有助于更好地使用和定制 Spring Boot。

---

## 一、自动配置核心机制

### 1.1 @SpringBootApplication 解析

```java
@SpringBootApplication
public class MyApplication {
    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }
}

// @SpringBootApplication 是一个组合注解
@SpringBootConfiguration    // = @Configuration
@EnableAutoConfiguration    // 核心：启用自动配置
@ComponentScan              // 组件扫描
public @interface SpringBootApplication { }
```

### 1.2 自动配置流程

```
1. SpringApplication.run() 启动
2. 加载 META-INF/spring.factories（2.x）或 META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports（3.x）
3. 过滤 @Conditional 条件
4. 注册符合条件的配置类
5. 创建 Bean 实例
```

---

## 二、@Conditional 条件注解

### 2.1 常用条件注解

| 注解 | 说明 |
|------|------|
| `@ConditionalOnClass` | 类路径存在指定类时生效 |
| `@ConditionalOnMissingBean` | 容器中不存在指定 Bean 时生效 |
| `@ConditionalOnProperty` | 配置文件存在指定属性时生效 |
| `@ConditionalOnWebApplication` | Web 应用时生效 |
| `@ConditionalOnExpression` | SpEL 表达式为 true 时生效 |

### 2.2 自动配置类示例

```java
@Configuration
@ConditionalOnClass(DataSource.class)
@EnableConfigurationProperties(DataSourceProperties.class)
public class DataSourceAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public DataSource dataSource(DataSourceProperties properties) {
        return DataSourceBuilder.create()
            .url(properties.getUrl())
            .username(properties.getUsername())
            .password(properties.getPassword())
            .build();
    }
}

@ConfigurationProperties(prefix = "spring.datasource")
@Data
public class DataSourceProperties {
    private String url;
    private String username;
    private String password;
    private String driverClassName;
}
```

---

## 三、Starter 开发

### 3.1 自定义 Starter 结构

```
mybatis-spring-boot-starter/
├── pom.xml
└── src/main/
    ├── java/
    │   └── com/example/mybatis/
    │       ├── MybatisAutoConfiguration.java
    │       └── MybatisProperties.java
    └── resources/
        └── META-INF/
            ── spring/
                └── org.springframework.boot.autoconfigure.AutoConfiguration.imports
```

### 3.2 实现步骤

```java
// 1. 配置属性类
@ConfigurationProperties(prefix = "myapp.redis")
@Data
public class RedisProperties {
    private String host = "localhost";
    private int port = 6379;
    private String password;
    private int timeout = 3000;
}

// 2. 自动配置类
@Configuration
@ConditionalOnClass(RedisTemplate.class)
@EnableConfigurationProperties(RedisProperties.class)
public class RedisAutoConfiguration {
    
    @Bean
    @ConditionalOnMissingBean
    public RedisTemplate<String, Object> redisTemplate(
            RedisProperties properties,
            RedisConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        return template;
    }
}

// 3. 注册自动配置
// META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
com.example.redis.RedisAutoConfiguration
```

---

## 四、配置优先级

```
1. 命令行参数
2. SPRING_APPLICATION_JSON 环境变量
3. ServletConfig 初始化参数
4. ServletContext 初始化参数
5. JNDI 属性
6. Java System 属性
7. 操作系统环境变量
8. RandomValuePropertySource
9. jar 包外 application-{profile}.properties/yml
10. jar 包内 application-{profile}.properties/yml
11. jar 包外 application.properties/yml
12. jar 包内 application.properties/yml
13. @PropertySource
14. 默认属性
```

---

## 五、Profile 配置

```yaml
# application.yml
spring:
  profiles:
    active: @spring.profiles.active@

---
# application-dev.yml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/dev
    username: root
    password: dev123

---
# application-prod.yml
spring:
  datasource:
    url: jdbc:mysql://prod-server:3306/prod
    username: prod_user
    password: ${DB_PASSWORD}
```

---

## 六、实战建议

1. **合理使用@Conditional**：避免不必要的 Bean 创建
2. **Starter 命名规范**：官方 `spring-boot-starter-*`，自定义 `*-spring-boot-starter`
3. **配置属性**：使用@ConfigurationProperties 统一管理
4. **文档完善**：为 Starter 提供清晰的配置说明
5. **测试覆盖**：编写 AutoConfiguration 的单元测试

---

> 💡 **提示**：自动配置是 Spring Boot 的灵魂，理解条件注解和配置优先级，才能灵活运用。
