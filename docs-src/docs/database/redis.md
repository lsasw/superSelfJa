---
title: Redis 缓存实战
icon: database
order: 3
---

# Redis 缓存实战指南

Redis 是最流行的内存数据库，广泛应用于缓存、会话存储、分布式锁等场景。

---

## 一、Redis 核心数据结构

### 1.1 数据类型

| 类型 | 说明 | 典型场景 |
|------|------|----------|
| String | 字符串 | 缓存、计数器 |
| Hash | 哈希表 | 对象存储 |
| List | 列表 | 消息队列 |
| Set | 集合 | 去重、交集 |
| ZSet | 有序集合 | 排行榜 |
| Bitmap | 位图 | 签到、活跃用户 |
| HyperLogLog | 基数统计 | UV 统计 |
| Geo | 地理位置 | 附近的人 |

### 1.2 基本操作

```bash
# String
SET user:1001 '{"name":"张三","age":28}'
GET user:1001
INCR visit:count
EXPIRE user:1001 3600

# Hash
HSET user:1001 name "张三" age 28 email "zhangsan@example.com"
HGET user:1001 name
HGETALL user:1001

# List
LPUSH queue:tasks "task1" "task2" "task3"
RPOP queue:tasks
LRANGE queue:tasks 0 -1

# Set
SADD tags:article:1 "Java" "Spring" "Redis"
SISMEMBER tags:article:1 "Java"
SINTER tags:article:1 tags:article:2

# ZSet
ZADD leaderboard 100 "user1" 200 "user2" 150 "user3"
ZREVRANGE leaderboard 0 9 WITHSCORES
```

---

## 二、Spring Boot 集成

### 2.1 依赖配置

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>

<!-- 连接池 -->
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-pool2</artifactId>
</dependency>

<!-- JSON 序列化 -->
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
</dependency>
```

### 2.2 配置文件

```yaml
spring:
  data:
    redis:
      host: localhost
      port: 6379
      password: 123456
      database: 0
      timeout: 5000ms
      lettuce:
        pool:
          max-active: 20
          max-idle: 10
          min-idle: 5
          max-wait: 3000ms
```

### 2.3 RedisTemplate 配置

```java
@Configuration
public class RedisConfig {
    
    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory factory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(factory);
        
        // Key 序列化
        template.setKeySerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        
        // Value 序列化
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        template.setHashValueSerializer(new GenericJackson2JsonRedisSerializer());
        
        template.afterPropertiesSet();
        return template;
    }
    
    @Bean
    public StringRedisTemplate stringRedisTemplate(RedisConnectionFactory factory) {
        return new StringRedisTemplate(factory);
    }
}
```

---

## 三、缓存实战

### 3.1 缓存注解

```java
@Service
public class UserService {
    
    @Cacheable(value = "users", key = "#id")
    public User getUserById(Long id) {
        // 先从缓存查，没有再查数据库
        return userRepository.findById(id);
    }
    
    @CachePut(value = "users", key = "#user.id")
    public User updateUser(User user) {
        // 先执行方法，再更新缓存
        return userRepository.save(user);
    }
    
    @CacheEvict(value = "users", key = "#id")
    public void deleteUser(Long id) {
        // 删除缓存
        userRepository.deleteById(id);
    }
    
    @Caching(
        evict = {
            @CacheEvict(value = "users", key = "#id"),
            @CacheEvict(value = "userList", allEntries = true)
        }
    )
    public void deleteUserWithList(Long id) {
        userRepository.deleteById(id);
    }
}
```

### 3.2 缓存穿透

```java
@Service
public class ProductService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public Product getProductById(Long id) {
        String key = "product:" + id;
        
        // 1. 查缓存
        Product product = (Product) redisTemplate.opsForValue().get(key);
        if (product != null) {
            return product;
        }
        
        // 2. 查数据库
        product = productRepository.findById(id).orElse(null);
        
        // 3. 写入缓存（防止穿透：空值也缓存）
        if (product != null) {
            redisTemplate.opsForValue().set(key, product, 30, TimeUnit.MINUTES);
        } else {
            // 缓存空值，防止恶意查询
            redisTemplate.opsForValue().set(key, new Product(), 5, TimeUnit.MINUTES);
        }
        
        return product;
    }
}
```

### 3.3 缓存击穿

```java
// 方案 1：互斥锁
public Product getProductWithLock(Long id) {
    String key = "product:" + id;
    String lockKey = "lock:product:" + id;
    
    Product product = (Product) redisTemplate.opsForValue().get(key);
    if (product != null) {
        return product;
    }
    
    // 尝试获取锁
    Boolean locked = redisTemplate.opsForValue().setIfAbsent(lockKey, "1", 10, TimeUnit.SECONDS);
    if (Boolean.TRUE.equals(locked)) {
        try {
            // 双重检查
            product = (Product) redisTemplate.opsForValue().get(key);
            if (product != null) {
                return product;
            }
            
            // 查数据库并缓存
            product = productRepository.findById(id).orElse(null);
            redisTemplate.opsForValue().set(key, product, 30, TimeUnit.MINUTES);
            return product;
        } finally {
            redisTemplate.delete(lockKey);
        }
    } else {
        // 等待重试
        try { Thread.sleep(100); } catch (InterruptedException e) {}
        return getProductWithLock(id);
    }
}

// 方案 2：逻辑过期
@Data
public class CacheObject<T> {
    private T data;
    private LocalDateTime expireTime;
    private LocalDateTime refreshTime;
}
```

### 3.4 缓存雪崩

```yaml
# 设置不同的过期时间，避免同时失效
spring:
  cache:
    redis:
      time-to-live: ${REDIS_TTL:1800000}  # 30 分钟 + 随机偏移
```

```java
// 随机过期时间
public void setWithRandomExpire(String key, Object value) {
    int baseExpire = 1800; // 30 分钟
    int randomOffset = new Random().nextInt(300); // 0-5 分钟随机
    redisTemplate.opsForValue().set(key, value, baseExpire + randomOffset, TimeUnit.SECONDS);
}
```

---

## 四、分布式锁

### 4.1 Redisson 实现

```xml
<dependency>
    <groupId>org.redisson</groupId>
    <artifactId>redisson-spring-boot-starter</artifactId>
    <version>3.23.5</version>
</dependency>
```

```java
@Service
public class InventoryService {
    
    @Autowired
    private RedissonClient redissonClient;
    
    public void deductStock(String productId, int quantity) {
        RLock lock = redissonClient.getLock("stock:lock:" + productId);
        
        try {
            // 尝试获取锁，最多等待 10 秒，锁自动释放时间 30 秒
            if (lock.tryLock(10, 30, TimeUnit.SECONDS)) {
                try {
                    // 业务逻辑
                    int stock = getStock(productId);
                    if (stock >= quantity) {
                        updateStock(productId, stock - quantity);
                    } else {
                        throw new RuntimeException("库存不足");
                    }
                } finally {
                    lock.unlock();
                }
            } else {
                throw new RuntimeException("获取锁失败");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("获取锁被中断");
        }
    }
}
```

---

## 五、实战场景

### 5.1 排行榜

```java
@Service
public class LeaderboardService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    // 添加分数
    public void addScore(String gameId, String userId, double score) {
        redisTemplate.opsForZSet().add("leaderboard:" + gameId, userId, score);
    }
    
    // 获取 Top N
    public Set<ZSetOperations.TypedTuple<Object>> getTopN(String gameId, int n) {
        return redisTemplate.opsForZSet().reverseRangeWithScores(
            "leaderboard:" + gameId, 0, n - 1);
    }
    
    // 获取用户排名
    public Long getRank(String gameId, String userId) {
        Long rank = redisTemplate.opsForZSet().reverseRank("leaderboard:" + gameId, userId);
        return rank != null ? rank + 1 : null;
    }
}
```

### 5.2 限流器

```java
@Service
public class RateLimiter {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public boolean tryAcquire(String key, int limit, int windowSeconds) {
        String script = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local current = redis.call('GET', key)
            if current and tonumber(current) >= limit then
                return 0
            end
            redis.call('INCR', key)
            redis.call('EXPIRE', key, window)
            return 1
            """;
        
        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(script, Long.class),
            Collections.singletonList(key),
            String.valueOf(limit),
            String.valueOf(windowSeconds)
        );
        
        return result != null && result == 1;
    }
}
```

---

## 六、实战建议

1. **缓存设计**：先查缓存，再查数据库，最后写缓存
2. **过期策略**：设置合理的过期时间，避免雪崩
3. **序列化**：使用 JSON 序列化，便于调试
4. **监控**：监控缓存命中率、内存使用情况
5. **降级策略**：Redis 宕机时要有降级方案

---

> 💡 **提示**：缓存是双刃剑，用得好提升性能，用不好引发灾难。理解缓存三兄弟（穿透、击穿、雪崩）是必修课。
