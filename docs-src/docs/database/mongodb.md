---
title: MongoDB 实战指南
icon: database
order: 2
---

# MongoDB 实战指南

MongoDB 是最流行的文档型数据库，适合存储半结构化和动态数据。

---

## 一、MongoDB 核心概念

### 1.1 基本概念对比

| MySQL | MongoDB | 说明 |
|-------|---------|------|
| 数据库 | 数据库 | 数据容器 |
| 表 | 集合（Collection） | 文档集合 |
| 行 | 文档（Document） | BSON 格式 |
| 列 | 字段（Field） | 键值对 |
| 索引 | 索引 | 加速查询 |

### 1.2 BSON 文档示例

```json
{
  "_id": ObjectId("507f191e810c19729de860ea"),
  "username": "zhangsan",
  "email": "zhangsan@example.com",
  "age": 28,
  "isActive": true,
  "tags": ["developer", "java", "spring"],
  "address": {
    "city": "北京",
    "district": "朝阳区",
    "street": "建国路 88 号"
  },
  "orders": [
    { "orderId": "ORD001", "amount": 299.00, "date": ISODate("2024-01-15") },
    { "orderId": "ORD002", "amount": 159.00, "date": ISODate("2024-02-20") }
  ],
  "createdAt": ISODate("2024-01-01T00:00:00Z"),
  "updatedAt": ISODate("2024-03-01T12:30:00Z")
}
```

---

## 二、基本操作

### 2.1 CRUD 操作

```javascript
// 创建
db.users.insertOne({
  username: "zhangsan",
  email: "zhangsan@example.com",
  age: 28
});

db.users.insertMany([
  { username: "lisi", email: "lisi@example.com", age: 25 },
  { username: "wangwu", email: "wangwu@example.com", age: 30 }
]);

// 查询
db.users.find({ age: { $gt: 25 } });
db.users.findOne({ username: "zhangsan" });

// 更新
db.users.updateOne(
  { username: "zhangsan" },
  { $set: { age: 29, email: "new@example.com" } }
);

db.users.updateMany(
  { age: { $lt: 26 } },
  { $inc: { age: 1 } }
);

// 删除
db.users.deleteOne({ username: "zhangsan" });
db.users.deleteMany({ age: { $lt: 18 } });
```

### 2.2 查询操作符

| 操作符 | 说明 | 示例 |
|--------|------|------|
| `$eq` | 等于 | `{age: {$eq: 25}}` |
| `$ne` | 不等于 | `{age: {$ne: 25}}` |
| `$gt` | 大于 | `{age: {$gt: 25}}` |
| `$gte` | 大于等于 | `{age: {$gte: 25}}` |
| `$lt` | 小于 | `{age: {$lt: 25}}` |
| `$lte` | 小于等于 | `{age: {$lte: 25}}` |
| `$in` | 包含 | `{age: {$in: [25, 30]}}` |
| `$nin` | 不包含 | `{age: {$nin: [25, 30]}}` |
| `$regex` | 正则 | `{username: {$regex: /^zhang/}}` |

---

## 三、Spring Data MongoDB

### 3.1 配置

```yaml
spring:
  data:
    mongodb:
      uri: mongodb://localhost:27017/mydb
      # 或者分开配置
      host: localhost
      port: 27017
      database: mydb
      username: admin
      password: 123456
```

### 3.2 实体类

```java
@Document(collection = "users")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {
    
    @Id
    private String id;
    
    @Indexed(unique = true)
    private String username;
    
    private String email;
    
    private Integer age;
    
    @Indexed
    private Boolean isActive;
    
    private List<String> tags;
    
    private Address address;
    
    private List<Order> orders;
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    private LocalDateTime updatedAt;
    
    @Data
    @Builder
    public static class Address {
        private String city;
        private String district;
        private String street;
    }
    
    @Data
    @Builder
    public static class Order {
        private String orderId;
        private Double amount;
        private LocalDateTime date;
    }
}
```

### 3.3 Repository 操作

```java
public interface UserRepository extends MongoRepository<User, String> {
    
    // 方法名派生查询
    List<User> findByUsername(String username);
    List<User> findByAgeGreaterThan(Integer age);
    List<User> findByEmailContaining(String keyword);
    List<User> findByTagsContaining(String tag);
    List<User> findByAddressCity(String city);
    
    // 组合条件
    List<User> findByAgeBetweenAndIsActive(Integer minAge, Integer maxAge, Boolean isActive);
    
    // 分页排序
    Page<User> findByIsActive(Boolean isActive, Pageable pageable);
    
    // @Query 注解
    @Query("{ 'age': { $gt: ?0 } }")
    List<User> findUsersOlderThan(Integer age);
    
    @Query(value = "{ 'username': ?0 }", fields = "{ 'email': 1, 'age': 1 }")
    User findEmailAndAgeByUsername(String username);
}
```

### 3.4 MongoTemplate 操作

```java
@Service
public class UserService {
    
    @Autowired
    private MongoTemplate mongoTemplate;
    
    // 复杂查询
    public List<User> searchUsers(SearchRequest request) {
        Query query = new Query();
        
        // 条件组合
        Criteria criteria = new Criteria();
        if (request.getMinAge() != null) {
            criteria.and("age").gte(request.getMinAge());
        }
        if (request.getMaxAge() != null) {
            criteria.and("age").lte(request.getMaxAge());
        }
        if (request.getCity() != null) {
            criteria.and("address.city").is(request.getCity());
        }
        if (request.getKeyword() != null) {
            criteria.orOperator(
                Criteria.where("username").regex(request.getKeyword(), "i"),
                Criteria.where("email").regex(request.getKeyword(), "i")
            );
        }
        
        query.addCriteria(criteria);
        
        // 排序分页
        query.with(Sort.by(Sort.Direction.DESC, "createdAt"));
        query.skip((request.getPage() - 1) * request.getSize());
        query.limit(request.getSize());
        
        return mongoTemplate.find(query, User.class);
    }
    
    // 聚合查询
    public List<CityStat> getUserStatsByCity() {
        Aggregation aggregation = Aggregation.newAggregation(
            Aggregation.group("address.city")
                .count().as("userCount")
                .avg("age").as("avgAge"),
            Aggregation.sort(Sort.Direction.DESC, "userCount"),
            Aggregation.limit(10)
        );
        
        return mongoTemplate.aggregate(aggregation, "users", CityStat.class)
            .getMappedResults();
    }
    
    // 更新操作
    public void updateUserTags(String userId, List<String> newTags) {
        Update update = new Update();
        update.addToSet("tags").each(newTags.toArray());
        
        mongoTemplate.updateFirst(
            Query.query(Criteria.where("id").is(userId)),
            update,
            User.class
        );
    }
}
```

---

## 四、索引优化

### 4.1 索引类型

```javascript
// 单字段索引
db.users.createIndex({ username: 1 });

// 复合索引
db.users.createIndex({ city: 1, age: -1 });

// 唯一索引
db.users.createIndex({ email: 1 }, { unique: true });

//  TTL 索引（自动过期）
db.sessions.createIndex({ createdAt: 1 }, { expireAfterSeconds: 3600 });

// 文本索引
db.articles.createIndex({ title: "text", content: "text" });

// 地理空间索引
db.locations.createIndex({ location: "2dsphere" });
```

### 4.2 索引最佳实践

```javascript
// 查看索引使用情况
db.users.find({ username: "zhangsan" }).explain("executionStats");

// 查看索引大小
db.users.totalIndexSize();

// 删除未使用的索引
db.users.dropIndex("username_1");
```

---

## 五、聚合管道

### 5.1 基本聚合

```javascript
db.orders.aggregate([
  // 匹配阶段
  { $match: { status: "completed" } },
  
  // 分组阶段
  { $group: {
      _id: "$userId",
      totalAmount: { $sum: "$amount" },
      orderCount: { $count: {} },
      avgAmount: { $avg: "$amount" }
  }},
  
  // 排序阶段
  { $sort: { totalAmount: -1 } },
  
  // 限制阶段
  { $limit: 10 }
]);
```

### 5.2 Java 聚合

```java
Aggregation aggregation = Aggregation.newAggregation(
    Aggregation.match(Criteria.where("status").is("completed")),
    Aggregation.group("userId")
        .sum("amount").as("totalAmount")
        .count().as("orderCount")
        .avg("amount").as("avgAmount"),
    Aggregation.sort(Sort.Direction.DESC, "totalAmount"),
    Aggregation.limit(10)
);

AggregationResults<OrderStats> results = mongoTemplate.aggregate(
    aggregation, "orders", OrderStats.class
);
```

---

## 六、实战场景

### 6.1 日志存储

```java
@Document(collection = "system_logs")
@Data
public class SystemLog {
    @Id
    private String id;
    
    private String level;      // INFO, WARN, ERROR
    private String module;
    private String message;
    private String traceId;
    private LocalDateTime timestamp;
    
    @TtlIndex(expireAfterSeconds = 2592000) // 30 天过期
    private LocalDateTime expireAt;
}
```

### 6.2 商品属性（动态字段）

```java
@Document(collection = "products")
@Data
public class Product {
    @Id
    private String id;
    
    private String name;
    private Double price;
    private String category;
    
    // 动态属性
    private Map<String, Object> attributes;
    
    // 示例：手机
    // { "screen": "6.1 英寸", "cpu": "A15", "ram": "8GB", "storage": "256GB" }
    
    // 示例：衣服
    // { "material": "纯棉", "size": ["S", "M", "L"], "color": "黑色" }
}
```

---

## 七、MySQL vs MongoDB

| 特性 | MySQL | MongoDB |
|------|-------|---------|
| 数据模型 | 关系型 | 文档型 |
| Schema | 固定 | 灵活 |
| 事务 | 完整支持 | 4.0+ 支持多文档事务 |
| 扩展性 | 垂直扩展 | 水平扩展（分片） |
| 适用场景 | 强一致性业务 | 灵活 schema、大数据量 |

---

## 八、实战建议

1. **合理设计文档结构**：嵌入式 vs 引用式
2. **索引优化**：避免过多索引，定期清理
3. **使用 TTL 索引**：自动清理过期数据
4. **聚合管道**：在数据库层面完成复杂计算
5. **监控性能**：使用 MongoDB Compass 分析查询

---

> 💡 **提示**：MongoDB 适合存储半结构化和动态数据，但对于强一致性要求的场景，MySQL 仍是首选。
