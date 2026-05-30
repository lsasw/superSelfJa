---
title: MySQL 进阶实战
icon: database
order: 1
---

# MySQL 进阶实战指南

MySQL 是最流行的关系型数据库，掌握其高级特性是后端开发的必备技能。

---

## 一、索引原理与优化

### 1.1 B+ 树索引结构

```
         根节点
        /      \
   中间节点    中间节点
   /    \      /    \
 叶子节点 叶子节点 叶子节点 叶子节点
 (数据行)  (数据行)  (数据行)  (数据行)
```

**B+ 树特点**：
- 只有叶子节点存储数据
- 叶子节点形成有序链表
- 树高度通常为 3-4 层
- 适合范围查询和排序

### 1.2 索引类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| 主键索引 | 唯一且非空 | 主键字段 |
| 唯一索引 | 值唯一 | 身份证号、邮箱 |
| 普通索引 | 无限制 | 频繁查询字段 |
| 组合索引 | 多列组合 | 多条件查询 |
| 全文索引 | 文本搜索 | 文章内容搜索 |

### 1.3 索引最佳实践

```sql
-- ✅ 推荐：组合索引 (a, b, c)
CREATE INDEX idx_abc ON table_name(a, b, c);

-- 可以使用索引的查询
SELECT * FROM table_name WHERE a = 1;                    -- ✅
SELECT * FROM table_name WHERE a = 1 AND b = 2;          -- ✅
SELECT * FROM table_name WHERE a = 1 AND b = 2 AND c = 3;-- ✅
SELECT * FROM table_name WHERE a = 1 AND c = 3;          -- ✅ (部分使用)

-- 无法使用索引的查询
SELECT * FROM table_name WHERE b = 2;                    -- ❌
SELECT * FROM table_name WHERE b = 2 AND c = 3;          -- ❌
```

### 1.4 覆盖索引

```sql
-- 表结构
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    email VARCHAR(100),
    INDEX idx_name_age (name, age)
);

-- 覆盖索引查询（只查询索引列）
SELECT name, age FROM users WHERE name = '张三';  -- ✅ 只需查索引

-- 回表查询（需要查数据行）
SELECT * FROM users WHERE name = '张三';          -- ❌ 需要回表
```

---

## 二、事务与锁机制

### 2.1 事务隔离级别

```sql
-- 查看当前隔离级别
SELECT @@transaction_isolation;

-- 设置隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 各隔离级别对比
-- READ UNCOMMITTED: 脏读、不可重复读、幻读都可能发生
-- READ COMMITTED:   解决脏读
-- REPEATABLE READ:  解决脏读、不可重复读（MySQL 默认）
-- SERIALIZABLE:     解决所有问题，性能最低
```

### 2.2 锁类型

| 锁类型 | 说明 | 场景 |
|--------|------|------|
| 行锁 | 锁定单行 | InnoDB 默认 |
| 表锁 | 锁定整表 | MyISAM |
| 间隙锁 | 锁定区间 | 防止幻读 |
| 临键锁 | 行锁+间隙锁 | RR 级别默认 |
| 意向锁 | 表级锁，表示行锁意向 | InnoDB 内部 |

```sql
-- 悲观锁
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;

-- 乐观锁
UPDATE accounts 
SET balance = balance - 100, version = version + 1 
WHERE id = 1 AND version = #{version};
```

### 2.3 死锁分析

```sql
-- 查看死锁信息
SHOW ENGINE INNODB STATUS;

-- 查看当前锁等待
SELECT * FROM information_schema.INNODB_LOCK_WAITS;

-- 查看当前事务
SELECT * FROM information_schema.INNODB_TRX;
```

---

## 三、SQL 优化

### 3.1 EXPLAIN 分析

```sql
EXPLAIN SELECT * FROM users WHERE name = '张三';
```

| 字段 | 说明 | 优化方向 |
|------|------|----------|
| type | 访问类型 | 至少达到 range |
| key | 使用的索引 | 确保使用合适索引 |
| rows | 扫描行数 | 越少越好 |
| Extra | 额外信息 | 避免 Using filesort |

**type 字段优劣**（从好到差）：
```
system > const > eq_ref > ref > range > index > ALL
```

### 3.2 慢查询优化

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- 超过 1 秒记录

-- 查看慢查询
SHOW VARIABLES LIKE 'slow_query%';

-- 使用 pt-query-digest 分析
pt-query-digest /var/log/mysql/slow.log
```

### 3.3 优化案例

```sql
-- ❌ 差：函数导致索引失效
SELECT * FROM users WHERE YEAR(create_time) = 2024;

-- ✅ 好：范围查询
SELECT * FROM users WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01';

-- ❌ 差：隐式类型转换
SELECT * FROM users WHERE phone = 13800138000;  -- phone 是 VARCHAR

-- ✅ 好：显式类型
SELECT * FROM users WHERE phone = '13800138000';

-- ❌ 差：OR 条件
SELECT * FROM users WHERE name = '张三' OR age = 25;

-- ✅ 好：UNION ALL
SELECT * FROM users WHERE name = '张三'
UNION ALL
SELECT * FROM users WHERE age = 25;

-- ❌ 差：LIKE 前缀通配符
SELECT * FROM users WHERE name LIKE '%张三%';

-- ✅ 好：前缀匹配
SELECT * FROM users WHERE name LIKE '张三%';
```

---

## 四、分库分表

### 4.1 垂直分表

```sql
-- 用户基本信息（热数据）
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50),
    password VARCHAR(100),
    email VARCHAR(100),
    status TINYINT
);

-- 用户扩展信息（冷数据）
CREATE TABLE user_profile (
    user_id BIGINT PRIMARY KEY,
    nickname VARCHAR(50),
    avatar VARCHAR(255),
    bio TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 4.2 水平分表

```sql
-- 按用户 ID 取模分表
-- users_0, users_1, users_2, users_3

-- 路由算法
int tableIndex = userId % 4;
String tableName = "users_" + tableIndex;
```

### 4.3 分库分表中间件

```yaml
# ShardingSphere 配置
spring:
  shardingsphere:
    datasource:
      names: ds0,ds1
    rules:
      sharding:
        tables:
          users:
            actual-data-nodes: ds$->{0..1}.users_$->{0..3}
            table-strategy:
              standard:
                sharding-column: user_id
                sharding-algorithm-name: user-table-inline
        sharding-algorithms:
          user-table-inline:
            type: INLINE
            props:
              algorithm-expression: users_$->{user_id % 4}
```

---

## 五、性能优化

### 5.1 连接池配置

```yaml
spring:
  datasource:
    hikari:
      minimum-idle: 5
      maximum-pool-size: 20
      connection-timeout: 30000
      idle-timeout: 600000
      max-lifetime: 1800000
      pool-name: HikariPool
```

### 5.2 批量操作

```java
//  差：循环插入
for (User user : users) {
    userRepository.save(user);
}

// ✅ 好：批量插入
userRepository.saveAll(users);

// SQL 层面
INSERT INTO users (name, email) VALUES 
  ('张三', 'zhangsan@example.com'),
  ('李四', 'lisi@example.com'),
  ('王五', 'wangwu@example.com');
```

### 5.3 分页优化

```sql
-- ❌ 差：深分页
SELECT * FROM users ORDER BY id LIMIT 1000000, 10;

-- ✅ 好：延迟关联
SELECT u.* FROM users u
INNER JOIN (SELECT id FROM users ORDER BY id LIMIT 1000000, 10) tmp
ON u.id = tmp.id;

-- ✅ 好：游标分页
SELECT * FROM users WHERE id > #{lastId} ORDER BY id LIMIT 10;
```

---

## 六、实战建议

1. **索引设计**：遵循最左前缀原则，避免过度索引
2. **SQL 编写**：避免 SELECT *，使用 EXPLAIN 分析
3. **事务控制**：尽量缩短事务时间，避免长事务
4. **分库分表**：数据量超过 500 万考虑分表
5. **监控告警**：监控慢查询、连接数、锁等待

---

> 💡 **提示**：索引不是越多越好，每个索引都会增加写入开销。根据查询场景合理设计索引。
