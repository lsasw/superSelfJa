---
title: MySQL 索引与 MVCC 深度
icon: database
order: 1
category:
  - 数据库
tag:
  - MySQL
  - 索引
  - MVCC
---

# MySQL 索引与 MVCC 深度

> 任务编号：MY-01 / MY-02 / MY-03

## 一、B+ 树索引结构（MY-01）

```
                    [30 | 60]              ← 根节点（非叶子）
                   /    |    \
          [10|20]    [40|50]   [70|80]    ← 内部节点
          /  |  \    /  |  \   /  |  \
        [1][10][20] [30][40][50] [60][70][80] ← 叶子节点（双向链表）
         数据在这里！

特点：
- 所有数据存在叶子节点
- 叶子节点用双向链表连接（范围查询 O(logN+K)）
- 非叶子节点只存索引 key（更小，一个节点能存更多 key）
```

### 聚簇索引 vs 二级索引

```
聚簇索引（主键索引）
┌──────────────────────────────────┐
│ B+ 树结构                         │
│ 非叶子：主键值                     │
│ 叶子：完整行数据                   │
└──────────────────────────────────┘

二级索引（普通索引）
┌──────────────────────────────────┐
│ 非叶子：索引列值                   │
│ 叶子：主键值（不是行数据！）        │
└──────────────┬───────────────────┘
               │ 回表
               ▼
        聚簇索引 → 获取完整行数据
```

```
SQL: SELECT * FROM users WHERE name = '张三';

① 走 name 索引（二级索引）
② 找到 name='张三' → 叶子节点存的是主键 id
③ 用 id 去聚簇索引查（回表）
④ 获取完整行数据

总 IO：二级索引高度 + 聚簇索引高度
```

## 二、索引失效场景（MY-02）

```sql
-- 假设索引：idx_name_age_status(name, age, status)

-- ✅ 最左前缀匹配
EXPLAIN SELECT * FROM users WHERE name = '张三';
-- key: idx_name_age_status, type: ref

-- ✅ 全值匹配
EXPLAIN SELECT * FROM users WHERE name = '张三' AND age = 25 AND status = 1;

-- ❌ 跳过第一列
EXPLAIN SELECT * FROM users WHERE age = 25 AND status = 1;
-- key: NULL, type: ALL ← 索引失效！

-- ❌ LIKE 前置通配符
EXPLAIN SELECT * FROM users WHERE name LIKE '%三';
-- key: NULL, type: ALL

-- ✅ LIKE 后置通配符（可以走索引）
EXPLAIN SELECT * FROM users WHERE name LIKE '张三%';

-- ❌ 索引列加函数
EXPLAIN SELECT * FROM users WHERE UPPER(name) = 'ZHANGSAN';

-- ❌ 类型隐式转换
-- name 是 VARCHAR，但传了数字
EXPLAIN SELECT * FROM users WHERE name = 123;
-- key: NULL ← MySQL 对 name 做了隐式转换！

-- ❌ OR 条件中有非索引列
EXPLAIN SELECT * FROM users WHERE name = '张三' OR email = 'a@b.com';
-- 如果 email 没有索引 → ALL
```

## 三、MVCC 原理实战（MY-03）

### 两个关键隐藏列

```
每一行（InnoDB）都有两个隐藏列：
- DB_TRX_ID（6 字节）：最后修改此行的事务 ID
- DB_ROLL_PTR（7 字节）：指向 Undo Log 的回滚指针
```

### ReadView 可见性判断

```sql
-- 事务 A（trx_id=100）：查询
SELECT * FROM users WHERE id = 1;

-- 事务 B（trx_id=101）：更新（未提交）
UPDATE users SET age = 26 WHERE id = 1;

-- 此时事务 A 再次查询，能看到 age=26 吗？
-- → 不能！MVCC 保证可重复读

-- ReadView 判断逻辑：
-- 如果 DB_TRX_ID >= ReadView.min_trx_id 
--   且 DB_TRX_ID 在活跃事务列表中 → 不可见 → 查 Undo Log
```

### MVCC 可见性流程图

```
查询一条记录
    │
    ▼
DB_TRX_ID < ReadView.creator_trx_id？
    │
    ├─ 是 → 此版本是本事务自己改的 → 可见
    │
    ├─ 否 → DB_TRX_ID < ReadView.min_trx_id？
    │        │
    │        ├─ 是 → 事务已提交 → 可见
    │        │
    │        └─ 否 → DB_TRX_ID 在活跃列表中？
    │                 │
    │                 ├─ 是 → 事务未提交 → 不可见
    │                 │         │
    │                 │         └─ 沿 DB_ROLL_PTR 找 Undo Log → 递归判断
    │                 │
    │                 └─ 否 → 事务已提交 → 可见
```

### 不同隔离级别

| 隔离级别 | ReadView 生成时机 | 现象 |
|---------|------------------|------|
| READ UNCOMMITTED | 不使用 MVCC | 脏读 |
| READ COMMITTED | 每次查询生成新 ReadView | 不可重复读 |
| REPEATABLE READ | **事务开始时生成一次** | 可重复读（MySQL 默认） |
| SERIALIZABLE | 加锁 | 串行 |
