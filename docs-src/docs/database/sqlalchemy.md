---
title: SQLAlchemy 深度指南
icon: cube
order: 9
category:
  - 数据库
tag:
  - Python
  - ORM
  - SQL
  - SQLAlchemy
---

# SQLAlchemy — Python SQL 工具箱与 ORM

> **The Python SQL Toolkit and Object Relational Mapper**  
> SQLAlchemy 是 Python 生态中最成熟、最强大的数据库抽象层，让开发者拥有 SQL 的全部力量和灵活性。

SQLAlchemy 提供了一套完整的企业级持久化模式，从底层的 DBAPI 封装到顶层的 ORM，以 Pythonic 的方式构建高效、高性能的数据库访问层。18,000+ 次提交，MIT 开源许可，当前版本 **2.1.0b4**。

## 设计哲学

SQLAlchemy 的设计理念贯穿其整个架构，也是它区别于其他 ORM 的根本原因：

| 原则 | 含义 |
|------|------|
| **不隐藏 SQL** | ORM 不应遮蔽关系数据库的丰富能力——JOIN、子查询、窗口函数等应完全暴露 |
| **开发者始终掌控** | 不存在"ORM 生成了糟糕查询"——查询结构、JOIN 组织、列选择都由你决定 |
| **不需要就别用 ORM** | Core 层独立可用，提供完整的 Pythonic SQL 构造能力 |
| **事务是常态** | 调用 `commit()` 之前，没有任何数据写入永久存储 |
| **绑定参数优先** | 永远不在 SQL 中渲染字面值，杜绝 SQL 注入，让查询优化器有效缓存计划 |

## Core vs ORM — 双核架构

SQLAlchemy 的最大特色是**分层的双核架构**——两个层次独立可用，也可无缝组合：

```
┌──────────────────────────────────────────────────────┐
│                     SQLAlchemy                        │
├─────────────────────┬────────────────────────────────┤
│        ORM          │             Core               │
│  (对象关系映射)        │     (SQL 表达式语言 + DBAPI)      │
├─────────────────────┼────────────────────────────────┤
│ • Declarative Base  │ • Schema / Metadata            │
│ • Session / 事务     │ • SQL Expression Language      │
│ • Relationship      │ • Connection Pool              │
│ • Identity Map      │ • Type System                  │
│ • Unit of Work      │ • Reflection / Introspection   │
│ • Eager Loading     │ • 方言系统 (Dialect)            │
│ • 继承映射           │ • Insert/Update/Delete 构造     │
├─────────────────────┴────────────────────────────────┤
│                    Engine                             │
│                (连接池 + 方言)                          │
├──────────────────────────────────────────────────────┤
│                    DBAPI                              │
│          (psycopg2 / pymysql / sqlite3 ...)           │
└──────────────────────────────────────────────────────┘
```

**关键区别**：

| 场景 | 推荐层次 | 原因 |
|------|----------|------|
| 业务领域模型丰富 | **ORM** | 对象自动追踪变更、级联持久化 |
| 报表 / 数据分析 | **Core** 或纯 SQL | 不需要 ORM 的 overhead |
| 微服务简单 CRUD | **Core** | 轻量、显式、直接 |
| 复杂查询 + 部分 ORM | **Core 查询 + ORM Session** | 混合使用二者最优 |
| 已有大量原生 SQL | **Core Textual SQL** | `text()` 直接写 SQL，享受连接池和类型转换 |

---

## 一、快速入门：声明式 ORM

SQLAlchemy 2.0 引入了 **声明式映射（Declarative Mapping）** 和 PEP 484 类型注解支持。

### 1.1 定义模型

```python
from typing import List, Optional
from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column,
    relationship, Session
)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user_account"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]

    # 一对多关系
    addresses: Mapped[List["Address"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"User(id={self.id!r}, name={self.name!r})"

class Address(Base):
    __tablename__ = "address"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))

    # 多对一关系
    user: Mapped["User"] = relationship(back_populates="addresses")

    def __repr__(self):
        return f"Address(id={self.id!r}, email={self.email_address!r})"
```

**类型注解映射规则**：

| Python 类型 | SQL 类型 | NULL 策略 |
|-------------|----------|-----------|
| `int` | INTEGER | NOT NULL |
| `str` | VARCHAR | NOT NULL |
| `Optional[str]` | VARCHAR | 允许 NULL |
| `float` | FLOAT | NOT NULL |
| `bool` | BOOLEAN | NOT NULL |
| `datetime` | DATETIME | 取决于 Optional |

### 1.2 创建表和执行 CRUD

```python
# 1. 创建 Engine（不立即连接）
engine = create_engine("sqlite:///example.db", echo=True)

# 2. 生成表结构
Base.metadata.create_all(engine)

# 3. 使用 Session 进行 CRUD
with Session(engine) as session:
    # CREATE
    spongebob = User(
        name="spongebob",
        fullname="Spongebob Squarepants",
        addresses=[Address(email_address="spongebob@example.com")],
    )
    session.add(spongebob)
    session.commit()

    # READ — 使用 select() 构造
    from sqlalchemy import select
    stmt = select(User).where(User.name == "spongebob")
    user = session.scalars(stmt).one()
    print(user)  # User(id=1, name='spongebob')

    # UPDATE — 修改属性即可，Session 自动追踪
    user.fullname = "SpongeBob SquarePants"
    session.commit()

    # DELETE
    session.delete(user)
    session.commit()
```

**核心方法速查**：

| 操作 | 方法 | 说明 |
|------|------|------|
| 新增 | `session.add()` / `session.add_all()` | 添加到 Session |
| 查询 | `session.scalars(select(...))` | 返回 ORM 对象迭代器 |
| 按主键 | `session.get(User, id)` | 最快速的按主键查询 |
| 刷新 | `session.flush()` | 发出 SQL 但不提交 |
| 提交 | `session.commit()` | 提交事务 |
| 回滚 | `session.rollback()` | 回滚事务 |

---

## 二、Engine — 连接管理的核心

`Engine` 是 SQLAlchemy 应用的起点，是数据库和 DBAPI 的"大本营"。它是一个**工厂**，通过**连接池**管理连接生命周期。

### 2.1 创建 Engine

```python
from sqlalchemy import create_engine

# 基本形式
engine = create_engine("postgresql+psycopg2://user:pass@localhost:5432/mydb")

# 带配置
engine = create_engine(
    "mysql+pymysql://user:pass@host/dbname",
    echo=True,           # 将 SQL 输出到 stdout
    pool_size=10,        # 连接池大小（默认 5）
    max_overflow=20,     # 溢出连接数（默认 10）
    pool_recycle=3600,   # 连接回收时间（秒）
    pool_pre_ping=True,  # 检出前测试连接是否存活
)
```

### 2.2 数据库 URL 格式

```
dialect+driver://username:password@host:port/database
```

| 数据库 | URL 示例 |
|--------|----------|
| PostgreSQL | `postgresql+psycopg2://user:pass@localhost/mydb` |
| MySQL | `mysql+pymysql://user:pass@localhost/mydb` |
| SQLite (文件) | `sqlite:///path/to/foo.db` |
| SQLite (内存) | `sqlite://` |
| SQL Server | `mssql+pyodbc://user:pass@host:1433/db` |
| Oracle | `oracle+oracledb://user:pass@host:1521/?service_name=x` |

### 2.3 编程式 URL 构造（处理特殊字符）

```python
from sqlalchemy import URL

url = URL.create(
    "postgresql+psycopg2",
    username="dbuser",
    password="kx@jj5/g",   # 无需手动转义
    host="localhost",
    port=5432,
    database="mydb",
)
engine = create_engine(url)
```

### 2.4 连接池核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_size` | 5 | 池中保持的连接数 |
| `max_overflow` | 10 | 超出 pool_size 的额外连接 |
| `pool_timeout` | 30 | 等待可用连接的超时秒数 |
| `pool_recycle` | -1 | 连接最大存活时间（-1 永不过期） |
| `pool_pre_ping` | False | 每次检出前发送 `SELECT 1` 测试连接 |

### 2.5 自定义连接行为

```python
from sqlalchemy import event

# 每个新连接创建时执行
@event.listens_for(engine, "connect")
def set_session_params(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("SET timezone = 'UTC'")
    cursor.close()

# 动态生成认证令牌
@event.listens_for(engine, "do_connect")
def provide_token(dialect, conn_rec, cargs, cparams):
    cparams["token"] = get_auth_token()
```

**最佳实践**：整个进程只维护一个 `Engine` 实例，不要为每次请求创建新引擎。

---

## 三、Session — 事务与工作单元

`Session` 是 ORM 和数据库之间的对话通道，实现了**身份映射（Identity Map）** 和**工作单元（Unit of Work）** 模式。

### 3.1 Session 生命周期

```
begin → 操作 (add / query / modify) → flush → commit / rollback → close
```

### 3.2 推荐使用方式

```python
# 方式一：with 上下文管理器（推荐）
with Session(engine) as session:
    user = session.get(User, 1)
    user.name = "new_name"
    session.commit()
# Session 自动关闭

# 方式二：sessionmaker 工厂
from sqlalchemy.orm import sessionmaker

SessionFactory = sessionmaker(bind=engine)
session = SessionFactory()
try:
    # ... 操作 ...
    session.commit()
except:
    session.rollback()
    raise
finally:
    session.close()
```

### 3.3 身份映射（Identity Map）

Session 内部的 identity map 保证**同一个主键只对应一个内存对象**：

```python
user_a = session.get(User, 1)
user_b = session.get(User, 1)

user_a is user_b  # True！同一个对象
```

### 3.4 会话状态

| 状态 | 含义 |
|------|------|
| **Transient** | 新创建、未关联任何 Session 的对象 |
| **Pending** | 已 `add()` 到 Session，等待 flush |
| **Persistent** | 已持久化，处于 Session 管理之中 |
| **Detached** | Session 已关闭，对象脱离管理 |

### 3.5 Flush 策略

```python
# autoflush（默认）：查询前自动 flush 挂起的变更
session = Session(engine, autoflush=True)

# 手动控制 flush
session.add(new_user)
session.flush()          # 立即发出 SQL，但不提交
print(new_user.id)       # flush 后 id 可用
# ... 继续操作 ...
session.commit()         # 最终提交
```

---

## 四、查询体系

SQLAlchemy 2.0 统一了 Core 和 ORM 的查询方式，**`select()` 是唯一的查询入口**。

### 4.1 基本查询

```python
from sqlalchemy import select

stmt = select(User).where(User.name == "spongebob")
user = session.scalars(stmt).one()

# 等价于：SELECT * FROM user_account WHERE name = 'spongebob'
```

### 4.2 常用查询模式

```python
# ===== 条件查询 =====
select(User).where(User.name.in_(["alice", "bob"]))
select(User).where(User.name.like("a%"))
select(User).where(User.age.between(18, 65))
select(User).where(User.email.is_(None))

# 多条件（AND 链接）
select(User).where(User.age > 18, User.status == "active")

# OR 条件
from sqlalchemy import or_
select(User).where(or_(User.name == "alice", User.name == "bob"))

# ===== 排序与分页 =====
select(User).order_by(User.name.asc()).limit(10).offset(20)

# ===== 聚合 =====
from sqlalchemy import func
select(func.count(User.id)).where(User.status == "active")
select(func.avg(User.age), User.department).group_by(User.department)

# ===== JOIN =====
stmt = select(User, Address).join(Address).where(Address.email_address.like("%@gmail.com"))
for user, address in session.execute(stmt):
    print(user.name, address.email_address)

# ===== 子查询 =====
subq = select(Address.user_id).where(Address.email_address.like("%@example.com")).subquery()
select(User).where(User.id.in_(subq))
```

### 4.3 结果获取方法

| 方法 | 返回 | 失败时 |
|------|------|--------|
| `.scalars(stmt).all()` | `List[User]` | - |
| `.scalars(stmt).first()` | `User` 或 `None` | - |
| `.scalars(stmt).one()` | `User` | 0 或多行 → 异常 |
| `.scalars(stmt).one_or_none()` | `User` 或 `None` | 多行 → 异常 |
| `.scalars(stmt).unique()` | 去重迭代器 | - |
| `.execute(stmt).all()` | `List[Row]` | - |

### 4.4 加载策略（Eager Loading）

控制关联对象的加载时机和方式，是性能调优的核心手段：

| 策略 | 使用方法 | SQL | 适用场景 |
|------|----------|-----|----------|
| **Lazy** (默认) | 无需配置 | 访问时额外 SELECT | 大多数场景 |
| **Joined** | `.options(joinedload(User.addresses))` | LEFT JOIN 一次加载 | 总是需要关联数据时 |
| **Select IN** | `.options(selectinload(User.addresses))` | 第二条 SELECT...WHERE id IN (...) | 一对多，避免笛卡尔积 |
| **Subquery** | `.options(subqueryload(User.addresses))` | 子查询 | 复杂排序分页 |
| **Raise** | `.options(raiseload(User.addresses))` | 禁止访问 | 严格避免 N+1 |

```python
from sqlalchemy.orm import joinedload, selectinload

# 一次 JOIN 加载用户及其所有地址
stmt = select(User).options(joinedload(User.addresses))
users = session.scalars(stmt).unique().all()
for user in users:
    print(user.addresses)  # 不再触发额外 SQL

# 对一对多关系用 selectinload 更高效
stmt = select(User).options(selectinload(User.addresses))
```

---

## 五、关系映射详解

### 5.1 一对多 / 多对一

```python
class Parent(Base):
    __tablename__ = "parent"
    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Child"]] = relationship(back_populates="parent")

class Child(Base):
    __tablename__ = "child"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent.id"))
    parent: Mapped["Parent"] = relationship(back_populates="children")
```

### 5.2 一对一

```python
class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True)
    profile: Mapped["Profile"] = relationship(back_populates="user", uselist=False)

class Profile(Base):
    __tablename__ = "profile"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), unique=True)
    user: Mapped["User"] = relationship(back_populates="profile")
```

### 5.3 多对多

```python
from sqlalchemy import Table, Column

# 关联表
user_role = Table(
    "user_role", Base.metadata,
    Column("user_id", ForeignKey("user.id"), primary_key=True),
    Column("role_id", ForeignKey("role.id"), primary_key=True),
)

class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True)
    roles: Mapped[List["Role"]] = relationship(secondary=user_role, back_populates="users")

class Role(Base):
    __tablename__ = "role"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    users: Mapped[List["User"]] = relationship(secondary=user_role, back_populates="roles")
```

### 5.4 Cascade（级联操作）

| 选项 | 含义 |
|------|------|
| `save-update` | add() 父对象时自动 add() 子对象 |
| `merge` | merge() 父对象时级联子对象 |
| `delete` | 删除父对象时删除子对象 |
| `delete-orphan` | 从集合中移除子对象时删除该子对象 |
| `all` | `save-update, merge, refresh-expire, expunge, delete` |
| `all, delete-orphan` | 最常用组合，完全级联 |

---

## 六、Core — SQL 表达式语言

Core 层是 ORM 的基础，提供 Pythonic 方式构造 SQL。当你不需要 ORM 时，Core 是完美的选择。

### 6.1 表定义（MetaData + Table）

```python
from sqlalchemy import MetaData, Table, Column, Integer, String, ForeignKey

metadata = MetaData()

user_table = Table(
    "user_account", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(30), nullable=False),
    Column("fullname", String),
)

address_table = Table(
    "address", metadata,
    Column("id", Integer, primary_key=True),
    Column("email_address", String, nullable=False),
    Column("user_id", ForeignKey("user_account.id")),
)

# 生成表
metadata.create_all(engine)
```

### 6.2 使用 Core 执行查询

```python
with engine.connect() as conn:
    # INSERT
    conn.execute(
        user_table.insert(),
        [{"name": "alice", "fullname": "Alice Smith"},
         {"name": "bob", "fullname": "Bob Jones"}]
    )
    conn.commit()

    # SELECT
    result = conn.execute(
        user_table.select().where(user_table.c.name == "alice")
    )
    row = result.first()
    print(row.name, row.fullname)

    # UPDATE
    conn.execute(
        user_table.update()
        .where(user_table.c.name == "alice")
        .values(fullname="Alice Johnson")
    )

    # DELETE
    conn.execute(
        user_table.delete().where(user_table.c.name == "bob")
    )
```

### 6.3 文本 SQL（Textual SQL）

```python
from sqlalchemy import text

with engine.connect() as conn:
    # 参数化查询（防 SQL 注入）
    result = conn.execute(
        text("SELECT * FROM user_account WHERE name = :name"),
        {"name": "alice"}
    )

    # 复杂查询
    result = conn.execute(text("""
        SELECT u.name, COUNT(a.id) as addr_count
        FROM user_account u
        LEFT JOIN address a ON u.id = a.user_id
        GROUP BY u.id
    """))
```

### 6.4 ORM 中使用 Core 查询

```python
# ORM Session 中也能执行 Core 风格的 SQL
session.execute(user_table.select().where(user_table.c.name == "alice"))

# 甚至可以混合：Core 查询，返回 ORM 对象
from sqlalchemy import select
stmt = select(User).where(User.name == "alice")
users = session.scalars(stmt).all()
```

---

## 七、数据库方言（Dialect）

SQLAlchemy 通过方言系统支持几乎所有主流数据库，**每个方言自动处理 SQL 差异**。

| 数据库 | 方言 | 驱动（部分） |
|--------|------|-------------|
| PostgreSQL | `postgresql` | psycopg2, asyncpg, pg8000 |
| MySQL | `mysql` | pymysql, mysqldb, mysqlclient |
| SQLite | `sqlite` | sqlite3 (内置) |
| Microsoft SQL Server | `mssql` | pyodbc, pymssql |
| Oracle | `oracle` | oracledb, cx_oracle |
| MariaDB | `mariadb` | mariadb-connector |

### 方言特定特性

```python
# PostgreSQL 特定类型
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB, INET

class Product(Base):
    __tablename__ = "product"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    tags: Mapped[list] = mapped_column(ARRAY(String))
    metadata_: Mapped[dict] = mapped_column(JSONB, name="metadata")

# PostgreSQL ON CONFLICT (upsert)
from sqlalchemy.dialects.postgresql import insert as pg_insert

stmt = pg_insert(user_table).values(id=1, name="alice")
stmt = stmt.on_conflict_do_update(
    index_elements=["id"],
    set_={"name": stmt.excluded.name}
)
conn.execute(stmt)
```

---

## 八、高级特性

### 8.1 继承映射

三种继承策略：

```python
# === 单表继承 ===
class Employee(Base):
    __tablename__ = "employee"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    type: Mapped[str]

    __mapper_args__ = {
        "polymorphic_on": "type",
        "polymorphic_identity": "employee",
    }

class Manager(Employee):
    manager_data: Mapped[str]
    __mapper_args__ = {"polymorphic_identity": "manager"}

class Engineer(Employee):
    engineer_info: Mapped[str]
    __mapper_args__ = {"polymorphic_identity": "engineer"}
```

### 8.2 事件系统

```python
from sqlalchemy import event

# 插入前填充时间戳
@event.listens_for(User, "before_insert")
def set_created_at(mapper, connection, target):
    target.created_at = datetime.utcnow()

# SQL 执行前拦截
@event.listens_for(engine, "before_cursor_execute")
def before_cursor(conn, cursor, statement, parameters, context, executemany):
    if "FOR UPDATE" in statement:
        log_warning("排他锁查询", statement)
```

### 8.3 批量操作

```python
# insertmanyvalues — 批量插入，自动分页
users = [{"name": f"user_{i}"} for i in range(10000)]
session.execute(user_table.insert(), users)

# 批量 UPDATE
session.execute(
    user_table.update().where(user_table.c.status == "inactive"),
    [{"status": "archived"}]
)

# 大表分批查询
stmt = select(User).execution_options(yield_per=1000)
for user in session.scalars(stmt):
    process(user)
```

### 8.4 异步支持

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

async_engine = create_async_engine("postgresql+asyncpg://user:pass@host/db")

async with AsyncSession(async_engine) as session:
    stmt = select(User).where(User.name == "alice")
    result = await session.scalars(stmt)
    user = result.one()
```

### 8.5 混合属性（Hybrid Attributes）

```python
from sqlalchemy.ext.hybrid import hybrid_property

class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True)
    first_name: Mapped[str]
    last_name: Mapped[str]

    @hybrid_property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @full_name.expression
    def full_name(cls):
        return func.concat(cls.first_name, " ", cls.last_name)

# 查询中使用
select(User).where(User.full_name == "Alice Smith")
```

---

## 九、框架对比

| 维度 | SQLAlchemy | Django ORM | Peewee | SQLModel | Tortoise ORM |
|------|------------|------------|--------|----------|-------------|
| **架构** | 双核：Core + ORM | 一体化 ORM | 轻量 ORM | 基于 SQLAlchemy | 异步原生 |
| **SQL 控制** | ⭐⭐⭐⭐⭐ 完全控制 | ⭐⭐ 有限 | ⭐⭐⭐ 较好 | ⭐⭐⭐⭐ 好 | ⭐⭐⭐ 较好 |
| **性能** | ⭐⭐⭐⭐⭐ 接近原生 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 好 | ⭐⭐⭐⭐ 好 | ⭐⭐⭐⭐ 好 |
| **异步** | ✅ (1.4+) | ⚠️ 有限 | ❌ | ✅ | ✅ 原生 |
| **数据库支持** | 6+ 方言 | 5 种 | 3 种 | 同 SQLAlchemy | 3 种 |
| **学习曲线** | 较陡 | 平缓 | 平缓 | 中等 | 中等 |
| **社区规模** | ⭐⭐⭐⭐⭐ 巨大 | ⭐⭐⭐⭐⭐ 巨大 | ⭐⭐ 较小 | ⭐⭐⭐ 增长中 | ⭐⭐⭐ 增长中 |
| **适用场景** | 全场景，复杂查询 | Django 项目 | 微型/个人项目 | FastAPI 项目 | 异步项目 |
| **类型注解** | ✅ PEP 484 | ❌ | ❌ | ✅ 原生支持 | ✅ Pydantic |

---

## 十、项目结构

```
sqlalchemy/
├── lib/sqlalchemy/        # 核心库代码
│   ├── engine/            # Engine、连接池、事务
│   ├── sql/               # SQL 表达式语言
│   ├── orm/               # ORM（Session、Mapper、Relationship）
│   ├── dialects/          # 数据库方言（6+ 数据库）
│   ├── ext/               # 扩展（async、hybrid、declarative）
│   └── testing/           # 测试工具
├── examples/              # 使用示例
├── doc/                   # 文档源文件
├── test/                  # 测试套件
├── pyproject.toml         # 项目配置
├── setup.py / setup.cfg   # 安装配置
├── noxfile.py             # Nox 测试运行器
├── .pre-commit-config.yaml # pre-commit hooks
├── AUTHORS                # 贡献者列表
├── CHANGES.rst            # 变更日志
└── LICENSE                # MIT 许可证
```

**技术栈**：Python（核心） + Cython（性能优化，提升行处理速度 15-30%）+ Black 26.3.1 + Nox + pre-commit

---

## 十一、关键设计决策

| 决策 | 理由 |
|------|------|
| **双核架构** | ORM 和 Core 独立可用，不强绑定。不需要 ORM 时无需承担其开销 |
| **连接池内置** | 内置 QueuePool，避免第三方依赖。每个 Engine 维护一个池 |
| **延迟连接** | `create_engine()` 不立即连接，首次 `connect()` 才建立，快速启动 |
| **Session ≠ 连接** | Session 管理事务边界但不绑定单个连接，连接从池中按需获取归还 |
| **身份映射** | 同一主键在 Session 内只有一个对象，避免数据不一致 |
| **事务默认** | 显式 `commit()` 才能持久化，防止意外写入 |
| **绑定参数** | 所有 SQL 使用参数化查询，杜绝 SQL 注入 |

---

## 十二、安装与版本

```bash
# 安装最新版
pip install sqlalchemy

# 带异步支持
pip install sqlalchemy[asyncio]

# 带特定数据库驱动
pip install sqlalchemy[postgresql]    # psycopg2
pip install sqlalchemy[mysql]         # pymysql
pip install sqlalchemy[mssql]         # pyodbc
```

- 当前版本：**2.1.0b4**（Beta 4）
- Python 版本兼容：CPython 3.9+
- 提交数：18,000+
- 许可证：MIT

---

## 参考资源

| 资源 | 链接 |
|------|------|
| 官网 | [https://www.sqlalchemy.org/](https://www.sqlalchemy.org/) |
| 官方文档 | [https://docs.sqlalchemy.org/](https://docs.sqlalchemy.org/) |
| 统一教程 (2.0+) | [https://docs.sqlalchemy.org/en/21/tutorial/](https://docs.sqlalchemy.org/en/21/tutorial/) |
| GitHub 仓库 | [https://github.com/sqlalchemy/sqlalchemy](https://github.com/sqlalchemy/sqlalchemy) |
| PyPI | [https://pypi.org/project/sqlalchemy/](https://pypi.org/project/sqlalchemy/) |
| 社区支持 | [https://www.sqlalchemy.org/support.html](https://www.sqlalchemy.org/support.html) |
