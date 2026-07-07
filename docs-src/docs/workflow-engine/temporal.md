---
title: Temporal — 持久化执行平台
icon: history
order: 1
---

# Temporal — 持久化执行平台

**Temporal** 是一个持久化执行（Durable Execution）平台，让开发者构建可弹性伸缩的分布式应用。核心能力是将业务逻辑以 **Workflow** 的形式执行，自动处理间歇性故障、重试失败操作，状态可持久化到数据库。MIT 协议。

- **GitHub**: https://github.com/temporalio/temporal
- **官网**: https://temporal.io
- **语言**: Go（服务端）、SDK 支持 Go / Java / Python / TypeScript / .NET
- **前身**: Uber Cadence（由 Cadence 创始人创办 Temporal Technologies）

---

## 一、核心概念

### 1.1 Durable Execution（持久化执行）

传统微服务的问题：进程崩溃 → 状态丢失 → 需要补偿逻辑。

Temporal 的方案：**Workflow 代码被"重放"**——进程崩溃后，新进程从数据库恢复状态，继续执行未完成的步骤。对开发者来说，Workflow 代码看起来就像从未中断过。

### 1.2 四大组件

| 组件 | 职责 |
|------|------|
| **Workflow** | 业务逻辑编排，确定性代码（纯函数式），编排 Activity 调用顺序 |
| **Activity** | 实际执行单元——调用外部 API、读写数据库、发送消息等 |
| **Worker** | 轮询 Task Queue、执行 Workflow 和 Activity 的宿主进程 |
| **Temporal Server** | 核心调度引擎：状态持久化、任务分发、重试、超时管理 |

```
┌─────────────────────────────────────────────┐
│                 Temporal Server               │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
│  │ History │  │ Matching │  │  Frontend   │  │
│  │ Service │  │ Service  │  │   Service   │  │
│  └─────────┘  └──────────┘  └────────────┘  │
└──────────────────────┬──────────────────────┘
                       │ Task Queue
┌──────────────────────▼──────────────────────┐
│                   Workers                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Workflow │  │ Activity │  │ Activity │  │
│  │  Worker  │  │  Worker  │  │  Worker  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

---

## 二、为什么需要 Temporal？

### 2.1 传统方式的痛点

以一个"用户注册后发送欢迎邮件"为例：

```python
# 脆弱的方式
def register_user(data):
    user = db.insert(data)       # ① 成功
    email.send_welcome(user)     # ② 失败 → 数据不一致！
    analytics.track_signup(user) # ③ 永远不会执行
```

需要大量补偿代码：try-catch、重试队列、幂等性处理、Saga 模式...

### 2.2 Temporal 的方式

```python
@workflow.defn
class UserRegistrationWorkflow:
    @workflow.run
    async def run(self, data):
        user = await workflow.execute_activity(
            create_user, data,
            start_to_close_timeout=timedelta(seconds=10)
        )
        await workflow.execute_activity(
            send_welcome_email, user,
            start_to_close_timeout=timedelta(seconds=10)
        )
        await workflow.execute_activity(
            track_signup, user,
            start_to_close_timeout=timedelta(seconds=5)
        )
```

如果 Worker 在第 2 步崩溃：Temporal 重启新 Worker，从数据库恢复状态，从第 2 步继续。**开发者无需写补偿逻辑**。

---

## 三、核心特性

| 特性 | 说明 |
|------|------|
| **自动重试** | Activity 失败后按指数退避重试，可自定义策略 |
| **超时控制** | 支持 Schedule-to-Close、Start-to-Close、Heartbeat 等多层级超时 |
| **状态持久化** | Workflow 历史完整记录，支持查询和重放 |
| **Saga 模式** | 内置补偿事务支持（Saga） |
| **信号与查询** | Signal 向运行中的 Workflow 发送异步事件；Query 读取当前状态 |
| **多语言 SDK** | Go / Java / Python / TypeScript / .NET |
| **可见性** | Web UI 查看所有 Workflow 执行历史、输入输出、堆栈跟踪 |
| **命名空间** | 多租户隔离 |
| **版本化** | Workflow 代码升级的版本策略（`GetVersion` / Patching API） |

---

## 四、Java SDK 快速入门

### 4.1 依赖

```xml
<dependency>
    <groupId>io.temporal</groupId>
    <artifactId>temporal-sdk</artifactId>
    <version>1.24.0</version>
</dependency>
```

### 4.2 定义 Workflow 接口

```java
@WorkflowInterface
public interface GreetingWorkflow {
    @WorkflowMethod
    String greet(String name);
}
```

### 4.3 实现 Workflow

```java
public class GreetingWorkflowImpl implements GreetingWorkflow {
    private final ActivityOptions options = ActivityOptions.newBuilder()
        .setStartToCloseTimeout(Duration.ofSeconds(5))
        .build();
    private final GreetingActivities activities = 
        Workflow.newActivityStub(GreetingActivities.class, options);

    @Override
    public String greet(String name) {
        // 调用 Activity（自动重试、状态持久化）
        String greeting = activities.composeGreeting(name);
        return "Result: " + greeting;
    }
}
```

### 4.4 定义 Activity 接口与实现

```java
@ActivityInterface
public interface GreetingActivities {
    String composeGreeting(String name);
}

public class GreetingActivitiesImpl implements GreetingActivities {
    @Override
    public String composeGreeting(String name) {
        return "Hello, " + name + "!";
    }
}
```

### 4.5 启动 Worker 和 Workflow

```java
// 启动 Worker
WorkflowServiceStubs service = WorkflowServiceStubs.newLocalServiceStubs();
WorkflowClient client = WorkflowClient.newInstance(service);
WorkerFactory factory = WorkerFactory.newInstance(client);
Worker worker = factory.newWorker("greeting-queue");
worker.registerWorkflowImplementationTypes(GreetingWorkflowImpl.class);
worker.registerActivitiesImplementations(new GreetingActivitiesImpl());
factory.start();

// 启动 Workflow
GreetingWorkflow workflow = client.newWorkflowStub(
    GreetingWorkflow.class,
    WorkflowOptions.newBuilder().setTaskQueue("greeting-queue").build()
);
String result = workflow.greet("World");
```

---

## 五、架构原理：Event Sourcing

Temporal 的核心是 **Event Sourcing** 模型：

1. Workflow 代码每次执行时生成一系列 **Command**（启动 Activity、设置 Timer 等）
2. Temporal Server 将 Command 转换为 **Event** 追加到 Workflow History
3. 需要重建状态时，从头重放 History，执行所有 Event 对应的 Command
4. 非确定性操作（随机数、当前时间、外部调用）由 Temporal 通过 `Workflow` 静态方法提供确定性版本

```
Workflow 代码 ──→ Commands ──→ Temporal Server ──→ Events (持久化)
                    ↑                                    │
                    └──── Replay (状态恢复) ──────────────┘
```

**关键约束**：Workflow 代码必须是确定性的——不能直接调用 `System.currentTimeMillis()` 或 `Math.random()`，必须使用 Temporal 提供的 `Workflow.currentTimeMillis()` 等。

---

## 六、适用场景

| 场景 | 为什么适合 Temporal |
|------|-------------------|
| **订单处理** | 多步骤（下单→支付→发货→确认），任何步骤失败自动补偿 |
| **用户注册流程** | 创建账号→发邮件→初始化数据→通知下游 |
| **数据管道** | ETL 多阶段编排，阶段失败仅重试失败部分 |
| **定时任务** | Cron Workflow 替代传统 Cron + 监控 |
| **审批流程** | Signal 驱动的人工审批节点，无限等待 |
| **CI/CD 编排** | 构建→测试→部署，步骤依赖和重试 |
| **AI Agent 编排** | 多 Agent 协调，状态持久化避免上下文丢失 |

---

## 七、快速体验

```bash
# 安装 CLI
brew install temporal

# 启动本地开发服务器（含所有依赖）
temporal server start-dev

# 查看 Web UI
open http://localhost:8233

# 运行 Java 示例
git clone https://github.com/temporalio/samples-java
cd samples-java
./gradlew hello-world:run
```

---

## 八、与类似方案对比

| 方案 | Temporal | Cadence | Airflow | AWS Step Functions |
|------|:--------:|:-------:|:-------:|:------------------:|
| 持久化执行 | ✅ | ✅ | 仅状态 | ✅ |
| 代码即流程 | ✅ | ✅ | DAG 配置 | JSON DSL |
| 多语言 SDK | ✅ | ✅ | ❌ (Python) | ❌ |
| 可测试性 | ✅ 优秀 | ✅ | 有限 | 有限 |
| 自托管 | ✅ | ✅ | ✅ | ❌ |
| 社区活跃度 | ⭐ 极高 | 低 | 高 | AWS 生态 |
| 核心语言 | Go | Go | Python | - |

---

## 九、关键设计决策

1. **Workflow 确定性**：必须使用 Temporal 提供的时钟/随机数 API，因为重放需要完全一致的结果
2. **Activity 非确定性**：Activity 是实际的副作用执行单元，允许任意操作，Temporal 负责重试和超时
3. **Task Queue 解耦**：Worker 通过轮询 Task Queue 接收任务，天然支持弹性伸缩
4. **历史裁剪**：长时间运行的 Workflow 历史会自动归档，避免无限增长
