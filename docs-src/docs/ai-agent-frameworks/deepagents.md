---
title: DeepAgents
icon: robot
order: 2
category:
  - Agent 框架
tag:
  - LangChain
  - LangGraph
  - Agent
  - AI
---

# DeepAgents — 开箱即用的智能体框架

> **Deep Agents** is an open source agent harness — an opinionated agent that runs out of the box.  
> 灵感源自 Claude Code，由 LangChain 团队维护。MIT 开源许可。

DeepAgents 是一个"电池已装好"的智能体框架（agent harness），在 LangChain 和 LangGraph 之上提供了一套预设的生产级能力：**文件系统、子代理、上下文管理、技能系统、人工审批**——开箱即用，同时每一项都可以扩展或替换。

## 技术栈层次

DeepAgents 位于 LangChain 生态的顶层，与下层组件形成清晰的层次关系：

| 层级 | 组件 | 角色 |
|------|------|------|
| **Harness（套件）** | **DeepAgents** | 内置文件系统 / 子代理 / 上下文管理 / 技能 / HITL |
| **Harness（轻量）** | LangChain `create_agent` | 最小化 agent 包装，无内置中间件 |
| **Runtime（运行时）** | LangGraph | 图运行时：流式、持久化、检查点、断点 |
| **Framework（框架）** | LangChain | 核心构建模块：模型、工具、提示词 |

三层之间可以组合使用：任何 LangGraph `CompiledStateGraph` 都可以作为子代理传入 DeepAgents，自定义编排能无缝接入框架的默认能力。

## 核心能力体系

DeepAgents 围绕四大支柱构建，覆盖智能体从执行到进化的全生命周期：

```
┌──────────────────────────────────────────────────────────────┐
│                     DeepAgents Harness                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  执行环境      │  上下文管理    │    委托        │  控制           │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ • Tools/MCP  │ • Skills     │ • Task       │ • HITL          │
│ • 虚拟文件系统  │   (渐进式披露) │   Planning    │   (interrupts)  │
│ • 权限控制     │ • Memory     │ • Sub-agents │ • 权限规则       │
│ • 代码执行     │   (AGENTS.md)│   (task tool)│                │
│   ─ Sandbox  │ • 摘要/卸载   │              │                │
│   ─ QuickJS  │ • Prompt 缓存 │              │                │
│ • 流式传输     │              │              │                │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                   LangGraph Runtime                         │
├─────────────────────────────────────────────────────────────┤
│                   LangChain Framework                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、执行环境（Execution Environment）

执行环境是 agent 行动的场所，包含五层能力。

### 1.1 工具与 MCP

通过 `tools=` 参数传入任意工具，完全支持 **Model Context Protocol (MCP)**：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, fetch_page, run_query],
)
```

MCP 服务器可连接数据库、API、文件系统等外部资源，agent 通过标准工具调用接口使用它们。

### 1.2 虚拟文件系统

DeepAgents 内置一组文件操作工具，由可插拔后端驱动：

| 工具 | 描述 |
|------|------|
| `ls` | 列出目录文件及元数据（大小、修改时间） |
| `read_file` | 读取文件内容（带行号），支持 offset/limit 分页；支持多模态内容（图片/视频/音频/PDF） |
| `write_file` | 创建新文件 |
| `edit_file` | 精确字符串替换（支持全局替换模式） |
| `glob` | 按模式匹配文件（如 `**/*.py`） |
| `grep` | 搜索文件内容（文件列表 / 带上下文 / 计数的多种输出模式） |
| `execute` | 在沙箱环境中运行 shell 命令（仅限沙箱后端） |

**支持的多模态文件类型：**

| 类型 | 扩展名 |
|------|--------|
| 图片 | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.heic` `.heif` |
| 视频 | `.mp4` `.mpeg` `.mov` `.avi` `.flv` `.mpg` `.webm` `.wmv` `.3gpp` |
| 音频 | `.wav` `.mp3` `.aiff` `.aac` `.ogg` `.flac` |
| 文档 | `.pdf` `.ppt` `.pptx` |

**可插拔后端类型：**

| 后端 | 适用场景 |
|------|----------|
| 内存状态 (in-memory) | 轻量测试、临时会话 |
| 本地磁盘 (local disk) | 本地开发、单机部署 |
| LangGraph Store | 持久化、多会话共享 |
| 组合路由 (composite) | 按路径规则路由到不同后端 |
| 自定义后端 | 带自定义权限规则的远程存储 |

### 1.3 文件系统权限

声明式权限规则，自上而下评估，**首个匹配规则生效**（first-match-wins）：

```python
from deepagents import FilesystemBackend

backend = FilesystemBackend(
    root_path="/workspace",
    permissions=[
        {"operations": ["read", "write"], "paths": ["/workspace/**"], "mode": "allow"},
        {"operations": ["read"], "paths": ["/data/**"], "mode": "allow"},
        {"operations": ["read", "write"], "paths": ["**/.env"], "mode": "deny"},
    ],
)
```

无规则匹配时默认允许。注意：权限控制不适用于沙箱后端（沙箱中 `execute` 可执行任意命令）。

### 1.4 代码执行

两种代码执行方式：

| 方式 | 工具 | 环境 | 适用场景 |
|------|------|------|----------|
| **沙箱后端** | `execute` | 隔离容器 | 安装依赖、运行测试、调用 CLI、操作系统文件 |
| **QuickJS 解释器** | `eval` | JavaScript 运行时 | 轻量编程（循环、批处理）、确定性数据转换、程序化工具调用 |

> QuickJS 解释器不提供 shell 访问、包安装、文件系统或网络访问。

### 1.5 流式传输

事件流将 agent 运行暴露为类型化投影，涵盖消息、工具调用、值、委托任务。**关键增强**：`stream.subagents`——每个委托任务获得独立句柄，包含独立的消息、工具调用和嵌套子代理流。

---

## 二、上下文管理（Context Management）

控制 agent 的知识范围、token 限制内的运行时长以及跨会话内容保留。

### 2.1 技能系统（Skills）

遵循 [Agent Skills 标准](https://agentskills.io/)，每个技能为一个独立目录，包含 `SKILL.md` 及脚本、模板等资源。

**渐进式披露（Progressive Disclosure）：**

```
启动时                             任务触发时
┌─────────────────┐              ┌─────────────────┐
│ 仅读取 SKILL.md   │  ──匹配──▶  │ 加载完整技能内容   │
│ frontmatter      │              │ + 脚本 + 模板     │
└─────────────────┘              └─────────────────┘
```

- 启动上下文紧凑，仅含技能元数据
- 任务需要时才加载完整内容
- 按需提供丰富能力，不膨胀上下文

### 2.2 记忆系统（Memory）

通过 `AGENTS.md` 文件管理持久记忆，遵循 [agents.md 标准](https://agents.md/)：

- **始终加载**：与 Skills 不同，记忆文件始终在上下文中
- **可更新**：agent 基于交互和反馈更新记忆，偏好和模式跨线程保留
- **存储后端**：`StateBackend` / `StoreBackend` / `FilesystemBackend`

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    memory=FilesystemBackend(root_path="/home/user/.deepagents"),
)
```

### 2.3 摘要与上下文卸载

上下文流通过四层机制支持超长对话：

| 机制 | 说明 |
|------|------|
| **输入上下文** | 系统提示词 + 记忆 + 技能 + 工具提示词 = agent 的起始知识 |
| **压缩** | 内置摘要机制压缩对话历史和大中间结果 |
| **隔离** | 子代理在隔离上下文中处理重型子任务，仅返回最终结果 |
| **长期记忆** | 虚拟文件系统中的持久存储，跨线程携带信息 |

这些机制共同支持**超出单个上下文窗口的多步任务**，减少手动裁剪和 token 消耗。

### 2.4 Prompt 缓存

- **自动启用**：Anthropic 模型和 Amazon Bedrock (Claude/Nova)
- **缓存范围**：系统提示词的静态部分——基础指令、记忆、技能内容
- **效果**：跨调用避免重复处理相同 token，降低延迟与成本
- **无需配置**：开箱即用

---

## 三、委托机制（Delegation）

将大问题拆分为更小的可并行工作单元。

### 3.1 任务规划

内置 `write_todos` 工具，支持 `pending` / `in_progress` / `completed` 状态追踪，任务持久化在 agent 状态中：

```python
# Agent 自动创建并追踪任务
result = agent.invoke({"messages": "Plan and implement a REST API with 5 endpoints"})
# Agent 使用 write_todos 规划子任务并逐步执行
```

### 3.2 子代理（Sub-agents）

内置 `task` 工具，创建具有隔离上下文的临时子代理：

| 特性 | 说明 |
|------|------|
| **全新上下文** | 每次调用创建新实例，独立上下文 |
| **自主执行** | 子代理独立运行直到完成 |
| **单次交接** | 仅向主代理返回一份最终报告 |
| **可配置策略** | 默认 `general-purpose` 或自定义专用子代理 |
| **Token 效率** | 重型子任务隔离，压缩为紧凑结果 |

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    subagents=[
        {"name": "code-reviewer", "system_prompt": "You review code for bugs and style."},
        {"name": "test-writer", "system_prompt": "You write comprehensive unit tests."},
    ],
)
```

> 任何 LangGraph `CompiledStateGraph` 都可作为子代理传入，自定义编排接入框架默认能力。

---

## 四、控制机制（Steering）

### 4.1 Human-in-the-Loop

集成 LangGraph interrupts，在关键决策点暂停等待人工审批：

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    interrupt_on={
        "edit_file": True,      # 编辑前暂停
        "execute": True,        # 执行命令前暂停
        "write_file": True,     # 写文件前暂停
    },
)
```

每次匹配工具调用前暂停，允许人工**审批、添加指导或修改工具输入**。适用于破坏性操作、昂贵 API 调用、交互式调试等场景。

---

## 五、Middleware 管线

DeepAgents 通过中间件栈组织功能，两个核心中间件不可绕过：

| 中间件 | 作用 | 可移除？ |
|--------|------|----------|
| `FilesystemMiddleware` | 文件系统功能基础脚手架 | ❌ 不可移除 |
| `SubAgentMiddleware` | 子代理功能 | ❌ 不可直接移除（可通过 harness profile 禁用） |

**自定义方式**——通过 `HarnessProfile` 隐藏工具表面或禁用子代理：

```python
from deepagents import HarnessProfile, register_harness_profile

register_harness_profile(
    "anthropic:claude-sonnet-4-6",
    HarnessProfile(
        excluded_tools=frozenset(
            {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}
        ),
    ),
)
```

---

## 六、模型支持

**所有支持 tool calling 的模型均可使用**：

| 提供商 | 类型 | 示例模型 |
|--------|------|----------|
| OpenAI | 云端 API | gpt-5.5, gpt-4o |
| Anthropic | 云端 API | claude-sonnet-4-6, claude-opus-4-6 |
| Google | 云端 API | gemini-3.5-flash |
| Baseten / Fireworks | 开源模型托管 | Llama, Mistral 系列 |
| Ollama / vLLM / llama.cpp | 本地部署 | 开源模型本地运行 |

使用任何 [LangChain chat model](https://docs.langchain.com/oss/python/langchain/models) 即可。

---

## 七、DeepAgents Code (dcode)

DeepAgents Code 是一个**预构建的编码智能体**，类似 Claude Code 或 Cursor，通过任何 LLM 驱动，在终端运行。

```bash
# 一键安装
curl -LsSf https://langch.in/dcode | bash

# 常用命令
dcode doctor     # 诊断环境，报告构建信息
dcode "实现一个 FastAPI 用户认证模块"
```

**特性**：
- 文件读写编辑、代码搜索
- 终端命令执行
- 子代理委托重型子任务
- 可配置模型和工具
- 通过 `dcode doctor` 报告精确构建 commit

---

## 八、deepagents-talon 运行时

`deepagents-talon` 是 DeepAgents 的**本地单操作员运行时**（当前版本 0.0.3），将 agent 作为持久化服务运行：

| 能力 | 说明 |
|------|------|
| **WhatsApp 通道** | Agent 通过 WhatsApp 接收和回复消息 |
| **Cron 调度** | 定时触发 agent 任务 |
| **MCP 加载** | 运行时加载 MCP 服务器配置 |
| **Docker Compose** | 提供容器化部署示例 |
| **Fleet zip 导入** | 批量导入 agent 上下文 |

适用于将 agent 作为后台服务运行的场景，而非一次性任务。

---

## 九、可观测性与生产化

### 9.1 LangSmith 集成

- **追踪**：请求级追踪，可视化 agent 行为链路
- **调试**：逐步回放 agent 决策过程
- **评估**：对 agent 输出进行批量评估
- **监控**：生产环境持续监控 agent 表现

### 9.2 生产部署

DeepAgents 基于 LangGraph（流式、持久化、检查点），天然支持生产部署。详见 [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) 完整指南。

---

## 十、与 LangChain/LangGraph 的关系

| 问题 | 答案 |
|------|------|
| **何时用 DeepAgents？** | 需要完整 harness：规划、上下文管理、委托——开箱即用 |
| **何时用 `create_agent`？** | 需要轻量 harness，不需要内置中间件 |
| **何时用 LangGraph？** | agent 循环本身不合适，需要自定义图编排 |

三者可组合：LangGraph 图 → DeepAgents 子代理 → 复用 harness 默认能力。

---

## 十一、框架对比

| 维度 | DeepAgents | LangChain Agent | LangGraph | CrewAI | AutoGen |
|------|------------|-----------------|-----------|--------|---------|
| **抽象层级** | Harness（高层套件） | 中间层 Agent | 底层运行时 | Multi-Agent 编排 | 对话式 Multi-Agent |
| **文件系统** | ✅ 内置 7 个工具 + 多模态 + 权限 | ❌ 需手动添加 | 需手动构建 | ❌ | ❌ |
| **子代理** | ✅ 内置 task 工具 + 隔离上下文 | ❌ | 可手动构建 | ✅ 核心概念 | ✅ 核心概念 |
| **上下文管理** | ✅ 摘要 + 卸载 + Prompt 缓存 | ❌ | 需手动实现 | 部分支持 | ❌ |
| **HITL** | ✅ interrupt_on 声明式 | 需手动配置 | ✅ `interrupt()` | 有限支持 | ✅ |
| **Skills** | ✅ agentskills.io 标准 + 渐进式披露 | ❌ | ❌ | ❌ | ❌ |
| **Memory** | ✅ AGENTS.md + 可更新 + 多后端 | 需手动 | ✅ Store | 基本 | ❌ |
| **MCP** | ✅ 完整支持 | 需手动集成 | 需手动 | ❌ | ❌ |
| **生产化** | ✅ LangSmith + 监控 + 评估 | 需自行搭建 | ✅ | ⚠️ 有限 | ⚠️ 有限 |
| **多语言** | Python + JS/TS | Python + JS | Python + JS | Python | Python + .NET |
| **定位** | 通用 Agent 框架，灵感来自 Claude Code | 基础 Agent 构建块 | 自定义工作流引擎 | 角色扮演 Multi-Agent | 对话式 Agent 协作 |

---

## 十二、快速入门

```python
# 安装
# uv add deepagents

from deepagents import create_deep_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 创建 agent
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

# 调用
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

```python
# 复杂示例：带子代理的研究助手
from deepagents import create_deep_agent

research_agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt="You are a research assistant. Plan your research, delegate to sub-agents, and produce a comprehensive report.",
    subagents=[
        {
            "name": "web-searcher",
            "system_prompt": "Search the web and summarize findings concisely. Return key facts with sources.",
            "tools": [web_search, fetch_page],
        },
        {
            "name": "code-analyzer",
            "system_prompt": "Analyze code repositories. Read, search, and explain code structure and patterns.",
            "tools": [read_file, grep_code, glob_files],
        },
    ],
    interrupt_on={"edit_file": True},
)

result = research_agent.invoke({
    "messages": "Research LangGraph's architecture and write a detailed analysis document."
})
```

---

## 十三、项目结构

```
deepagents/
├── .github/              # CI/CD 工作流、发布文档
├── .vscode/              # IDE 设置
├── examples/             # 可运行的 agent 示例与模式
├── libs/                 # 核心库代码
│   ├── deepagents/       # 核心 SDK 包
│   ├── deepagents-code/  # 编码智能体 (dcode CLI)
│   ├── deepagents-cli/   # CLI 工具
│   ├── deepagents-talon/ # 本地运行时宿主 (v0.0.3)
│   └── partners/         # 合作伙伴包 (Daytona, Modal, Runloop)
├── AGENTS.md             # Agent 行为说明
├── README.md             # 项目主文档
├── LICENSE               # MIT 许可证
├── .mcp.json             # MCP 服务器配置参考
├── hatch_build.py        # 构建钩子，CI 构建时嵌入 commit SHA
└── release-please-config.json  # 自动发布配置
```

---

## 十四、设计决策与安全模型

### 核心设计决策

| 决策 | 理由 |
|------|------|
| **"Trust the LLM" 模型** | Agent 可执行工具允许的任何操作，边界在工具/沙箱层面 |
| **文件系统为根基** | 持久化能力通过文件系统抽象提供，而非独立存储层 |
| **子代理 == 隔离** | 子代理是最小隔离单元，重型任务不污染主上下文 |
| **渐进式披露** | 技能和记忆按需加载，保持 token 经济性 |

### 安全模型

安全边界应在**工具和沙箱层面**强制执行，不应依赖模型自我约束。执行危险操作时启用 `interrupt_on` 人工审批。

---

## 十五、路线图

DeepAgents 由 LangChain 团队持续维护（截至 2026.07，已积累 2600+ 提交），目前处于稳定迭代阶段。

**近期更新（2026 Q2-Q3）：**
- [x] `deepagents-talon` v0.0.3：视频帧提取、Fleet zip 导入
- [x] `dcode doctor`：报告精确构建 commit
- [x] CI 本地对等（`make check` / `make bootstrap`）
- [x] `deepagentsjs` JavaScript/TypeScript 版本
- [ ] 更多沙箱后端集成
- [ ] 增强多模态能力

---

## 参考资源

| 资源 | 链接 |
|------|------|
| 官方文档 | [https://docs.langchain.com/oss/python/deepagents/overview](https://docs.langchain.com/oss/python/deepagents/overview) |
| API 参考 | [https://reference.langchain.com/python/deepagents/](https://reference.langchain.com/python/deepagents/) |
| GitHub 仓库 | [https://github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) |
| JS/TS 版本 | [https://github.com/langchain-ai/deepagentsjs](https://github.com/langchain-ai/deepagentsjs) |
| 示例代码 | [https://github.com/langchain-ai/deepagents/tree/main/examples](https://github.com/langchain-ai/deepagents/tree/main/examples) |
| 社区论坛 | [https://forum.langchain.com/c/oss-product-help-lc-and-lg/deep-agents/18](https://forum.langchain.com/c/oss-product-help-lc-and-lg/deep-agents/18) |
| LangChain Academy | [https://academy.langchain.com/](https://academy.langchain.com/) |
| 生产部署指南 | [https://docs.langchain.com/oss/python/deepagents/going-to-production](https://docs.langchain.com/oss/python/deepagents/going-to-production) |
