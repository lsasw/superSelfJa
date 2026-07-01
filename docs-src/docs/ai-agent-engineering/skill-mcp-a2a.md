---
title: Skill vs MCP vs A2A
icon: balance-scale
order: 4
category:
  - AI Agent
tag:
  - Skill
  - MCP
  - A2A
---

# Skill vs MCP vs A2A — 三层抽象的边界

> 任务编号：SM-01 / SM-02
> 这是面试中被追问最深的一道题（有赞追问 7 层），必须彻底搞懂。

## 一、一句话区分

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Skill  =  "教 Agent 怎么思考"（约束 + Prompt 组合体）       │
│   MCP    =  "给 Agent 提供工具"（工具能力的标准化协议）       │
│   A2A    =  "让 Agent 之间对话"（Agent 间通信标准）           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 二、核心对比表

| 维度 | Skill | MCP Tool | A2A |
|------|-------|----------|-----|
| **本质** | 约束 + System Prompt | 工具能力标准化协议 | Agent 间通信协议 |
| **定位** | 限定 Agent 的行为边界和领域知识 | 暴露外部能力（API/数据库/文件系统） | 实现多 Agent 协作 |
| **通信方式** | 注入到 LLM 上下文（无网络调用） | JSON-RPC 2.0（双向流） | HTTP + JSON（RESTful） |
| **谁在调用** | LLM 自行遵循约束 | LLM 决定何时调用 Tool | Agent 主动发起 |
| **协议方** | 无统一标准（各家自定义） | Anthropic 主导的开放标准 | Google 主导的开放标准 |
| **典型实现** | CodeBuddy Skill、Cursor Rules | MCP Server（文件/数据库/Git） | A2A Server + Agent Card |
| **状态** | 无状态 | 无状态（每次调用独立） | 有状态（Task 生命周期） |
| **适用场景** | 限定 Agent 角色、注入领域知识 | 读写外部数据、调用 API | 复杂任务的多 Agent 分解 |

## 三、逐层深入

### 3.1 Skill — 约束 + Prompt 组合体

```yaml
# Skill 示例：将 Agent 限制为「代码审查专家」
name: code-reviewer
description: 代码审查专家 Skill
constraints:
  - 只审查代码质量和安全性
  - 不修改代码，只给出建议
  - 使用中文回答
  - 每条建议标注严重程度（严重/警告/建议）

tools_allowed:
  - read_file
  - search_code
  # 不允许：write_file、execute_command

system_prompt_override: |
  你是一个资深代码审查专家。审查时重点关注：
  1. 安全漏洞（SQL 注入、XSS、敏感信息泄露）
  2. 并发安全（竞态条件、死锁风险）
  3. 错误处理（异常捕获、资源释放）
  4. 代码可读性（命名、注释、复杂度）
  
  每条审查意见必须标注严重程度：🔴严重 / 🟡警告 / 🟢建议
```

### 3.2 MCP Tool — 工具能力的标准化协议

```python
# MCP Server 示例：暴露文件系统工具
# 客户端通过 JSON-RPC 调用

import asyncio
from mcp.server import Server, stdio_server
from mcp.types import Tool, TextContent

server = Server("filesystem-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="读取文件内容。参数 path 必须是项目目录内的相对路径。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件相对路径"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="search_content",
            description="搜索文件内容（基于 ripgrep）。返回匹配行和文件路径。",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"}
                },
                "required": ["pattern"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "read_file":
        path = arguments["path"]
        content = open(path).read()
        return [TextContent(type="text", text=content)]
    elif name == "search_content":
        # 执行搜索逻辑...
        pass

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

asyncio.run(main())
```

### 3.3 A2A — Agent 间通信标准

```python
# A2A Agent 示例

from a2a.server import A2AServer
from a2a.types import AgentCard, AgentSkill

# 定义 Agent 的身份卡片
agent_card = AgentCard(
    name="ResearchAgent",
    description="研究 Agent，擅长信息收集和文档分析",
    url="http://localhost:8001",
    skills=[
        AgentSkill(
            id="web_research",
            name="网络调研",
            description="搜索和整理网络信息"
        ),
        AgentSkill(
            id="document_analysis",
            name="文档分析",
            description="分析长文档并提取关键信息"
        )
    ]
)

server = A2AServer(agent_card=agent_card)

@server.task_handler
async def handle_task(task):
    """处理来自其他 Agent 的任务"""
    # Agent 间通过标准 HTTP API 通信
    # 支持 streaming、状态查询、中断恢复
    pass

server.start()
```

## 四、面试追问链：7 层深度

```
Q1: "Skill 和 MCP 有什么区别？"
    ↓
Q2: "都是在约束 Agent 行为，为什么需要两个概念？"
    ↓
Q3: "Skill 里的 tools_allowed 和 MCP 的 list_tools，谁说了算？"
    ↓
Q4: "如果 Skill 说只能用 read_file，但 MCP Server 提供了 write_file，Agent 能用吗？"
    ↓
Q5: "A2A 和 MCP 都是协议，为什么不能二合一？"
    ↓
Q6: "一个 Skill 可以绑定多个 MCP Server 吗？反过来呢？"
    ↓
Q7: "设计一个 Skill → MCP → A2A 的完整分层架构，每一层做什么？"
```

### 第 7 层答案：完整分层架构

```
┌──────────────────────────────────────────────────────────────┐
│                        用户请求                              │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  L1: Skill 层（约束 & 角色）                                  │
│  - 定义 Agent 的身份、行为边界、领域知识                        │
│  - 限制可用的 MCP Server 白名单                               │
│  - 注入 System Prompt 模板                                    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  L2: MCP 层（工具能力）                                       │
│  - 暴露外部能力：文件、数据库、API、搜索                        │
│  - 标准化 Tool schema 注册                                    │
│  - JSON-RPC 协议通信                                          │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  L3: A2A 层（Agent 协作）                                     │
│  - 当任务超出单个 Agent 能力时，分发给其他 Agent                │
│  - Task 生命周期管理（创建/执行/状态/完成）                     │
│  - Agent Card 服务发现                                        │
└──────────────────────────────────────────────────────────────┘
```

**关键规则：**
- Skill 是"守门员"：定义 Agent 能做什么、不能做什么
- MCP 是"工具箱"：提供具体的工具实现
- A2A 是"协作网"：当自己的能力不够时，找其他 Agent 帮忙
- Skill 的白名单可以覆盖 MCP 的注册：即使 MCP Server 提供了 write_file，如果 Skill 禁止写操作，Agent 就不能调用

## 五、最小 Demo 合集（SM-02）

### Skill Demo：约束 Agent 行为

```python
# 一个简单的 Skill 实现
class Skill:
    def __init__(self, name: str, constraints: list[str], system_prompt: str):
        self.name = name
        self.constraints = constraints
        self.system_prompt = system_prompt
    
    def apply(self, messages: list) -> list:
        """将 Skill 约束注入到对话中"""
        constraint_text = "\n".join(f"- {c}" for c in self.constraints)
        system_msg = {
            "role": "system",
            "content": f"{self.system_prompt}\n\n你必须遵守以下约束：\n{constraint_text}"
        }
        return [system_msg] + messages

# 使用
code_reviewer = Skill(
    name="代码审查专家",
    constraints=["只审查不修改", "使用中文", "标注严重程度"],
    system_prompt="你是资深代码审查专家，重点关注安全性和可维护性。"
)

messages = code_reviewer.apply([
    {"role": "user", "content": "审查这段代码：def foo(): pass"}
])
```

### MCP Tool Demo：暴露计算器

```python
# 最小 MCP Tool：计算器
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("calculator-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="calculate",
        description="执行数学计算，支持 + - * / 和括号",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "calculate":
        result = eval(arguments["expression"])
        return [TextContent(type="text", text=str(result))]
```

### A2A Demo：Agent 互调

```python
# 最小 A2A：Agent A 调用 Agent B
import requests

class AgentA:
    def handle(self, task: str):
        if "需要研究" in task:
            # 自己不会做，找 AgentB
            result = requests.post(
                "http://agent-b:8001/tasks",
                json={"task": task, "sender": "agent-a"}
            ).json()
            return f"AgentA 委托给 AgentB，结果：{result}"
        return f"AgentA 直接处理：{task}"

class AgentB:
    def handle(self, task: str):
        return f"AgentB 完成了研究：" + task.replace("研究", "").upper()

# 调用
a = AgentA()
print(a.handle("帮我研究一下 MCP 协议"))
# 输出：AgentA 委托给 AgentB，结果：AgentB 完成了研究： MCP 协议
```

## 📚 延伸阅读

- [MCP 官方 Spec](https://spec.modelcontextprotocol.io)
- [A2A 协议 (Google)](https://github.com/google/A2A)
- [Multi-Agent 编排模式](./multi-agent-orchestration.md)
