---
title: Function Call 全链路
icon: code
order: 2
category:
  - AI Agent
tag:
  - Function Call
  - Tool Calling
  - LangChain
---

# Function Call 全链路

> 任务编号：FC-01 / FC-02 / FC-03

## 一、全链路时序图

```
调用方                    LLM                     Tool Registry            Tool Executor
  │                        │                         │                        │
  │  ① 注册 Tool schema    │                         │                        │
  │ ──────────────────────→│                         │                        │
  │                        │  ② 存储 schema          │                        │
  │                        │ ────────────────────────→│                        │
  │  ③ 发送 User Message   │                         │                        │
  │ ──────────────────────→│                         │                        │
  │                        │  ④ 判断需要调用 Tool     │                        │
  │                        │  ⑤ 查询 Tool schema     │                        │
  │                        │ ────────────────────────→│                        │
  │                        │ ←────────────────────────│                        │
  │                        │                         │                        │
  │  ⑥ 返回 Function Call  │                         │                        │
  │ ←──────────────────────│                         │                        │
  │   (tool_call JSON)     │                         │                        │
  │                        │                         │                        │
  │  ⑦ 执行 Tool 调用      │                         │                        │
  │ ──────────────────────────────────────────────────────────────────────────→│
  │                        │                         │                        │
  │  ⑧ 返回 Tool Result    │                         │                        │
  │ ←──────────────────────────────────────────────────────────────────────────│
  │                        │                         │                        │
  │  ⑨ 将结果回灌上下文     │                         │                        │
  │ ──────────────────────→│                         │                        │
  │  (含 tool_call_id)    │                         │                        │
  │                        │                         │                        │
  │  ⑩ 生成最终回复        │                         │                        │
  │ ←──────────────────────│                         │                        │
```

## 二、核心概念拆解

### 2.1 Schema 定义

Tool 的 JSON Schema 是 Function Call 的"接口契约"：

```python
# OpenAI Function Call Schema 定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气，返回温度和天气状况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'",
                        "enum": ["北京", "上海", "深圳", "广州"]
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认摄氏度"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

**Schema 设计原则：**

| 原则 | 说明 |
|------|------|
| `name` 语义化 | 动词+名词，如 `search_documents`、`send_email` |
| `description` 详尽 | LLM 依赖 description 决定何时调用；写清楚"什么时候该调、什么时候不该调" |
| `parameters` 约束 | 用 `enum` 限制可选值，减少幻觉 |
| `required` 精确 | 只标记真正必填的字段 |

### 2.2 大模型解析

LLM 收到 message 后做两件事：

1. **判断意图**：这个请求需要调用工具吗？
2. **提取参数**：从用户输入中抽取参数值

```json
// LLM 返回的 Function Call 响应
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"北京\"}"
  }
}
```

### 2.3 执行器调度

```python
import json

class ToolExecutor:
    """通用 Tool 执行器"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, func):
        """注册一个 Tool"""
        self.tools[name] = func
    
    def execute(self, tool_call: dict) -> str:
        """根据 LLM 返回的 tool_call 执行函数"""
        name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        if name not in self.tools:
            return json.dumps({"error": f"Tool '{name}' not found"})
        
        try:
            result = self.tools[name](**arguments)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})
```

### 2.4 结果回灌

```python
# 将 Tool 执行结果回灌到对话上下文
messages.append({
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": tool_result
})

# 再次调用 LLM，让它基于 Tool 结果生成最终回复
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
```

## 三、三种 Tool 方案的 Schema 对比

| 维度 | OpenAI Function Call | LangChain Tool | MCP Tool |
|------|---------------------|---------------|----------|
| **Schema 格式** | JSON Schema (function.parameters) | Pydantic BaseModel | JSON Schema (inputSchema) |
| **注册方式** | `tools` 数组传入 API | `@tool` 装饰器 | MCP Server `list_tools()` |
| **执行位置** | 客户端自行执行 | 客户端自行执行 | MCP Server 远程执行 |
| **传输协议** | REST API (一次性) | 内存调用 | JSON-RPC (流式) |
| **适用场景** | 简单单步调用 | Python 生态内部 | 跨语言/跨服务调用 |

### 代码对比

```python
# === OpenAI 原生 Function Call ===
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 25}

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

# === LangChain Tool ===
from langchain.tools import tool

@tool
def get_weather(city: str) -> dict:
    """查询指定城市的天气"""
    return {"city": city, "temp": 25}

# === MCP Tool（服务端）===
# mcp_server.py
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="查询天气",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        )
    ]
```

## 四、LangChain Tool 调用 Demo

> 任务 FC-02：让 Agent 调用「查天气」工具，打印每一步的原始 JSON。

```python
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

# 1. 定义 Tool
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的天气。city 必须是城市名称。"""
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 28, "condition": "晴"},
        "上海": {"temp": 32, "condition": "多云"},
        "深圳": {"temp": 35, "condition": "雷阵雨"},
    }
    data = weather_data.get(city, {"temp": 25, "condition": "未知"})
    return json.dumps(data, ensure_ascii=False)

# 2. 创建 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3. 创建 Agent（带 verbose 输出每一步）
agent = create_tool_calling_agent(llm, [get_weather], prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_weather],
    verbose=True,  # 打印每一步！
    handle_parsing_errors=True
)

# 4. 执行
result = agent_executor.invoke({
    "input": "北京今天天气怎么样？"
})
```

**Verbose 输出示例（每一步的原始信息）：**

```
> Entering new AgentExecutor chain...

Invoking: `get_weather` with `{'city': '北京'}`
responded: 我需要调用天气工具来查询北京的天气

{"temp": 28, "condition": "晴"}

北京今天的天气是晴天，气温 28 摄氏度。
```

## 五、常见踩坑与排查

| 问题 | 原因 | 解决 |
|------|------|------|
| LLM 调了不该调的工具 | `description` 不够精确 | 在 description 中写明"什么场景下使用"和"什么场景下不要使用" |
| 参数传错了类型 | schema 缺少 `type` 约束 | 给每个参数明确声明类型 |
| Tool 执行超时 | 外部 API 响应慢 | 设置 timeout + 重试机制 |
| 多轮 Tool 调用死循环 | Agent 无法判断何时停止 | 设置 `max_iterations` 上限 + 添加 finish tool |
| Tool 结果太大 | 返回了完整文档 | 截断 + 摘要，控制 token 消耗 |

## 📚 延伸阅读

- [Multi-Agent 编排模式](./multi-agent-orchestration.md)
- [Skill vs MCP vs A2A 对比](./skill-mcp-a2a.md)
- [OpenAI Function Calling 官方文档](https://platform.openai.com/docs/guides/function-calling)
