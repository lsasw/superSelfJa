---
title: MCP 协议深度解析
icon: book-open
order: 1
category:
  - MCP
tag:
  - MCP
  - JSON-RPC
  - Tools
---

# MCP 协议深度解析

> 任务编号：MP-01 / MP-02

## 一、MCP 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                    MCP Host (宿主)                        │
│  例如：CodeBuddy、Claude Desktop、VS Code                │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                │
│  │   MCP Client A  │  │   MCP Client B  │                │
│  │  (文件系统)      │  │  (数据库)       │                │
│  └────────┬────────┘  └────────┬────────┘                │
│           │ JSON-RPC           │ JSON-RPC                │
│           │ (stdio/SSE)        │ (WebSocket)             │
└───────────┼────────────────────┼─────────────────────────┘
            ▼                    ▼
┌───────────────────────┐ ┌───────────────────────┐
│   MCP Server A        │ │   MCP Server B        │
│   (文件系统服务器)      │ │   (数据库服务器)       │
│                       │ │                       │
│ Tools:                │ │ Tools:                │
│  - read_file          │ │  - query_sql          │
│  - write_file         │ │  - list_tables        │
│  - search_files       │ │                       │
└───────────────────────┘ └───────────────────────┘
```

## 二、四大核心原语

MCP 协议定义了四种核心交互原语：

### 2.1 Resources — 资源暴露

将数据暴露为 LLM 可读取的"资源"（类 REST 的 GET）。

```json
// Server → Client: 列出可用资源
{
  "method": "resources/list",
  "result": {
    "resources": [
      {
        "uri": "file:///project/src/main.py",
        "name": "main.py",
        "mimeType": "text/x-python",
        "description": "主程序入口"
      },
      {
        "uri": "postgres://users/table",
        "name": "users 表",
        "mimeType": "application/json",
        "description": "用户表结构"
      }
    ]
  }
}

// Client → Server: 读取资源
{
  "method": "resources/read",
  "params": { "uri": "file:///project/src/main.py" }
}
```

### 2.2 Tools — 工具调用

暴露 LLM 可调用的函数（类 REST 的 POST）。

```json
// Server → Client: 列出可用工具
{
  "method": "tools/list",
  "result": {
    "tools": [
      {
        "name": "search_code",
        "description": "搜索代码库",
        "inputSchema": {
          "type": "object",
          "properties": {
            "pattern": { "type": "string" },
            "directory": { "type": "string" }
          },
          "required": ["pattern"]
        }
      }
    ]
  }
}

// Client → Server: 调用工具
{
  "method": "tools/call",
  "params": {
    "name": "search_code",
    "arguments": { "pattern": "TODO", "directory": "/src" }
  }
}
```

### 2.3 Prompts — 提示词模板

提供预定义的 Prompt 模板。

```json
// Server → Client: 列出可用 Prompt 模板
{
  "method": "prompts/list",
  "result": {
    "prompts": [
      {
        "name": "code_review",
        "description": "代码审查 Prompt 模板",
        "arguments": [
          { "name": "language", "required": true },
          { "name": "focus", "description": "审查重点" }
        ]
      }
    ]
  }
}

// Client → Server: 获取渲染后的 Prompt
{
  "method": "prompts/get",
  "params": {
    "name": "code_review",
    "arguments": { "language": "Python", "focus": "安全性" }
  }
}
```

### 2.4 Sampling — 反向 LLM 调用

Server 可以向 Client 请求 LLM 生成（Server 主动发起）。

```json
// Server → Client: 请求 LLM 生成
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      { "role": "user", "content": "总结以下代码的功能..." }
    ],
    "maxTokens": 500
  }
}
```

## 三、MCP vs A2A vs Function Call 完整对比

| 维度 | Function Call | MCP | A2A |
|------|:---:|:---:|:---:|
| **协议标准** | OpenAI 私有 | Anthropic 主导 | Google 主导 |
| **传输方式** | REST API | JSON-RPC 2.0 (stdio/SSE) | HTTP + JSON |
| **通信模式** | 请求-响应 | 双向流 | 请求-响应 + 流式 |
| **谁执行** | 客户端 | MCP Server | 远端 Agent |
| **状态** | 无状态 | 无状态 | 有状态（Task） |
| **服务发现** | Tool schema 内联 | `tools/list` 动态 | Agent Card |
| **适用场景** | 单步工具调用 | 能力标准化暴露 | 多 Agent 协作 |
| **复杂度** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 决策指南

```
需要调用外部 API/数据库？
  ├─ 单次调用、简单场景 → Function Call
  ├─ 需要标准化暴露、跨语言 → MCP
  └─ 需要 Agent 协作 → A2A
```

## 四、MCP 协议通信流程

```
Client                                Server
  │                                      │
  │  ① initialize                        │
  │ ────────────────────────────────────→│
  │ ←────────────────────────────────────│
  │  ② initialized (capabilities)        │
  │                                      │
  │  ③ tools/list                        │
  │ ────────────────────────────────────→│
  │ ←────────────────────────────────────│
  │  ④ [Tool1, Tool2, ...]               │
  │                                      │
  │  ⑤ LLM 决定调用 Tool1                │
  │                                      │
  │  ⑥ tools/call {name: "Tool1", ...}  │
  │ ────────────────────────────────────→│
  │                                      │  ⑦ 执行逻辑
  │ ←────────────────────────────────────│
  │  ⑧ 返回结果                           │
  │                                      │
  │  ⑨ 结果回灌 LLM 上下文                │
```

## 五、传输方式选择

| 传输方式 | 适用场景 | 优点 | 缺点 |
|----------|---------|------|------|
| **stdio** | 本地进程通信 | 零配置、安全 | 只能本机 |
| **SSE (HTTP)** | 远程服务 | 可跨网络、简单 | 单向流 |
| **WebSocket** | 需要双向实时 | 全双工、低延迟 | 配置复杂 |

```json
// MCP Client 配置示例
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "transport": "stdio"         // 本地
    },
    "database": {
      "url": "http://localhost:3001/sse",
      "transport": "sse"           // 远程 HTTP
    },
    "realtime": {
      "url": "ws://localhost:3002",
      "transport": "websocket"     // 远程 WebSocket
    }
  }
}
```
