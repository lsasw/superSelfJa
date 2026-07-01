---
title: MCP 协议与网关
icon: server
order: 1
---

# MCP 协议与网关

> Model Context Protocol — AI Agent 的「USB 接口」

MCP（Model Context Protocol）是 Anthropic 提出的开放协议，让 LLM 能够标准化地连接外部工具和数据源。本模块从协议原理到网关实战，覆盖完整落地路径。

## 📚 文档列表

- **[MCP 协议深度解析](protocol-deep-dive.md)** — 四大原语（Initialize/Tools/Resources/Prompts）、JSON-RPC 2.0 消息格式、传输层抽象、与 REST/GraphQL 对比
- **[MCP 网关搭建实战](gateway-build.md)** — Nginx 反向代理 + Redis 缓存 + 三种限流策略 + JWT/API Key 鉴权 + Prometheus 可观测性
- **[OpenAPI → MCP Tool 转换](openapi-to-mcp.md)** — OpenAPI 规范自动映射为 MCP Tool Schema、参数类型转换、批量导入脚本

## 🎯 学习路线

```
协议原理 (深度解析)
    ↓
网关搭建 (Nginx + Redis)
    ↓
工具转换 (OpenAPI → MCP)
```

## 📌 核心要点

| 主题 | 关键概念 |
|------|---------|
| **四大原语** | Initialize / Tools / Resources / Prompts |
| **传输层** | stdio（本地） / HTTP（远程，SSE） |
| **鉴权方式** | API Key / JWT Bearer / OAuth |
| **限流策略** | 固定窗口 / 滑动日志 / 令牌桶 |
| **可观测** | Prometheus metrics + Grafana dashboard |

## 🔗 相关阅读

- [AI Agent 工程化](../ai-agent-engineering/) — Function Call 与 Skill/MCP/A2A 的关系
- [RAG 与向量数据库](../rag-vector-db/) — MCP Tool 调用 RAG 检索管道
