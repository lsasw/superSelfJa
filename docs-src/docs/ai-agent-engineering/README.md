---
title: AI Agent 架构与工程化
icon: robot
order: 1
index: true
category:
  - AI Agent
tag:
  - Function Call
  - Multi-Agent
  - MCP
  - Memory
---

# AI Agent 架构与工程化

> 来源：[团队技术提升执行清单](../team-process/) · 一、AI Agent 架构与工程化（13 条任务）

## 📖 章节

<Catalog base="/docs/ai-agent-engineering/" />

## 🎯 学习目标

完成本模块后，你将能够：

1. **画出 Function Call 全链路**：从 schema 定义到结果回灌的完整时序
2. **实现三种 Multi-Agent 编排模式**：Sequential / Parallel / Router
3. **清晰区分 Skill / MCP / A2A 三层抽象**：定位、通信方式、适用场景
4. **实现三种 Agent Memory**：短期 / 长期 / 工作记忆

## 🗺️ 知识地图

```
Function Call 全链路
    ├── Schema 定义（Tool 注册）
    ├── 大模型解析（JSON 输出）
    ├── 执行器调度（函数调用）
    └── 结果回灌（上下文拼接）

Multi-Agent 编排
    ├── Sequential（顺序执行）
    ├── Parallel（并行搜索）
    ├── Router（意图路由）
    ├── 文件传输协议
    └── 失败处理策略

Skill vs MCP vs A2A
    ├── Skill：约束 + Prompt 组合体
    ├── MCP：工具能力的标准化协议
    └── A2A：Agent 间通信标准

Agent Memory
    ├── 短期记忆 → Context Window
    ├── 长期记忆 → 向量数据库
    └── 工作记忆 → KV 存储 / Redis
```
