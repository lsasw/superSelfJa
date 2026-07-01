---
title: RAG 与向量数据库
icon: search
order: 2
---

# RAG 与向量数据库

> Retrieval-Augmented Generation — 让大模型拥有「长期记忆」和「知识库」

RAG 是目前企业级 AI 应用最核心的架构模式之一。本模块从端到端实现出发，深入 Embedding 选型和向量数据库对比。

## 📚 文档列表

- **[RAG 端到端搭建](rag-end-to-end.md)** — Python 完整实现：文档加载 → 切分 → Embedding → 向量存储 → 检索 → 生成；三种切分策略对比（递归字符/语义/基于结构）；混合检索（稠密+稀疏）
- **[Embedding 选型指南](embedding-guide.md)** — OpenAI vs Cohere vs 本地模型；多语言支持；维度选择 Trade-off（128d~4096d）；Matryoshka Representation Learning；缓存与批处理优化
- **[向量数据库选型](vector-database.md)** — Milvus/Qdrant/Chroma/Pgvector 对比；内存 vs磁盘 vs混合索引；HNSW/IVF/Flat 原理；音频/视频异构数据扩展方案

## 🎯 学习路线

```
端到端 RAG 实现 (Python)
    ↓
Embedding 选型 (模型+维度)
    ↓
向量数据库 (索引+部署)
```

## 📌 核心架构

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│ 文档加载     │ →   │ 文档切分  │ →   │ Embedding   │ →   │ 向量存储  │
│ PDF/HTML/TXT │     │ Chunking │     │ 模型推理     │     │ Indexing │
└─────────────┘     └──────────┘     └─────────────┘     └──────────┘
                                                               ↓
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│ 最终答案     │ ←   │ LLM 生成  │ ←   │ 上下文组装   │ ←   │ 相似检索  │
│ 引用+来源   │     │ Prompt   │     │ Top-K        │     │ ANN Search│
└─────────────┘     └──────────┘     └─────────────┘     └──────────┘
```

## 🔗 相关阅读

- [AI Agent 工程化](../ai-agent-engineering/) — Agent 调用 RAG 作为 Memory 层
- [MCP 协议与网关](../mcp-gateway/) — 通过 MCP Tool 暴露 RAG 能力
- [大语言模型](../ai-llm/74-rag.html) — RAG 检索增强生成理论
