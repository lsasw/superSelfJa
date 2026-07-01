---
title: 系统设计
icon: project-diagram
order: 3
---

# 系统设计

> 面试高频题 + 工程实战框架 — 从「能答出来」到「能设计出来」

系统设计是高级工程师的核心能力。本模块提供标准化的七步答题框架和精选面试题库。

## 📚 文档列表

- **[系统设计框架与题库](framework-and-cases.md)** — 七步法：需求明确→粗略估算→数据模型→核心 API→高层架构→瓶颈分析→扩展方案；12 周题库覆盖 URL Shortener/Chat/Distributed Cache/Notification/Feed/Search/Rate Limiter/News Feed/Video Stream/Key-Value Store/File System/Unique ID

## 🎯 七步框架

```
① 明确需求范围          "需要哪些功能？用户规模？"
       ↓
② 粗略估算             "QPS？存储量？读写比？"
       ↓
③ 数据模型设计          "关系型？NoSQL？Schema？"
       ↓
④ 核心 API 定义         "RESTful？参数？返回值？"
       ↓
⑤ 高层架构设计          "负载均衡→服务层→缓存→DB→消息队列"
       ↓
⑥ 瓶颈识别与解决        "热点数据？单点故障？一致性？"
       ↓
⑦ 扩展性与权衡         "水平分片？最终一致？CQRS？"
```

## 📌 高频题目

| 难度 | 题目 | 核心考点 |
|------|------|---------|
| ⭐⭐ | URL Shortener | Base62 编码 + 缓存策略 |
| ⭐⭐⭐ | Chat System | WebSocket + 消息持久化 + 已读回执 |
| ⭐⭐⭐⭐ | 分布式缓存 | 一致性哈希 + 故障检测 |
| ⭐⭐⭐ | Notification Service | 推送队列 + 速率限制 |
| ⭐⭐⭐⭐ | News Feed | Fan-out/on-model + Timeline 合并 |
| ⭐⭐⭐⭐⭐ | 搜索引擎 | 倒排索引 + 分面搜索 + Ranking |

## 🔗 相关阅读

- [Java 并发与 JVM](../java-concurrency-jvm/) — 并发设计在系统中的应用
- [数据库进阶](../database/) — 数据库选型对系统设计的影响
- [AI Agent 工程化](../ai-agent-engineering/) — Agent 系统的设计考量
