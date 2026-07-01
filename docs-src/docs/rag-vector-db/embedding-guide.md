---
title: Embedding 维度选型与模型对比
icon: chart-bar
order: 2
category:
  - RAG
tag:
  - Embedding
  - 向量
---

# Embedding 维度选型与模型对比

> 任务编号：EM-01 / EM-02

## 一、维度 Trade-off 表

| 维度 | 精度 | 速度 | 存储 | 成本 | 推荐场景 |
|------|:----:|:----:|:----:|:----:|---------|
| **128** | ⭐⭐ | ⚡⚡⚡ | 极小 | 💰 | 粗粒度分类、去重 |
| **512** | ⭐⭐⭐ | ⚡⚡⚡ | 小 | 💰 | 简单 QA、标签匹配 |
| **768** | ⭐⭐⭐⭐ | ⚡⚡ | 中 | 💰💰 | **通用推荐**（bge 系列） |
| **1024** | ⭐⭐⭐⭐ | ⚡⚡ | 中 | 💰💰 | 标准 RAG 场景 |
| **1536** | ⭐⭐⭐⭐⭐ | ⚡ | 大 | 💰💰💰 | OpenAI 默认 |
| **3072** | ⭐⭐⭐⭐⭐ | ⚡ | 很大 | 💰💰💰💰 | 高精度语义搜索 |

### 存储成本估算（100 万条向量）

| 维度 | 单条大小 | 总存储 | 索引大小 |
|------|---------|--------|---------|
| 128 | 0.5 KB | 512 MB | ~100 MB |
| 768 | 3 KB | 3 GB | ~600 MB |
| 1536 | 6 KB | 6 GB | ~1.2 GB |
| 3072 | 12 KB | 12 GB | ~2.4 GB |

## 二、三大 Embedding 模型横向对比

```python
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class EmbeddingBenchmark:
    """Embedding 模型对比基准测试"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def load_models(self):
        self.models = {
            "bge-large-zh-v1.5": SentenceTransformer(
                "BAAI/bge-large-zh-v1.5"
            ),
            "m3e-base": SentenceTransformer(
                "moka-ai/m3e-base"
            ),
        }
    
    def benchmark(self, queries: list[str], corpus: list[str]):
        """对比各模型的 Top-5 命中率"""
        
        for model_name, model in self.models.items():
            start = time.time()
            
            # 编码所有文档
            corpus_embeddings = model.encode(
                corpus, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            # 编码查询
            query_embeddings = model.encode(
                queries, 
                normalize_embeddings=True
            )
            
            encode_time = time.time() - start
            
            # 计算相似度 + Top-5 命中率
            hits = 0
            for i, query_emb in enumerate(query_embeddings):
                scores = np.dot(corpus_embeddings, query_emb)
                top_5 = np.argsort(scores)[-5:][::-1]
                if i in top_5:
                    hits += 1
            
            self.results[model_name] = {
                "hit_rate": hits / len(queries),
                "encode_time": encode_time,
                "dimension": model.get_sentence_embedding_dimension(),
                "avg_query_time": (time.time() - start) / len(queries),
            }
        
        return self.results

# === 运行基准 ===
benchmark = EmbeddingBenchmark()
benchmark.load_models()

# 模拟数据：query 和对应的正确答案在同一位置
results = benchmark.benchmark(
    queries=["MCP 协议是什么？", "Redis 持久化方式有哪些？"],
    corpus=[
        "MCP 协议由 Anthropic 提出...",    # 正确答案
        "Spring Boot 自动配置...",
        "Redis 支持 RDB 和 AOF...",         # 正确答案
        "Docker 容器化部署..."
    ]
)

for name, result in results.items():
    print(f"{name}: 命中率={result['hit_rate']:.1%}, "
          f"维度={result['dimension']}, 编码耗时={result['encode_time']:.1f}s")
```

### 预期对比结果

| 模型 | 维度 | Top-5 命中率 | 编码速度 | 推荐场景 |
|------|:---:|:---:|:---:|------|
| **bge-large-zh-v1.5** | 1024 | 92%+ | 中等 | **中文 RAG 首选** |
| **text-embedding-3-small** | 512 | 88% | 快 | 成本敏感 + 英文为主 |
| **m3e-base** | 768 | 89% | 快 | 轻量级中文场景 |

### bge 模型特殊注意事项

```python
# bge 系列模型需要在 query 前加指令前缀
query_prefix = "为这个句子生成表示以用于检索相关文章："

# 文档编码时不需要前缀
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 查询编码时需要前缀！
query_embeddings = model.encode(
    [query_prefix + q for q in queries], 
    normalize_embeddings=True
)
```

## 三、维度选择决策树

```
你的 RAG 场景是？
  ├─ 简单 FAQ（几百条）
  │   └─ → 512 维足够，选 m3e-base（速度快）
  │
  ├─ 知识库搜索（几千-几万条）
  │   └─ → 768-1024 维，选 bge-large-zh（精度高）
  │
  ├─ 专业领域搜索（十万+）
  │   └─ → 1024-1536 维，加 Reranker
  │
  └─ 多模态混合检索
      └─ → 不同模态用不同维度，多路召回
```
