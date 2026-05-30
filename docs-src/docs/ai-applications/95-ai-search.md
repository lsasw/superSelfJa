---
title: AI 搜索技术
icon: search
order: 95
---

# 95. AI 搜索技术

## 概述

AI 搜索技术代表了信息检索领域的一次根本性变革。从传统的关键词匹配到语义理解，从静态索引到动态推理，AI 正在重新定义"搜索"的含义。在代码大语言模型能够生成和理解代码之后，AI 搜索技术则让 AI 具备了从海量信息中精准定位、理解和整合知识的能力。传统的搜索引擎返回一堆链接，而 AI 搜索引擎直接给出经过理解、总结和引用的答案。

### 搜索技术的演进历程

| 阶段 | 时间 | 核心技术 | 代表产品 | 特点 |
|------|------|---------|---------|------|
| 关键词匹配 | 1990s | TF-IDF、倒排索引 | Google、百度 | 基于关键词匹配 |
| 语义搜索 | 2010s | Word Embedding、BERT | Google BERT | 理解词义相关性 |
| 向量检索 | 2020s | 稠密向量、ANN | Elastic ML | 语义相似度匹配 |
| AI 增强搜索 | 2023+ | RAG、Agent | Perplexity、New Bing | 生成式回答+引用 |
| 自主研究 | 2024+ | 多步推理、深度研究 | Deep Research | 自主规划、多源验证 |

### AI 搜索 vs 传统搜索

| 维度 | 传统搜索 | AI 搜索 |
|------|---------|--------|
| 输入方式 | 关键词 | 自然语言问题 |
| 处理逻辑 | 关键词匹配 + PageRank | 语义理解 + 向量检索 |
| 输出形式 | 链接列表 | 综合回答 + 引用 |
| 交互模式 | 单次查询 | 多轮对话、追问 |
| 理解深度 | 浅层匹配 | 深层语义推理 |
| 知识整合 | 不提供 | 跨文档整合 |
| 引用来源 | 链接列表 | 精确段落引用 |

## AI 搜索核心技术栈

### 1. 语义嵌入模型（Embedding Models）

语义嵌入是 AI 搜索的基石，它将文本转换为稠密向量表示，使得语义相似的文本在向量空间中也相互接近：

```python
"""
语义嵌入模型应用
支持文本向量化和相似度计算
"""
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
import hashlib
import json
import os

class EmbeddingService:
    """嵌入向量服务"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._cache: Dict[str, np.ndarray] = {}

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        将文本转换为嵌入向量

        Args:
            text: 输入文本
            use_cache: 是否使用缓存
        Returns:
            嵌入向量
        """
        # 缓存检查
        if use_cache:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

        # 调用 API
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        # 缓存
        if use_cache:
            self._cache[cache_key] = embedding

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """批量嵌入"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            for data in response.data:
                all_embeddings.append(data.embedding)

        return np.array(all_embeddings, dtype=np.float32)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def search_similar(self,
                       query_embedding: np.ndarray,
                       doc_embeddings: np.ndarray,
                       top_k: int = 5) -> List[Tuple[int, float]]:
        """
        在文档集合中搜索最相似的文档

        Args:
            query_embedding: 查询向量
            doc_embeddings: 文档向量集合 [n_docs, dim]
            top_k: 返回结果数
        Returns:
            [(文档索引, 相似度)] 列表
        """
        # 计算所有相似度
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, float(similarities[idx])) for idx in top_indices]


class LocalEmbeddingService:
    """本地嵌入模型服务（无需 API）"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """本地嵌入"""
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 [CLS] token 的表示
            embedding = outputs.last_hidden_state[:, 0, :]
            # 归一化
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.squeeze().numpy()
```

### 2. 混合搜索架构（BM25 + 向量检索）

AI 搜索的最佳实践是结合传统的关键词匹配（BM25）和语义向量检索：

```python
"""
混合搜索引擎实现
结合 BM25 关键词匹配和语义向量检索
"""
import math
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import re

class BM25Index:
    """BM25 全文索引"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # 词频饱和参数
        self.b = b    # 文档长度归一化参数
        self.documents: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.term_freqs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.N: int = 0

    def add_document(self, doc_id: str, text: str):
        """添加文档到索引"""
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.N += 1

        # 统计词频
        term_freq = Counter(tokens)
        self.term_freqs[doc_id] = term_freq

        # 更新文档频率
        for term in term_freq:
            self.doc_freqs[term] += 1

        # 更新平均长度
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.N

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 搜索

        Args:
            query: 查询文本
            top_k: 返回结果数
        Returns:
            [(文档ID, 分数)] 列表
        """
        query_terms = self._tokenize(query)
        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            df = self.doc_freqs[term]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, term_freq in self.term_freqs.items():
                if term not in term_freq:
                    continue

                tf = term_freq[term]
                doc_len = self.doc_lengths[doc_id]

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_length
                )
                scores[doc_id] += idf * numerator / denominator

        # 排序返回 top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（实际项目应使用专业分词器）"""
        text = text.lower()
        # 中文简单分词（按字符）
        if any('一' <= c <= '鿿' for c in text):
            # 简单处理：提取连续中文字符
            tokens = re.findall(r'[一-鿿]+|[a-z]+', text)
            return tokens
        # 英文分词
        return re.findall(r'[a-z]+', text)


class HybridSearchEngine:
    """混合搜索引擎"""

    def __init__(self, embedding_service, alpha: float = 0.5):
        """
        Args:
            embedding_service: 嵌入向量服务
            alpha: BM25 权重（1-alpha 为向量检索权重）
        """
        self.bm25 = BM25Index()
        self.embedding_service = embedding_service
        self.alpha = alpha
        self.doc_embeddings: Dict[str, np.ndarray] = {}

    def add_document(self, doc_id: str, title: str, content: str):
        """添加文档"""
        full_text = f"{title} {content}"
        self.bm25.add_document(doc_id, full_text)

        # 生成嵌入
        embedding = self.embedding_service.embed_text(full_text)
        self.doc_embeddings[doc_id] = embedding

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        混合搜索

        Args:
            query: 查询文本
            top_k: 返回结果数
        Returns:
            搜索结果列表
        """
        # BM25 搜索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}

        # 向量搜索
        query_embedding = self.embedding_service.embed_text(query)
        vector_results = self.embedding_service.search_similar(
            query_embedding,
            np.array([self.doc_embeddings[doc_id] for doc_id, _ in bm25_results]),
            top_k=top_k * 2
        )

        # 获取文档ID列表
        bm25_doc_ids = [doc_id for doc_id, _ in bm25_results]

        # 归一化分数
        max_bm25 = max(s for _, s in bm25_results) if bm25_results else 1
        max_vector = max(s for _, s in vector_results) if vector_results else 1

        # 融合分数
        final_scores = defaultdict(float)

        for doc_id, score in bm25_results:
            normalized_score = score / max_bm25 if max_bm25 > 0 else 0
            final_scores[doc_id] += self.alpha * normalized_score

        for idx, score in vector_results:
            doc_id = bm25_doc_ids[idx]
            normalized_score = score / max_vector if max_vector > 0 else 0
            final_scores[doc_id] += (1 - self.alpha) * normalized_score

        # 排序
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "doc_id": doc_id,
                "score": score,
                "title": doc_id.split("/")[0] if "/" in doc_id else doc_id,
                "snippet": self.bm25.documents.get(doc_id, "")[:200] + "..."
            }
            for doc_id, score in sorted_results[:top_k]
        ]
```

### 3. RAG 搜索系统（检索增强生成）

RAG 将检索到的文档内容注入到大语言模型的上下文中，使模型能够基于最新、最准确的信息生成回答：

```python
"""
RAG 搜索引擎完整实现
"""
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    source: str
    relevance_score: float
    metadata: dict

@dataclass
class SearchResponse:
    """搜索响应"""
    answer: str
    sources: List[SearchResult]
    thinking_process: Optional[str] = None

class RAGSearchEngine:
    """基于 RAG 的智能搜索引擎"""

    SYSTEM_PROMPT = """你是一个智能搜索助手。请基于提供的参考信息回答用户的问题。

回答要求：
1. 只使用提供的参考信息，不要编造内容
2. 如果参考信息不足以回答问题，请明确说明
3. 在回答中引用具体信息的来源
4. 回答要准确、简洁、有帮助

参考信息格式：
[来源1] 内容...
[来源2] 内容...
...
"""

    def __init__(self, hybrid_search_engine, llm_client, model: str = "gpt-4o"):
        self.search_engine = hybrid_search_engine
        self.llm_client = llm_client
        self.model = model

    def search_and_answer(self, query: str, top_k: int = 5) -> SearchResponse:
        """
        搜索并生成回答

        Args:
            query: 用户查询
            top_k: 检索文档数
        Returns:
            搜索响应
        """
        # 步骤 1: 检索相关文档
        search_results = self.search_engine.search(query, top_k=top_k)

        if not search_results:
            return SearchResponse(
                answer="抱歉，我没有找到与您的问题相关的信息。",
                sources=[]
            )

        # 步骤 2: 构建参考上下文
        context = self._build_context(search_results)

        # 步骤 3: 生成回答
        answer = self._generate_answer(query, context)

        # 步骤 4: 构建响应
        sources = [
            SearchResult(
                content=result.get("snippet", ""),
                source=result.get("title", ""),
                relevance_score=result.get("score", 0),
                metadata=result
            )
            for result in search_results
        ]

        return SearchResponse(
            answer=answer,
            sources=sources
        )

    def _build_context(self, search_results: List[dict]) -> str:
        """构建参考上下文字符串"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[来源{i}] {result.get('title', '')}\n{result.get('snippet', '')}"
            )
        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """基于上下文生成回答"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"参考信息：\n{context}\n\n问题：{query}"}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )

        return response.choices[0].message.content.strip()


class AgenticSearchEngine:
    """多步推理的 Agent 搜索引擎"""

    def __init__(self, llm_client, search_tools: List):
        self.llm_client = llm_client
        self.search_tools = search_tools  # 可用的搜索工具列表
        self.search_history: List[dict] = []

    def deep_search(self, query: str, max_steps: int = 5) -> dict:
        """
        深度搜索：多步自主搜索

        Args:
            query: 查询问题
            max_steps: 最大搜索步数
        Returns:
            综合回答
        """
        # 步骤 1: 分析问题并制定搜索计划
        plan = self._analyze_and_plan(query)

        # 步骤 2: 执行搜索计划
        search_results = []
        for step in plan.get("search_queries", []):
            if len(search_results) >= max_steps:
                break

            result = self._execute_search(step)
            search_results.append({
                "query": step,
                "result": result
            })
            self.search_history.append({
                "step": len(search_results),
                "query": step,
                "result_summary": result[:200] if result else "无结果"
            })

        # 步骤 3: 综合所有搜索结果生成最终回答
        final_answer = self._synthesize(query, search_results)

        return {
            "answer": final_answer,
            "search_steps": self.search_history,
            "total_results": len(search_results)
        }

    def _analyze_and_plan(self, query: str) -> dict:
        """分析问题并制定搜索计划"""
        prompt = f"""请分析以下问题，并制定搜索计划：

问题：{query}

请输出 JSON 格式：
{{
    "analysis": "问题分析",
    "search_queries": ["子查询1", "子查询2", ...],
    "reasoning": "为什么需要这些查询"
}}
"""
        # 调用 LLM 获取搜索计划
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        import json
        return json.loads(response.choices[0].message.content)

    def _execute_search(self, query: str) -> str:
        """执行单个搜索"""
        # 实际项目中调用搜索引擎 API
        return f"搜索 '{query}' 的结果..."

    def _synthesize(self, query: str, results: List[dict]) -> str:
        """综合搜索结果生成回答"""
        context = "\n\n".join([
            f"查询：{r['query']}\n结果：{r['result']}"
            for r in results
        ])

        prompt = f"""请综合以下搜索结果来回答问题：

问题：{query}

搜索结果：
{context}

请提供全面、准确的回答，并注明信息来源。
"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
```

## AI 搜索应用场景

| 应用场景 | 技术方案 | 代表产品 | 关键指标 |
|---------|---------|---------|---------|
| 企业知识库搜索 | RAG + 混合检索 | 内部知识助手 | 准确率 > 90% |
| 电商搜索 | 向量检索 + 推荐排序 | Amazon、淘宝 | 转化率提升 |
| 学术文献搜索 | 语义检索 + 引用分析 | Semantic Scholar | 文献召回率 |
| 法律文档搜索 | 专业嵌入 + BM25 | 法大大 | 精确匹配率 |
| 医疗问答搜索 | 医学知识图谱 + RAG | AI 医疗助手 | 诊断准确率 |
| 新闻聚合搜索 | 多源检索 + 摘要生成 | Perplexity AI | 回答完整性 |

## 总结

AI 搜索技术正在从"找到链接"进化到"给出答案"。通过语义嵌入、混合检索和 RAG 技术的组合，现代 AI 搜索系统能够理解用户的真实意图，从海量文档中检索最相关的信息，并生成准确的综合回答。多步推理的 Agent 搜索进一步扩展了搜索的深度和广度。掌握这些技术，是构建智能信息系统的核心能力。

---

**下一篇**: [96. AI 推荐系统](./96-recommendation-ai.md)
