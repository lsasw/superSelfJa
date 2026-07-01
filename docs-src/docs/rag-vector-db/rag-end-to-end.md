---
title: RAG 端到端搭建
icon: search
order: 1
category:
  - RAG
tag:
  - RAG
  - LangChain
  - LlamaIndex
---

# RAG 端到端搭建

> 任务编号：RG-01 / RG-02 / RG-03

## 一、RAG 全链路

```
文档导入 → 文本切分 → Embedding → 向量入库 → 检索 → 重排序 → 生成
  │           │          │          │         │        │        │
  ▼           ▼          ▼          ▼         ▼        ▼        ▼
 PDF/       固定长度    模型选择    PG Vector  余弦/    Reranker  LLM
 Markdown/  语义切分   bge/       Milvus     BM25    Cross-    拼接
 数据库      递归切分   OpenAI     Qdrant     混合     Encoder  生成
```

## 二、端到端实现（RG-01）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class RAGSystem:
    """端到端 RAG 系统"""
    
    def __init__(self, connection_string: str):
        # Embedding 模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # LLM
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # 向量数据库
        self.connection_string = connection_string
        self.vectorstore = None
    
    def ingest_documents(self, documents: list[str], 
                         metadatas: list[dict] = None):
        """步骤 1-2：文档切分 + Embedding + 入库"""
        
        # 文本切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        chunks = []
        chunk_metadatas = []
        
        for i, doc in enumerate(documents):
            doc_chunks = splitter.split_text(doc)
            chunks.extend(doc_chunks)
            if metadatas:
                chunk_metadatas.extend([metadatas[i]] * len(doc_chunks))
        
        print(f"[RAG] 文档切分为 {len(chunks)} 个 chunk")
        
        # Embedding + 入库
        self.vectorstore = PGVector.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            connection_string=self.connection_string,
            collection_name="rag_docs",
            metadatas=chunk_metadatas if chunk_metadatas else None
        )
        
        print(f"[RAG] 已入库 {len(chunks)} 条向量")
    
    def query(self, question: str, top_k: int = 5) -> dict:
        """步骤 4-6：检索 + 重排序 + 生成"""
        if not self.vectorstore:
            raise ValueError("请先导入文档")
        
        # BM25 + 向量混合检索
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 2}  # 多召回一些给 Reranker
        )
        
        # Reranker 重排序
        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-large"
            ),
            top_n=top_k
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever
        )
        
        # 构建 QA 链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=compression_retriever,
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [
                {"content": doc.page_content[:200], 
                 "metadata": doc.metadata}
                for doc in result["source_documents"]
            ]
        }

# === 使用 ===
rag = RAGSystem(connection_string="postgresql://...")

# 导入文档
documents = [
    "Spring Boot 是一个基于 Spring 框架的快速开发工具...",
    "MCP 协议由 Anthropic 提出，用于标准化 AI-工具的交互...",
    "Redis 是一个高性能的键值对数据库，支持多种数据结构..."
]
rag.ingest_documents(documents)

# 查询
result = rag.query("MCP 协议是什么？由谁提出？")
print(result["answer"])
```

## 三、三种切分策略对比（RG-02）

```python
import time
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

def compare_splitters(document: str, query: str, vectorstore_class):
    """对比三种切分策略的检索效果"""
    
    splitters = {
        "固定长度(500)": CharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separator="\n"
        ),
        "递归切分": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        ),
        "语义切分(按段落)": TokenTextSplitter(
            chunk_size=500, chunk_overlap=50
        ),
    }
    
    results = {}
    for name, splitter in splitters.items():
        chunks = splitter.split_text(document)
        
        # 创建向量库并检索
        vs = vectorstore_class.from_texts(chunks, embeddings)
        docs = vs.similarity_search_with_score(query, k=5)
        
        results[name] = {
            "chunks": len(chunks),
            "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks),
            "top_scores": [score for _, score in docs],
            "top_contents": [doc.page_content[:100] for doc, _ in docs],
        }
    
    return results
```

### 切分策略选择指南

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 固定长度 | 简单、可控 | 可能切断语义 | 格式统一的文档 |
| 递归切分 | 尽可能保持语义完整 | 可能产生过长/过短的 chunk | **通用推荐** |
| 语义切分 | 语义边界最清晰 | 依赖模型质量、速度慢 | 高质量长文档 |

## 四、混合检索 Pipeline（RG-03）

```python
from rank_bm25 import BM25Okapi
import jieba

class HybridRetriever:
    """混合检索器：BM25 + 向量 + Reranker"""
    
    def __init__(self, vectorstore, documents: list[str]):
        self.vectorstore = vectorstore
        
        # BM25 索引
        tokenized = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = documents
    
    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """混合检索"""
        
        # 1. BM25 关键词检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(
            enumerate(bm25_scores), 
            key=lambda x: x[1], reverse=True
        )[:top_k * 2]
        
        # 2. 向量相似度检索
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=top_k * 2
        )
        
        # 3. 融合（RRF: Reciprocal Rank Fusion）
        fused = self._rrf_fusion(bm25_top, vector_results, k=60)
        
        # 4. 取 Top-K
        sorted_results = sorted(fused.items(), 
                                key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc_id": doc_id, "score": score} 
                for doc_id, score in sorted_results]
    
    def _rrf_fusion(self, bm25_results, vector_results, k=60):
        """RRF 算法融合两种排序"""
        scores = {}
        
        # BM25 贡献
        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # 向量检索贡献
        for rank, (doc, score) in enumerate(vector_results):
            doc_id = doc.metadata.get("doc_id", hash(doc.page_content))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        return scores
```
