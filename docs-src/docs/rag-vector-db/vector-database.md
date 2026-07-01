---
title: 向量数据库选型与实战
icon: database
order: 3
category:
  - RAG
tag:
  - PG Vector
  - Milvus
  - Qdrant
---

# 向量数据库选型与实战

> 任务编号：VD-01 / VD-02 / VD-03

## 一、PG Vector：IVFFlat vs HNSW（VD-01）

```sql
-- 创建测试表
CREATE TABLE vector_test (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(1024),
    text TEXT
);

-- === IVFFlat 索引 ===
-- 先插入数据，再建索引
CREATE INDEX idx_ivfflat ON vector_test 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- === HNSW 索引 ===
CREATE INDEX idx_hnsw ON vector_test 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

### 性能对比（10 万条 1024 维向量）

| 指标 | IVFFlat | HNSW |
|------|:---:|:---:|
| 建索引耗时 | 8s | 45s |
| 索引大小 | 120 MB | 180 MB |
| QPS（top-10） | 150 | 480 |
| 召回率 (Recall@10) | 95% | 99.5% |
| 写入 QPS | 800 | 500 |

### 选择指南

```
数据量 < 10 万  → IVFFlat（简单够用）
数据量 10-100 万 → HNSW（高性能）
数据量 > 100 万  → 考虑专用向量数据库
```

## 二、向量数据库横向对比（VD-02）

| 维度 | PG Vector | Milvus | Qdrant | Weaviate |
|------|:---:|:---:|:---:|:---:|
| **部署复杂度** | ⭐（已有 PG 就无额外成本） | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **查询性能** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **过滤能力** | ⭐⭐⭐⭐⭐（SQL 原生） | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **亿级支持** | 吃力 | ✅ | ✅ | ✅ |
| **生态集成** | LangChain/Python 成熟 | Python/Java/Go | Rust/Python | Python/Java |
| **推荐场景** | 数据在 PG 中、中等规模 | **大规模专用** | 中小团队快速启动 | 需要内置模型 |

### 选型决策

```
你的数据在哪？
  ├─ PostgreSQL 中 → PG Vector（零迁移成本）
  │
  ├─ 新项目、快速启动 → Qdrant（Docker 一条命令）
  │
  ├─ 大规模（百万+） → Milvus（性能最强）
  │
  └─ 需要内置 Embedding 模型 → Weaviate
```

## 三、异构数据处理链路（VD-03）

### 音频处理链路

```python
import whisper
from sentence_transformers import SentenceTransformer

class AudioProcessor:
    """语音→文本→Embedding"""
    
    def __init__(self):
        self.asr_model = whisper.load_model("medium")  # ASR
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-large-zh-v1.5"
        )
    
    def process(self, audio_path: str) -> list[dict]:
        # 1. ASR 语音转文本
        result = self.asr_model.transcribe(audio_path)
        full_text = result["text"]
        
        # 2. 按段落切分（用时间戳）
        segments = result["segments"]
        chunks = []
        
        for seg in segments:
            text = seg["text"].strip()
            if len(text) < 10:
                continue
            
            # 3. 文本 → Embedding
            embedding = self.embedding_model.encode(
                text, normalize_embeddings=True
            )
            
            chunks.append({
                "text": text,
                "start": seg["start"],
                "end": seg["end"],
                "embedding": embedding.tolist(),
            })
        
        return chunks
```

### 视频处理链路

```python
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class VideoProcessor:
    """视频→抽帧→视觉 Embedding"""
    
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    
    def extract_keyframes(self, video_path: str, 
                          interval: int = 30) -> list:
        """每隔 interval 帧抽取关键帧"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps * 2  # 每 2 秒抽一帧
        
        keyframes = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # OpenCV BGR → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.append({
                    "timestamp": frame_count / fps,
                    "frame": Image.fromarray(frame_rgb)
                })
            
            frame_count += 1
        
        cap.release()
        return keyframes
    
    def embed_keyframes(self, keyframes: list) -> list[dict]:
        """关键帧 → CLIP Embedding"""
        results = []
        
        for kf in keyframes:
            inputs = self.clip_processor(
                images=kf["frame"], return_tensors="pt"
            )
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            results.append({
                "timestamp": kf["timestamp"],
                "embedding": embedding[0].tolist(),
            })
        
        return results
```

### 完整数据流图

```
┌─────────────────────────────────────────────────────────┐
│                     异构数据输入                          │
├──────────────┬──────────────┬──────────────┬────────────┤
│   语音文件    │   视频文件    │   PDF 文档   │   数据库    │
│   .mp3/.wav  │   .mp4       │   .pdf       │   SQL      │
└──────┬───────┴──────┬───────┴──────┬───────┴─────┬──────┘
       │              │              │             │
       ▼              ▼              ▼             ▼
  ┌─────────┐   ┌──────────┐   ┌──────────┐  ┌──────────┐
  │ Whisper │   │   CLIP   │   │ PDF→Text │  │ SQL→Text │
  │  ASR    │   │  视觉模型 │   │  提取     │  │  提取    │
  └────┬────┘   └────┬─────┘   └────┬─────┘  └────┬─────┘
       │              │              │             │
       ▼              ▼              ▼             ▼
  ┌──────────────────────────────────────────────────────┐
  │              Text + Metadata                         │
  │  "今天讨论 MCP 协议..." + {source: "audio", ts: 120}  │
  └──────────────────────┬───────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │            Embedding Model（bge/OpenAI）              │
  │              文本 → 1024 维向量                       │
  └──────────────────────┬───────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │              向量数据库（PG Vector/Milvus）            │
  │   存储：向量 + 文本 + 元数据（source, timestamp...）   │
  └──────────────────────────────────────────────────────┘
```
