---
title: 多模态 AI 开发
icon: image
order: 92
---

# 92. 多模态 AI 开发

## 概述

多模态 AI（Multimodal AI）是指能够同时处理和理解多种不同类型数据模态（如文本、图像、音频、视频等）的人工智能系统。在 AI Agent 掌握了自主规划与工具调用的能力之后，多模态 AI 进一步扩展了智能体的感知边界，使其能够"看到"图像、"听到"声音、"读懂"视频，从而实现更接近人类认知的综合理解能力。

### 从单模态到多模态的演进

| 阶段 | 代表技术 | 处理能力 | 局限性 |
|------|---------|---------|--------|
| 单模态（文本） | GPT-3、BERT | 纯文本理解与生成 | 无法处理非文本数据 |
| 双模态（图文） | CLIP、DALL-E | 图文关联理解 | 模态间交互有限 |
| 多模态统一 | GPT-4V、Gemini | 全模态统一理解 | 计算成本高 |
| 全模态交互 | 多模态 Agent | 跨模态推理与生成 | 实时性待提升 |

### 多模态 AI 的核心价值

多模态 AI 的核心优势在于**模态互补**：不同数据模态携带的信息可以相互补充和验证。例如，在视频理解任务中，视觉信息提供画面内容，音频信息提供环境声音和语音，文本信息（字幕）提供语义上下文，三者结合才能形成完整理解。

```
┌──────────────────────────────────────────┐
│           多模态 AI 融合架构              │
├──────────────────────────────────────────┤
│  输入层：文本 | 图像 | 音频 | 视频 | 传感器 │
│  编码层：Text Encoder | Vision Encoder    │
│          | Audio Encoder | Video Encoder  │
│  融合层：Cross-Attention | 联合嵌入空间    │
│  推理层：跨模态推理 | 模态间信息互补       │
│  输出层：文本 | 图像 | 代码 | 多模态响应   │
└──────────────────────────────────────────┘
```

## 多模态模型架构

### Transformer 架构的多模态扩展

现代多模态模型大多基于 Transformer 架构进行扩展。关键创新在于**统一编码空间**和**跨模态注意力机制**：

```python
"""
多模态 Transformer 架构实现
演示如何将文本和图像编码到统一的嵌入空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class VisionEncoder(nn.Module):
    """视觉编码器：使用 ViT（Vision Transformer）"""

    def __init__(self, image_size: int = 224, embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12):
        super().__init__()
        self.patch_size = 16
        self.num_patches = (image_size // self.patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch 嵌入层
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像张量 [batch, 3, H, W]
        Returns:
            视觉嵌入 [batch, num_patches + 1, embed_dim]
        """
        batch_size = x.shape[0]

        # Patch 嵌入
        x = self.patch_embed(x)  # [batch, embed_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # Transformer 编码
        x = self.transformer(x)
        x = self.layer_norm(x)

        return x


class TextEncoder(nn.Module):
    """文本编码器"""

    def __init__(self, vocab_size: int = 50257, embed_dim: int = 768, max_length: int = 77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: 文本 token IDs [batch, seq_len]
        Returns:
            文本嵌入 [batch, seq_len, embed_dim]
        """
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


class MultimodalFusion(nn.Module):
    """多模态融合模块"""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()

        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 模态投影层（确保视觉和文本在同一维度空间）
        self.vision_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

        # 融合后处理
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, vision_embeds: torch.Tensor,
                text_embeds: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行跨模态注意力融合

        Args:
            vision_embeds: 视觉嵌入 [batch, num_patches, embed_dim]
            text_embeds: 文本嵌入 [batch, seq_len, embed_dim]
            attention_mask: 注意力掩码
        Returns:
            融合后的多模态表示 [batch, seq_len, embed_dim]
        """
        # 投影到统一空间
        vision_proj = self.vision_projection(vision_embeds)
        text_proj = self.text_projection(text_embeds)

        # 跨模态注意力：文本作为 query，视觉作为 key/value
        cross_attn_output, _ = self.cross_attention(
            query=text_proj,
            key=vision_proj,
            value=vision_proj,
            key_padding_mask=attention_mask
        )

        # 残差连接
        cross_attn_output = self.layer_norm(cross_attn_output + text_proj)

        # 拼接并融合
        fused = torch.cat([cross_attn_output, text_proj], dim=-1)
        output = self.fusion_layer(fused)

        return output


class MultimodalModel(nn.Module):
    """完整的多模态模型"""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.fusion = MultimodalFusion(embed_dim=embed_dim)
        self.output_head = nn.Linear(embed_dim, 1000)  # 输出层

    def forward(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 编码
        vision_embeds = self.vision_encoder(images)
        text_embeds = self.text_encoder(input_ids)

        # 融合
        fused_embeds = self.fusion(vision_embeds, text_embeds, attention_mask)

        # 输出（使用 CLS token 的表示）
        cls_output = fused_embeds[:, 0, :]
        logits = self.output_head(cls_output)

        return logits
```

## 多模态主流模型对比

| 模型 | 开发者 | 模态支持 | 参数规模 | 主要特点 | 开源情况 |
|------|--------|---------|---------|---------|---------|
| GPT-4V | OpenAI | 文本+图像 | 未公开 | 视觉推理能力极强 | 闭源 |
| Gemini 1.5 | Google | 文本+图像+音频+视频 | 未公开 | 100万 token 上下文 | 闭源/API |
| LLaVA | 开源社区 | 文本+图像 | 7B-34B | 开源多模态标杆 | 开源 |
| Qwen-VL | 阿里巴巴 | 文本+图像 | 7B-72B | 中文理解优秀 | 开源 |
| InternVL | 上海AI实验室 | 文本+图像 | 多种规格 | 视觉编码能力强 | 开源 |
| CLIP | OpenAI | 文本+图像 | ViT-L/14 | 图文匹配基础模型 | 开源 |

## 使用开源模型实战：LLaVA

```python
"""
使用 LLaVA 进行视觉问答（VQA）实战
LLaVA = Large Language-and-Vision Assistant
"""
import torch
from PIL import Image
import requests
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq
)

class LLaVAVQA:
    """基于 LLaVA 的视觉问答系统"""

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        初始化 LLaVA 模型

        Args:
            model_name: HuggingFace 模型名称
        """
        print(f"正在加载模型: {model_name}")
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("模型加载完成")

    def ask(self, image_path: str, question: str) -> str:
        """
        对图片进行提问

        Args:
            image_path: 图片路径或 URL
            question: 问题文本
        Returns:
            模型回答
        """
        # 加载图片
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        # 构建提示词
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        # 处理输入
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)

        # 生成回答
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

        # 解码输出
        response = self.processor.decode(output[0], skip_special_tokens=True)
        return response.split("ASSISTANT:")[-1].strip()


# 使用示例
if __name__ == "__main__":
    # vqa = LLaVAVQA()
    # answer = vqa.ask("photo.jpg", "这张图片中有什么？请详细描述。")
    # print(f"回答: {answer}")
    pass
```

## 多模态 RAG（检索增强生成）

多模态 RAG 将检索增强生成技术扩展到多模态场景，允许模型基于多模态知识库进行回答：

```python
"""
多模态 RAG 系统实现
支持图像和文本的混合检索与生成
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from transformers import CLIPProcessor, CLIPModel
import chromadb
from chromadb.config import Settings

class MultimodalRAG:
    """多模态检索增强生成系统"""

    def __init__(self, clip_model: str = "openai/clip-vit-large-patch14"):
        # 加载 CLIP 模型用于多模态编码
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)

        # 初始化向量数据库
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_multimodal"
        ))
        self.text_collection = self.client.get_or_create_collection("text_docs")
        self.image_collection = self.client.get_or_create_collection("image_docs")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """编码图像为向量"""
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        )
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        return embedding.numpy().flatten()

    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        inputs = self.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        return embedding.numpy().flatten()

    def add_image_document(self, image_path: str, metadata: Dict[str, Any]):
        """添加图像文档到知识库"""
        image = Image.open(image_path).convert("RGB")
        embedding = self.encode_image(image)

        self.image_collection.add(
            embeddings=[embedding.tolist()],
            documents=[metadata.get("description", "")],
            metadatas=[metadata],
            ids=[metadata.get("id", os.path.basename(image_path))]
        )

    def add_text_document(self, text: str, metadata: Dict[str, Any]):
        """添加文本文档到知识库"""
        embedding = self.encode_text(text)

        self.text_collection.add(
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[metadata],
            ids=[metadata.get("id", text[:20])]
        )

    def search(self, query: str, modality: str = "both", top_k: int = 3) -> Dict[str, List]:
        """
        多模态检索

        Args:
            query: 查询文本
            modality: 检索模态 ("text", "image", "both")
            top_k: 返回结果数量
        Returns:
            检索结果
        """
        query_embedding = self.encode_text(query)
        results = {"text": [], "image": []}

        if modality in ["text", "both"]:
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            results["text"] = text_results

        if modality in ["image", "both"]:
            image_results = self.image_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            results["image"] = image_results

        return results

    def generate_answer(self, query: str, context: Dict[str, List]) -> str:
        """
        基于检索上下文生成回答
        （实际项目中接入 LLM API）
        """
        # 构建提示词
        prompt_parts = ["请基于以下信息回答问题：\n"]

        # 添加文本上下文
        if context["text"]:
            prompt_parts.append("相关文档：")
            for doc in context["text"].get("documents", [[]])[0]:
                prompt_parts.append(f"- {doc}")

        # 添加图像上下文描述
        if context["image"]:
            prompt_parts.append("\n相关图片：")
            for doc in context["image"].get("documents", [[]])[0]:
                prompt_parts.append(f"- {doc}")

        prompt_parts.append(f"\n问题：{query}")

        return "\n".join(prompt_parts)
```

## 多模态开发关键技术

### 1. 图像预处理与增强

```python
"""
图像预处理管道
"""
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def get_image_preprocessor(image_size: int = 224):
    """构建图像预处理管道"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
```

### 2. 多模态数据融合策略

| 融合策略 | 实现方式 | 优点 | 缺点 |
|---------|---------|------|------|
| 早期融合 | 拼接原始特征 | 实现简单 | 可能丢失模态特性 |
| 晚期融合 | 独立处理后合并 | 保留模态特性 | 交互建模不足 |
| 交叉注意力 | Cross-Attention 交互 | 深度模态交互 | 计算成本高 |
| 门控融合 | 自适应选择模态权重 | 灵活高效 | 训练复杂 |

## 多模态应用场景

| 应用场景 | 技术栈 | 典型案例 |
|---------|--------|---------|
| 图像描述生成 | VLM + 语言模型 | Google Lookout |
| 视觉问答 | VQA 模型 | 教育辅导系统 |
| 文档理解 | OCR + 多模态 LLM | 票据自动识别 |
| 医疗影像分析 | 医学图像 + 文本推理 | 辅助诊断系统 |
| 工业质检 | 视觉检测 + 异常描述 | 制造缺陷检测 |
| 自动驾驶 | 多摄像头融合 + 语言导航 | 场景理解与决策 |
| 内容审核 | 图像+文本联合判断 | 平台内容安全 |

## 总结

多模态 AI 突破了单一文本模态的限制，使 AI 系统能够像人类一样综合处理视觉、听觉和语言信息。CLIP 架构奠定了图文统一编码的基础，LLaVA 等开源模型让多模态开发触手可及。多模态 RAG 技术进一步扩展了 Agent 的知识边界。随着模型能力的持续提升，多模态 AI 将成为构建下一代智能应用的核心基础设施。

---

**下一篇**: [93. 视觉语言模型](./93-vision-language.md)
