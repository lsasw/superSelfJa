---
title: AI 推荐系统
icon: thumbs-up
order: 96
---

# 96. AI 推荐系统

## 概述

AI 推荐系统是人工智能在商业领域应用最广泛、最成功的方向之一。从电商平台的"猜你喜欢"到短视频平台的"推荐流"，从音乐应用的"每日推荐"到新闻资讯的"个性化推送"，推荐系统无处不在。在 AI 搜索技术教会系统如何检索信息之后，AI 推荐系统则进一步解决了一个更主动的问题：在用户还没有明确搜索意图时，如何预测并推荐他们可能感兴趣的内容。

### 推荐系统的核心问题

推荐系统本质上解决的是**信息过载**问题。当内容供给远远超过用户的消费能力时，推荐系统充当了智能过滤器的角色，为每个用户找到最相关的内容。

### 推荐系统发展历程

| 阶段 | 时间 | 核心技术 | 代表方法 | 特点 |
|------|------|---------|---------|------|
| 规则推荐 | 2000s | 人工规则 | 热门推荐、编辑精选 | 简单但缺乏个性化 |
| 协同过滤 | 2000s | 矩阵分解 | UserCF、ItemCF、SVD | 基于用户行为相似性 |
| 深度学习推荐 | 2015+ | 深度神经网络 | Wide&Deep、DeepFM | 特征自动学习 |
| 序列推荐 | 2018+ | RNN/Transformer | SASRec、BERT4Rec | 利用行为序列 |
| 大模型推荐 | 2023+ | LLM + 推荐 | LLM-RAG 推荐 | 语义理解 + 推理 |

### 推荐系统的核心组件

```
┌────────────────────────────────────────────────┐
│              AI 推荐系统架构                     │
├────────────────────────────────────────────────┤
│  数据采集层：用户行为 | 物品属性 | 上下文信息     │
│  特征工程层：用户画像 | 物品表示 | 交叉特征       │
│  模型计算层：召回 → 粗排 → 精排 → 重排          │
│  服务部署层：在线推理 | AB 测试 | 实时监控        │
│  反馈优化层：数据回流 | 模型更新 | 策略迭代       │
└────────────────────────────────────────────────┘
```

## 推荐算法详解

### 1. 协同过滤（Collaborative Filtering）

协同过滤是最经典的推荐算法，核心思想是利用用户群体的集体智慧进行推荐：

```python
"""
协同过滤推荐算法实现
包含基于用户和基于物品的协同过滤
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    """协同过滤推荐系统"""

    def __init__(self):
        self.user_item_matrix: np.ndarray = None
        self.user_similarity: np.ndarray = None
        self.item_similarity: np.ndarray = None
        self.user_means: np.ndarray = None

    def fit(self, interactions: List[Tuple[str, str, float]]):
        """
        训练模型

        Args:
            interactions: 用户-物品-评分列表 [(user_id, item_id, rating)]
        """
        # 构建用户和物品的索引映射
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}

        for user_id, item_id, _ in interactions:
            if user_id not in self.user_to_idx:
                idx = len(self.user_to_idx)
                self.user_to_idx[user_id] = idx
                self.idx_to_user[idx] = user_id
            if item_id not in self.item_to_idx:
                idx = len(self.item_to_idx)
                self.item_to_idx[item_id] = idx
                self.idx_to_item[idx] = item_id

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        # 构建用户-物品评分矩阵
        self.user_item_matrix = np.zeros((n_users, n_items))
        for user_id, item_id, rating in interactions:
            u_idx = self.user_to_idx[user_id]
            i_idx = self.item_to_idx[item_id]
            self.user_item_matrix[u_idx, i_idx] = rating

        # 计算用户均值（用于中心化）
        self.user_means = np.zeros(n_users)
        for u in range(n_users):
            rated = self.user_item_matrix[u, :] > 0
            if rated.any():
                self.user_means[u] = np.mean(self.user_item_matrix[u, rated])

        # 计算相似度
        self._compute_similarities()

    def _compute_similarities(self):
        """计算用户和物品的相似度"""
        # 用户相似度（基于余弦相似度）
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        # 将对角线置零（用户不与自己相似）
        np.fill_diagonal(self.user_similarity, 0)

        # 物品相似度
        item_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)
        np.fill_diagonal(self.item_similarity, 0)

    def predict_user_cf(self, user_id: str, item_id: str, k: int = 20) -> float:
        """
        基于用户的协同过滤预测

        Args:
            user_id: 用户ID
            item_id: 物品ID
            k: 最近邻数量
        Returns:
            预测评分
        """
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[item_id]

        # 找到对当前物品有过评分的相似用户
        users_who_rated = np.where(self.user_item_matrix[:, i_idx] > 0)[0]
        users_who_rated = users_who_rated[users_who_rated != u_idx]

        if len(users_who_rated) == 0:
            return self.user_means[u_idx]

        # 取最相似的 k 个用户
        similarities = self.user_similarity[u_idx, users_who_rated]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_users = users_who_rated[top_k_indices]
        top_k_sims = similarities[top_k_indices]

        # 加权平均预测
        numerator = np.sum(
            top_k_sims * (self.user_item_matrix[top_k_users, i_idx] - self.user_means[top_k_users])
        )
        denominator = np.sum(np.abs(top_k_sims))

        if denominator == 0:
            return self.user_means[u_idx]

        prediction = self.user_means[u_idx] + numerator / denominator
        return prediction

    def predict_item_cf(self, user_id: str, item_id: str, k: int = 20) -> float:
        """
        基于物品的协同过滤预测
        """
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[item_id]

        # 找到用户评分过的物品
        items_user_rated = np.where(self.user_item_matrix[u_idx, :] > 0)[0]

        if len(items_user_rated) == 0:
            return 0.0

        # 取与目标物品最相似的 k 个物品
        similarities = self.item_similarity[i_idx, items_user_rated]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_items = items_user_rated[top_k_indices]
        top_k_sims = similarities[top_k_indices]

        # 加权平均预测
        numerator = np.sum(top_k_sims * self.user_item_matrix[u_idx, top_k_items])
        denominator = np.sum(np.abs(top_k_sims))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def recommend(self, user_id: str, n_items: int = 10,
                  method: str = "item_cf") -> List[Tuple[str, float]]:
        """
        为用户推荐物品

        Args:
            user_id: 用户ID
            n_items: 推荐数量
            method: 方法 (user_cf / item_cf)
        Returns:
            [(物品ID, 预测评分)] 列表
        """
        if user_id not in self.user_to_idx:
            return []

        u_idx = self.user_to_idx[user_id]
        already_rated = set(np.where(self.user_item_matrix[u_idx, :] > 0)[0])

        predictions = []
        for i_idx in range(self.user_item_matrix.shape[1]):
            if i_idx in already_rated:
                continue

            item_id = self.idx_to_item[i_idx]
            if method == "user_cf":
                score = self.predict_user_cf(user_id, item_id)
            else:
                score = self.predict_item_cf(user_id, item_id)

            predictions.append((item_id, score))

        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_items]
```

### 2. 矩阵分解（Matrix Factorization）

```python
"""
矩阵分解推荐算法（SVD 风格）
"""
class MatrixFactorization:
    """基于梯度下降的矩阵分解"""

    def __init__(self, n_factors: int = 50, learning_rate: float = 0.005,
                 regularization: float = 0.02, n_epochs: int = 20):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs

        # 模型参数
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = 0

    def fit(self, interactions: List[Tuple[str, str, float]]):
        """训练模型"""
        # 构建索引
        self.user_to_idx = {}
        self.item_to_idx = {}

        for user_id, item_id, _ in interactions:
            if user_id not in self.user_to_idx:
                self.user_to_idx[user_id] = len(self.user_to_idx)
            if item_id not in self.item_to_idx:
                self.item_to_idx[item_id] = len(self.item_to_idx)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        # 初始化参数
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        ratings = np.array([r for _, _, r in interactions])
        self.global_bias = np.mean(ratings)

        # 转换数据
        train_data = [
            (self.user_to_idx[u], self.item_to_idx[i], r)
            for u, i, r in interactions
        ]

        # 梯度下降训练
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)
            total_loss = 0

            for u_idx, i_idx, rating in train_data:
                # 预测
                pred = (self.global_bias +
                        self.user_biases[u_idx] +
                        self.item_biases[i_idx] +
                        np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

                # 误差
                error = rating - pred
                total_loss += error ** 2

                # 更新参数
                self.user_biases[u_idx] += self.lr * (error - self.reg * self.user_biases[u_idx])
                self.item_biases[i_idx] += self.lr * (error - self.reg * self.item_biases[i_idx])

                user_factor_old = self.user_factors[u_idx].copy()
                self.user_factors[u_idx] += self.lr * (
                    error * self.item_factors[i_idx] - self.reg * self.user_factors[u_idx]
                )
                self.item_factors[i_idx] += self.lr * (
                    error * user_factor_old - self.reg * self.item_factors[i_idx]
                )

            # 添加正则化损失
            reg_loss = self.reg * (
                np.sum(self.user_factors ** 2) +
                np.sum(self.item_factors ** 2) +
                np.sum(self.user_biases ** 2) +
                np.sum(self.item_biases ** 2)
            )
            total_loss += reg_loss

            rmse = np.sqrt(total_loss / len(train_data))
            print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

    def predict(self, user_id: str, item_id: str) -> float:
        """预测评分"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_bias

        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[item_id]

        return (self.global_bias +
                self.user_biases[u_idx] +
                self.item_biases[i_idx] +
                np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
```

### 3. 深度学习推荐模型（DeepFM）

```python
"""
DeepFM 推荐模型实现
结合 FM（因子分解机）和 Deep Neural Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class FactorizationMachine(nn.Module):
    """因子分解机层"""

    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_features, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FM 二阶交互计算

        Args:
            x: 稀疏特征索引 [batch, n_features]
        Returns:
            FM 二阶输出 [batch, 1]
        """
        # 获取嵌入向量 [batch, n_features, embed_dim]
        embeddings = self.embedding(x)

        # FM 公式简化计算：
        # sum(vi * vj) = 0.5 * ((sum(vi))^2 - sum(vi^2))
        sum_embedding = torch.sum(embeddings, dim=1)  # [batch, embed_dim]
        sum_squared = torch.sum(embeddings ** 2, dim=1)  # [batch, embed_dim]

        fm_output = 0.5 * torch.sum(
            sum_embedding ** 2 - sum_squared,
            dim=1,
            keepdim=True
        )  # [batch, 1]

        return fm_output


class DeepFM(nn.Module):
    """DeepFM 推荐模型"""

    def __init__(self,
                 field_dims: List[int],
                 embed_dim: int = 16,
                 mlp_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.2):
        """
        Args:
            field_dims: 每个特征域的维度大小
            embed_dim: 嵌入向量维度
            mlp_dims: MLP 各层维度
            dropout: Dropout 概率
        """
        super().__init__()

        total_features = sum(field_dims)

        # 线性部分（一阶特征交互）
        self.linear = nn.Linear(total_features, 1)

        # FM 部分（二阶特征交互）
        self.fm = FactorizationMachine(total_features, embed_dim)

        # MLP 部分（高阶特征交互）
        mlp_input_dim = len(field_dims) * embed_dim
        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(mlp_input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            mlp_input_dim = dim
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

        # 偏移
        self.bias = nn.Parameter(torch.zeros(1))

        # 计算偏移量用于特征域分隔
        self.offsets = [0]
        for dim in field_dims[:-1]:
            self.offsets.append(self.offsets[-1] + dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, n_fields]
        Returns:
            点击率预测 [batch, 1]
        """
        # 线性部分
        linear_output = self.linear(x.float())

        # FM 部分
        fm_output = self.fm(x)

        # MLP 部分
        # 获取嵌入并展平
        embeddings = []
        for i in range(len(self.offsets)):
            start = self.offsets[i]
            end = self.offsets[i + 1] if i < len(self.offsets) - 1 else x.shape[1]
            # 简化处理：直接使用 one-hot 索引
            field_indices = x[:, i] + start
            embeddings.append(field_indices)

        embed_tensor = torch.stack(embeddings, dim=1)  # [batch, n_fields]
        # 实际项目中需要 Embedding 层
        embed_flat = embed_tensor.float().view(embed_tensor.shape[0], -1)
        mlp_output = self.mlp(embed_flat)

        # 组合输出
        output = linear_output + fm_output + mlp_output + self.bias

        return torch.sigmoid(output)
```

## AI 大模型增强的推荐系统

### LLM 增强的推荐流程

```python
"""
基于大语言模型的推荐系统增强
利用 LLM 的语义理解能力提升推荐效果
"""
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    history_items: List[str]
    preferences: Dict[str, float]
    demographic: Dict[str, str]

@dataclass
class ItemProfile:
    """物品画像"""
    item_id: str
    title: str
    description: str
    category: str
    tags: List[str]
    attributes: Dict[str, str]

class LLMEnhancedRecommender:
    """LLM 增强的推荐系统"""

    def __init__(self, llm_client, base_recommender):
        self.llm = llm_client
        self.base_recommender = base_recommender

    def explain_recommendation(self,
                                user_profile: UserProfile,
                                item: ItemProfile) -> str:
        """
        生成推荐理由解释

        Args:
            user_profile: 用户画像
            item: 推荐物品
        Returns:
            推荐理由文本
        """
        prompt = f"""你是一个个性化推荐助手。请为用户推荐这个物品，并给出令人信服的理由。

用户历史行为：
- 浏览/购买过的物品：{', '.join(user_profile.history_items[:5])}
- 偏好：{user_profile.preferences}

推荐物品：
- 名称：{item.title}
- 描述：{item.description}
- 分类：{item.category}
- 标签：{', '.join(item.tags)}

请用一句话说明推荐理由，语气自然友好。
"""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    def cold_start_recommend(self,
                              query: str,
                              items: List[ItemProfile],
                              top_k: int = 5) -> List[ItemProfile]:
        """
        冷启动推荐（无用户历史数据）

        Args:
            query: 用户的自然语言描述
            items: 候选物品池
            top_k: 推荐数量
        Returns:
            推荐物品列表
        """
        # 利用 LLM 理解用户需求
        prompt = f"""用户说："{query}"

请从以下物品中选择最符合用户需求的 {top_k} 个，按匹配度排序。
只输出物品 ID，用逗号分隔。

物品列表：
{chr(10).join([f"- {item.item_id}: {item.title} ({item.description[:50]}...)" for item in items[:50]])}
"""

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        selected_ids = [id.strip() for id in response.choices[0].message.content.split(",")]
        item_map = {item.item_id: item for item in items}

        return [item_map[pid] for pid in selected_ids if pid in item_map][:top_k]

    def diversify_recommendations(self,
                                   recommendations: List[ItemProfile],
                                   diversity_factor: float = 0.3) -> List[ItemProfile]:
        """
        推荐结果多样性优化（MMR - Maximal Marginal Relevance）

        Args:
            recommendations: 初始推荐列表
            diversity_factor: 多样性因子（0-1，越大越多样）
        Returns:
            优化后的推荐列表
        """
        if len(recommendations) <= 2:
            return recommendations

        selected = []
        remaining = list(recommendations)

        # 首先选择最相关的
        selected.append(remaining.pop(0))

        while remaining and len(selected) < len(recommendations):
            best_score = -float('inf')
            best_item = None

            for item in remaining:
                # 相关性分数（假设已排序）
                relevance = len(recommendations) - remaining.index(item)

                # 多样性分数（与已选物品的最大相似度取反）
                max_similarity = 0
                for sel in selected:
                    similarity = self._compute_item_similarity(item, sel)
                    max_similarity = max(max_similarity, similarity)

                diversity = 1 - max_similarity

                # MMR 公式
                score = diversity_factor * relevance + (1 - diversity_factor) * diversity

                if score > best_score:
                    best_score = score
                    best_item = item

            if best_item:
                remaining.remove(best_item)
                selected.append(best_item)

        return selected

    def _compute_item_similarity(self, item1: ItemProfile, item2: ItemProfile) -> float:
        """计算物品间的相似度"""
        # 简单实现：基于标签 Jaccard 相似度
        tags1 = set(item1.tags)
        tags2 = set(item2.tags)

        if not tags1 and not tags2:
            return 0.0

        intersection = tags1 & tags2
        union = tags1 | tags2

        return len(intersection) / len(union) if union else 0.0
```

## 推荐系统评估指标

| 指标 | 全称 | 含义 | 计算方式 |
|------|------|------|---------|
| Precision@K | 精确率 | 推荐列表中用户实际喜欢的比例 | 相关数 / K |
| Recall@K | 召回率 | 用户喜欢的物品中被推荐的比例 | 推荐相关数 / 总相关数 |
| NDCG@K | 归一化折损累计增益 | 考虑排序位置的相关性 | DCG / IDCG |
| MRR | 平均倒数排名 | 第一个相关物品的排名倒数 | 1/rank 的平均值 |
| Hit Rate | 命中率 | 推荐列表中包含至少一个相关物品的用户比例 | 命中用户数 / 总用户数 |
| Coverage | 覆盖率 | 被推荐到的物品占总物品的比例 | 推荐物品数 / 总物品数 |
| Diversity | 多样性 | 推荐结果中物品类别的丰富程度 | 类别数 / K |

## 总结

AI 推荐系统已经从简单的协同过滤演进为融合深度学习、序列建模和大语言模型的复杂系统。协同过滤和矩阵分解提供了经典的用户-物品匹配框架，DeepFM 等深度学习模型能够捕捉高阶特征交互，而 LLM 的引入则为冷启动、解释生成和语义推荐带来了质的飞跃。构建一个生产级推荐系统需要兼顾召回、排序、多样性和实时性等多个维度。

---

**下一篇**: [97. AI for Science](./97-ai-for-science.md)
