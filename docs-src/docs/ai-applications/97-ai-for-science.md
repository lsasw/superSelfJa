---
title: AI for Science
icon: flask
order: 97
---

# 97. AI for Science

## 概述

AI for Science（科学智能）是指利用人工智能技术加速科学研究和创新的方法论与实践。它代表了 AI 从"商业应用"向"科学发现"的重大跨越。从蛋白质结构预测到新材料设计，从药物发现到气候模拟，AI 正在深刻改变科学研究范式。当推荐系统能够在海量商品中精准匹配用户偏好时，AI for Science 则要在浩瀚的科学知识空间中"推荐"出最具潜力的研究假设和发现。

### AI for Science 的核心范式转变

| 科学范式 | 方法 | 代表 | 局限 |
|---------|------|------|------|
| 实验科学 | 观察和实验 | 牛顿时代 | 依赖实验条件 |
| 理论科学 | 数学建模 | 爱因斯坦 | 复杂系统难以建模 |
| 计算科学 | 数值模拟 | 气象模型 | 算力限制、近似误差 |
| 数据驱动科学 | 大数据分析 | 生物信息学 | 仅发现相关性 |
| AI for Science | AI + 物理规律 | AlphaFold | 融合知识与数据 |

### AI 在各科学领域的应用概览

| 科学领域 | AI 应用 | 突破性成果 | 影响 |
|---------|---------|-----------|------|
| 生物学 | 蛋白质结构预测 | AlphaFold2/3 | 解决 50 年难题 |
| 化学 | 分子生成与性质预测 | 自主化学实验室 | 加速药物研发 |
| 物理学 | 量子多体问题求解 | Neural Quantum States | 超越传统方法 |
| 材料科学 | 新材料发现 | GNoME（Google DeepMind） | 发现 220 万种新材料 |
| 天文学 | 星系分类、引力波检测 | 深度学习分析 | 发现效率提升 10 倍 |
| 气象学 | 天气预报 | 盘古气象大模型 | 超越传统数值方法 |
| 数学 | 定理证明辅助 | GPT-f、Lean | 辅助发现新定理 |

## 核心技术与应用

### 1. 蛋白质结构预测（AlphaFold 范式）

蛋白质结构预测是 AI for Science 最标志性的成就。AlphaFold2 在 2020 年的 CASP14 竞赛中达到实验级精度：

```python
"""
简化版蛋白质结构预测系统
基于 AlphaFold 核心思想实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class EvoformerBlock(nn.Module):
    """
    Evoformer 模块：AlphaFold2 的核心组件
    处理 MSA（多序列比对）和 pair 表示的信息交互
    """

    def __init__(self, msa_dim: int = 256, pair_dim: int = 128,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # MSA 行注意力（在序列维度上）
        self.msa_row_attn = nn.MultiheadAttention(
            embed_dim=msa_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.msa_row_norm = nn.LayerNorm(msa_dim)

        # MSA 列注意力（在残基维度上）
        self.msa_col_attn = nn.MultiheadAttention(
            embed_dim=msa_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.msa_col_norm = nn.LayerNorm(msa_dim)

        # 三角乘法注意力（处理 pair 表示）
        self.triangle_attn_starting = TriangleAttention(pair_dim, n_heads)
        self.triangle_attn_ending = TriangleAttention(pair_dim, n_heads)

        # 三角乘法更新
        self.triangle_mul_update = TriangleMultiplication(pair_dim)

        # Pair 更新
        self.pair_transition = nn.Sequential(
            nn.Linear(pair_dim, pair_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim)
        )
        self.pair_norm = nn.LayerNorm(pair_dim)

        # MSA 到 Pair 的信息传递
        self.outer_product_mean = OuterProductMean(msa_dim, pair_dim)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None,
                pair_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            msa: MSA 表示 [batch, n_seq, n_res, msa_dim]
            pair: Pair 表示 [batch, n_res, n_res, pair_dim]
            msa_mask: MSA 注意力掩码
            pair_mask: Pair 注意力掩码
        Returns:
            更新后的 msa 和 pair
        """
        batch, n_seq, n_res, msa_dim = msa.shape

        # MSA 行注意力
        msa_flat = msa.view(batch * n_seq, n_res, msa_dim)
        attn_output, _ = self.msa_row_attn(
            msa_flat, msa_flat, msa_flat,
            key_padding_mask=msa_mask
        )
        msa_flat = self.msa_row_norm(msa_flat + attn_output)
        msa = msa_flat.view(batch, n_seq, n_res, msa_dim)

        # MSA 列注意力
        msa_transposed = msa.transpose(1, 2).contiguous().view(batch * n_res, n_seq, msa_dim)
        attn_output, _ = self.msa_col_attn(
            msa_transposed, msa_transposed, msa_transposed
        )
        msa_transposed = self.msa_col_norm(msa_transposed + attn_output)
        msa = msa_transposed.view(batch, n_res, n_seq, msa_dim).transpose(1, 2)

        # 更新 Pair 表示
        pair = pair + self.outer_product_mean(msa, msa_mask)

        # 三角乘法注意力
        pair = self.triangle_attn_starting(pair, pair_mask) + pair
        pair = self.triangle_attn_ending(pair, pair_mask) + pair

        # 三角乘法更新
        pair = self.triangle_mul_update(pair, pair_mask) + pair

        # Pair transition
        pair = self.pair_norm(pair + self.pair_transition(pair))

        return msa, pair


class TriangleAttention(nn.Module):
    """三角乘法注意力"""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, pair: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, n_res, _, dim = pair.shape

        # 重塑为序列对序列的注意力
        pair_flat = pair.view(batch * n_res, n_res, dim)
        output, _ = self.attention(pair_flat, pair_flat, pair_flat, key_padding_mask=mask)
        output = self.norm(output + pair_flat)

        return output.view(batch, n_res, n_res, dim)


class TriangleMultiplication(nn.Module):
    """三角乘法更新"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.proj_g = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, pair: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        三角乘法：类似矩阵乘法的操作
        O_ik = sum_j (A_ij * B_jk) * G_ik
        """
        a = self.proj_a(pair)  # [B, N, N, D]
        b = self.proj_b(pair)  # [B, N, N, D]
        g = torch.sigmoid(self.proj_g(pair))  # [B, N, N, D]

        # 类似矩阵乘法
        # [B, N, N, D] x [B, N, N, D] -> [B, N, N, D]
        output = torch.einsum('bijd,bjkd->bikd', a, b)
        output = output * g

        output = self.proj_out(output)
        return self.norm(output)


class OuterProductMean(nn.Module):
    """MSA 到 Pair 的外积均值"""

    def __init__(self, msa_dim: int, pair_dim: int):
        super().__init__()
        self.proj = nn.Linear(msa_dim, pair_dim * 2)
        self.out_proj = nn.Linear(pair_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, msa: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, n_seq, n_res, msa_dim = msa.shape

        # 投影
        a = self.proj(msa)  # [B, S, N, 2D]
        a, b = a.chunk(2, dim=-1)  # [B, S, N, D], [B, S, N, D]

        # 外积均值
        output = torch.einsum('bsid,bsjd->bijcd', a, b)
        output = output.sum(dim=1) / n_seq  # 在序列维度上平均
        output = output.view(batch, n_res, n_res, -1)

        return self.norm(self.out_proj(output))


class StructureModule(nn.Module):
    """结构模块：从 pair 表示预测 3D 结构"""

    def __init__(self, pair_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # 预测距离分布
        self.distogram = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64)  # 64 个距离 bin
        )

        # 预测角度（torsion angles）
        self.angle_predictor = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7 * 2)  # 7 个主 torsion angles, 每个用 sin/cos 表示
        )

    def forward(self, pair: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测蛋白质结构

        Args:
            pair: Pair 表示 [batch, n_res, n_res, pair_dim]
        Returns:
            距离矩阵、角度预测
        """
        # 预测距离分布
        dist_logits = self.distogram(pair)  # [B, N, N, 64]

        # 预测角度
        angles = self.angle_predictor(pair[:, 0])  # 取对角线附近的信息 [B, 7*2]

        return {
            "distogram": dist_logits,
            "angles": angles.view(-1, 7, 2)
        }


class ProteinStructurePredictor(nn.Module):
    """完整的蛋白质结构预测模型"""

    def __init__(self, msa_dim: int = 256, pair_dim: int = 128,
                 n_evoformer_layers: int = 48):
        super().__init__()

        # 输入嵌入
        self.msa_embedding = nn.Linear(49, msa_dim)  # one-hot + 额外特征
        self.pair_embedding = nn.Linear(25, pair_dim)  # 残基对特征

        # Evoformer 堆叠
        self.evoformer = nn.ModuleList([
            EvoformerBlock(msa_dim, pair_dim)
            for _ in range(n_evoformer_layers)
        ])

        # 结构模块
        self.structure_module = StructureModule(pair_dim)

    def forward(self, msa_features: torch.Tensor,
                pair_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测蛋白质 3D 结构

        Args:
            msa_features: MSA 特征 [batch, n_seq, n_res, 49]
            pair_features: 残基对特征 [batch, n_res, n_res, 25]
        Returns:
            结构预测结果
        """
        # 嵌入
        msa = self.msa_embedding(msa_features)
        pair = self.pair_embedding(pair_features)

        # Evoformer 迭代
        for layer in self.evoformer:
            msa, pair = layer(msa, pair)

        # 结构预测
        predictions = self.structure_module(pair)

        return predictions
```

### 2. AI 分子设计与药物发现

```python
"""
AI 驱动的分子生成与药物发现系统
"""
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw, rdMolTransforms
from typing import List, Optional

class MolecularVAE(nn.Module):
    """分子变分自编码器"""

    def __init__(self, vocab_size: int = 50, hidden_dim: int = 256,
                 latent_dim: int = 64, max_length: int = 128):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.GRU(hidden_dim, hidden_dim, batch_first=True),
        )

        # 隐变量映射
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder = nn.GRU(hidden_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, smiles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码 SMILES 字符串到隐空间

        Args:
            smiles: tokenized SMILES [batch, seq_len]
        Returns:
            mu, logvar
        """
        _, hidden = self.encoder(smiles)
        hidden = hidden[-1]  # 取最后一层

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从隐变量解码生成 SMILES

        Args:
            z: 隐变量 [batch, latent_dim]
        Returns:
            生成的 token 序列 [batch, max_length, vocab_size]
        """
        batch_size = z.shape[0]

        # 初始输入：起始 token
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim)
        hidden = z.unsqueeze(0)  # [1, batch, latent_dim]

        outputs = []
        for _ in range(self.max_length):
            # 拼接隐变量和当前输入
            decoder_input = torch.cat([decoder_input, z.unsqueeze(1).expand(-1, decoder_input.shape[1], -1)], dim=-1)

            output, hidden = self.decoder(decoder_input, hidden)
            logits = self.fc_out(output)
            outputs.append(logits)

            # 取最大概率的 token 作为下一步输入（贪婪解码）
            next_token = logits.argmax(dim=-1)
            decoder_input = self.decoder_input[:, 0:1, :].clone()  # 重置

        return torch.stack(outputs, dim=1)

    def forward(self, smiles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        mu, logvar = self.encode(smiles)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    def generate_molecules(self, n_samples: int = 100,
                           temperature: float = 1.0) -> List[str]:
        """生成新的分子结构"""
        z = torch.randn(n_samples, self.latent_dim)
        logits = self.decode(z)

        # 采样
        probs = torch.softmax(logits / temperature, dim=-1)
        tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
        tokens = tokens.view(n_samples, self.max_length)

        # 转换为 SMILES 字符串
        smiles_list = self.tokens_to_smiles(tokens)
        return smiles_list
```

### 3. 材料发现系统

```python
"""
AI 驱动的材料发现系统
基于图神经网络的晶体性质预测
"""
class CrystalGraphNetwork(nn.Module):
    """晶体图神经网络"""

    def __init__(self, node_dim: int = 64, edge_dim: int = 32,
                 hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()

        # 原子特征嵌入
        self.atom_embedding = nn.Linear(92, node_dim)  # 元素周期表特征

        # 键特征嵌入
        self.bond_embedding = nn.Linear(10, edge_dim)

        # 消息传递层
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # 读出层
        self.readout = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 预测性质（如形成能）
        )

    def forward(self, atom_features: torch.Tensor,
                bond_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        预测晶体性质

        Args:
            atom_features: 原子特征 [n_atoms, 92]
            bond_features: 键特征 [n_edges, 10]
            edge_index: 边索引 [2, n_edges]
        Returns:
            预测性质
        """
        # 嵌入
        node_h = self.atom_embedding(atom_features)
        edge_h = self.bond_embedding(bond_features)

        # 消息传递
        for layer in self.message_layers:
            node_h = layer(node_h, edge_h, edge_index)

        # 图级读出
        graph_repr = torch.mean(node_h, dim=0)  # 全局平均池化
        prediction = self.readout(graph_repr)

        return prediction


class MessagePassingLayer(nn.Module):
    """消息传递层"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()

        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.update_net = nn.GRUCell(node_dim, node_dim)

    def forward(self, node_h: torch.Tensor,
                edge_h: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_h: 节点特征 [n_nodes, node_dim]
            edge_h: 边特征 [n_edges, edge_dim]
            edge_index: 边索引 [2, n_edges]
        Returns:
            更新后的节点特征
        """
        source, target = edge_index

        # 构建消息
        messages = self.message_net(
            torch.cat([
                node_h[source],
                node_h[target],
                edge_h
            ], dim=1)
        )

        # 聚合消息
        aggregated = torch.zeros_like(node_h)
        aggregated.index_add_(0, target, messages)

        # 更新节点状态
        new_h = self.update_net(aggregated, node_h)

        return node_h + new_h  # 残差连接
```

## AI for Science 的挑战与机遇

| 挑战 | 说明 | 应对策略 |
|------|------|---------|
| 数据稀缺 | 科学数据量小、获取成本高 | 迁移学习、物理约束、主动学习 |
| 可解释性 | 科学发现需要可解释性 | 符号回归、可解释 AI、因果推理 |
| 泛化能力 | 模型需要外推到未知空间 | 物理先验、对称性约束 |
| 不确定性量化 | 科学决策需要可靠的不确定性估计 | 贝叶斯深度学习、集成方法 |
| 多尺度建模 | 从原子到宏观的多尺度问题 | 层次化建模、多保真融合 |

## 总结

AI for Science 正在重新定义科学研究的边界。AlphaFold 解决了困扰生物学 50 年的蛋白质折叠问题，GNoME 发现了数百万种新材料，盘古气象大模型实现了超越传统数值方法的天气预报。AI 不仅仅是加速了科学计算，更重要的是它提供了发现新规律、新假设的新范式。随着 AI 与物理、化学、生物等学科基础理论的深度融合，科学发现的速度和精度都将得到质的飞跃。

---

**下一篇**: [98. 具身智能](./98-embodied-ai.md)
