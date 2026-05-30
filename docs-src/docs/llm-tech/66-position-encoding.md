---
title: 位置编码
icon: ruler
order: 6
---

# 位置编码（Position Encoding）

Transformer 架构中的自注意力机制本质上是排列不变的（permutation-invariant）——它不关心 Token 的顺序。然而，语言是有严格顺序的："猫追狗"和"狗追猫"含义完全不同。为了解决这个问题，位置编码（Position Encoding）被引入 Transformer，为每个 Token 注入位置信息。本文将深入分析各种位置编码方法，从原始的正弦编码到最新的 RoPE 和 YaRN。

## 为什么需要位置编码

自注意力的计算方式决定了它无法区分位置：

```python
# 自注意力计算
def self_attention(Q, K, V):
    """
    Q, K, V: [batch, seq_len, d_model]
    注意力矩阵: softmax(QK^T / sqrt(d_k))  - 只依赖内容，不依赖位置
    """
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, V)

# 问题：以下两个序列会产生相同的注意力矩阵
# "猫 追 狗" 和 "狗 追 猫"
# 因为 QK^T 只计算内容相似度，不感知顺序
```

| 问题 | 说明 |
|------|------|
| 排列不变性 | 交换 Token 顺序，注意力结果不变 |
| 相对距离 | 模型需要知道"相邻"和"遥远"的区别 |
| 外推需求 | 推理时需要处理比训练时更长的序列 |

## 位置编码方法全景图

| 方法 | 类型 | 绝对/相对 | 外推能力 | 计算复杂度 | 代表模型 |
|------|------|-----------|----------|-----------|----------|
| 正弦编码 | 确定性 | 绝对 | 差 | O(1) | 原始 Transformer |
| 学习式编码 | 可训练 | 绝对 | 差 | O(1) | GPT、BERT |
| RoPE | 确定性 | 相对 | 优秀 | O(d) | LLaMA、PaLM |
| ALiBi | 确定性 | 相对 | 优秀 | O(1) | MPT、GLM |
| T5 偏差 | 可训练 | 相对 | 中等 | O(1) | T5 |
| Fire | 可训练 | 相对 | 优秀 | O(1) | — |
| YaRN | 确定性 | 相对（基于 RoPE） | 极强 | O(d) | 长上下文 LLaMA |
| LeX | 确定性 | 相对（基于 RoPE） | 极强 | O(d) | 超长上下文 |

## 绝对位置编码

### 正弦/余弦编码（原始 Transformer）

Vaswani 等人在 "Attention Is All You Need" 中提出的经典方法：

```python
import torch
import math

class SinusoidalPositionEncoding(nn.Module):
    """正弦/余弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 不同维度使用不同频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        返回: x + position_encoding
        """
        return x + self.pe[:, :x.size(1), :]

# 使用示例
pos_encoding = SinusoidalPositionEncoding(d_model=512, max_len=100)
tokens = torch.randn(2, 10, 512)  # 2 个句子，每句 10 个 token
output = pos_encoding(tokens)
```

#### 数学原理

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: Token 的位置（0, 1, 2, ...）
- i: 维度索引（0, 1, 2, ..., d_model/2-1）
- 10000: 最大波长超参数
```

#### 为什么使用正弦和余弦

```
正弦编码的关键性质：

1. 唯一性：每个位置有唯一的编码
2. 平滑性：相邻位置的编码相似
3. 线性关系：PE(pos+k) 可以用 PE(pos) 的线性变换表示
   PE(pos+k) = T_k × PE(pos)  （对固定偏移 k）
   这使得模型可以学习到相对位置关系

4. 频率梯度：低频维度捕获长距离关系，高频维度捕获局部关系
   最低频率波长：10000
   最高频率波长：2π
```

### 学习式位置编码

GPT 系列采用更简单的方法——直接学习位置嵌入：

```python
class LearnedPositionEncoding(nn.Module):
    """学习式位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        # 位置嵌入是一个可训练参数
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.position_embeddings[:, :seq_len, :]

# GPT-2 配置
learned_pe = LearnedPositionEncoding(
    d_model=768, 
    max_len=1024  # GPT-2 最大上下文
)
```

| 对比项 | 正弦编码 | 学习式编码 |
|--------|----------|-----------|
| 可训练 | 否（固定） | 是（端到端学习） |
| 外推 | 理论可以，实际差 | 不能（固定大小） |
| 表达力 | 固定模式 | 自适应数据 |
| 参数量 | 0 | d_model × max_len |
| 长度变化 | 自动适应 | 需要插值或截断 |

## 相对位置编码

绝对位置编码的一个根本问题是：**模型真正需要的是相对位置信息**。两个 Token 之间的距离（如"相隔 3 个词"）比它们的绝对位置（如"第 5 个和第 8 个词"）更重要。

### T5 相对位置偏差

T5 采用将相对位置作为注意力偏置的方式：

```python
class T5RelativePositionBias(nn.Module):
    """T5 相对位置偏置"""
    
    def __init__(self, num_buckets: int = 32, max_distance: int = 128, 
                 num_heads: int = 12):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    def _relative_position_bucket(self, relative_position):
        """将相对位置映射到桶索引"""
        ret = 0
        n = -relative_position
        
        num_buckets = self.num_buckets // 2
        ret += (n < 0).to(relative_position.dtype) * num_buckets
        n = torch.abs(n)
        
        # 小距离：精确桶
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        # 大距离：对数桶
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).to(relative_position.dtype)
        
        val_if_large = torch.min(val_if_large, 
                                  torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, qlen, klen):
        """计算相对位置偏置"""
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # [qlen, klen]
        
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)  # [qlen, klen, num_heads]
        
        return values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, qlen, klen]
```

### ALiBi（Attention with Linear Biases）

ALiBi 不添加位置嵌入，而是在注意力分数上添加与距离成正比的偏置：

```python
class AliBiAttention(nn.Module):
    """ALiBi：线性偏置注意力"""
    
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # 每个头使用不同的斜率
        self.slopes = self._get_slopes(num_heads)
    
    def _get_slopes(self, num_heads):
        """计算每个头的斜率"""
        # 找到最接近 2 的幂
        n = 2 ** math.floor(math.log2(num_heads))
        m_0 = 2.0 ** (-8 / n)  # 初始斜率
        slopes = [m_0 ** (i + 1) for i in range(n)]
        
        # 如果 num_heads 不是 2 的幂，插值补充
        if n < num_heads:
            m_1 = 2.0 ** (-4 / n)
            slopes += [m_1 ** (2 * (i - n) + 1) for i in range(n, num_heads)]
        
        return torch.tensor(slopes)
    
    def forward(self, Q, K, V, causal=True):
        """
        Q, K, V: [batch, num_heads, seq_len, d_k]
        """
        batch_size, num_heads, q_len, d_k = Q.shape
        k_len = K.size(2)
        
        # 标准注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [b, h, q, k]
        
        # ALiBi 偏置
        # 相对距离矩阵
        relative_distance = torch.arange(k_len, device=Q.device) - torch.arange(q_len, device=Q.device).unsqueeze(1)
        # 因果掩码：只保留负距离（过去的 token）
        if causal:
            relative_distance = torch.clamp(relative_distance, max=0)
        
        # 偏置 = 斜率 × 距离
        alibi_bias = self.slopes.view(1, num_heads, 1, 1) * relative_distance.view(1, 1, q_len, k_len)
        
        # 应用偏置
        scores = scores + alibi_bias
        
        # Softmax + 加权
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
```

#### ALiBi 的优势

| 优势 | 说明 |
|------|------|
| 外推能力强 | 推理时可处理任意长度序列 |
| 计算高效 | 偏置可以预先计算 |
| 无参数 | 不增加可训练参数 |
| 直觉清晰 | 距离越远，注意力越小 |

```
ALiBi 的注意力偏置可视化：

注意力分数（无 ALiBi）：   注意力分数（+ ALiBi）：
[0.2  0.3  0.5]          [0.3  0.3  0.15]  ← 远距离 token 被惩罚
[0.1  0.6  0.3]          [0.15 0.4  0.2]
```

## RoPE（Rotary Position Embedding）

RoPE 是当前最主流的位置编码方法，被 LLaMA、PaLM、Falcon、Qwen 等众多模型采用。

### 核心思想

RoPE 通过旋转操作将位置信息注入 Q 和 K 向量：

```
核心公式：
q_m = f_q(x_m, m) = R_Θ^m · W_q · x_m
k_n = f_k(x_n, n) = R_Θ^n · W_k · x_n

其中 R_Θ^m 是旋转矩阵：
R_Θ^m = diag(R(θ_0, m), R(θ_1, m), ..., R(θ_{d/2-1}, m))

R(θ_i, m) = [[cos(mθ_i), -sin(mθ_i)],
              [sin(m_i),  cos(mθ_i)]]

关键性质：
<q_m, k_n> = g(x_m, x_n, m-n)  ← 内积只依赖相对位置 m-n
```

### 实现代码

```python
class RotaryPositionEmbedding(nn.Module):
    """RoPE 位置编码"""
    
    def __init__(self, dim: int, max_position: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # 计算频率: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device):
        """预计算 cos 和 sin 缓存"""
        if seq_len == self._seq_len_cached:
            return
        
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).float()
        
        # 外积：[seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将向量后半部分旋转"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_heads, seq_len, head_dim]
        """
        seq_len = q.size(2)
        self._update_cos_sin_cache(seq_len, q.device)
        
        cos = self._cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # 应用旋转
        q_rotated = (q * cos) + (self.rotate_half(q) * sin)
        k_rotated = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rotated, k_rotated

# 使用示例
rope = RotaryPositionEmbedding(dim=128, max_position=4096, base=10000)
q = torch.randn(2, 32, 10, 128)  # [batch, heads, seq, head_dim]
k = torch.randn(2, 32, 10, 128)

q_rot, k_rot = rope(q, k)
```

### RoPE 的优势

| 优势 | 说明 |
|------|------|
| 相对位置 | 内积天然编码相对位置信息 |
| 外推能力 | 可以处理比训练时更长的序列 |
| 计算高效 | 只需要 cos/sin 查表和矩阵乘法 |
| 无参数 | 不增加可训练参数 |
| 兼容性好 | 可直接替代绝对位置编码 |

### RoPE 的参数设置

```python
# 不同模型的 RoPE base 值
ROPE_CONFIGS = {
    "LLaMA-1/2": {
        "base": 10000,
        "max_position": 2048,
        "theta_formula": "10000^(-2i/d)"
    },
    "LLaMA-3": {
        "base": 500000,  # 更大的 base 值
        "max_position": 8192,
        "reason": "提高高频分辨率"
    },
    "PaLM": {
        "base": 10000,
        "max_position": 8192,
    },
    "CodeLLaMA": {
        "base": 1000000,  # 极大 base 值
        "max_position": 16384,
        "reason": "代码需要更精确的位置感知"
    }
}

# base 值的影响
# base 越大 → 低频波长越长 → 长距离位置区分能力越强
# base 过小 → 高频振荡过快 → 难以学习远距离关系
```

## 长上下文位置编码扩展

当模型需要处理比训练时更长的序列时，需要特殊的位置编码扩展技术。

### 线性插值（Position Interpolation）

```python
class PositionInterpolatedRoPE(nn.Module):
    """通过线性插值扩展 RoPE 的上下文长度"""
    
    def __init__(self, dim: int, original_max: int, extended_max: int, base: int = 10000):
        super().__init__()
        self.scale = original_max / extended_max  # 缩放因子
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, q, k, seq_len):
        """
        关键：位置索引乘以 scale 因子
        原始位置 [0, 1, 2, ..., extended_max]
        缩放位置 [0, scale, 2*scale, ..., extended_max*scale]
                                = [0, scale, ..., original_max]
        """
        t = torch.arange(seq_len, device=q.device).float() * self.scale
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        
        return q_rot, k_rot

# 示例：将 4K 训练的 LLaMA 扩展到 32K
interpolation = PositionInterpolatedRoPE(
    dim=128,
    original_max=4096,
    extended_max=32768,
    base=10000
)
# scale = 4096 / 32768 = 0.125
```

### YaRN（Yet another RoPE extensioN）

YaRN 是对 Position Interpolation 的改进，使用非线性缩放：

```python
class YaRNScaledRoPE(nn.Module):
    """YaRN：使用对数线性缩放的 RoPE"""
    
    def __init__(self, dim: int, original_max: int, extended_max: int,
                 base: int = 10000, scale_factor: float = 1.0,
                 beta_fast: float = 32, beta_slow: float = 1):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_factor = scale_factor
        
        # YaRN 核心：对频率进行非线性缩放
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # 计算 YaRN 缩放因子
        # 低频部分（大波长）使用插值
        # 高频部分（小波长）保持不变
        self.inv_freq = self._yarn_correction(
            inv_freq, beta_fast, beta_slow, scale_factor, original_max
        )
        
        self.register_buffer('inv_freq_original', inv_freq)
        self.register_buffer('inv_freq', self.inv_freq)
    
    def _yarn_correction(self, inv_freq, beta_fast, beta_slow, s, original_max):
        """YaRN 频率校正"""
        dim = len(inv_freq) * 2
        
        # 计算临界频率
        lambda_low = 2 * torch.pi / inv_freq[0]  # 最低频率波长
        lambda_high = 2 * torch.pi / inv_freq[-1]  # 最高频率波长
        
        # 确定哪些频率需要缩放
        freq_idx = torch.arange(len(inv_freq), dtype=torch.float)
        freq = 1.0 / inv_freq  # 实际频率值
        
        # 低频（大波长）需要插值
        # 高频（小波长）不需要
        scale = torch.ones_like(freq)
        
        for i in range(len(freq)):
            if freq[i] < beta_slow:
                scale[i] = s  # 缩放
            elif freq[i] > beta_fast:
                scale[i] = 1.0  # 不缩放
            else:
                # 过渡区域：平滑插值
                ratio = (freq[i] - beta_slow) / (beta_fast - beta_slow)
                scale[i] = s + (1.0 - s) * ratio
        
        return inv_freq / scale

# YaRN 参数设置
YARN_CONFIGS = {
    "LLaMA-2-4K→32K": {
        "scale_factor": 8,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max": 4096,
        "extended_max": 32768
    },
    "LLaMA-2-4K→128K": {
        "scale_factor": 32,
        "beta_fast": 128,
        "beta_slow": 1,
        "original_max": 4096,
        "extended_max": 128000
    }
}
```

### NTK-Aware 缩放

```python
def ntk_aware_rope(base: int, original_context: int, new_context: int, dim: int):
    """NTK-Aware RoPE 缩放"""
    # NTK 理论：插值会改变模型的神经正切核
    # 解决方案：缩放 base 参数而非直接插值位置
    
    scale = new_context / original_context
    
    # 调整 base 参数
    base_scaled = base * (scale ** (dim / (dim - 2)))
    
    inv_freq = 1.0 / (base_scaled ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq

# 示例
inv_freq = ntk_aware_rope(
    base=10000,
    original_context=4096,
    new_context=16384,
    dim=128
)
```

## 位置编码对比实验

### 外推能力测试

```python
def test_extrapolation(encoding_type: str, train_len: int, test_len: int):
    """测试不同位置编码的外推能力"""
    
    results = {
        "learned_absolute": {
            "train_512_test_1024": "性能严重下降（~50% PPL 增长）",
            "reason": "训练时只见过 0-511 的位置"
        },
        "sinusoidal": {
            "train_512_test_1024": "有所下降但可用（~20% PPL 增长）",
            "reason": "数学定义可以扩展，但未见过的模式效果差"
        },
        "RoPE": {
            "train_512_test_1024": "轻微下降（~10% PPL 增长）",
            "reason": "相对位置编码，不依赖绝对位置范围"
        },
        "ALiBi": {
            "train_512_test_1024": "几乎无影响（~5% PPL 增长）",
            "reason": "偏置只依赖相对距离，天然支持任意长度"
        },
        "RoPE + YaRN": {
            "train_512_test_8192": "可控下降（~15% PPL 增长）",
            "reason": "通过频率缩放实现外推"
        }
    }
    
    return results[encoding_type]
```

### 计算效率对比

| 方法 | 预计算 | 运行时开销 | 内存占用 |
|------|--------|-----------|----------|
| 正弦编码 | 可以 | 向量加法 | O(d × L) |
| 学习式编码 | 不可（参数） | 向量加法 | O(d × L) |
| RoPE | 可以 | 旋转向量 | O(d × L) |
| ALiBi | 可以 | 标量乘法+加法 | O(L²)（注意力偏置） |
| T5 偏差 | 不可（参数） | 查表+加法 | O(num_buckets × num_heads) |

## 总结

位置编码是 Transformer 架构中至关重要的组件，直接影响模型对序列顺序的理解能力：

1. **绝对位置编码**（正弦/学习式）简单直接，但外推能力差
2. **相对位置编码**（ALiBi、T5）更自然地捕获 Token 间关系
3. **RoPE** 结合了两者的优点，成为当前最主流的选择
4. **长上下文扩展**（YaRN、NTK-Aware）使模型能够处理远超训练长度的序列

位置编码选择的关键考虑因素：
- 如果只需要固定长度推理 → 任何方法都可以
- 如果需要外推到更长序列 → RoPE 或 ALiBi
- 如果需要极端长上下文（100K+）→ RoPE + YaRN
- 如果训练效率优先 → ALiBi（无参数、简单）

RoPE 已被 LLaMA、PaLM 等主流模型采用，其核心操作是在 Q 和 K 上进行旋转变换。下一步，我们将深入了解注意力机制本身的各种变体。

💡 **提示**：RoPE 的 base 参数对模型性能有显著影响。LLaMA 3 将 base 从 10000 提升到 500000，是为了在 8K+ 的上下文长度下获得更好的位置区分能力。在微调或扩展现有模型时，不要随意改变 base 值。

## 下一篇

继续阅读 [注意力变体](./67-attention-variants.md)，了解 Multi-Head Attention、MQA、GQA 和 Flash Attention 等技术。
