---
title: 注意力机制变体
icon: eye
order: 7
---

# 注意力机制变体（Attention Variants）

注意力机制是 Transformer 架构的核心。自 2017 年提出以来，研究者们在标准多头注意力（MHA）的基础上发展出了多种变体，旨在提升效率、减少显存占用或增强特定能力。本文将系统介绍 MHA、MQA、GQA、滑动窗口注意力、Flash Attention 等关键技术，以及它们在主流模型中的应用。

## 标准注意力回顾

在深入变体之前，先回顾标准的缩放点积注意力：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    标准缩放点积注意力
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    d_k = Q.size(-1)
    
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码（因果或填充）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax 归一化
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output
```

注意力计算的复杂度：

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| QK^T | O(L² × d) | O(L² × H) |
| Softmax | O(L²) | O(L² × H) |
| × V | O(L² × d) | O(L × H × d) |

其中 L 是序列长度，d 是 head_dim，H 是注意力头数。

## 多头注意力（MHA）

多头注意力通过多个独立的"头"并行计算注意力，每个头学习不同的表示：

```python
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 每个头有独立的 Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 投影并分头
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并头并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)

# MHA 参数量：
# Q: d_model × d_model
# K: d_model × d_model  
# V: d_model × d_model
# O: d_model × d_model
# 总计: 4 × d_model²
```

### MHA 的特点

| 优点 | 缺点 |
|------|------|
| 表达能力最强 | KV Cache 占用大 |
| 每个头独立学习 | 推理时显存带宽瓶颈 |
| 并行度高 | 长序列时计算量爆炸 |

## 多查询注意力（MQA）

MQA 让所有头共享一组 K 和 V，大幅减少 KV Cache 的大小：

```python
class MultiQueryAttention(nn.Module):
    """多查询注意力：多个 Q 头，共享 K 和 V"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q 有 num_heads 个
        self.W_q = nn.Linear(d_model, d_model)
        # K 和 V 只有 1 个（所有头共享）
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Q: 多头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # K, V: 单头
        K = self.W_k(x).unsqueeze(2)  # [batch, seq, 1, d_k]
        V = self.W_v(x).unsqueeze(2)
        
        # 广播 K, V 到所有头
        K = K.expand(-1, -1, self.num_heads, -1)
        V = V.expand(-1, -1, self.num_heads, -1)
        
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.W_o(attn_output)
```

### MQA 的 KV Cache 对比

```
KV Cache 大小比较（以 70B 模型为例）：

MHA（每个头独立 K, V）：
- KV Cache = 2 × num_heads × head_dim × seq_len
- = 2 × 64 × 128 × 4096 = 64 MB/层
- 80 层 = 5.12 GB

MQA（共享 K, V）：
- KV Cache = 2 × 1 × head_dim × seq_len
- = 2 × 1 × 128 × 4096 = 1 MB/层
- 80 层 = 80 MB
- 减少约 64 倍！

代价：
- 表达能力略有下降
- 训练稳定性稍差
```

### MHA vs MQA 对比

| 指标 | MHA | MQA |
|------|-----|-----|
| Q 头数 | H | H |
| K 头数 | H | 1 |
| V 头数 | H | 1 |
| KV Cache | 2 × H × d × L | 2 × d × L |
| 训练质量 | 最高 | 略低（约 1-2%） |
| 推理速度 | 慢（显存带宽瓶颈） | 快 |
| 代表模型 | BERT、T5 | PaLM、Falcon |

## 分组查询注意力（GQA）

GQA 是 MHA 和 MQA 的折中方案，将头分成若干组，每组共享 K 和 V：

```python
class GroupedQueryAttention(nn.Module):
    """分组查询注意力（GQA）"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads  # 每组重复数
        
        # Q: num_heads 个
        self.W_q = nn.Linear(d_model, d_model)
        # K, V: num_kv_heads 个
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Q: [batch, num_heads, seq, head_dim]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        
        # K, V: [batch, num_kv_heads, seq, head_dim]
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        K = K.transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = V.transpose(1, 2)
        
        # 将 K, V 重复 n_rep 次
        K = K.repeat_interleave(self.n_rep, dim=1)  # [batch, num_heads, seq, head_dim]
        V = V.repeat_interleave(self.n_rep, dim=1)
        
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.W_o(attn_output)
```

### GQA 的 KV Cache 效率

```
GQA KV Cache 大小：
- KV Cache = 2 × num_kv_heads × head_dim × seq_len

LLaMA 3 配置示例：
- num_heads = 32
- num_kv_heads = 8
- head_dim = 128
- seq_len = 8192

KV Cache = 2 × 8 × 128 × 8192 = 16 MB/层
32 层 = 512 MB

对比 MHA（32 个 KV 头）：
KV Cache = 2 × 32 × 128 × 8192 = 64 MB/层
32 层 = 2 GB

GQA 减少了 4 倍 KV Cache，同时保持了接近 MHA 的质量
```

### 三种注意力对比

| 指标 | MHA | GQA | MQA |
|------|-----|-----|-----|
| KV 头数 | 32 | 8 | 1 |
| KV Cache | 2 GB | 512 MB | 64 MB |
| 质量 | 100% | 99% | 97% |
| 推理速度 | 基准 | 快 2 倍 | 快 4 倍 |
| 采用模型 | BERT | LLaMA 3 | PaLM |

## 滑动窗口注意力（SWA）

滑动窗口注意力限制每个 Token 只能关注其附近的固定窗口内的 Token：

```python
class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力"""
    
    def __init__(self, d_model: int, num_heads: int, window_size: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(d_model, num_heads)
    
    def create_sliding_mask(self, seq_len):
        """创建滑动窗口掩码"""
        # 位置矩阵
        positions = torch.arange(seq_len)
        
        # 距离矩阵
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        # dist[i, j] = j - i（j 相对于 i 的距离）
        
        # 滑动窗口掩码：只关注 window_size 范围内的过去 token
        mask = (dist >= -self.window_size) & (dist <= 0)
        # 添加因果性（只看左边）
        mask = mask & (dist <= 0)
        
        return mask.float()
    
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.create_sliding_mask(seq_len).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # 广播到 batch 和 heads
        return self.attention(x, mask=mask)

# 滑动窗口注意力的复杂度：
# 标准注意力：O(L² × d)
# 滑动窗口：  O(L × W × d)，W 是窗口大小
# 当 W << L 时，计算量大幅减少
```

### 混合注意力策略

Mistral 等模型采用混合策略：

```python
class HybridAttention(nn.Module):
    """Mistral 的混合注意力：局部窗口 + 全局 token"""
    
    def __init__(self, config):
        super().__init__()
        self.local_attn = SlidingWindowAttention(
            config.hidden_size, config.num_heads,
            window_size=config.sliding_window
        )
        self.global_tokens = config.num_global_tokens
    
    def forward(self, x):
        # 1. 大部分 token 使用滑动窗口注意力
        local_output = self.local_attn(x)
        
        # 2. 特殊全局 token 使用全注意力
        #    （如 [CLS]、段落标记等）
        global_output = self.global_attention(x, self.global_tokens)
        
        return local_output + global_output
```

## Flash Attention

Flash Attention 是一种 IO 感知的注意力算法，通过优化显存访问模式大幅提升速度：

```python
# Flash Attention 的核心思想：分块计算（Tiling）

def flash_attention_v1(Q, K, V, block_size=64):
    """
    Flash Attention v1 简化实现
    
    核心优化：
    1. 分块计算，避免将整个注意力矩阵加载到 SRAM
    2. 在线 Softmax，不需要存储完整的注意力矩阵
    3. 利用 GPU 的层次化显存结构
    """
    N = Q.shape[0]  # seq_len
    
    # 输出和统计量初始化
    O = torch.zeros_like(Q)
    L = torch.zeros(N, 1)  # Softmax 归一化因子
    M = torch.full((N, 1), float('-inf'))  # Softmax 最大值
    
    # 分块遍历
    for i in range(0, N, block_size):
        # 加载 Q 块到 SRAM
        Qi = Q[i:i+block_size]
        
        Oi = torch.zeros_like(Qi)
        Li = torch.zeros(Qi.shape[0], 1)
        Mi = torch.full((Qi.shape[0], 1), float('-inf'))
        
        for j in range(0, N, block_size):
            # 加载 K, V 块到 SRAM
            Kj = K[j:j+block_size]
            Vj = V[j:j+block_size]
            
            # 计算注意力分数块
            Sij = Qi @ Kj.T / math.sqrt(Qi.shape[-1])
            
            # 在线 Softmax 更新
            Mi_new = torch.max(Mi, Sij.max(dim=1, keepdim=True).values)
            Pij = torch.exp(Sij - Mi_new)
            Oi = Oi * torch.exp(Mi - Mi_new) + Pij @ Vj
            Li = Li * torch.exp(Mi - Mi_new) + Pij.sum(dim=1, keepdim=True)
            
            Mi = Mi_new
        
        O[i:i+block_size] = Oi / Li
    
    return O

# Flash Attention 2 的关键改进：
# 1. 重新排列计算顺序，减少 non-matmul FLOPs
# 2. 利用更多 GPU 并行度
# 3. 支持 MQA/GQA 的优化
```

### Flash Attention 性能

| 指标 | 标准 Attention | Flash Attention 2 | 提升 |
|------|---------------|-------------------|------|
| 时间复杂度 | O(L²d) | O(L²d)（但常数更小） | 2-4 倍 |
| 空间复杂度 | O(L²)（显存） | O(L)（显存） | L 倍 |
| IO 复杂度 | O(L²d/B) | O(L²d/(BM)) | M 倍 |

其中 B 是块大小，M 是 SM 数量。

### 在推理中的应用

```python
# vLLM 使用 PagedAttention（Flash Attention 的变体）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size=1,
    max_num_seqs=256,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True  # KV Cache 复用
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

outputs = llm.generate(prompts, sampling_params)
```

## 其他注意力变体

### 线性注意力

```python
class LinearAttention(nn.Module):
    """线性注意力：避免 O(L²) 复杂度"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_map = nn.ReLU()  # 特征映射函数
    
    def forward(self, Q, K, V):
        # 应用特征映射
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # 线性注意力：(Q × K^T) × V = Q × (K^T × V)
        # 改变计算顺序：先计算 KV，再与 Q 相乘
        # 复杂度：O(L × d²) 而非 O(L² × d)
        
        # K^T × V: [d_k, L] × [L, d_k] = [d_k, d_k]
        KV = torch.einsum('bhld,bhlm->bhdm', K, V)
        
        # Q × KV: [L, d_k] × [d_k, d_k] = [L, d_k]
        output = torch.einsum('bhld,bhdm->bhlm', Q, KV)
        
        return output
```

### 稀疏注意力

```python
class SparseAttention(nn.Module):
    """稀疏注意力：只计算部分注意力分数"""
    
    def __init__(self, d_model: int, num_heads: int, 
                 sparse_pattern: str = "fixed"):
        super().__init__()
        self.sparse_pattern = sparse_pattern
    
    def create_sparse_mask(self, seq_len):
        """创建稀疏注意力掩码"""
        if self.sparse_pattern == "fixed":
            # 固定稀疏：随机选择一部分位置
            mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                # 局部窗口 + 随机全局
                local = slice(max(0, i-64), i+1)
                mask[i, local] = 1
                # 随机选择 10 个全局位置
                global_idx = torch.randperm(seq_len)[:10]
                mask[i, global_idx] = 1
            return mask
        
        elif self.sparse_pattern == "strided":
            # 步幅稀疏：固定步长的位置
            mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(0, i+1, 32):  # 每 32 个位置
                    mask[i, j] = 1
                # 局部
                mask[i, max(0, i-32):i+1] = 1
            return mask
```

### 多尺度注意力

```python
class MultiScaleAttention(nn.Module):
    """多尺度注意力：在不同粒度上计算注意力"""
    
    def __init__(self, d_model: int, num_heads: int, scales: list = [1, 4, 16]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads) for _ in scales
        ])
        self.fusion = nn.Linear(d_model * len(scales), d_model)
    
    def forward(self, x):
        outputs = []
        for scale, attn in zip(self.scales, self.attentions):
            if scale == 1:
                # 原始粒度
                out = attn(x)
            else:
                # 下采样 → 注意力 → 上采样
                pooled = self.pool(x, scale)
                attn_out = attn(pooled)
                out = self.upsample(attn_out, scale)
            outputs.append(out)
        
        # 融合多尺度输出
        fused = torch.cat(outputs, dim=-1)
        return self.fusion(fused)
```

## 注意力变体在主流模型中的应用

| 模型 | 注意力类型 | KV 头数 | 特殊优化 |
|------|-----------|---------|----------|
| GPT-3 | MHA | 96（= Q 头） | — |
| GPT-4 | GQA（推测） | 推测 16-32 | Flash Attention |
| LLaMA-1 | MHA | 32（= Q 头） | RoPE |
| LLaMA-2 70B | MQA | 1 | MQA、RoPE |
| LLaMA-3 | GQA | 8 | GQA、RoPE、Flash |
| Mistral | GQA | 8 | 滑动窗口、RoPE |
| Mixtral | GQA + MoE | 8 | MoE、滑动窗口 |
| PaLM | MQA | 1 | MQA、ALiBi |
| Falcon | MQA | 1 | MQA、ALiBi |
| MPT | MQA | 1 | MQA、ALiBi |

## KV Cache 优化

注意力变体对 KV Cache 的影响：

```python
class KVCacheManager:
    """KV Cache 管理器"""
    
    def __init__(self, num_layers: int, num_kv_heads: int, 
                 head_dim: int, max_seq_len: int, dtype=torch.float16):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # 预分配 KV Cache
        # 形状：[num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
        self.cache = torch.zeros(
            (num_layers, 2, 1, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype
        )
        self.current_seq_len = 0
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """将新的 K, V 写入 Cache"""
        seq_len = k.size(2)
        start = self.current_seq_len
        end = start + seq_len
        
        self.cache[layer_idx, 0, :, :, start:end, :] = k
        self.cache[layer_idx, 1, :, :, start:end, :] = v
        self.current_seq_len = end
    
    def get(self, layer_idx: int):
        """获取当前层的完整 KV Cache"""
        return (
            self.cache[layer_idx, 0, :, :, :self.current_seq_len, :],
            self.cache[layer_idx, 1, :, :, :self.current_seq_len, :]
        )
    
    def memory_usage(self):
        """计算 KV Cache 显存占用"""
        return (
            self.num_layers * 2 * self.num_kv_heads * 
            self.current_seq_len * self.head_dim * 2  # float16 = 2 bytes
        )

# 不同配置的显存占用对比（seq_len=4096, head_dim=128, layers=32）
configs = {
    "MHA (32 heads)": 32 * 2 * 32 * 4096 * 128 * 2 / 1e9,   # ~1.07 GB
    "GQA (8 heads)": 32 * 2 * 8 * 4096 * 128 * 2 / 1e9,     # ~0.27 GB
    "MQA (1 head)": 32 * 2 * 1 * 4096 * 128 * 2 / 1e9,      # ~0.03 GB
}
```

## 总结

注意力机制的变体主要围绕两个核心目标：

1. **效率优化**：MQA、GQA 减少 KV Cache，Flash Attention 优化 IO
2. **能力增强**：滑动窗口支持长上下文，稀疏注意力减少计算量

GQA 已经成为现代模型的主流选择，它在质量和效率之间取得了良好的平衡。LLaMA 3 采用 8 个 KV 头（GQA），相比 MHA 减少了 4 倍的 KV Cache，同时保持了几乎相同的模型质量。

Flash Attention 是推理加速的关键技术，通过 IO 感知的设计将注意力计算的速度提升了 2-4 倍。

理解了注意力机制之后，我们将继续探索 Transformer 的另一个核心组件——前馈神经网络（FFN）及其演进到 MoE 架构的历程。

💡 **提示**：在实际部署中，KV Cache 的大小往往决定了最大 batch size。如果显存受限，优先考虑使用 GQA 或 MQA 模型，并启用 KV Cache 量化（如 FP8 或 INT8）。

## 下一篇

继续阅读 [FFN 与 MoE](./68-ffn-moe.md)，了解前馈网络的演进和混合专家架构。
