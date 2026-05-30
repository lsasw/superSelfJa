---
title: LLaMA 家族
icon: code
order: 3
---

# LLaMA 家族（LLaMA Family）

LLaMA（Large Language Model Meta AI）是 Meta AI 于 2023 年 2 月发布的一系列开源大语言模型。与 OpenAI 的 GPT 系列不同，LLaMA 选择了对学术界和开源社区开放，这一决策直接催生了开源 LLM 的繁荣生态。从 LLaMA 1 到 LLaMA 3.1，从 70 亿到 4050 亿参数，LLaMA 家族不断追赶甚至在某些方面超越闭源模型。本文将深入分析 LLaMA 家族的技术细节、训练策略和生态影响。

## LLaMA 家族的演进时间线

| 模型 | 发布时间 | 参数规模 | 上下文长度 | 训练数据 | 关键创新 |
|------|----------|----------|-----------|----------|----------|
| LLaMA 1 | 2023.02 | 7B/13B/33B/65B | 2048 | 1.4T token | 开源标杆、高效训练 |
| LLaMA 2 | 2023.07 | 7B/13B/70B | 4096 | 2T token | 对话优化、安全对齐 |
| LLaMA 3 | 2024.04 | 8B/70B | 8192 | 15T token | 新 Tokenizer、高质量数据 |
| LLaMA 3.1 | 2024.07 | 8B/70B/405B | 128K | 15T+ token | 工具使用、多语言 |
| LLaMA 3.2 | 2024.09 | 1B/3B/11B/90B | 128K | — | 视觉、端侧部署 |
| LLaMA 3.3 | 2024.12 | 70B | 128K | — | 对齐优化 |

## LLaMA 1：开源标杆

### 发布背景

2023 年 2 月，Meta 发布了 LLaMA，目标是为研究社区提供一个高质量的开源基座模型。与当时主流的闭源模型（如 GPT-3、Claude）不同，LLaMA 的权重对研究人员开放。

### 模型架构

LLaMA 1 在 Transformer 架构的基础上进行了多项优化，这些优化后来成为现代 LLM 的标准配置：

```python
class LLaMAConfig:
    """LLaMA 1 配置（以 7B 为例）"""
    hidden_size = 4096        # 隐藏层维度
    intermediate_size = 11008 # FFN 中间层维度
    num_hidden_layers = 32    # Transformer 层数
    num_attention_heads = 32  # 注意力头数
    vocab_size = 32000        # SentencePiece 词表
    max_position_embeddings = 2048  # 最大序列长度
    
    # LLaMA 特有的架构选择
    rms_norm_eps = 1e-6       # RMSNorm epsilon
    hidden_act = "silu"       # SwiGLU 激活
    use_rotary_embeddings = True   # RoPE 位置编码
```

| 架构选择 | 选择 | 对比 GPT | 优势 |
|----------|------|----------|------|
| 归一化 | RMSNorm | LayerNorm | 计算更高效、效果相当 |
| 激活函数 | SwiGLU | GeLU | 表达能力更强 |
| 位置编码 | RoPE | 绝对位置编码 | 更好的外推能力 |
| 注意力 | 高效注意力 | 标准注意力 | 减少计算冗余 |

### SwiGLU 实现

```python
class SwiGLUFFN(nn.Module):
    """LLaMA 使用的 SwiGLU FFN"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # 三个线性层：gate、up、down
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
    
    def forward(self, x):
        # SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
```

### RoPE 位置编码

```python
def rotate_half(x):
    """将张量沿最后一个维度对折并旋转"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码（RoPE）"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def compute_rope_embeddings(hidden_states, position_ids, inv_freq):
    """计算 RoPE 的 cos 和 sin 值"""
    # inv_freq: [d/2] = 1 / (10000^(2i/d))
    seq_len = position_ids.max().item() + 1
    t = position_ids.float().type_as(inv_freq)
    
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin
```

### 训练数据策略

LLaMA 1 的训练数据配比：

| 数据源 | 占比 | Token 量 | 说明 |
|--------|------|----------|------|
| CommonCrawl | 67% | 940B | 网页数据，经 CCNet 清洗 |
| C4 | 15% | 210B | 清洗后的网页数据 |
| GitHub | 4.5% | 63B | 代码数据 |
| Wikipedia | 4.5% | 63B | 多语言百科 |
| Books | 4.5% | 63B | Gutenberg + Books3 |
| ArXiv | 2.5% | 35B | 学术论文 |
| StackExchange | 2% | 28B | 问答数据 |

### 高效训练技巧

LLaMA 论文强调了以下训练效率优化：

```python
# 1. 因果注意力（仅关注左侧）
def causal_attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1))
    # 上三角掩码：禁止关注未来 token
    mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ value

# 2. RMSNorm（计算高效）
def rms_norm(x, weight, eps=1e-6):
    """Root Mean Square Layer Normalization"""
    # RMSNorm 只计算均方根，不计算均值
    norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / norm * weight

# 3. 激活检查点（节省显存）
from torch.utils.checkpoint import checkpoint

class CheckpointedLLaMALayer(nn.Module):
    def forward(self, hidden_states, attention_mask):
        return checkpoint(
            super().forward,
            hidden_states, attention_mask,
            use_reentrant=False
        )
```

### LLaMA 1 模型规格

| 规格 | 7B | 13B | 33B | 65B |
|------|-----|-----|-----|-----|
| 层数 | 32 | 40 | 60 | 80 |
| 隐藏层 | 4096 | 5120 | 6656 | 8192 |
| FFN 维度 | 11008 | 13824 | 17920 | 22016 |
| 注意力头 | 32 | 40 | 52 | 64 |
| 学习率 | 3e-4 | 3e-4 | 1.5e-4 | 1.5e-4 |
| 训练步数 | — | — | — | 350K |

## LLaMA 2：对话优化与安全对齐

### 版本区分

LLaMA 2 分为两个版本：

```
LLaMA 2 家族：
├── LLaMA-2-Base（基座模型）
│   ├── 7B：基础研究和微调
│   ├── 13B：平衡性能和资源
│   └── 70B：最强性能
│
└── LLaMA-2-Chat（对话模型）
    ├── 7B-Chat：轻量对话
    ├── 13B-Chat：中等对话
    └── 70B-Chat：最强对话（使用 RLHF/RLAIF）
```

### 关键改进

| 改进项 | LLaMA 1 | LLaMA 2 | 影响 |
|--------|---------|---------|------|
| 训练数据 | 1.4T token | 2T token | +43% |
| 上下文长度 | 2048 | 4096 | 2 倍 |
| 70B 模型 | 无 | 新增 | 缩小与闭源差距 |
| 注意力 | 标准 MHA | MQA（70B） | 推理加速 |
| 微调方法 | SFT | SFT + RLHF | 对话质量提升 |
| 安全训练 | 无 | 安全微调+红队 | 有害输出减少 |

### MQA（Multi-Query Attention）

LLaMA 2 的 70B 版本采用了 MQA 来优化推理效率：

```python
class MultiQueryAttention(nn.Module):
    """多查询注意力：所有头共享 K 和 V"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Q 仍然有多个头
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # K 和 V 只有单个头
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    
    def forward(self, x, kv_cache=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).unsqueeze(2)  # [batch, seq, 1, head_dim]
        v = self.v_proj(x).unsqueeze(2)
        
        # KV Cache 处理
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # 注意力计算（广播 K, V 到所有头）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).expand(-1, -1, self.num_heads, -1)
        v = v.transpose(1, 2).expand(-1, -1, self.num_heads, -1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)
```

### 对话模板

```python
# LLaMA-2-Chat 对话格式
LLAMA2_CHAT_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST] {assistant_response}</s>"""

# 使用示例
prompt = LLAMA2_CHAT_TEMPLATE.format(
    system_prompt="You are a helpful, respectful and honest assistant.",
    user_message="什么是 Transformer 的注意力机制？",
    assistant_response=""  # 留空供模型生成
)
```

### 安全微调

```python
# LLaMA 2 安全训练策略
safety_training_data = {
    "safe_examples": [
        # 大量安全对话示例
    ],
    "unsafe_examples": [
        # 标注有害内容的示例
    ],
    "refusals": [
        # 模型正确拒绝有害请求的示例
    ],
    "red_team": [
        # 红队测试中发现的问题及修复
    ]
}

# 安全微调目标：
# 1. 学习拒绝有害请求
# 2. 保持对安全请求的帮助性
# 3. 减少过度拒绝（false positive）
```

## LLaMA 3：高质量数据与更大词表

### 架构升级

LLaMA 3 进行了重大架构升级：

```python
class LLaMA3Config:
    """LLaMA 3 配置（以 8B 为例）"""
    hidden_size = 4096
    intermediate_size = 14336     # 更大的 FFN
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8       # 使用 GQA
    vocab_size = 128256           # 词表扩大 4 倍！
    max_position_embeddings = 8192
    rms_norm_eps = 1e-5
    hidden_act = "silu"
    
    # 新增
    rope_theta = 500000           # 更高的 RoPE base
    use_grouped_query_attention = True  # GQA
```

### GQA（Grouped Query Attention）

```python
class GroupedQueryAttention(nn.Module):
    """分组查询注意力（GQA）：MQA 和 MHA 的折中"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads       # 32
        self.num_kv_heads = config.num_key_value_heads    # 8
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_rep = self.num_heads // self.num_kv_heads  # 每组重复数
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 将 K, V 重复 n_rep 次匹配 Q 的头数
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        
        # 注意力计算...
```

### 新 Tokenizer

LLaMA 3 将词表大小从 32000 扩大到 128256，使用 TikToken 的分词器：

| 指标 | LLaMA 2 | LLaMA 3 | 影响 |
|------|---------|---------|------|
| 词表大小 | 32000 | 128256 | 更精细的分词 |
| 平均词长 | — | 更短 | 同样序列容纳更多内容 |
| 编码效率 | 基准 | +15% | 更少的 token 表示相同内容 |
| 多语言支持 | 有限 | 更好 | 覆盖更多语言 |

```python
# Tokenizer 对比
from transformers import AutoTokenizer

llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

text = "人工智能正在改变世界"
print(f"LLaMA 2 tokens: {len(llama2_tokenizer.encode(text))}")
print(f"LLaMA 3 tokens: {len(llama3_tokenizer.encode(text))}")
# LLaMA 3 通常能用更少的 token 编码相同内容
```

### 数据筛选策略

LLaMA 3 使用了 15T token 的训练数据，其中核心创新在于数据筛选：

```python
# LLaMA 3 数据筛选流程
def llama3_data_pipeline():
    """LLaMA 3 训练数据处理流程"""
    
    raw_data = load_raw_web_data()  # 数万亿美元的原始文本
    
    # 1. 启发式过滤
    data = heuristic_filter(raw_data)
    
    # 2. 去重（MinHash + LSH）
    data = deduplicate(data)
    
    # 3. 使用小型分类器过滤低质量内容
    data = filter_with_classifier(data)
    
    # 4. 混合高质量子集（学术、代码、百科等）
    data = mix_with_curated_sources(data)
    
    # 5. 多语言数据按比例混合
    data = multilingual_mix(data)
    
    return data

# 数据配比（推测）
data_mix = {
    "english_web": 80%,      # 英文网页
    "code": 5%,              # 代码
    "math": 3%,              # 数学
    "multilingual": 7%,      # 多语言
    "curated": 5%            # 精选数据
}
```

## LLaMA 3.1：工具使用与超长上下文

### 新特性

```
LLaMA 3.1 关键更新：
✅ 128K 上下文窗口（从 8K 扩展到 128K）
✅ 支持工具使用（Tool Use）
✅ 多语言支持（8 种语言优化）
✅ 405B 超大参数模型
✅ 思维链推理优化
✅ 更长的预填充上下文（long context）
```

### 模型规格对比

| 规格 | 8B | 70B | 405B |
|------|-----|-----|------|
| 隐藏层 | 4096 | 8192 | 16384 |
| FFN 维度 | 14336 | 28672 | 53248 |
| 层数 | 32 | 80 | 126 |
| 注意力头 | 32 | 64 | 128 |
| KV 头 | 8 | 8 | 8 |
| 头维度 | 128 | 128 | 128 |

### 工具使用模板

```python
# LLaMA 3.1 工具调用格式
LLAMA3_TOOL_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant with access to the following tools:

{tool_descriptions}

Use the following format:

Thought: consider the available tools
Action: tool_name
Action Input: {{"key": "value"}}
Observation: tool output
...
Thought: I have the answer
Final Answer: the answer<|eot_id|>

<|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|>"""

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "搜索维基百科",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    }
]
```

### 128K 上下文处理

```python
# 长上下文注意力优化
class LongContextAttention(nn.Module):
    """优化 128K 上下文的注意力计算"""
    
    def __init__(self):
        super().__init__()
        # 使用滑动窗口 + 全局注意力的混合策略
        self.sliding_window = 4096  # 局部窗口大小
    
    def forward(self, q, k, v):
        seq_len = q.size(1)
        
        if seq_len <= self.sliding_window:
            # 短序列：使用标准注意力
            return standard_attention(q, k, v)
        else:
            # 长序列：混合注意力
            # 1. 局部滑动窗口注意力
            local_attn = sliding_window_attention(q, k, v, self.sliding_window)
            
            # 2. 全局 token 的全局注意力
            global_attn = global_attention(q, k, v)
            
            # 3. 混合
            return local_attn + global_attn
```

## LLaMA 生态模型

基于 LLaMA 基座模型，社区涌现了大量衍生模型：

### 知名衍生模型

| 模型 | 基于 | 特点 | 用途 |
|------|------|------|------|
| Alpaca | LLaMA-7B | 52K Self-Instruct 数据 | 指令跟随 |
| Vicuna | LLaMA-13B | ShareGPT 对话数据 | 对话 |
| OpenOrca | LLaMA | FLAN 数据集 | 通用指令 |
| CodeLLaMA | LLaMA-2 | 代码专用训练 | 代码生成 |
| MedLLaMA | LLaMA-2 | 医疗数据微调 | 医疗问答 |
| WizardLM | LLaMA | Evol-Instruct | 复杂指令 |
| Nous-Hermes | LLaMA | 高质量对话 | 通用对话 |

### LoRA 微调示例

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                        # LoRA 秩
    lora_alpha=32,               # 缩放系数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标层
    lora_dropout=0.05,
    bias="none",
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 0.15% of all parameters
```

## 性能对比

### LLaMA 3.1 与竞品对比

| 模型 | MMLU | HumanEval | GSM8K | MATH |
|------|------|-----------|-------|------|
| LLaMA 3.1 8B | 66.7 | 72.0 | 84.5 | 51.4 |
| LLaMA 3.1 70B | 79.3 | 80.5 | 93.2 | 68.0 |
| LLaMA 3.1 405B | 85.2 | 89.0 | 96.8 | 73.8 |
| GPT-4o | 88.0 | 90.2 | 95.3 | 74.6 |
| Claude 3.5 Sonnet | 88.3 | 92.0 | 96.4 | 78.3 |

## 对开源社区的影响

LLaMA 系列模型的发布对开源社区产生了深远影响：

```
开源 LLM 生态图谱：

模型层：
├── 基座模型：LLaMA → Mistral → Qwen → DeepSeek
├── 微调框架：Axolotl、Unsloth、LLaMA-Factory
├── 推理引擎：vLLM、TGI、MLX、llama.cpp
└── 量化工具：bitsandbytes、GGUF

应用层：
├── 本地部署：Ollama、LM Studio
├── RAG 框架：LangChain、LlamaIndex
├── Agent 框架：LangGraph、CrewAI、AutoGen
└── 评估工具：lm-eval、OpenCompass

数据层：
├── 指令数据：Alpaca、Evol-Instruct、UltraChat
├── 偏好数据：HH-RLHF、UltraFeedback
└── 合成数据：Self-Instruct、Self-Play
```

## 总结

LLaMA 家族的发展历程展示了开源大模型的崛起之路：
1. **LLaMA 1** 证明了开源模型可以达到接近闭源模型的水平
2. **LLaMA 2** 引入了对话优化和安全对齐
3. **LLaMA 3** 通过数据质量和架构优化实现质的飞跃
4. **LLaMA 3.1** 达到了接近 GPT-4 的水平，并支持工具使用和超长上下文

LLaMA 的成功在于：
- **开放权重**：让研究者和开发者可以自由实验
- **持续迭代**：每代都有明确的改进方向
- **生态共建**：催生了大量下游工具和应用

理解 LLaMA 家族有助于我们把握开源 LLM 的技术路线。而闭源模型的另一个重要代表是 Anthropic 的 Claude 系列，它以其独特的 Constitutional AI 理念著称。

💡 **提示**：LLaMA 系列模型需要在 Meta 的许可协议下使用。商业用途需要单独申请许可。对于个人研究和非商业用途，可以直接从 HuggingFace 下载。

## 下一篇

继续阅读 [Claude 分析](./64-claude-analysis.md)，了解 Anthropic 的技术理念和 Claude 模型的发展。
