---
title: GPT 模型演进
icon: rocket
order: 2
---

# GPT 模型演进（GPT Evolution）

GPT（Generative Pre-trained Transformer）系列是 OpenAI 开发的最具影响力的大语言模型家族。从 2018 年的 GPT-1 到 2024 年的 GPT-4o，每一代模型都在参数规模、训练策略、能力边界和应用场景上取得了显著进步。本文将全面回顾 GPT 系列的技术演进，分析每一代模型的关键创新，并展望未来的发展方向。

## GPT 系列发展总览

| 模型 | 发布时间 | 参数量 | 上下文长度 | 训练数据量 | 关键创新 |
|------|----------|--------|-----------|-----------|----------|
| GPT-1 | 2018.06 | 1.17 亿 | 512 | 约 5 GB | 两阶段训练范式 |
| GPT-2 | 2019.02 | 15 亿 | 1024 | 约 40 GB | 规模扩展 |
| GPT-3 | 2020.05 | 1750 亿 | 2048 | 约 570 GB | 涌现能力、ICL |
| InstructGPT | 2022.01 | 13 亿 | 2048 | — | RLHF 首次应用 |
| GPT-3.5 | 2022.11 | ~1750 亿 | 4096 | — | 代码能力增强 |
| GPT-4 | 2023.03 | 未公开 | 8K/32K | — | 多模态、推理 |
| GPT-4 Turbo | 2023.11 | 未公开 | 128K | — | 长上下文、降价 |
| GPT-4o | 2024.05 | 未公开 | 128K | — | 原生多模态 |
| GPT-4.5 | 2025.02 | 未公开 | 128K | — | 直觉推理 |
| o1 系列 | 2024.09 | 未公开 | 128K | — | 推理模型、思维链 |

## GPT-1：两阶段训练范式

2018 年 6 月，OpenAI 发表了 **"Improving Language Understanding by Generative Pre-Training"** 论文，提出了 GPT 的第一个版本。GPT-1 的核心思想是 **两阶段训练**：

### 架构设计

GPT-1 基于 Transformer 的解码器部分，使用 12 层、768 维隐藏层、12 个注意力头，总计 1.17 亿参数。

```python
class GPT1Config:
    """GPT-1 配置参数"""
    n_layer = 12          # Transformer 层数
    n_embd = 768          # 隐藏层维度
    n_head = 12           # 注意力头数
    n_positions = 512     # 最大序列长度
    vocab_size = 40000    # 词表大小
    n_ctx = 512           # 上下文窗口
```

### 训练流程

```
阶段一：无监督预训练
    ↓
目标：最大化语言建模似然
    L1(U) = Σ log P(ui | ui-k, ..., ui-1; θ)
    ↓
阶段二：有监督微调
    ↓
目标：最大化下游任务似然
    L2(D) = Σ log P(y | x1, ..., xn; θ)
```

GPT-1 的创新之处在于认识到：**大量无标签文本中蕴含着语言结构知识**，可以先通过无监督学习获取这些知识，再用少量标注数据适配到具体任务上。这一思路后来成为所有 LLM 的标准范式。

### 实验结果

在四个 NLP 基准上的表现：

| 任务 | GPT-1 | 此前 SOTA | 提升 |
|------|-------|----------|------|
| ROCNLI（文本蕴涵） | 87.4% | 86.0% | +1.4% |
| SST（情感分析） | 93.2% | 93.2% | 持平 |
| QQP（语义相似度） | 91.3% | 91.3% | 持平 |
| MNLI（多领域蕴涵） | 82.1% | 81.4% | +0.7% |

## GPT-2：规模即力量

2019 年 2 月，OpenAI 发布了 GPT-2，提出了一个影响深远的观点：**单纯增加模型规模就能显著提升能力**。

### 模型变体

GPT-2 提供了四种不同规模的版本：

| 版本 | 参数 | 层数 | 隐藏层 | 注意力头 |
|------|------|------|--------|----------|
| Small | 1.24 亿 | 12 | 768 | 12 |
| Medium | 3.55 亿 | 24 | 1024 | 16 |
| Large | 7.74 亿 | 36 | 1280 | 20 |
| XL | 15.42 亿 | 48 | 1600 | 25 |

### 关键改进

GPT-2 在架构上做了以下调整：

```python
# GPT-2 相对于原始 Transformer 的关键改进
class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 使用 LayerNorm 替代原始 LayerNorm（改为 Pre-LN）
        # GPT-2 将 LayerNorm 放在每个子层之前而非之后
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Pre-LN 架构：残差连接更稳定
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

| 改进点 | 原始 Transformer | GPT-2 | 效果 |
|--------|------------------|-------|------|
| 归一化位置 | Post-LN | Pre-LN | 训练更稳定 |
| 词表大小 | 37000 | 50257 | 更好的分词覆盖 |
| 上下文长度 | 512 | 1024 | 更长的依赖建模 |
| 初始化 | Xavier | 修改后的初始化 | 深层训练更稳定 |

### 零样本能力

GPT-2 最大的贡献是展示了 **零样本（Zero-Shot）迁移** 能力：无需任何微调，直接在 Prompt 中描述任务即可让模型执行新任务。

```
请翻译以下内容为法语：

英文：The cat sat on the mat.
法文：Le chat s'est assis sur le tapis.

英文：I love programming in Python.
法文：
```

## GPT-3：涌现能力的发现

2020 年 5 月，GPT-3 的发布标志着 LLM 发展的重要转折点。1750 亿参数规模带来了前所未有的**涌现能力（Emergent Abilities）**。

### 模型配置

```python
class GPT3Config:
    n_layer = 96            # 96 层 Transformer
    n_embd = 12288          # 12288 维隐藏层
    n_head = 96             # 96 个注意力头
    n_positions = 2048      # 上下文窗口
    vocab_size = 50257      # BPE 词表
```

### In-Context Learning

GPT-3 系统性地研究了三种上下文学习范式：

```
Few-Shot: [示例1][示例2][示例3][新输入] → 模型预测
    ↓
One-Shot: [1个示例][新输入] → 模型预测
    ↓
Zero-Shot: [任务描述][新输入] → 模型预测
```

实验表明，Few-Shot 性能显著优于 Zero-Shot，且随着模型规模增大，ICL 能力单调增长。

### Few-Shot 示例代码

```python
from openai import OpenAI

client = OpenAI()

# Few-Shot 翻译任务
response = client.chat.completions.create(
    model="gpt-3.5-turbo-instruct",  # 使用 GPT-3.5 的 completion 接口
    prompt="""将英文翻译为中文。

English: The weather is beautiful today.
Chinese: 今天天气很好。

English: Machine learning is transforming the world.
Chinese: 机器学习正在改变世界。

English: The quick brown fox jumps over the lazy dog.
Chinese:""",
    max_tokens=50,
    temperature=0,
)

print(response.choices[0].text.strip())
```

###  Scaling Laws

OpenAI 在 GPT-3 论文中验证了 **Scaling Laws** 的存在：

```
Loss(N, D) = (N_c / N)^a + (D_c / D)^b + L_∞

其中：
- N: 模型参数量
- D: 训练数据量
- a, b, c, L_∞: 拟合常数
```

这意味着，只要持续扩大模型和数据，损失就会持续降低。这一发现为后续的模型竞赛提供了理论依据。

## InstructGPT：RLHF 的引入

2022 年 1 月，OpenAI 发表了 **"Training language models to follow instructions with human feedback"** 论文，提出了 InstructGPT，这是 RLHF 技术在 LLM 中的首次系统性应用。

### 训练流程

```
GPT-3 (13B)
    ↓
【阶段一】SFT：在人工标注的指令数据上微调
    ↓
【阶段二】奖励模型：训练 RM 对输出排序
    ↓
【阶段三】PPO：用奖励信号优化策略模型
    ↓
InstructGPT
```

### RLHF 三阶段详解

```python
# 阶段一：SFT 数据格式
sft_data = [
    {
        "instruction": "用三句话解释量子计算",
        "input": "",
        "output": "量子计算是一种利用量子力学原理进行信息处理的计算方式。它使用量子比特代替传统比特，可以同时处于多种状态的叠加。这使得量子计算机能够并行处理大量可能性，在某些问题上实现指数级加速。"
    }
]

# 阶段二：奖励模型训练
# 对同一指令，收集多个模型输出，人工标注排序
rlhf_data = {
    "prompt": "写一首关于秋天的诗",
    "ranked_outputs": [
        "output_3（最好）",  # 排名 1
        "output_1",           # 排名 2
        "output_4",           # 排名 3
        "output_2（最差）"    # 排名 4
    ]
}

# 阶段三：PPO 训练伪代码
def ppo_train(policy_model, reward_model, prompts):
    for step in range(num_steps):
        # 采样：从 prompts 中采样一批
        batch_prompts = sample(prompts, batch_size)
        
        # 生成：用 policy_model 生成回复
        responses = policy_model.generate(batch_prompts)
        
        # 评估：用 reward_model 给出奖励分数
        rewards = reward_model.score(batch_prompts, responses)
        
        # 计算 KL 散度惩罚（防止偏离原始模型太远）
        kl_penalty = compute_kl_divergence(policy_model, reference_model)
        
        # PPO 更新
        loss = ppo_loss(rewards - kl_penalty)
        policy_model.update(loss)
```

### 效果对比

InstructGPT 相比于 GPT-3 在人类评估中的表现：

| 指标 | GPT-3 175B | InstructGPT 1.3B | 提升 |
|------|-----------|------------------|------|
| 有用性 | 基准 | +85% | 显著 |
| 真实性 | 基准 | +35% | 中等 |
| 无害性 | 基准 | +60% | 显著 |

值得注意的是，仅 13 亿参数的 InstructGPT 在人类评估中表现优于 1750 亿的 GPT-3。

## GPT-4：多模态与推理突破

2023 年 3 月，GPT-4 发布，虽然 OpenAI 没有公开详细的模型规格，但从技术报告和使用体验可以推断出以下关键信息。

### 关键特性

| 特性 | 描述 |
|------|------|
| 多模态 | 可以接收图像和文本输入，输出文本 |
| 上下文窗口 | 基础版 8192 token，扩展版 32768 token |
| 参数规模 | 推测为 MoE 架构，总参数约 1.8 万亿，激活参数约 2800 亿 |
| 知识截止 | 2023 年 4 月 |
| 安全训练 | 大幅减少有害内容的生成 |

### 推理能力

GPT-4 在多个专业考试基准上达到了人类顶尖水平：

| 考试 | GPT-4 百分位 | GPT-3.5 百分位 |
|------|-------------|----------------|
| 美国律师资格考试 | ~90% | ~10% |
| SAT 数学 | ~89% | ~63% |
| SAT 阅读写作 | ~93% | ~81% |
| GRE 分析写作 | ~54% | ~12% |

### GPT-4 的代码能力示例

```python
# GPT-4 可以完成复杂的编程任务
# 用户请求："实现一个线程安全的 LRU 缓存"

import threading
from collections import OrderedDict

class ThreadSafeLRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
```

## GPT-4 Turbo 与 GPT-4o

### GPT-4 Turbo（2023.11）

```
关键更新：
- 上下文窗口扩大到 128K token
- 知识更新到 2023 年 4 月
- 支持 JSON 模式（JSON mode）
- 支持 reproducible outputs（seed 参数）
- 价格降低约 3 倍
- 可以调用外部工具（Function Calling）
```

```python
# Function Calling 示例
tools = [{
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "查询数据库中的用户信息",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户ID"},
                "fields": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["user_id"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "查询用户 U12345 的姓名和邮箱"}],
    tools=tools,
    tool_choice="auto",
    seed=42  # 可复现输出
)
```

### GPT-4o（2024.05）

GPT-4o 中的 "o" 代表 "omni"（全能），是第一个**原生多模态**模型：

```
GPT-4o 能力矩阵：
┌──────────┬──────────┬──────────┬──────────┐
│   输入    │  文本    │  图像    │  音频    │
├──────────┼──────────┼──────────┼──────────┤
│  文本输出 │   ✅     │   ✅     │   ✅     │
│  图像输出 │   ✅     │   ✅     │   ✅     │
│  音频输出 │   ✅     │   ✅     │   ✅     │
└──────────┴──────────┴──────────┴──────────┘
```

与 GPT-4 Turbo 相比，GPT-4o 的关键改进：

| 指标 | GPT-4 Turbo | GPT-4o | 提升 |
|------|-------------|--------|------|
| 文本推理 | 基准 | 同等或略优 | — |
| 视觉理解 | 需要单独模型 | 原生支持 | 架构统一 |
| 响应速度 | 基准 | 快 2 倍 | 延迟减半 |
| API 价格 | 基准 | 降低 50% | 成本优化 |
| 多语言 | 强 | 更强 | 99 种语言 |

## o1 推理模型（2024.09）

2024 年 9 月，OpenAI 发布了全新的 **o1 系列**，这是一个专门针对复杂推理优化的模型。

### 核心理念

o1 系列与传统 GPT 模型的关键区别在于训练方式：

```
传统 GPT 系列：
- 预训练：海量数据上的语言建模
- 微调：SFT + RLHF
- 特点：快速响应，适合对话和一般任务

o1 系列：
- 预训练：同样使用大规模语言建模
- 微调：大规模强化学习训练
- 特点：生成内部思维链，逐步推理，适合复杂问题
```

### 使用方式

```python
# o1 模型的使用方式
response = client.chat.completions.create(
    model="o1",
    messages=[{
        "role": "user",
        "content": """一个水池有两个进水管和一个出水管。
单开进水管A，6小时可以注满；单开进水管B，8小时可以注满；
单开出水管，12小时可以排空。如果三个管同时打开，
多长时间可以注满水池？"""
    }]
)

# o1 会在内部进行逐步推理，然后给出答案
# 推理过程对用户不可见（在 o1 中），但结果更准确
```

### 性能对比

| 基准 | GPT-4o | o1 | 提升 |
|------|--------|-----|------|
| MATH（数学） | 60.3% | 85.5% | +25.2% |
| GPQA（科学） | 53.6% | 77.3% | +23.7% |
| AIME 2024 | 9.3% | 42.0% | +32.7% |
| Codeforces | 11% | 63% | +52% |

## GPT-4.5（2025.02）

GPT-4.5 是 OpenAI 发布的最后一个 GPT-系列模型，定位为"直觉推理"模型。

```
GPT-4.5 特点：
- 更大的参数规模（推测远超 GPT-4）
- 直觉推理能力（Intuitive Reasoning）
- 更强的创意和共情能力
- 更好的对话风格自然性
- 作为 o 系列和 GPT-5 之间的过渡
```

## 技术演进的关键趋势

通过分析 GPT 系列的发展，我们可以总结出以下趋势：

| 趋势 | 表现 | 未来方向 |
|------|------|----------|
| 规模扩大 | 从 1.17 亿到数千亿/万亿参数 | 继续增长但边际收益递减 |
| 架构优化 | Decoder-only → MoE → 多模态 | 更高效的架构设计 |
| 训练策略 | 纯 LM → SFT → RLHF → RL | 更复杂的对齐方法 |
| 多模态 | 纯文本 → 图文 → 原生多模态 | 视频、音频、3D |
| 推理能力 | 浅层理解 → 链式推理 → 深度思考 | 更强的规划与验证 |
| 安全性 | 无 → 基础过滤 → 系统对齐 | 更严格的安全保障 |

## 对开发者的启示

GPT 系列的演进给开发者带来了以下启示：

### 1. 规模效应仍然有效

```
模型规模 vs 性能关系（近似）：
Loss ∝ N^(-0.34) × D^(-0.28)

即使进入"后 Scaling"时代，增加参数和数据仍然能带来可预测的提升。
```

### 2. 数据质量越来越重要

```python
# 高质量数据筛选示例
def filter_high_quality_data(texts):
    """筛选高质量训练数据"""
    filtered = []
    for text in texts:
        # 1. 去除重复
        # 2. 去除低质量内容
        # 3. 保留结构良好的文本
        # 4. 添加元数据标注
        if is_high_quality(text):
            filtered.append(text)
    return filtered
```

### 3. 推理优化同样关键

```python
# 量化推理：使用 vLLM 部署
from vllm import LLM, SamplingParams

llm = LLM(model="openai/gpt-4o", tensor_parallel_size=4)
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=2048
)
outputs = llm.generate(prompts, sampling_params)
```

## 总结

GPT 系列的发展史几乎就是大语言模型技术的发展史。从 GPT-1 的两阶段训练，到 GPT-3 的涌现能力，再到 InstructGPT 的 RLHF 对齐，以及 GPT-4 的多模态和 o1 的深度推理，每一代模型都带来了范式级的创新。

理解 GPT 的演进对于把握整个 LLM 领域至关重要，因为：
1. **它定义了标准**：大多数后续模型都参考 GPT 的架构和训练方式
2. **它验证了 Scaling Law**：规模与能力的关系成为行业共识
3. **它推动了应用**：从学术研究到商业产品的转化

GPT 系列的成功也催生了开源社区的追赶，其中最具代表性的就是 Meta 的 LLaMA 系列。接下来我们将深入了解开源 LLM 的发展脉络。

💡 **提示**：GPT 的很多详细架构参数（如确切参数量、训练数据量）OpenAI 并未公开披露。本文中的推测数据来自技术报告和学术论文的综合分析。

## 下一篇

继续阅读 [LLaMA 家族](./63-llama-family.md)，了解开源大模型的发展全貌。
