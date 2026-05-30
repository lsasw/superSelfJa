---
title: RNN、LSTM 与 GRU
icon: repeat
order: 38
---

# RNN、LSTM 与 GRU

循环神经网络（Recurrent Neural Network，RNN）是处理序列数据的核心架构。与 CNN 不同，RNN 具有"记忆"能力，能够利用历史信息来理解当前输入。本章将系统讲解 RNN 的原理、长短期记忆网络（LSTM）和门控循环单元（GRU）。

## 序列数据的特点

序列数据无处不在，且数据点之间存在时间或逻辑上的依赖关系：

| 数据类型 | 示例 | 序列长度 |
|----------|------|----------|
| 自然语言 | 句子、段落、文档 | 可变 |
| 时间序列 | 股票价格、气温记录 | 可变 |
| 音频信号 | 语音、音乐 | 长（数千帧） |
| 视频帧 | 连续的视频画面 | 长 |
| DNA 序列 | 碱基对序列 | 长 |

全连接网络和 CNN 处理序列数据时，需要将输入展平或固定大小，破坏了序列的时序信息。RNN 天然支持变长输入，并且通过隐藏状态维持对历史信息的记忆。

## 标准 RNN 的结构

### 循环连接

RNN 的核心思想是在网络中引入循环连接，使得当前时刻的输出不仅取决于当前输入，还取决于之前时刻的隐藏状态。

**展开形式**：

```
         h(t-1)       h(t)         h(t+1)
           |           |             |
           v           v             v
      +--------+  +--------+   +--------+
x(t-1)->| RNN    |->| RNN    |-->| RNN    |
      | Cell   |  | Cell   |   | Cell   |
      +--------+  +--------+   +--------+
           |           |             |
           v           v             v
         y(t-1)      y(t)         y(t+1)
```

### 数学公式

在时间步 $t$，RNN 的计算如下：

$$\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$

$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

其中：
- $\mathbf{x}_t$：时间步 $t$ 的输入
- $\mathbf{h}_t$：时间步 $t$ 的隐藏状态
- $\mathbf{y}_t$：时间步 $t$ 的输出
- $\mathbf{W}_{xh}$：输入到隐藏状态的权重
- $\mathbf{W}_{hh}$：隐藏状态到隐藏状态的权重（循环权重）
- $\mathbf{W}_{hy}$：隐藏状态到输出的权重

### RNN 的三种模式

| 模式 | 输入-输出 | 典型应用 |
|------|-----------|----------|
| 多对一 | 序列 -> 单个输出 | 情感分析、文本分类 |
| 一对一 | 单个输入 -> 序列 | 图像描述生成 |
| 多对多 | 序列 -> 序列 | 机器翻译、命名实体识别 |
| 多对多（同步） | 每个输入 -> 对应输出 | 视频分类、词性标注 |

### 使用 NumPy 实现 RNN

```python
import numpy as np

class SimpleRNN:
    """
    简单 RNN 的手动实现
    支持多对一模式（序列分类）
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 权重初始化
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_h = np.zeros(hidden_size)
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.1
        self.b_y = np.zeros(output_size)

        # 缓存
        self.hidden_states = []

    def forward(self, sequence):
        """
        前向传播
        sequence: 形状 (seq_len, input_size) 的输入序列
        """
        self.hidden_states = []
        h_t = np.zeros(self.hidden_size)  # 初始隐藏状态

        for x_t in sequence:
            h_t = np.tanh(
                x_t @ self.W_xh + h_t @ self.W_hh + self.b_h
            )
            self.hidden_states.append(h_t.copy())

        # 使用最后一个隐藏状态进行预测（多对一）
        y = h_t @ self.W_hy + self.b_y
        return y

    def forward_all(self, sequence):
        """每个时间步都输出预测（多对多）"""
        self.hidden_states = []
        h_t = np.zeros(self.hidden_size)
        outputs = []

        for x_t in sequence:
            h_t = np.tanh(
                x_t @ self.W_xh + h_t @ self.W_hh + self.b_h
            )
            self.hidden_states.append(h_t.copy())
            y_t = h_t @ self.W_hy + self.b_y
            outputs.append(y_t)

        return np.array(outputs)
```

## RNN 的反向传播：BPTT

RNN 的反向传播称为**随时间反向传播（Backpropagation Through Time，BPTT）**。核心思想是将循环网络在时间维度上展开，然后应用标准的反向传播。

对于包含 $T$ 个时间步的 RNN，损失函数为：

$$L = \sum_{t=1}^{T} L_t$$

梯度计算需要考虑到每个时间步的隐藏状态都依赖于之前所有时间步：

$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial \mathbf{W}_{hh}}$$

$$\frac{\partial L_t}{\partial \mathbf{W}_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

其中跨时间步的梯度传播涉及连乘：

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^{t} \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=k+1}^{t} \mathbf{W}_{hh}^T \cdot \text{diag}(\tanh'(\mathbf{z}_j))$$

## RNN 的核心问题：长期依赖

标准 RNN 在处理长序列时面临严重问题：

### 梯度消失

由于 $\tanh$ 的导数最大为 1，权重矩阵连乘时，如果特征值小于 1，梯度会指数级衰减。这导致 RNN 无法学习时间跨度超过一定范围的依赖关系。

### 梯度爆炸

相反，如果权重矩阵的特征值大于 1，梯度会指数级增长。

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 梯度消失 | 早期时间步的权重不更新 | LSTM/GRU |
| 梯度爆炸 | 训练不稳定，NaN 损失 | 梯度裁剪（Gradient Clipping） |

### 梯度裁剪

```python
def clip_gradients(gradients, max_norm=5.0):
    """梯度裁剪"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
    clip_coeff = max_norm / (total_norm + 1e-8)

    if clip_coeff < 1.0:
        return [g * clip_coeff for g in gradients]
    return gradients


# PyTorch 中的梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## LSTM（长短期记忆网络）

LSTM 由 Hochreiter 和 Schmidhuber 在 1997 年提出，通过引入门控机制和细胞状态解决了长期依赖问题。

### 核心概念

LSTM 引入了两个关键创新：

1. **细胞状态（Cell State）**：贯穿整个序列的"信息高速公路"，允许信息无损传递
2. **门控机制（Gating）**：三个门控制信息的流动

### 三个门的详细解析

**遗忘门（Forget Gate）**：决定哪些信息从细胞状态中丢弃

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**输入门（Input Gate）**：决定哪些新信息加入细胞状态

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)$$

**输出门（Output Gate）**：决定输出什么信息

$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

### 完整计算流程

```
细胞状态更新: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
隐藏状态:     h_t = o_t ⊙ tanh(C_t)
输出:         y_t = W_hy h_t + b_y
```

其中 $\odot$ 表示逐元素乘法。

### LSTM 的实现

```python
class LSTMCell:
    """手动实现的 LSTM 单元"""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 四个门共享相同的输入（为了效率，合并为一个矩阵运算）
        # 权重矩阵: [hidden_size, input_size + hidden_size]
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.1
        self.b = np.zeros(4 * hidden_size)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_t, h_prev, c_prev):
        """
        LSTM 前向传播
        x_t: 当前输入，形状 (input_size,)
        h_prev: 上一时刻隐藏状态，形状 (hidden_size,)
        c_prev: 上一时刻细胞状态，形状 (hidden_size,)
        """
        # 拼接输入和隐藏状态
        combined = np.concatenate([x_t, h_prev])

        # 计算四个门的值
        gates = self.W @ combined + self.b

        # 分割为四个门
        chunk_size = self.hidden_size
        f_t = self.sigmoid(gates[0:chunk_size])       # 遗忘门
        i_t = self.sigmoid(gates[chunk_size:2*chunk_size])  # 输入门
        g_t = np.tanh(gates[2*chunk_size:3*chunk_size])     # 候选细胞状态
        o_t = self.sigmoid(gates[3*chunk_size:])       # 输出门

        # 细胞状态更新
        c_t = f_t * c_prev + i_t * g_t

        # 隐藏状态
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t, (f_t, i_t, g_t, o_t)


class LSTM:
    """完整的 LSTM 网络"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建多层 LSTM 单元
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

        # 输出层
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.1
        self.b_y = np.zeros(output_size)

    def forward(self, sequence):
        """
        前向传播
        sequence: 形状 (seq_len, input_size)
        """
        seq_len = len(sequence)
        h_all = [np.zeros((self.num_layers, self.hidden_size))]
        c_all = [np.zeros((self.num_layers, self.hidden_size))]

        for t in range(seq_len):
            h_next = []
            c_next = []
            x_t = sequence[t]

            for layer in range(self.num_layers):
                h_t, c_t, gates = self.cells[layer].forward(
                    x_t, h_all[-1][layer], c_all[-1][layer]
                )
                h_next.append(h_t)
                c_next.append(c_t)
                x_t = h_t  # 下一层的输入

            h_all.append(np.array(h_next))
            c_all.append(np.array(c_next))

        # 使用最后一层最后一个时间步的隐藏状态
        final_h = h_all[-1][-1]
        output = final_h @ self.W_hy + self.b_y
        return output
```

## GRU（门控循环单元）

GRU 由 Cho 等人在 2014 年提出，是 LSTM 的简化版本，只有两个门。

### 两个门的计算

**更新门（Update Gate）**：决定多少旧信息保留、多少新信息加入

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**重置门（Reset Gate）**：决定在计算候选隐藏状态时忽略多少历史信息

$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

**隐藏状态更新**：

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### LSTM vs GRU 对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3 个（遗忘门、输入门、输出门） | 2 个（更新门、重置门） |
| 状态变量 | 2 个（隐藏状态 + 细胞状态） | 1 个（隐藏状态） |
| 参数量 | 更多 | 约少 25% |
| 训练速度 | 稍慢 | 稍快 |
| 表达能力 | 更强（更精细的信息控制） | 稍弱但足够 |
| 效果 | 通常略好 | 差距不大 |
| 推荐场景 | 长序列、复杂依赖 | 中等序列、需要快速训练 |

## RNN 的变体结构

### 双向 RNN（Bi-RNN）

双向 RNN 同时使用正向和反向两个方向的 RNN，可以捕获过去和未来的上下文信息。

```python
# PyTorch 中实现双向 RNN
rnn = nn.LSTM(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,  # 双向
    batch_first=True
)
# 输出 hidden_size * 2，因为正向和反向各有一个隐藏状态
```

### 多层 RNN（Stacked RNN）

多层 RNN 将多层堆叠在一起，每层处理上一层的输出。

```python
# PyTorch 中实现多层 RNN
rnn = nn.LSTM(
    input_size=100,
    hidden_size=128,
    num_layers=3,     # 3 层
    dropout=0.3,      # 层间 Dropout
    batch_first=True
)
```

### 序列对序列（Seq2Seq）

Seq2Seq 架构由编码器（Encoder）和解码器（Decoder）组成：

```
Encoder:  输入序列 -> [RNN/LSTM] -> 上下文向量
Decoder:  上下文向量 -> [RNN/LSTM] -> 输出序列
```

```python
class Seq2Seq(nn.Module):
    """基础的 Seq2Seq 模型"""

    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, src, trg):
        # 编码
        src_embedded = self.encoder_embedding(src)
        _, (hidden, cell) = self.encoder(src_embedded)

        # 解码
        trg_embedded = self.decoder_embedding(trg)
        outputs, _ = self.decoder(trg_embedded, (hidden, cell))
        outputs = self.fc(outputs)
        return outputs
```

## RNN 变体总结

| 类型 | 信息流 | 适用场景 |
|------|--------|----------|
| 单向 RNN | 从前到后 | 时间序列预测（只看历史） |
| 双向 RNN | 从前到后 + 从后到前 | 文本分类、NER（利用全文上下文） |
| 多层 RNN | 逐层抽象 | 复杂序列到序列任务 |
| 深层双向 RNN | 双向 + 多层 | 最强大的 RNN 架构 |

## 在 PyTorch 中使用 RNN

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    """使用 LSTM 的文本分类器"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # 双向 LSTM 输出维度为 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        """
        x: 词索引序列，形状 (batch, seq_len)
        lengths: 每个序列的实际长度
        """
        # 嵌入
        embedded = self.dropout(self.embedding(x))

        # 打包序列（避免对 padding 部分计算）
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        _, (hidden, _) = self.lstm(packed)

        # 拼接双向最后一个隐藏状态
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # 分类
        output = self.fc(self.dropout(hidden))
        return output
```

## 总结

RNN 通过循环连接实现了对序列数据的记忆能力。本章核心要点：

1. RNN 通过隐藏状态维持对历史信息的记忆，天然支持变长序列
2. BPTT 是 RNN 的训练算法，在时间维度上展开后应用反向传播
3. 标准 RNN 存在严重的梯度消失问题，无法学习长期依赖
4. LSTM 通过细胞状态和门控机制有效解决了长期依赖问题
5. GRU 是 LSTM 的简化版本，参数量更少，训练更快
6. 双向 RNN 同时利用正向和反向信息，适合文本理解任务
7. Seq2Seq 架构是机器翻译、文本摘要等任务的基础

RNN 虽然强大，但其序列化的计算方式效率较低。接下来我们将学习一种更高效的信息压缩方法：自编码器。

---

**上一篇**: [37. CNN 基础](37-cnn-basics.md)
**下一篇**: [39. 自编码器](39-autoencoder.md)
