---
title: 激活函数详解
icon: wave-square
order: 33
---

# 激活函数详解

激活函数（Activation Function）是神经网络中引入非线性的关键组件。没有激活函数，无论网络有多深，都等价于单层线性变换。本章将系统讲解各类激活函数的数学原理、优缺点、适用场景以及实现细节。

## 激活函数的作用

激活函数的核心作用是为神经网络引入**非线性变换**。这种非线性使得神经网络能够：

1. **拟合复杂函数**：Universal Approximation Theorem 证明，带有非线性激活函数的多层网络可以以任意精度逼近任何连续函数
2. **实现特征分层**：不同层学习不同抽象级别的特征，从简单的边缘到复杂的语义概念
3. **控制信号传播**：影响梯度在前向和反向传播中的行为

### 理想激活函数的特性

| 特性 | 说明 | 重要性 |
|------|------|--------|
| 非线性 | 使网络能够学习复杂的映射关系 | 必须 |
| 可微性 | 反向传播需要计算导数 | 必须 |
| 单调性 | 保证单层网络是凸的 | 推荐 |
| 计算效率 | 前向和反向计算都应高效 | 重要 |
| 输出范围 | 有界输出可稳定训练 | 有益 |
| 零中心化 | 使梯度更新更加均衡 | 有益 |

## Sigmoid 函数

Sigmoid 函数是最早被广泛使用的激活函数之一，其输出被压缩到 (0, 1) 区间。

### 数学定义

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 导数

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

这个性质非常优雅：导数可以用函数值本身表示。

### 代码实现

```python
import numpy as np

def sigmoid(x):
    """数值稳定的 Sigmoid 实现"""
    # 使用 np.where 避免溢出
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

def sigmoid_derivative(a):
    """
    Sigmoid 导数
    a: 已经是 sigmoid 的输出值
    """
    return a * (1 - a)
```

### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 输出有界 (0, 1)，适合概率解释 | 存在严重的梯度消失问题（导数最大仅 0.25） |
| 平滑连续，处处可导 | 输出非零中心化，梯度更新存在"Z字形"震荡 |
| 计算导数非常方便 | 指数运算计算成本较高 |
| 适合二分类输出层 | 在深层网络中信号迅速衰减 |

## Tanh 函数（双曲正切）

Tanh 是 Sigmoid 的改进版本，输出范围为 (-1, 1)，实现了零中心化。

### 数学定义

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

### 导数

$$\tanh'(x) = 1 - \tanh^2(x)$$

### 代码实现

```python
def tanh(x):
    """数值稳定的 Tanh 实现"""
    return np.tanh(x)  # NumPy 内置的数值稳定版本

def tanh_derivative(a):
    """Tanh 导数"""
    return 1 - a ** 2
```

### 与 Sigmoid 对比

| 特性 | Sigmoid | Tanh |
|------|---------|------|
| 输出范围 | (0, 1) | (-1, 1) |
| 零中心化 | 否 | 是 |
| 最大导数 | 0.25 | 1.0 |
| 关系 | 原始形式 | $\tanh(x) = 2\sigma(2x) - 1$ |
| 梯度消失 | 严重 | 仍然存在但有所缓解 |

## ReLU 函数（Rectified Linear Unit）

ReLU 是当前深度学习中最常用的激活函数，由 Hinton 实验室在 2011 年推广使用。

### 数学定义

$$\text{ReLU}(x) = \max(0, x)$$

### 导数

$$\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}$$

### 代码实现

```python
def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 导数"""
    return (x > 0).astype(float)
```

### 优势

| 优势 | 说明 |
|------|------|
| 计算极快 | 只需比较和取最大值，无指数运算 |
| 梯度恒定 | 正区间的梯度为 1，有效缓解梯度消失 |
| 稀疏激活 | 约 50% 的神经元被抑制为零，网络具有稀疏性 |
| 实现简单 | 一行代码即可实现 |

### 缺陷：神经元死亡（Dying ReLU）

当神经元的输入持续为负值时，ReLU 的梯度始终为 0，导致该神经元永远无法被激活，权重不再更新。这在高学习率下尤其严重。

💡 **提示**：实验发现，使用 ReLU 的网络中可能有高达 40% 的神经元在训练过程中"死亡"。如果训练时准确率良好但测试时表现骤降，可能是 Dying ReLU 问题。

## ReLU 的变体

为了解决 Dying ReLU 问题，研究者提出了多种改进版本。

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases}$$

其中 $\alpha$ 通常取 0.01。负区间有小的斜率，保证梯度不会为零。

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU 激活函数"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU 导数"""
    return np.where(x > 0, 1.0, alpha)
```

### PReLU（Parametric ReLU）

PReLU 将 Leaky ReLU 中的 $\alpha$ 也作为可学习参数，通过反向传播自动学习最佳值。

### ELU（Exponential Linear Unit）

$$\text{ELU}(x) = \begin{cases} x, & x > 0 \\ \alpha(e^x - 1), & x \leq 0 \end{cases}$$

ELU 在负区间使用指数函数，使得激活均值更接近零，加速训练。

```python
def elu(x, alpha=1.0):
    """ELU 激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 0)) - 1))
```

### GELU（Gaussian Error Linear Unit）

GELU 在 Transformer 架构中被广泛使用（BERT、GPT 系列均采用）：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

近似实现：

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

```python
def gelu(x):
    """GELU 激活函数（近似实现）"""
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    ))
```

### SiLU / Swish

由 Google Brain 在 2017 年提出：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```python
def silu(x):
    """SiLU / Swish 激活函数"""
    return x / (1 + np.exp(-x))

def silu_derivative(x):
    """SiLU 导数"""
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 + x * (1 - sig))
```

### 激活函数全面对比

| 函数 | 公式 | 输出范围 | 零中心化 | 计算成本 | 推荐场景 |
|------|------|----------|----------|----------|----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | (0, 1) | 否 | 高 | 二分类输出层 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | (-1, 1) | 是 | 高 | RNN 隐藏层 |
| ReLU | $\max(0, x)$ | [0, ∞) | 否 | 极低 | CNN/MLP 隐藏层（默认） |
| Leaky ReLU | $\max(\alpha x, x)$ | (-∞, ∞) | 否 | 极低 | 防止 Dying ReLU |
| ELU | $x>0: x, \text{else}: \alpha(e^x-1)$ | (-α, ∞) | 接近 | 中 | 需要更快收敛时 |
| GELU | $x \cdot \Phi(x)$ | (-∞, ∞) | 接近 | 中 | Transformer |
| SiLU | $x \cdot \sigma(x)$ | (-∞, ∞) | 否 | 中 | 现代架构（EfficientNet） |

## Softmax 函数

Softmax 是专门用于多分类输出层的激活函数，将输出转换为概率分布。

### 数学定义

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

### 性质

- 所有输出值在 (0, 1) 之间
- 所有输出值之和为 1
- 保持输入的相对大小顺序

### 数值稳定实现

直接计算 $e^{x_i}$ 可能导致溢出。通过减去最大值来保证数值稳定性：

```python
def softmax(x, axis=-1):
    """数值稳定的 Softmax 实现"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 与交叉熵的组合

Softmax 与交叉熵损失函数组合使用时，梯度计算可以简化为：

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

其中 $p_i$ 是 Softmax 输出，$y_i$ 是真实标签。这种简洁性使得两者的组合成为多分类任务的标准选择。

## 激活函数的选择策略

| 场景 | 推荐激活函数 | 理由 |
|------|--------------|------|
| 隐藏层（通用） | ReLU | 计算快、效果好，是最安全的默认选择 |
| CNN 隐藏层 | ReLU | 在图像任务中表现优异 |
| RNN 隐藏层 | Tanh | 零中心化有利于循环网络的稳定 |
| Transformer | GELU | 在自注意力机制中表现最佳 |
| 二分类输出层 | Sigmoid | 输出可直接解释为概率 |
| 多分类输出层 | Softmax | 输出为概率分布 |
| 回归输出层 | 线性（无激活） | 需要无界的输出范围 |
| 存在 Dying ReLU | Leaky ReLU / ELU | 防止神经元死亡 |

## 在 PyTorch 中使用激活函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernCNN(nn.Module):
    """使用现代激活函数的 CNN 示例"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        # 现代网络常用 GELU 或 SiLU
        self.act = nn.GELU()

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
            # 输出层不使用激活函数，与 CrossEntropyLoss 配合使用
        )

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = self.act(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## 激活函数的可视化

```python
import matplotlib.pyplot as plt

def plot_activation_functions():
    """绘制常见激活函数图像"""
    x = np.linspace(-5, 5, 500)

    functions = {
        'Sigmoid': sigmoid,
        'Tanh': np.tanh,
        'ReLU': relu,
        'Leaky ReLU': lambda x: leaky_relu(x, 0.1),
        'ELU': lambda x: elu(x, 1.0),
        'GELU': gelu,
        'SiLU': silu
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(functions.items()):
        axes[idx].plot(x, func(x), linewidth=2)
        axes[idx].set_title(name, fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='k', linewidth=0.5)
        axes[idx].axvline(x=0, color='k', linewidth=0.5)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150)
    plt.show()
```

## 总结

激活函数是神经网络的非线性引擎，其选择直接影响模型的表达能力和训练效率。本章核心要点：

1. 激活函数为神经网络引入非线性，没有非线性就没有深度学习的强大能力
2. Sigmoid 和 Tanh 存在梯度消失问题，不适合深层网络
3. ReLU 是默认首选，计算快且效果好，但要注意 Dying ReLU 问题
4. Leaky ReLU、ELU、GELU、SiLU 等变体各有特点，在不同架构中有不同偏好
5. 输出层根据任务类型选择激活函数：分类用 Sigmoid/Softmax，回归用线性
6. 现代架构如 Transformer 偏好 GELU，CNN 偏好 ReLU/SiLU

理解了激活函数后，下一步需要学习如何衡量模型预测与真实值之间的差距，即损失函数的设计与选择。

---

**上一篇**: [32. 反向传播算法](32-backpropagation.md)
**下一篇**: [34. 损失函数详解](34-loss-functions.md)
