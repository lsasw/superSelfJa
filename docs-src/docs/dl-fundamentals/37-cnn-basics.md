---
title: CNN 基础
icon: grid-2
order: 37
---

# CNN 基础

卷积神经网络（Convolutional Neural Network，CNN）是计算机视觉领域的核心技术，也是深度学习中最成功的架构之一。本章将系统讲解 CNN 的核心组件、工作原理以及现代 CNN 架构的演进。

## 为什么需要 CNN

全连接网络处理图像面临两个根本问题：

1. **参数量爆炸**：一张 $224 \times 224 \times 3$ 的图像展平后有 150,528 个特征，单个隐藏层就需要上千万参数
2. **空间信息丢失**：展平操作破坏了像素间的空间结构关系

CNN 通过两个核心设计解决了这些问题：
- **局部连接**：每个神经元只连接输入的局部区域
- **权重共享**：同一个卷积核在整张图像上滑动使用

## 卷积操作

### 一维卷积

卷积（Convolution）是信号处理中的基本操作。在一维情况下，输入序列 $x$ 与核（kernel）$w$ 的卷积定义为：

$$(x * w)[n] = \sum_{m} x[m] \cdot w[n - m]$$

在深度学习中，实际使用的是**互相关（Cross-Correlation）**操作，不翻转核：

$$(x \star w)[n] = \sum_{m} x[m] \cdot w[m - n]$$

### 二维卷积

图像处理使用的是二维卷积。卷积核在输入特征图上滑动，在每个位置执行逐元素相乘再求和的操作。

```
输入 (5x5)              卷积核 (3x3)             输出 (3x3)
+---+---+---+---+---+    +---+---+---+
| 1 | 2 | 3 | 0 | 1 |    | 1 | 0 |-1 |
+---+---+---+---+---+    +---+---+---+
| 2 | 3 | 1 | 2 | 0 |    | 1 | 0 | 1 |
+---+---+---+---+---+    +---+---+---+
| 0 | 1 | 2 | 3 | 1 |    |-1 | 1 | 0 |
+---+---+---+---+---+    +---+---+---+
| 1 | 0 | 3 | 1 | 2 |
+---+---+---+---+---+
| 2 | 1 | 0 | 2 | 3 |
+---+---+---+---+---+
```

每个输出值的计算：
```
输出[0][0] = 1*1 + 2*0 + 3*(-1) + 2*1 + 3*0 + 1*1 + 0*(-1) + 1*1 + 2*0 = 3
```

### 使用 NumPy 实现二维卷积

```python
import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    """
    二维卷积的朴素实现
    input: 形状 (H_in, W_in) 的单通道特征图
    kernel: 形状 (K, K) 的卷积核
    stride: 步长
    padding: 填充大小
    """
    # 添加填充
    if padding > 0:
        input = np.pad(input, padding, mode='constant', constant_values=0)

    H_in, W_in = input.shape
    K = kernel.shape[0]

    # 计算输出尺寸
    H_out = (H_in - K) // stride + 1
    W_out = (W_in - K) // stride + 1

    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + K
            w_end = w_start + K

            # 逐元素相乘再求和
            output[i, j] = np.sum(input[h_start:h_end, w_start:w_end] * kernel)

    return output


# 使用示例
input_image = np.array([
    [1, 2, 3, 0, 1],
    [2, 3, 1, 2, 0],
    [0, 1, 2, 3, 1],
    [1, 0, 3, 1, 2],
    [2, 1, 0, 2, 3]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, 1],
    [-1, 1, 0]
])

output = conv2d(input_image, kernel, stride=1, padding=0)
print(f"输出形状: {output.shape}")
print(output)
```

## 卷积层的关键概念

### 多通道卷积

真实图像是 RGB 三通道（或多通道特征图）。多通道卷积需要为每个输入通道使用不同的卷积核，然后将结果相加。

```python
def conv2d_multi_channel(input, kernels, bias=0, stride=1, padding=0):
    """
    多通道卷积
    input: 形状 (C_in, H_in, W_in)
    kernels: 形状 (C_out, C_in, K, K)
    bias: 形状 (C_out,)
    """
    C_out, C_in, K, _ = kernels.shape

    if padding > 0:
        input = np.pad(input, ((0, 0), (padding, padding), (padding, padding)),
                      mode='constant')

    H_in, W_in = input.shape[1], input.shape[2]
    H_out = (H_in - K) // stride + 1
    W_out = (W_in - K) // stride + 1

    output = np.zeros((C_out, H_out, W_out))

    for c_out in range(C_out):
        for c_in in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    h_s, w_s = i * stride, j * stride
                    output[c_out, i, j] += np.sum(
                        input[c_in, h_s:h_s+K, w_s:w_s+K] * kernels[c_out, c_in]
                    )
        output[c_out] += bias[c_out]

    return output
```

### 步长（Stride）

步长控制卷积核在输入上滑动的距离。

| 步长 | 效果 | 输出尺寸变化 |
|------|------|-------------|
| Stride=1 | 逐像素滑动，信息损失最小 | 几乎不变 |
| Stride=2 | 跳过 1 个像素 | 约减半 |
| Stride=s | 跳过 s-1 个像素 | 约缩小 s 倍 |

输出尺寸公式：

$$H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

### 填充（Padding）

| 填充方式 | 说明 | 输出尺寸 |
|----------|------|----------|
| Valid（无填充） | 不使用填充 | $\lfloor \frac{H - K}{S} \rfloor + 1$ |
| Same（同尺寸填充） | 填充使得输出与输入尺寸相同 | $\lceil \frac{H}{S} \rceil$ |

## 池化层（Pooling）

池化层对特征图进行降采样，减少空间维度和计算量。

### 最大池化（Max Pooling）

在窗口内取最大值，提取最显著的特征。

```python
def max_pool2d(input, pool_size=2, stride=None):
    """最大池化"""
    if stride is None:
        stride = pool_size

    C, H_in, W_in = input.shape
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1

    output = np.zeros((C, H_out, W_out))

    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * stride, j * stride
                output[c, i, j] = np.max(
                    input[c, h_s:h_s+pool_size, w_s:w_s+pool_size]
                )

    return output
```

### 平均池化（Average Pooling）

在窗口内取平均值，保留更多背景信息。

### 池化对比

| 特性 | 最大池化 | 平均池化 |
|------|----------|----------|
| 操作 | 取窗口内最大值 | 取窗口内平均值 |
| 效果 | 突出最显著特征 | 保留整体信息 |
| 抗噪能力 | 较弱 | 较强 |
| 常用场景 | 特征提取 | 全局平均池化 |

## 经典 CNN 架构

### LeNet-5（1998）

Yann LeCun 提出的第一个实用 CNN，用于手写数字识别。

```
输入(32x32) -> Conv(5x5) -> AvgPool(2x2) -> Conv(5x5)
            -> AvgPool(2x2) -> FC(120) -> FC(84) -> FC(10) -> Softmax
```

### AlexNet（2012）

ImageNet 竞赛冠军，开启了深度学习革命。

| 层 | 配置 |
|----|------|
| Conv1 | 96 @ 11x11, stride=4 |
| MaxPool1 | 3x3, stride=2 |
| Conv2 | 256 @ 5x5 |
| MaxPool2 | 3x3, stride=2 |
| Conv3 | 384 @ 3x3 |
| Conv4 | 384 @ 3x3 |
| Conv5 | 256 @ 3x3 |
| MaxPool3 | 3x3, stride=2 |
| FC6 | 4096 + Dropout(0.5) |
| FC7 | 4096 + Dropout(0.5) |
| FC8 | 1000 (输出) |

### VGG（2014）

核心贡献：使用 $3 \times 3$ 小卷积核替代大卷积核。

| 特性 | 说明 |
|------|------|
| 卷积核 | 全部使用 3x3 |
| 池化 | 全部使用 2x2 MaxPool, stride=2 |
| 通道数 | 64 -> 128 -> 256 -> 512 |
| 深度 | VGG-16 有 16 层（含权重） |
| 参数量 | 约 1.38 亿（主要在 FC 层） |

两个 $3 \times 3$ 卷积的感受野等效于一个 $5 \times 5$ 卷积，但参数更少、非线性更强。

### GoogLeNet / Inception（2014）

核心思想：多尺度特征并行提取。

```python
# Inception 模块的简化实现
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, out3x3_reduce, out3x3,
                 out5x5_reduce, out5x5, out_pool_proj):
        super().__init__()

        # 1x1 卷积分支
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        # 3x3 卷积分支（先降维再卷积）
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out3x3_reduce, out3x3, kernel_size=3, padding=1),
        )

        # 5x5 卷积分支
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out5x5_reduce, out5x5, kernel_size=5, padding=2),
        )

        # 池化分支
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
```

### ResNet（2015）

Kaiming He 提出的残差网络，通过残差连接解决了超深网络的退化问题。

**残差块**：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \mathcal{W}) + \mathbf{x}$$

其中 $\mathcal{F}(\mathbf{x}, \mathcal{W})$ 是残差函数（两层或三层卷积），$\mathbf{x}$ 是恒等映射。

```python
class BasicBlock(nn.Module):
    """ResNet 基本残差块（两层卷积）"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

### CNN 架构演进对比

| 架构 | 年份 | 深度 | Top-1 错误率 | 创新点 |
|------|------|------|-------------|--------|
| LeNet-5 | 1998 | 7 层 | - | 第一个 CNN |
| AlexNet | 2012 | 8 层 | 37.5% | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 16 层 | 26.2% | 3x3 小卷积核 |
| GoogLeNet | 2014 | 22 层 | 21.2% | Inception 模块 |
| ResNet-50 | 2015 | 50 层 | 22.0% | 残差连接 |
| ResNet-152 | 2015 | 152 层 | 21.6% | 极深网络 |
| EfficientNet | 2019 | 可变 | 20.0% | 复合缩放 |

## 在 PyTorch 中构建 CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernCNN(nn.Module):
    """
    现代 CNN 实现
    包含：Conv + BN + ReLU + Pooling + Dropout + FC
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # 特征提取器
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 使用示例
model = ModernCNN(num_classes=10)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 测试前向传播
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"输出形状: {output.shape}")
```

## 感受野（Receptive Field）

感受野是指输出特征图上某个点能够"看到"的输入区域大小。

| 层级 | 累积感受野 | 说明 |
|------|-----------|------|
| 第 1 层 3x3 | 3x3 | 直接看到的区域 |
| 第 2 层 3x3 | 5x5 | (3-1)*1 + 3 = 5 |
| 第 3 层 3x3 | 7x7 | (5-1)*1 + 3 = 7 |
| 第 N 层 3x3 | (2N+1)x(2N+1) | 每增加一层增加 2 个像素 |

感受野的计算公式：

$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

其中 $K_l$ 是第 $l$ 层的卷积核大小，$S_i$ 是前面层的步长。

## 总结

CNN 通过局部连接和权重共享，高效地处理了图像等具有空间结构的数据。本章核心要点：

1. 卷积操作通过局部连接和权重共享大幅减少了参数数量
2. 步长控制输出尺寸，填充保持空间信息
3. 池化层进行空间降采样，最大池化提取最显著特征
4. 经典 CNN 架构从 LeNet 到 ResNet，深度不断增加，性能持续提升
5. 残差连接解决了深层网络的退化问题
6. 现代 CNN 普遍采用 Conv+BN+ReLU 的标准组合

CNN 擅长处理空间数据，但对于序列数据，我们需要另一种专门的网络结构：循环神经网络。

---

**上一篇**: [36. 批归一化](36-batch-normalization.md)
**下一篇**: [38. RNN、LSTM 与 GRU](38-rnn-lstm-gru.md)
