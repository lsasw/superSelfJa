---
title: 批归一化
icon: bars-sort
order: 36
---

# 批归一化

批归一化（Batch Normalization，简称 BatchNorm 或 BN）是深度学习中最重要的技术之一。它通过对每层的输入进行归一化处理，大幅加速训练收敛、提高模型稳定性，并具有一定的正则化效果。

## 深度网络训练的根本问题

在深度神经网络中，随着网络的加深，每一层输入数据的分布都在不断变化。这种现象被称为**内部协变量偏移（Internal Covariate Shift）**：

> 网络中每一层的输入分布随着前面层参数的更新而不断变化，导致后面的层需要持续适应新的数据分布。

这带来了三个主要问题：

1. **训练速度慢**：每层需要适应不断变化的输入分布，学习率必须设置得很小
2. **梯度消失/爆炸**：多层复合变换导致梯度信号不稳定
3. **对初始化敏感**：权重初始化不当会导致训练失败

## Batch Normalization 的原理

Sergey Ioffe 和 Christian Szegedy 在 2015 年提出的 BatchNorm 通过在每个 mini-batch 上对激活值进行归一化来解决上述问题。

### 计算步骤

对于一个 mini-batch 中某个特征维度上的激活值 $\mathcal{B} = \{x_1, x_2, ..., x_m\}$：

**步骤 1：计算均值和方差**

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**步骤 2：归一化**

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

**步骤 3：缩放和平移（可学习参数）**

$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\gamma$（缩放）和 $\beta$（平移）是可学习的参数，初始值 $\gamma = 1, \beta = 0$。

### 为什么需要可学习的 $\gamma$ 和 $\beta$

归一化操作可能限制了网络的表达能力。例如，Sigmoid 函数在零附近的线性区域最敏感，强制归一化到零均值可能导致激活集中在非敏感区域。可学习的 $\gamma$ 和 $\beta$ 允许网络在需要时"撤销"归一化，恢复原始的激活分布。

💡 **提示**：如果最优的激活分布正好是零均值单位方差，网络可以学习到 $\gamma = 1, \beta = 0$。如果有其他需求，网络也可以学习到不同的值。这是 BatchNorm 设计的精妙之处。

## BatchNorm 的完整实现

```python
import numpy as np

class BatchNorm2D:
    """
    手动实现的 2D 批归一化层
    适用于卷积网络，对每个通道独立计算 BN
    """

    def __init__(self, num_channels, momentum=0.1, epsilon=1e-5):
        self.num_channels = num_channels
        self.momentum = momentum
        self.epsilon = epsilon

        # 可学习参数
        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))

        # 运行时统计量（推理时使用）
        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))

        # 缓存（反向传播使用）
        self.cache = None
        self.training = True

    def forward(self, x):
        """
        前向传播
        x: 形状 (batch, channels, height, width)
        """
        if self.training:
            # 计算 mini-batch 均值和方差（对每个通道独立计算）
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # 更新运行时统计量（指数移动平均）
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * batch_var
            )
        else:
            # 推理时使用运行时统计量
            batch_mean = self.running_mean
            batch_var = self.running_var

        # 归一化
        x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

        # 缩放和平移
        out = self.gamma * x_norm + self.beta

        # 缓存用于反向传播
        if self.training:
            self.cache = (x, x_norm, batch_mean, batch_var)

        return out

    def backward(self, dout):
        """
        反向传播
        dout: 上游梯度，形状 (batch, channels, height, width)
        """
        x, x_norm, mean, var = self.cache
        m = x.shape[0] * x.shape[2] * x.shape[3]  # mini-batch 总元素数

        # gamma 和 beta 的梯度
        dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        # 对归一化输入的梯度
        dx_norm = dout * self.gamma

        # 对 var 的梯度
        dvar = np.sum(
            dx_norm * (x - mean) * (-0.5) * (var + self.epsilon) ** (-1.5),
            axis=(0, 2, 3), keepdims=True
        )

        # 对 mean 的梯度
        dmean = np.sum(
            dx_norm * (-1.0 / np.sqrt(var + self.epsilon)),
            axis=(0, 2, 3), keepdims=True
        ) + dvar * np.mean(-2.0 * (x - mean), axis=(0, 2, 3), keepdims=True)

        # 对输入的梯度
        dx = (
            dx_norm / np.sqrt(var + self.epsilon)
            + dvar * 2.0 * (x - mean) / m
            + dmean / m
        )

        return dx, dgamma, dbeta
```

## BatchNorm 在不同网络中的应用

### 在全连接层中使用

对于全连接层，BatchNorm 对每个特征维度独立计算均值和方差。

```python
# 输入形状: (batch_size, num_features)
# BatchNorm 对每个 feature 维度独立计算统计量
```

### 在卷积层中使用

对于卷积层，BatchNorm 对每个通道独立计算均值和方差，在所有空间位置和 batch 维度上聚合统计量。

```python
# 输入形状: (batch, channels, height, width)
# 均值和方差的形状: (1, channels, 1, 1)
# 对所有空间位置 (H x W) 和 batch 求平均/方差
```

💡 **提示**：卷积层使用 BatchNorm 时，一个 $3 \times 3$ 卷积核在 BN 中只有 3 个可学习参数（gamma 和 beta，每个通道各一个），无论输入图像多大。这保证了平移不变性。

## 在 PyTorch 中使用 BatchNorm

```python
import torch
import torch.nn as nn

class CNNWithBatchNorm(nn.Module):
    """带批归一化的 CNN"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 第一组卷积
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),       # BN 在卷积和激活之间
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第二组卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 使用示例
model = CNNWithBatchNorm(num_classes=10)

# 训练模式（默认）
model.train()
train_output = model(torch.randn(32, 3, 32, 32))

# 评估模式（使用 running_mean 和 running_var）
model.eval()
eval_output = model(torch.randn(32, 3, 32, 32))
```

## 训练模式 vs 推理模式

BatchNorm 在训练和推理时的行为完全不同：

| 阶段 | 均值 | 方差 | 行为 |
|------|------|------|------|
| 训练 | mini-batch 均值 | mini-batch 方差 | 对当前 batch 归一化，更新 running 统计量 |
| 推理 | running_mean（EMA） | running_var（EMA） | 使用训练期间累积的全局统计量 |

```python
# 重要：推理时必须调用 model.eval()！
model.eval()  # 切换到评估模式，BN 使用 running 统计量
with torch.no_grad():
    predictions = model(test_data)
```

## BatchNorm 的效果

| 效果 | 说明 |
|------|------|
| 加速收敛 | 允许使用更大的学习率，减少训练轮数 |
| 缓解梯度消失 | 保持激活值在敏感区间 |
| 降低初始化敏感度 | 对初始权重的选择不再极端敏感 |
| 轻微正则化 | mini-batch 统计量的噪声有正则化效果 |
| 改善深层网络训练 | 使 100+ 层的网络训练成为可能 |

## BatchNorm 的注意事项

| 问题 | 描述 | 解决方案 |
|------|------|----------|
| 小 batch size | batch size < 8 时统计量不准确 | 使用 GroupNorm 或增大 batch size |
| 与 Dropout 冲突 | 两者都有正则化效果，叠加可能过度正则化 | 减少 Dropout 率或只用其一 |
| RNN 中不适用 | 序列长度可变，难以定义 batch 统计量 | 使用 LayerNorm |
| 推理依赖 batch 统计 | 如果推理时 batch size 为 1，running 统计量可能不够准确 | 确保训练和推理的 batch 统计一致 |
| 在线学习不适用 | 数据逐个到达时无法计算 batch 统计量 | 使用其他归一化方法 |

## 其他归一化方法对比

| 方法 | 归一化维度 | 适用场景 | 对 batch size 敏感度 |
|------|------------|----------|---------------------|
| BatchNorm | 对每个通道，在 batch + 空间维度上 | CNN（大 batch） | 高 |
| LayerNorm | 对每个样本，在特征维度上 | RNN、Transformer | 无 |
| InstanceNorm | 对每个通道，在空间维度上 | 风格迁移 | 无 |
| GroupNorm | 对每组通道，在空间维度上 | 小 batch CNN | 无 |

```python
# 各种归一化的 PyTorch 实现
nn.BatchNorm2d(num_features)   # BatchNorm
nn.LayerNorm(normalized_shape) # LayerNorm
nn.InstanceNorm2d(num_features) # InstanceNorm
nn.GroupNorm(num_groups, num_channels)  # GroupNorm
```

## BatchNorm 位置的选择

BatchNorm 放在激活函数之前还是之后？两种方案都有支持者：

| 方案 | 结构 | 支持者 |
|------|------|--------|
| 前置 BN（Pre-activation） | BN -> ReLU -> Conv | ResNet v2 |
| 后置 BN（Post-activation） | Conv -> BN -> ReLU | 原始 ResNet v1 |

目前更推荐**前置 BN**（Pre-activation），因为信号传播更稳定，尤其在极深网络中。

## 总结

批归一化是深度学习最重要的技术创新之一。本章核心要点：

1. BatchNorm 通过对 mini-batch 激活值归一化，解决内部协变量偏移问题
2. 计算步骤：求均值方差 -> 归一化 -> 可学习的缩放和平移
3. 训练时使用 batch 统计量，推理时使用指数移动平均的全局统计量
4. 必须注意 `model.train()` 和 `model.eval()` 的切换
5. BatchNorm 加速收敛、允许更大学习率、降低初始化敏感度
6. 小 batch size 时效果下降，可考虑 LayerNorm 或 GroupNorm 替代

归一化技术解决了深层网络的训练稳定性问题。接下来，我们将学习深度学习中另一类基础架构：卷积神经网络。

---

**上一篇**: [35. 正则化技术](35-regularization.md)
**下一篇**: [37. CNN 基础](37-cnn-basics.md)
