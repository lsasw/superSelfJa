---
title: 损失函数详解
icon: chart-line
order: 34
---

# 损失函数详解

损失函数（Loss Function）是衡量模型预测值与真实值之间差距的数学工具。它定义了优化的目标，是训练神经网络的核心要素。选择合适的损失函数直接决定了模型能否有效学习以及最终的预测质量。

## 损失函数的核心作用

损失函数在机器学习中扮演着三个关键角色：

1. **优化目标**：定义了什么才是"好"的模型，训练过程就是最小化损失函数
2. **梯度来源**：损失对模型参数的梯度指导了参数更新的方向和幅度
3. **任务定义**：不同的损失函数隐含了对任务的不同假设

💡 **提示**：损失函数（Loss Function）与代价函数（Cost Function）、目标函数（Objective Function）经常混用。严格来说，损失函数针对单个样本，代价函数是整个数据集上损失的平均值，目标函数还可能包含正则化项。实践中通常不做严格区分。

## 损失函数的分类

| 任务类型 | 常用损失函数 | 适用场景 |
|----------|--------------|----------|
| 回归 | MSE、MAE、Huber、Smooth L1 | 预测连续值 |
| 二分类 | BCE（二元交叉熵） | 两个类别的判断 |
| 多分类 | 交叉熵、Focal Loss | 多个互斥类别 |
| 多标签分类 | 多标签 BCE | 多个独立标签 |
| 排序/检索 | 对比损失、三元组损失 | 相似度学习 |
| 序列生成 | CTC Loss、序列交叉熵 | 语音识别、机器翻译 |

## 回归任务的损失函数

### 均方误差（MSE, Mean Squared Error）

MSE 是最常用的回归损失函数。

**数学定义**：

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**导数**：

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{2}{n} (y_i - \hat{y}_i)$$

**代码实现**：

```python
import numpy as np

def mse_loss(y_true, y_pred):
    """
    均方误差损失
    y_true: 真实值，形状 (n,) 或 (n, 1)
    y_pred: 预测值，形状 (n,) 或 (n, 1)
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_gradient(y_true, y_pred):
    """MSE 对预测值的梯度"""
    n = y_true.shape[0]
    return -2.0 / n * (y_true - y_pred)
```

**特点**：

| 优点 | 缺点 |
|------|------|
| 处处可导，梯度计算简单 | 对异常值（Outliers）极度敏感 |
| 大误差受到更大惩罚 | 误差分布非高斯时表现不佳 |
| 在正态噪声假设下是最优的 | 平方运算可能导致数值不稳定 |

### 平均绝对误差（MAE, Mean Absolute Error）

也称为 L1 损失。

**数学定义**：

$$L_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**代码实现**：

```python
def mae_loss(y_true, y_pred):
    """平均绝对误差损失"""
    return np.mean(np.abs(y_true - y_pred))

def mae_loss_gradient(y_true, y_pred):
    """MAE 对预测值的梯度（次梯度）"""
    return np.sign(y_pred - y_true) / y_true.shape[0]
```

**特点**：

| 优点 | 缺点 |
|------|------|
| 对异常值鲁棒 | 在零点不可导（使用次梯度） |
| 等价于中位数回归 | 梯度恒定，收敛速度可能较慢 |
| 对拉普拉斯噪声最优 | 优化难度比 MSE 大 |

### Huber Loss

Huber Loss 结合了 MSE 和 MAE 的优点，在小误差时使用 MSE，大误差时使用 MAE。

**数学定义**：

$$L_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2, & |y - \hat{y}| > \delta \end{cases}$$

**代码实现**：

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss
    delta: 切换 MSE/MAE 的阈值
    """
    error = y_true - y_pred
    is_small = np.abs(error) <= delta

    # MSE 部分
    mse_part = 0.5 * error ** 2
    # MAE 部分
    mae_part = delta * np.abs(error) - 0.5 * delta ** 2

    return np.mean(np.where(is_small, mse_part, mae_part))

def huber_loss_gradient(y_true, y_pred, delta=1.0):
    """Huber Loss 梯度"""
    error = y_true - y_pred
    is_small = np.abs(error) <= delta

    # MSE 梯度部分
    mse_grad = -error
    # MAE 梯度部分
    mae_grad = -delta * np.sign(error)

    return np.where(is_small, mse_grad, mae_grad) / y_true.shape[0]
```

**特点**：

| 特性 | 说明 |
|------|------|
| 小误差区域 | 表现如 MSE，提供精确的梯度 |
| 大误差区域 | 表现如 MAE，对异常值鲁棒 |
| 可导性 | 在整个定义域上处处可导 |
| 超参数 | $\delta$ 控制切换点，通常取 1.0 |

### Smooth L1 Loss

Smooth L1 是 Huber Loss 在 $\delta = 1$ 时的特例，在目标检测（如 Faster R-CNN）中广泛使用。

```python
def smooth_l1_loss(y_true, y_pred, beta=1.0):
    """
    Smooth L1 Loss（Faster R-CNN 中使用）
    beta: 切换点参数
    """
    diff = y_true - y_pred
    abs_diff = np.abs(diff)

    # 判断条件
    smooth_mask = abs_diff < beta

    # Smooth 部分
    smooth = 0.5 * (diff ** 2) / beta
    # L1 部分
    l1 = abs_diff - 0.5 * beta

    return np.mean(np.where(smooth_mask, smooth, l1))
```

## 分类任务的损失函数

### 二元交叉熵（BCE, Binary Cross-Entropy）

用于二分类任务。

**数学定义**：

$$L_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$

**代码实现**：

```python
def binary_cross_entropy(y_true, y_pred):
    """二元交叉熵损失"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

def bce_gradient(y_true, y_pred):
    """BCE 对预测值的梯度"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
```

### 多分类交叉熵（Categorical Cross-Entropy）

用于多分类任务，配合 Softmax 激活函数使用。

**数学定义**：

$$L_{\text{CE}} = -\sum_{c=1}^{C} y_{c} \log(\hat{y}_{c})$$

其中 $C$ 是类别数，$y_c$ 是 one-hot 编码的真实标签。

**代码实现**：

```python
def categorical_cross_entropy(y_true, y_pred):
    """
    多分类交叉熵损失
    y_true: one-hot 编码，形状 (n, num_classes)
    y_pred: Softmax 输出，形状 (n, num_classes)
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

### Softmax + 交叉熵的合并梯度

当 Softmax 和交叉熵组合使用时，梯度计算简化为：

```python
def softmax_cross_entropy_with_logits(logits, y_true):
    """
    数值稳定的 Softmax + 交叉熵
    logits: 网络原始输出（未经过 Softmax），形状 (n, num_classes)
    y_true: one-hot 编码的真实标签
    返回值: loss, gradient
    """
    # 数值稳定的 Softmax
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 计算损失
    eps = 1e-15
    softmax_output = np.clip(softmax_output, eps, 1 - eps)
    loss = -np.mean(np.sum(y_true * np.log(softmax_output), axis=1))

    # 梯度（极其简洁的形式）
    gradient = (softmax_output - y_true) / y_true.shape[0]

    return loss, gradient
```

💡 **提示**：PyTorch 的 `nn.CrossEntropyLoss` 和 TensorFlow 的 `tf.nn.softmax_cross_entropy_with_logits` 都内置了这个合并操作，在代码中不需要手动加 Softmax 层。

### Focal Loss

Focal Loss 由 Kaiming He 等人在 2017 年提出，专门解决类别不平衡问题。它在交叉熵的基础上添加了调制因子，降低易分类样本的权重，使模型更关注难分类的样本。

**数学定义**：

$$L_{\text{FL}} = -\alpha_t (1 - \hat{y}_t)^{\gamma} \log(\hat{y}_t)$$

其中：
- $\hat{y}_t$ 是正确类别的预测概率
- $\gamma$ 是聚焦参数（通常取 2.0），控制对难样本的关注程度
- $\alpha_t$ 是类别平衡因子

**代码实现**：

```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss
    alpha: 类别平衡因子
    gamma: 聚焦参数
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 计算交叉熵部分
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # 调制因子
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    modulating_factor = (1 - p_t) ** gamma

    # 类别平衡因子
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    return np.mean(alpha_factor * modulating_factor * bce)
```

### 损失函数选择决策表

| 场景 | 推荐损失函数 | 理由 |
|------|--------------|------|
| 标准回归 | MSE | 计算简单，在噪声为正态分布时最优 |
| 含异常值的回归 | Huber / Smooth L1 | 对异常值鲁棒 |
| 二分类 | BCE | 信息论解释，梯度性质好 |
| 多分类（平衡数据） | 交叉熵 | 标准选择，与 Softmax 配合完美 |
| 多分类（不平衡数据） | Focal Loss / 加权交叉熵 | 缓解类别不平衡 |
| 目标检测边框回归 | Smooth L1 | 对异常边界值鲁棒 |
| 多标签分类 | 多标签 BCE | 每个标签独立计算损失 |

## 在 PyTorch 中使用损失函数

```python
import torch
import torch.nn as nn

# 回归任务
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
huber_loss = nn.HuberLoss(delta=1.0)
smooth_l1_loss = nn.SmoothL1Loss()

# 分类任务
bce_loss = nn.BCELoss()           # 需要输入经过 Sigmoid
bce_with_logits = nn.BCEWithLogitsLoss()  # 内置 Sigmoid，更数值稳定
ce_loss = nn.CrossEntropyLoss()   # 内置 Softmax
nll_loss = nn.NLLLoss()           # 需要输入是 log_softmax

# 目标检测
focal_loss = nn.FocalLoss()       # 需要 torchvision

# 使用示例
predictions = torch.randn(32, 10)  # batch_size=32, num_classes=10
targets = torch.randint(0, 10, (32,))

loss = ce_loss(predictions, targets)
print(f"Cross-Entropy Loss: {loss.item():.4f}")
```

💡 **提示**：优先使用带 Logits 的损失函数版本（如 `BCEWithLogitsLoss` 而非 `BCELoss`），它们内部使用了 Log-Sum-Exp 技巧，数值稳定性更好。

## 自定义损失函数

在 PyTorch 中实现自定义损失函数非常简单：

```python
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """加权均方误差损失"""

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        if self.weights is None:
            return torch.mean((y_pred - y_true) ** 2)
        return torch.mean(self.weights * (y_pred - y_true) ** 2)


class ContrastiveLoss(nn.Module):
    """对比损失（用于相似度学习）"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        output1, output2: 两个样本的特征向量
        label: 1 表示相似，0 表示不相似
        """
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class TripletLoss(nn.Module):
    """三元组损失"""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor: 锚样本
        positive: 与 anchor 同类
        negative: 与 anchor 不同类
        """
        pos_dist = nn.functional.pairwise_distance(anchor, positive)
        neg_dist = nn.functional.pairwise_distance(anchor, negative)

        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)
```

## 损失函数与信息论的联系

交叉熵损失的理论基础来自信息论。理解信息论视角有助于深入理解为什么交叉熵是分类任务的天然选择：

- **熵（Entropy）**：度量信息的不确定性 $H(p) = -\sum p(x) \log p(x)$
- **交叉熵（Cross-Entropy）**：使用分布 $q$ 编码分布 $p$ 所需的平均比特数
- **KL 散度**：两个分布之间的"距离" $D_{KL}(p \| q) = H(p, q) - H(p)$

最小化交叉熵等价于最小化 KL 散度，即使模型预测分布尽可能接近真实分布。

## 总结

损失函数定义了模型的优化目标，是连接预测与真实值的桥梁。本章核心要点：

1. 回归任务首选 MSE（无异常值时）或 Huber（有异常值时）
2. 分类任务首选交叉熵，配合对应的激活函数使用
3. 使用带 Logits 的版本（如 `BCEWithLogitsLoss`）数值稳定性更好
4. 类别不平衡问题使用 Focal Loss 或加权损失
5. 自定义损失函数只需继承 `nn.Module` 并实现 `forward` 方法
6. 交叉熵有深刻的信息论基础，最小化交叉熵等价于最小化分布间的 KL 散度

选择合适的损失函数后，需要防止模型过度拟合训练数据，这就需要正则化技术来约束模型的复杂度。

---

**上一篇**: [33. 激活函数详解](33-activation-functions.md)
**下一篇**: [35. 正则化技术](35-regularization.md)
