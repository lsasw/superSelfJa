---
title: 正则化技术
icon: shield-halved
order: 35
---

# 正则化技术

正则化（Regularization）是防止模型过拟合的核心技术。过拟合是指模型在训练数据上表现优异，但在未见数据上表现显著下降的现象。本章将系统讲解深度学习中常用的正则化方法及其原理、实现和应用策略。

## 过拟合与欠拟合

在深入正则化之前，需要理解模型的两个基本状态：

| 状态 | 训练误差 | 测试误差 | 原因 | 解决方案 |
|------|----------|----------|------|----------|
| 欠拟合（Underfitting） | 高 | 高 | 模型太简单、训练不充分 | 增大模型、训练更久 |
| 恰好拟合（Good Fit） | 低 | 低 | 模型复杂度匹配任务 | -- |
| 过拟合（Overfitting） | 很低 | 高 | 模型太复杂、数据太少 | 正则化、增大数据 |

### 偏差-方差权衡

模型的泛化误差可以分解为三个部分：

$$\text{泛化误差} = \text{偏差}^2 + \text{方差} + \text{不可约误差}$$

| 成分 | 含义 | 与模型复杂度的关系 |
|------|------|--------------------|
| 偏差（Bias） | 模型预测的系统性偏差 | 模型越简单，偏差越大 |
| 方差（Variance） | 模型对不同训练集的敏感度 | 模型越复杂，方差越大 |
| 不可约误差 | 数据本身的噪声 | 与模型无关 |

💡 **提示**：正则化本质上是在偏差和方差之间寻找最优平衡。通过引入有偏的估计（如缩小权重），换取方差的降低，从而减少总体泛化误差。

## L1 正则化（Lasso）

L1 正则化在损失函数中添加权重绝对值之和作为惩罚项。

### 数学定义

$$L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} |w_i|$$

### 特性

L1 正则化的显著特性是产生**稀疏解**，即很多权重值恰好为零。这源于 L1 正则化的等高线是菱形的，最优点往往落在坐标轴上。

### 代码实现

```python
import numpy as np

def l1_regularization(weights, lambda_l1=0.01):
    """计算 L1 正则化项"""
    return lambda_l1 * np.sum(np.abs(weights))

def l1_gradient(weights, lambda_l1=0.01):
    """L1 正则化的梯度（次梯度）"""
    return lambda_l1 * np.sign(weights)


class L1RegularizedLinearModel:
    """带 L1 正则化的线性模型"""

    def __init__(self, n_features, lambda_l1=0.01, learning_rate=0.01):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lambda_l1 = lambda_l1
        self.lr = learning_rate

    def predict(self, X):
        return X @ self.weights + self.bias

    def train_step(self, X, y):
        """带 L1 正则化的梯度下降"""
        n = X.shape[0]
        predictions = self.predict(X)
        errors = predictions - y

        # 原始梯度
        grad_w = (1.0 / n) * X.T @ errors

        # L1 正则化梯度（次梯度）
        grad_w += self.lambda_l1 * np.sign(self.weights)

        # 更新
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * (1.0 / n) * np.sum(errors)

        return np.mean(errors ** 2)
```

## L2 正则化（Ridge / 权重衰减）

L2 正则化在损失函数中添加权重平方和作为惩罚项，是最常用的正则化方法。

### 数学定义

$$L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} w_i^2$$

在 PyTorch 中，L2 正则化通常通过优化器的 `weight_decay` 参数实现，也称为**权重衰减（Weight Decay）**。

### 代码实现

```python
def l2_regularization(weights, lambda_l2=0.01):
    """计算 L2 正则化项"""
    return lambda_l2 * np.sum(weights ** 2)

def l2_gradient(weights, lambda_l2=0.01):
    """L2 正则化的梯度"""
    return 2 * lambda_l2 * weights


class L2RegularizedLinearModel:
    """带 L2 正则化的线性模型"""

    def __init__(self, n_features, lambda_l2=0.01, learning_rate=0.01):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lambda_l2 = lambda_l2
        self.lr = learning_rate

    def predict(self, X):
        return X @ self.weights + self.bias

    def train_step(self, X, y):
        """带 L2 正则化的梯度下降（权重衰减形式）"""
        n = X.shape[0]
        predictions = self.predict(X)
        errors = predictions - y

        # 原始梯度 + L2 正则化
        grad_w = (1.0 / n) * X.T @ errors + 2 * self.lambda_l2 * self.weights

        # 更新（可以重写为权重衰减形式）
        # self.weights -= self.lr * grad_w
        # 等价于:
        self.weights = self.weights * (1 - 2 * self.lr * self.lambda_l2) \
                       - self.lr * (1.0 / n) * X.T @ errors

        self.bias -= self.lr * (1.0 / n) * np.sum(errors)

        return np.mean(errors ** 2)
```

### L1 vs L2 对比

| 特性 | L1 正则化 | L2 正则化 |
|------|-----------|-----------|
| 惩罚项 | $\sum |w_i|$ | $\sum w_i^2$ |
| 解的性质 | 稀疏解（很多权重为 0） | 权重趋于小但非零 |
| 特征选择 | 自动进行特征选择 | 不产生稀疏性 |
| 可导性 | 在 0 点不可导 | 处处可导 |
| 异常值敏感度 | 不敏感 | 较敏感 |
| 常用场景 | 特征工程、稀疏模型 | 深度学习默认正则化 |
| PyTorch 实现 | 需手动添加 | `weight_decay` 参数 |

## Dropout

Dropout 由 Geoffrey Hinton 等人在 2014 年提出，是深度学习中最著名的正则化技术之一。

### 工作原理

在训练过程中，Dropout 以概率 $p$ 随机"丢弃"（置零）部分神经元，同时在推理时将保留的神经元输出乘以 $p$（或使用 Inverted Dropout 在训练时放大输出）。

### 为什么 Dropout 有效

1. **打破神经元共适应**：防止神经元之间形成依赖关系，每个神经元必须学习独立有用的特征
2. **集成学习视角**：每次前向传播相当于训练一个子网络，推理时使用所有子网络的集成
3. **噪声注入**：Dropout 相当于在激活值上添加噪声，增强模型鲁棒性

### 代码实现

```python
class Dropout:
    """手动实现 Dropout 层"""

    def __init__(self, dropout_rate=0.5):
        """
        dropout_rate: 丢弃概率（如 0.5 表示丢弃 50% 的神经元）
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x):
        if not self.training or self.dropout_rate == 0:
            return x

        # 生成随机掩码（Inverted Dropout）
        self.mask = (np.random.rand(*x.shape) > self.dropout_rate)
        # 除以 (1 - dropout_rate) 保持期望不变
        return x * self.mask / (1 - self.dropout_rate)

    def backward(self, grad_output):
        if not self.training or self.dropout_rate == 0:
            return grad_output

        return grad_output * self.mask / (1 - self.dropout_rate)
```

### Inverted Dropout

上述实现使用的是 Inverted Dropout，在训练时就放大输出，使得推理时不需要修改网络。这是现代框架的默认实现方式。

```python
# 传统 Dropout（推理时需要缩放）
x_dropped = x * mask  # 训练时
x_test = x * p        # 推理时

# Inverted Dropout（推理时不需要操作）
x_dropped = x * mask / p  # 训练时
x_test = x                # 推理时
```

### 在 PyTorch 中使用

```python
import torch
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),   # Dropout 层
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.network(x)


# 使用示例
model = NetworkWithDropout(784, 256, 10, dropout_rate=0.5)

model.train()  # Dropout 生效
train_output = model(torch.randn(32, 784))

model.eval()   # Dropout 关闭
eval_output = model(torch.randn(32, 784))
```

### Dropout 率的选择

| 网络层类型 | 推荐 Dropout 率 | 说明 |
|------------|-----------------|------|
| 全连接层 | 0.3 - 0.5 | 经典取值范围 |
| 卷积层 | 0.1 - 0.3 | 卷积本身已有正则化效果 |
| 循环层（RNN） | 0.2 - 0.5 | 使用 Variational Dropout |
| Transformer 注意力 | 0.1 - 0.3 | attention_dropout |
| Transformer FFN | 0.1 - 0.3 | hidden_dropout |
| 输入层（Input Dropout） | 0.0 - 0.2 | 对输入特征做 Dropout |

## 早停法（Early Stopping）

早停法是最简单的正则化方法：在验证集性能不再提升时停止训练。

### 实现

```python
class EarlyStopping:
    """早停法实现"""

    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        """
        patience: 验证集性能不提升的最大轮数
        min_delta: 认为性能有提升的最小变化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = self._get_model_state(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = self._get_model_state(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _get_model_state(self, model):
        """保存模型状态"""
        return {k: v.clone() for k, v in model.state_dict().items()}

    def restore_best_weights(self, model):
        """恢复最佳模型权重"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# 使用示例
early_stopping = EarlyStopping(patience=15, min_delta=1e-4, verbose=True)

for epoch in range(200):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        early_stopping.restore_best_weights(model)
        break
```

## 数据增强（Data Augmentation）

数据增强通过对训练数据施加变换来增加数据多样性，是最有效的正则化方法之一。

### 图像数据增强

```python
import torchvision.transforms as transforms

# 常见的图像数据增强组合
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # 随机裁剪 + 缩放
    transforms.RandomHorizontalFlip(0.5),    # 随机水平翻转
    transforms.RandomRotation(15),           # 随机旋转
    transforms.ColorJitter(                  # 颜色扰动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                         std=[0.229, 0.224, 0.225])
])

# 验证集不使用随机增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### 高级数据增强技术

| 技术 | 原理 | 效果 |
|------|------|------|
| Mixup | 对两个样本及其标签做线性插值 | 简单有效，线性正则化 |
| CutMix | 将一个样本的部分区域替换为另一样本 | 定位能力更强 |
| Cutout | 随机遮挡图像的矩形区域 | 强制模型关注全局 |
| RandAugment | 自动搜索增强策略 | 减少手动调参 |
| AutoAugment | 基于 NAS 搜索的增强 | 效果最好但计算成本高 |

```python
# Mixup 实现
def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 使用示例
for images, labels in train_loader:
    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

    optimizer.zero_grad()
    outputs = model(images)
    loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, labels_a, labels_b, lam)
    loss.backward()
    optimizer.step()
```

## 标签平滑（Label Smoothing）

标签平滑通过将硬标签（hard label）转换为软标签（soft label），防止模型过度自信。

### 原理

原始 one-hot 标签：$[0, 0, 1, 0, 0]$
标签平滑后：$[\frac{\epsilon}{K}, \frac{\epsilon}{K}, 1 - \epsilon + \frac{\epsilon}{K}, \frac{\epsilon}{K}, \frac{\epsilon}{K}]$

其中 $\epsilon$ 是平滑因子（通常取 0.1），$K$ 是类别数。

```python
import torch.nn.functional as F

# PyTorch 内置标签平滑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 手动实现
def label_smoothing_one_hot(labels, num_classes, smoothing=0.1):
    """将 one-hot 标签转换为平滑标签"""
    with torch.no_grad():
        smooth_labels = torch.full(
            (labels.size(0), num_classes),
            smoothing / num_classes,
            device=labels.device
        )
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing + smoothing / num_classes)
    return smooth_labels
```

## 正则化方法综合对比

| 方法 | 实现难度 | 计算开销 | 正则化强度 | 推荐组合 |
|------|----------|----------|------------|----------|
| L2 正则化 | 极低 | 无 | 中等 | 所有场景默认使用 |
| Dropout | 极低 | 无（推理时） | 强 | 全连接层首选 |
| 早停法 | 低 | 低（需验证集） | 中等 | 所有场景推荐 |
| 数据增强 | 中 | 低 | 很强 | 图像/文本任务必选 |
| 标签平滑 | 极低 | 无 | 弱-中 | 多分类任务推荐 |
| Batch Normalization | 低 | 低 | 中等 | 与 Dropout 配合需谨慎 |
| L1 正则化 | 低 | 无 | 中等 | 需要特征选择时 |

## 正则化策略建议

💡 **提示**：正则化不是越多越好。过度正则化会导致欠拟合。建议按以下顺序添加正则化：

1. **基础层**：L2 正则化（weight_decay）+ 合理的学习率
2. **增强层**：数据增强 + Dropout（根据网络类型选择位置）
3. **精细层**：早停法 + 标签平滑（根据验证集表现调整）

如果一个模型在训练集上表现差（欠拟合），应该**先减少正则化**，确保模型有足够的拟合能力，再逐步增加防止过拟合的措施。

## 总结

正则化是控制模型复杂度、提升泛化能力的必要手段。本章核心要点：

1. 过拟合是模型复杂度超过数据承载能力的表现，正则化在偏差和方差间寻求平衡
2. L2 正则化（权重衰减）是深度学习的默认正则化方式
3. L1 正则化产生稀疏解，适合特征选择
4. Dropout 通过随机丢弃神经元防止共适应，是最有效的深度正则化方法
5. 数据增强是最强力的正则化方法之一，应该优先考虑
6. 早停法简单有效，是所有训练流程的标准配置
7. 标签平滑防止模型过度自信，在多分类任务中表现优秀

正则化控制模型复杂度后，如果网络较深，还需要批归一化来稳定训练过程。

---

**上一篇**: [34. 损失函数详解](34-loss-functions.md)
**下一篇**: [36. 批归一化](36-batch-normalization.md)
