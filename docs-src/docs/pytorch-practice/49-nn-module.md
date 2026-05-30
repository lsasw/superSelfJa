---
title: 神经网络模块 torch.nn
icon: cpu
order: 49
---

# 神经网络模块 torch.nn

`torch.nn` 是 PyTorch 中构建神经网络的核心模块，提供了丰富的层（Layer）、损失函数（Loss Function）和实用工具。本文将系统学习如何使用 `torch.nn` 构建各种神经网络。

## torch.nn 模块概览

PyTorch 的 `torch.nn` 模块按照功能可分为以下几大类：

| 类别 | 主要模块 | 说明 |
|------|---------|------|
| 容器类 | `nn.Module`, `nn.Sequential` | 组织和组合网络层 |
| 线性层 | `nn.Linear`, `nn.Bilinear` | 全连接层 |
| 卷积层 | `nn.Conv1d/2d/3d` | 卷积神经网络核心 |
| 池化层 | `nn.MaxPool2d`, `nn.AvgPool2d` | 空间降采样 |
| 循环层 | `nn.RNN`, `nn.LSTM`, `nn.GRU` | 序列模型 |
| 归一化层 | `nn.BatchNorm2d`, `nn.LayerNorm` | 加速训练收敛 |
| 激活函数 | `nn.ReLU`, `nn.Sigmoid`, `nn.GELU` | 引入非线性 |
| 丢弃层 | `nn.Dropout`, `nn.Dropout2d` | 防止过拟合 |
| Transformer | `nn.Transformer`, `nn.MultiheadAttention` | 注意力机制 |
| 损失函数 | `nn.CrossEntropyLoss`, `nn.MSELoss` | 计算损失 |

## nn.Module：所有网络层的基类

`nn.Module` 是 PyTorch 神经网络的基类，所有自定义的网络层都必须继承它：

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    """简单的两层全连接网络"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 在 __init__ 中定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """自定义权重初始化"""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # 在 forward 中定义前向传播逻辑
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建网络实例
model = SimpleNetwork(input_size=784, hidden_size=128, output_size=10)
print(model)
```

### Module 的核心方法

```python
model = SimpleNetwork(784, 128, 10)

# 训练/评估模式切换
model.train()      # 启用 Dropout 和 BatchNorm 的训练行为
model.eval()       # 固定 Dropout 和 BatchNorm

# 参数管理
print(f"Parameters: {list(model.parameters())}")
print(f"Named parameters: {list(model.named_parameters())}")

# 状态字典
state_dict = model.state_dict()
print(f"State dict keys: {state_dict.keys()}")

# 加载状态字典
# model.load_state_dict(state_dict)

# 设备迁移
model.cuda()       # 迁移到 GPU
model.cpu()        # 迁移回 CPU
```

### Module 的自动参数注册

```python
class DemoModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 方式 1：作为属性赋值（自动注册为参数）
        self.linear = nn.Linear(10, 5)

        # 方式 2：使用 ParameterList
        self.params_list = nn.ParameterList([
            nn.Parameter(torch.randn(5, 3)),
            nn.Parameter(torch.randn(3, 2)),
        ])

        # 方式 3：使用 ParameterDict
        self.params_dict = nn.ParameterDict({
            'weight_a': nn.Parameter(torch.randn(4, 4)),
            'weight_b': nn.Parameter(torch.randn(4, 2)),
        })

        # 错误方式：普通列表不会注册
        # self.my_params = [nn.Parameter(torch.randn(3))]  # 不会被注册！

model = DemoModule()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

💡 **提示**：只有 `nn.Parameter` 类型或 `nn.Module` 子类的实例，赋值给 `self.xxx` 时才会被自动注册。使用 Python 原生 list 或 dict 存储参数时，必须用 `nn.ParameterList` 或 `nn.ParameterDict` 包装。

## nn.Sequential：快速构建顺序网络

```python
# 方式 1：直接传入层
model1 = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 方式 2：使用 OrderedDict 命名每一层
from collections import OrderedDict
model2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(128, 10))
]))

# 方式 3：使用 add_module
model3 = nn.Sequential()
model3.add_module('fc1', nn.Linear(784, 256))
model3.add_module('relu1', nn.ReLU())
model3.add_module('fc2', nn.Linear(256, 128))
model3.add_module('relu2', nn.ReLU())
model3.add_module('fc3', nn.Linear(128, 10))

# 验证输出
x = torch.randn(4, 784)
out1 = model1(x)
out2 = model2(x)
out3 = model3(x)

print(f"Outputs equal: {torch.allclose(out1, out2) and torch.allclose(out2, out3)}")
```

## 核心网络层详解

### 线性层（Linear / Fully Connected）

```python
# 基本用法
linear = nn.Linear(in_features=10, out_features=5)
print(f"Weight shape: {linear.weight.shape}")    # [5, 10]
print(f"Bias shape: {linear.bias.shape}")        # [5]

x = torch.randn(32, 10)  # batch_size=32
y = linear(x)
print(f"Output shape: {y.shape}")  # [32, 5]

# 偏置禁用
linear_no_bias = nn.Linear(10, 5, bias=False)
print(f"No bias: {linear_no_bias.bias is None}")
```

### 卷积层（Convolution）

```python
# 1D 卷积（序列数据）
conv1d = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x_1d = torch.randn(32, 3, 64)  # batch, channels, length
out_1d = conv1d(x_1d)
print(f"Conv1D output: {out_1d.shape}")

# 2D 卷积（图像数据）
conv2d = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1
)
x_2d = torch.randn(32, 3, 224, 224)  # batch, channels, height, width
out_2d = conv2d(x_2d)
print(f"Conv2D output: {out_2d.shape}")

# 深度可分离卷积（groups=in_channels）
depthwise = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
print(f"Depthwise params: {depthwise.weight.numel()}")
```

### 卷积参数详解

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| in_channels | 输入通道数 | - | 必须指定 |
| out_channels | 输出通道数 | - | 决定卷积核数量 |
| kernel_size | 卷积核大小 | - | 感受野大小 |
| stride | 步长 | 1 | 输出尺寸 |
| padding | 填充 | 0 | 保留边界信息 |
| dilation | 空洞率 | 1 | 扩大感受野 |
| groups | 分组数 | 1 | 减少参数量 |
| bias | 是否使用偏置 | True | 参数量 |

### 池化层（Pooling）

```python
# 最大池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(32, 64, 28, 28)
out = max_pool(x)
print(f"MaxPool2D: {out.shape}")  # [32, 64, 14, 14]

# 平均池化
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
out = avg_pool(x)
print(f"AvgPool2D: {out.shape}")

# 自适应池化（自动计算池化大小）
adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
out = adaptive_pool(x)
print(f"AdaptiveAvgPool2D: {out.shape}")

# 全局平均池化
global_avg_pool = nn.AdaptiveAvgPool2d(1)
out = global_avg_pool(x)
print(f"GlobalAvgPool: {out.shape}")  # [32, 64, 1, 1]
```

### 激活函数

```python
x = torch.linspace(-3, 3, 10)

# 常见激活函数对比
activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(negative_slope=0.1),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'GELU': nn.GELU(),
    'SiLU': nn.SiLU(),     # Swish 的 PyTorch 实现
    'Softmax': nn.Softmax(dim=-1),
    'Softplus': nn.Softplus(),
}

for name, act in activations.items():
    if name == 'Softmax':
        y = act(x.unsqueeze(0)).squeeze()
    else:
        y = act(x)
    print(f"{name}: {y.tolist()}")
```

### 激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 | 典型场景 |
|----------|------|------|------|---------|
| ReLU | max(0, x) | 计算快、无梯度消失 | 死亡 ReLU 问题 | 隐藏层默认选择 |
| LeakyReLU | max(αx, x) | 缓解死亡 ReLU | 额外超参数 | ReLU 的改进版 |
| GELU | xΦ(x) | Transformer 默认 | 计算稍慢 | BERT/GPT 等 Transformer |
| SiLU/Swish | x·σ(x) | 平滑、性能好 | 计算较慢 | 计算机视觉 |
| Sigmoid | 1/(1+e^{-x}) | 输出 [0,1] | 梯度消失 | 二分类输出层 |
| Tanh | (e^x-e^{-x})/(e^x+e^{-x}) | 输出 [-1,1] | 梯度消失 | RNN 内部 |

## 归一化层

### Batch Normalization

```python
# BatchNorm2d 通常放在卷积层之后、激活函数之前
bn = nn.BatchNorm2d(num_features=64)

# 训练模式：使用 batch 统计量
bn.train()
x = torch.randn(32, 64, 28, 28)
y = bn(x)
print(f"Running mean shape: {bn.running_mean.shape}")
print(f"Running var shape: {bn.running_var.shape}")

# 评估模式：使用滑动平均统计量
bn.eval()
y_eval = bn(x)

# 训练时的典型用法
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

### Layer Normalization

```python
# LayerNorm 常用于 NLP 和 Transformer
ln = nn.LayerNorm(normalized_shape=128)

# 输入最后维度必须匹配 normalized_shape
x = torch.randn(32, 64, 128)  # batch, seq, features
y = ln(x)
print(f"LayerNorm mean (per sample): {y.mean(dim=-1).mean():.4f}")  # 接近 0
print(f"LayerNorm std (per sample): {y.std(dim=-1).mean():.4f}")   # 接近 1
```

### 归一化层对比

| 归一化方法 | 归一化维度 | 适用场景 | 对 batch size 敏感 |
|-----------|-----------|---------|-------------------|
| BatchNorm | 跨 batch 维度 | CNN | 是 |
| LayerNorm | 特征维度 | Transformer/RNN | 否 |
| InstanceNorm | 单样本通道维度 | 风格迁移 | 否 |
| GroupNorm | 分组通道维度 | 小 batch CNN | 否 |

## Dropout 与正则化

```python
# Dropout
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)

# 训练模式：随机置零
dropout.train()
y_train = dropout(x)
print(f"Dropout (train): {y_train}")

# 评估模式：恒等映射
dropout.eval()
y_eval = dropout(x)
print(f"Dropout (eval): {y_eval}")  # 与输入相同

# Dropout2d：对整个通道随机置零
dropout2d = nn.Dropout2d(p=0.3)
x_2d = torch.randn(32, 64, 28, 28)
y_2d = dropout2d(x_2d)
print(f"Dropout2D: {y_2d.shape}")

# Alpha Dropout：配合 SELU 激活函数
alpha_dropout = nn.AlphaDropout(p=0.5)
```

## 完整网络示例：LeNet-5

```python
class LeNet5(nn.Module):
    """经典 LeNet-5 网络，用于手写数字识别"""

    def __init__(self):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # C1: 卷积层，1 -> 6 通道，5x5 卷积
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Sigmoid(),
            # S2: 最大池化，2x2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: 卷积层，6 -> 16 通道，5x5 卷积
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            # S4: 最大池化，2x2
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# 验证
model = LeNet5()
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

x = torch.randn(8, 1, 32, 32)  # MNIST 风格输入
output = model(x)
print(f"Output shape: {output.shape}")  # [8, 10]
```

## 损失函数

```python
# 交叉熵损失（分类任务）
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(4, 10)     # batch=4, classes=10
targets = torch.tensor([3, 7, 1, 5])
loss = ce_loss(logits, targets)
print(f"CrossEntropy Loss: {loss.item():.4f}")

# 均方误差损失（回归任务）
mse_loss = nn.MSELoss()
pred = torch.randn(4, 5)
target = torch.randn(4, 5)
loss = mse_loss(pred, target)
print(f"MSE Loss: {loss.item():.4f}")

# 二元交叉熵（二分类）
bce_loss = nn.BCEWithLogitsLoss()  # 推荐：内部包含 Sigmoid
logits = torch.randn(4, 1)
targets = torch.tensor([[1], [0], [1], [0]], dtype=torch.float)
loss = bce_loss(logits, targets)
print(f"BCE Loss: {loss.item():.4f}")

# 自定义损失函数
class FocalLoss(nn.Module):
    """Focal Loss，用于类别不平衡问题"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * ce_loss
        return loss.mean()
```

## 总结

本文系统学习了 `torch.nn` 模块的核心内容：

- **nn.Module**：所有网络层的基类，通过继承实现自定义网络
- **nn.Sequential**：快速构建顺序网络
- **核心网络层**：Linear、Conv、Pooling、Activation 等常用层
- **归一化**：BatchNorm、LayerNorm 等加速训练的技术
- **Dropout**：防止过拟合的正则化方法
- **损失函数**：CrossEntropyLoss、MSELoss 等常用损失
- **实战示例**：完整的 LeNet-5 实现

掌握这些核心组件后，你已经能够构建大多数常见的神经网络架构。在下一篇文章中，我们将学习 Dataset 和 DataLoader，了解如何高效地加载和处理训练数据。

[上一篇：自动微分 Autograd 机制](./48-autograd.md) | [下一篇：数据集与数据加载 →](./50-dataloader-dataset.md)
