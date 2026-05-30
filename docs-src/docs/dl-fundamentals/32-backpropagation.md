---
title: 反向传播算法
icon: arrows-rotate
order: 32
---

# 反向传播算法

反向传播（Backpropagation，简称 Backprop）是训练神经网络的核心算法。它高效地计算了损失函数对每个参数的梯度，使得梯度下降优化成为可能。理解反向传播是深入掌握深度学习的关键。

## 反向传播的历史背景

反向传播算法的思想最早可以追溯到 1960 年代。1960 年，Henry Kelley 提出了连续形式的反向传播。1970 年，Seppo Linnainmaa 提出了链式法则的自动微分形式。1986 年，David Rumelhart、Geoffrey Hinton 和 Ronald Williams 发表的论文将反向传播应用于神经网络训练，使其广为人知。

反向传播的本质是**链式法则（Chain Rule）**在计算图上的系统应用。它不是独立的算法，而是微积分基本原理在大规模参数计算中的高效实现方式。

## 链式法则回顾

链式法则是微积分中复合函数求导的基本法则：

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

对于多层复合函数：

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

在神经网络中，损失函数 $L$ 是通过多层嵌套计算得到的，链式法则允许我们将复杂的导数分解为一系列简单导数的乘积。

## 计算图视角

将神经网络的前向计算过程表示为**计算图（Computational Graph）**是理解反向传播的最佳方式。计算图是一个有向无环图（DAG），其中：

- **节点（Node）**：表示数学运算
- **边（Edge）**：表示数据（张量）的流动

以一个简单的两层网络为例：

```
输入 X → 线性变换(z1) → 激活(a1) → 线性变换(z2) → 激活(a2) → 损失 L
```

每个计算节点都需要实现两个函数：
1. **前向函数**：根据输入计算输出
2. **反向函数**：根据上游梯度计算对输入的梯度

### 基本运算的梯度计算

| 运算 | 前向公式 | 梯度公式 |
|------|----------|----------|
| 加法 $z = x + y$ | $z = x + y$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}$, $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z}$ |
| 乘法 $z = x \times y$ | $z = xy$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot y$, $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot x$ |
| ReLU $z = \max(0, x)$ | $z = \max(0, x)$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \mathbb{I}(x > 0)$ |
| Sigmoid $z = \sigma(x)$ | $z = \frac{1}{1+e^{-x}}$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot z(1-z)$ |

💡 **提示**：加法节点的反向传播只是将上游梯度原样传递（梯度分发器），而乘法节点则需要交换输入值再相乘（梯度交换器）。理解这些基本运算的梯度行为是构建复杂网络的基础。

## 反向传播的完整推导

考虑一个具有 $L$ 层的全连接网络，损失函数为 $\mathcal{L}$。我们需要计算 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ 和 $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$。

### 前向传播回顾

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

### 输出层梯度

从输出层 $L$ 开始，假设使用均方误差损失函数：

$$\mathcal{L} = \frac{1}{2m} \sum_{i=1}^{m} \|\mathbf{a}^{[L](i)} - \mathbf{y}^{(i)}\|^2$$

则输出层的误差信号为：

$$\mathbf{d}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})$$

其中 $\odot$ 表示逐元素乘法（Hadamard 积）。

### 隐藏层梯度递推

对于任意隐藏层 $l$，误差信号从后一层递推而来：

$$\mathbf{d}^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = ((\mathbf{W}^{[l+1]})^T \mathbf{d}^{[l+1]}) \odot f'^{[l]}(\mathbf{z}^{[l]})$$

### 参数梯度

有了误差信号后，参数梯度的计算变得直接：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \mathbf{d}^{[l]} (\mathbf{a}^{[l-1]})^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{d}^{[l](i)}$$

### 维度分析

维度分析是验证反向传播正确性的有效方法：

| 变量 | 维度 |
|------|------|
| $\mathbf{W}^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ |
| $\mathbf{b}^{[l]}$ | $(n^{[l]}, 1)$ |
| $\mathbf{z}^{[l]}, \mathbf{a}^{[l]}$ | $(n^{[l]}, m)$ |
| $\mathbf{d}^{[l]}$ | $(n^{[l]}, m)$ |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ | $(n^{[l]}, n^{[l-1]})$ |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$ | $(n^{[l]}, 1)$ |

## 梯度消失与梯度爆炸

反向传播在深层网络中面临两个严重问题：

### 梯度消失（Vanishing Gradient）

当网络层数较深时，梯度从输出层向输入层传播的过程中，由于连续乘以小于 1 的数值（如 Sigmoid 函数的导数最大值为 0.25），梯度会指数级衰减，导致靠近输入层的参数几乎得不到更新。

| 激活函数 | 导数最大值 | 梯度消失风险 |
|----------|------------|--------------|
| Sigmoid | 0.25 | 严重 |
| Tanh | 1.0 | 中等 |
| ReLU | 1.0 | 较低 |
| Leaky ReLU | 1.0 | 很低 |

### 梯度爆炸（Exploding Gradient）

相反，当权重初始化过大或使用某些激活函数时，梯度可能指数级增长，导致参数更新过大，网络训练不稳定。

💡 **提示**：解决梯度消失的方法包括：使用 ReLU 系列激活函数、合理的权重初始化（He/Xavier）、残差连接（Residual Connection）和批归一化（Batch Normalization）。解决梯度爆炸的方法包括：梯度裁剪（Gradient Clipping）和权重正则化。

## 反向传播的数值验证

在实现反向传播时，使用数值梯度检验可以验证解析梯度的正确性：

```python
import numpy as np

def numerical_gradient(f, x, epsilon=1e-7):
    """
    使用中心差分法计算数值梯度
    f: 函数，输入 x 返回标量损失值
    x: 参数数组
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + epsilon
        f_plus = f(x)

        x[idx] = old_value - epsilon
        f_minus = f(x)

        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        x[idx] = old_value
        it.iternext()

    return grad

def gradient_check(analytical_grad, numerical_grad, threshold=1e-7):
    """
    检查解析梯度与数值梯度的差异
    """
    # 计算相对误差
    numerator = np.linalg.norm(analytical_grad - numerical_grad)
    denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = numerator / max(denominator, 1e-18)

    if relative_error < threshold:
        print(f"梯度检验通过! 相对误差: {relative_error:.2e}")
        return True
    else:
        print(f"梯度检验失败! 相对误差: {relative_error:.2e}")
        return False
```

## 完整实现：从头构建反向传播

下面使用纯 NumPy 实现一个具有两个隐藏层的前馈网络，包含完整的前向传播和反向传播。

```python
import numpy as np

class DeepNeuralNetwork:
    """
    使用 NumPy 实现的多层前馈神经网络（含完整反向传播）
    支持任意数量的隐藏层
    """

    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        layer_sizes: 列表，如 [784, 128, 64, 10] 表示
                     输入784维，隐藏层128和64维，输出10维
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.lr = learning_rate
        self.params = {}
        self.cache = {}

        # He 初始化
        for l in range(1, self.num_layers + 1):
            self.params[f'W{l}'] = np.random.randn(
                layer_sizes[l], layer_sizes[l-1]
            ) * np.sqrt(2.0 / layer_sizes[l-1])
            self.params[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def relu(self, z):
        """ReLU 激活函数"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU 导数"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """数值稳定的 Softmax"""
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        """
        前向传播
        X: 输入数据，形状 (n_features, m_samples)
        """
        self.cache['A0'] = X

        for l in range(1, self.num_layers + 1):
            self.cache[f'Z{l}'] = (
                self.params[f'W{l}'] @ self.cache[f'A{l-1}']
                + self.params[f'b{l}']
            )

            if l == self.num_layers:
                # 输出层使用 Softmax
                self.cache[f'A{l}'] = self.softmax(self.cache[f'Z{l}'])
            else:
                # 隐藏层使用 ReLU
                self.cache[f'A{l}'] = self.relu(self.cache[f'Z{l}'])

        return self.cache[f'A{self.num_layers}']

    def backward(self, Y):
        """
        反向传播
        Y: 真实标签，形状 (n_classes, m_samples)，one-hot 编码
        """
        m = Y.shape[1]
        grads = {}

        # 输出层梯度（Softmax + 交叉熵的合并梯度）
        L = self.num_layers
        grads[f'dZ{L}'] = self.cache[f'A{L}'] - Y
        grads[f'dW{L}'] = (1.0 / m) * grads[f'dZ{L}'] @ self.cache[f'A{L-1}'].T
        grads[f'db{L}'] = (1.0 / m) * np.sum(grads[f'dZ{L}'], axis=1, keepdims=True)

        # 隐藏层梯度（从后向前）
        for l in range(L - 1, 0, -1):
            grads[f'dA{l}'] = self.params[f'W{l+1}'].T @ grads[f'dZ{l+1}']
            grads[f'dZ{l}'] = grads[f'dA{l}'] * self.relu_derivative(self.cache[f'Z{l}'])
            grads[f'dW{l}'] = (1.0 / m) * grads[f'dZ{l}'] @ self.cache[f'A{l-1}'].T
            grads[f'db{l}'] = (1.0 / m) * np.sum(grads[f'dZ{l}'], axis=1, keepdims=True)

        # 更新参数
        for l in range(1, L + 1):
            self.params[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= self.lr * grads[f'db{l}']

    def compute_loss(self, Y_hat, Y):
        """交叉熵损失"""
        m = Y.shape[1]
        eps = 1e-15
        Y_hat = np.clip(Y_hat, eps, 1 - eps)
        loss = -(1.0 / m) * np.sum(Y * np.log(Y_hat))
        return loss

    def train(self, X, Y, epochs=1000, print_interval=100):
        """训练循环"""
        for epoch in range(epochs):
            # 前向传播
            Y_hat = self.forward(X)

            # 计算损失
            loss = self.compute_loss(Y_hat, Y)

            # 反向传播
            self.backward(Y)

            # 打印进度
            if epoch % print_interval == 0:
                accuracy = self.evaluate(X, Y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

    def predict(self, X):
        """预测类别"""
        probs = self.forward(X)
        return np.argmax(probs, axis=0)

    def evaluate(self, X, Y):
        """计算准确率"""
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=0)
        return np.mean(predictions == true_labels)


# 使用示例：MNIST 手写数字分类（简化版）
if __name__ == "__main__":
    np.random.seed(42)

    # 生成模拟数据（10个类别，每个样本784维）
    n_samples = 1000
    n_features = 784
    n_classes = 10

    X_train = np.random.randn(n_features, n_samples)
    Y_train = np.zeros((n_classes, n_samples))
    for i in range(n_samples):
        Y_train[np.random.randint(0, n_classes), i] = 1

    # 创建网络
    nn = DeepNeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=0.01
    )

    # 训练
    nn.train(X_train, Y_train, epochs=500, print_interval=100)
```

## 自动微分与现代框架

现代深度学习框架（PyTorch、TensorFlow、JAX）都内置了自动微分（Automatic Differentiation）引擎，可以自动计算梯度，无需手动推导和实现反向传播。

### PyTorch 的 autograd

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建数据和模型
x = torch.randn(100, 784)
y = torch.randint(0, 10, (100,))

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()       # 清零梯度
    output = model(x)           # 前向传播
    loss = criterion(output, y) # 计算损失
    loss.backward()             # 反向传播（自动计算所有梯度）
    optimizer.step()            # 更新参数

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

`loss.backward()` 一行代码就完成了所有参数的梯度计算。框架自动构建了前向计算图，并在反向传播时按拓扑逆序应用链式法则。

## 反向传播的内存优化

反向传播需要保存前向传播的中间结果（激活值），这导致内存消耗与网络深度成正比。常见的内存优化策略包括：

| 优化策略 | 原理 | 内存节省 | 计算开销 |
|----------|------|----------|----------|
| 梯度累积 | 累积多个小 batch 的梯度再更新 | 无直接节省 | 无 |
| 激活检查点 | 只保存部分激活值，其余在反向时重新计算 | 约 $\sqrt{n}$ | 增加约 33% |
| 混合精度训练 | 使用 FP16 存储激活值 | 约 50% | 轻微 |
| 离线卸载 | 将激活值卸载到 CPU 内存 | 取决于 GPU 显存 | 通信开销 |

## 总结

反向传播是深度学习得以工作的核心机制。本章核心要点：

1. 反向传播的本质是链式法则在计算图上的系统应用
2. 基本运算（加法、乘法、激活函数）各有特定的反向规则
3. 梯度计算从输出层向输入层逐层递推，复用中间结果避免重复计算
4. 梯度消失和梯度爆炸是深层网络训练的主要挑战
5. 数值梯度检验可以验证反向传播实现的正确性
6. 现代框架通过自动微分消除了手动推导梯度的需求

理解反向传播后，我们需要深入了解激活函数的选择，因为不同的激活函数对梯度的传播和网络的学习能力有直接影响。

---

**上一篇**: [31. 神经网络基础](31-neural-network-basics.md)
**下一篇**: [33. 激活函数详解](33-activation-functions.md)
