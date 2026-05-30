---
title: 神经网络基础
icon: brain
order: 31
---

# 神经网络基础

神经网络是深度学习的核心基石。本章将从生物神经元的启发入手，逐步讲解人工神经网络的基本结构、数学原理以及实现方式，为后续的深度学习知识体系打下坚实基础。

## 从生物神经元到人工神经元

生物大脑中的神经元是信息处理的基本单元。一个生物神经元由细胞体、树突（接收信号）和轴突（发送信号）组成。神经元之间通过突触连接，信号以电化学脉冲的形式传递。

人工神经元（Artificial Neuron）是对生物神经元的高度简化数学模型。1943 年，Warren McCulloch 和 Walter Pitts 提出了第一个数学模型，称为 M-P 神经元模型。

一个典型的人工神经元接收多个输入 $x_1, x_2, ..., x_n$，每个输入对应一个权重 $w_1, w_2, ..., w_n$。神经元首先计算加权求和，然后加上偏置项 $b$，最后通过激活函数 $f$ 产生输出：

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$a = f(z)$$

其中 $a$ 是神经元的最终输出，$f$ 是激活函数（Activation Function）。

💡 **提示**：偏置项 $b$ 的作用类似于线性方程 $y = wx + b$ 中的截距，它允许模型在输入为零时仍然产生非零输出，增加了模型的表达能力。

## 神经网络的基本结构

人工神经网络由大量人工神经元按层次组织而成。最基本的网络结构是多层前馈神经网络（Multi-Layer Feedforward Neural Network），也称为多层感知机（MLP, Multi-Layer Perceptron）。

一个典型的 MLP 包含三种类型的层：

| 层类型 | 英文名称 | 作用 |
|--------|----------|------|
| 输入层 | Input Layer | 接收原始数据，不进行计算 |
| 隐藏层 | Hidden Layer | 进行特征提取和非线性变换 |
| 输出层 | Output Layer | 产生最终的预测结果 |

网络的工作流程如下：

1. 数据从输入层进入网络
2. 每一层对输入进行线性变换（加权求和 + 偏置）
3. 通过激活函数引入非线性
4. 结果传递到下一层
5. 最终在输出层得到预测结果

### 前向传播

前向传播（Forward Propagation）是指数据从输入层经过各隐藏层最终到达输出层的过程。对于一个具有 $L$ 层的网络，第 $l$ 层的计算可以表示为：

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

其中：
- $\mathbf{W}^{[l]}$ 是第 $l$ 层的权重矩阵
- $\mathbf{b}^{[l]}$ 是第 $l$ 层的偏置向量
- $\mathbf{a}^{[l-1]}$ 是前一层的激活输出（输入层时 $\mathbf{a}^{[0]} = \mathbf{x}$）
- $f^{[l]}$ 是第 $l$ 层的激活函数
- $\mathbf{a}^{[L]}$ 是网络的最终输出

## 为什么需要非线性激活函数

如果神经网络中只使用线性变换，那么无论网络有多少层，整个网络仍然等价于一个单一的线性变换。这是因为多个线性变换的复合仍然是线性变换：

$$f(f(x)) = W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$$

这本质上还是 $\mathbf{W}'\mathbf{x} + \mathbf{b}'$ 的形式。

引入非线性激活函数后，神经网络才能够逼近任意复杂的函数。Universal Approximation Theorem（通用近似定理）证明：具有单个隐藏层和足够多神经元的前馈神经网络，可以以任意精度逼近任何定义在紧致集上的连续函数。

## 常用激活函数概览

神经网络中常用的激活函数包括：

| 激活函数 | 公式 | 输出范围 | 特点 |
|----------|------|----------|------|
| Sigmoid | $\frac{1}{1 + e^{-x}}$ | (0, 1) | 平滑可导，存在梯度消失问题 |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | 零中心化，仍存在梯度消失 |
| ReLU | $\max(0, x)$ | [0, +∞) | 计算简单，缓解梯度消失 |
| Leaky ReLU | $\max(\alpha x, x)$ | (-∞, +∞) | 解决 ReLU 死亡问题 |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | (0, 1) | 多分类输出概率分布 |

激活函数的详细对比将在后续章节深入讲解。

## 神经网络的信息流向

神经网络的学习过程可以分为两个阶段：

### 前向传播阶段

在前向传播中，输入数据逐层经过加权求和和激活函数运算，最终得到网络的预测输出。这个过程是确定性的计算过程。

### 反向传播阶段

反向传播（Backpropagation）是训练神经网络的核心算法。它通过计算损失函数对各个参数的梯度，利用链式法则从输出层向输入层逐层传播误差，从而更新网络参数。

反向传播的详细原理将在后续章节专门讲解。

## 损失函数与优化目标

神经网络的训练需要定义一个损失函数（Loss Function）来衡量预测值与真实值之间的差距。常见的损失函数包括：

- **均方误差（MSE）**：适用于回归任务
- **交叉熵损失（Cross-Entropy）**：适用于分类任务
- **二元交叉熵（BCE）**：适用于二分类任务

损失函数的选择直接影响网络的训练效果和收敛速度。

## 使用 Python 和 NumPy 实现简单神经网络

下面我们通过纯 NumPy 实现一个具有单隐藏层的多层感知机，用于二分类任务。

```python
import numpy as np

class SimpleNeuralNetwork:
    """使用 NumPy 实现的多层感知机（MLP）"""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # 使用 He 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        """Sigmoid 函数的导数"""
        return a * (1 - a)

    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        """计算二元交叉熵损失"""
        m = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -(1.0 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, X, y_true):
        """反向传播"""
        m = X.shape[0]

        # 输出层梯度
        dz2 = self.a2 - y_true
        dW2 = (1.0 / m) * np.dot(self.a1.T, dz2)
        db2 = (1.0 / m) * np.sum(dz2, axis=0, keepdims=True)

        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = (1.0 / m) * np.dot(X.T, dz1)
        db1 = (1.0 / m) * np.sum(dz1, axis=0, keepdims=True)

        # 更新参数
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        """训练网络"""
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)

            # 计算损失
            loss = self.compute_loss(y_pred, y)

            # 反向传播
            self.backward(X, y)

            # 打印训练进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """预测"""
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)


# 使用示例：XOR 问题
if __name__ == "__main__":
    np.random.seed(42)

    # XOR 训练数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # 创建并训练网络
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    nn.train(X, y, epochs=1000)

    # 测试
    print("\n预测结果:")
    for i in range(4):
        pred = nn.predict(X[i])
        print(f"输入: {X[i]}, 预测: {pred[0][0]}, 真实: {y[i][0]}")
```

## 使用 PyTorch 实现相同网络

PyTorch 提供了高级的神经网络 API，大大简化了实现过程。

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """使用 PyTorch 实现的多层感知机"""

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# 使用示例
if __name__ == "__main__":
    torch.manual_seed(42)

    # 训练数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # 创建网络、损失函数和优化器
    model = MLP(input_size=2, hidden_size=4, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)

    # 训练
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 测试
    print("\n预测结果:")
    with torch.no_grad():
        y_pred = model(X)
        for i in range(4):
            pred = 1 if y_pred[i].item() >= 0.5 else 0
            print(f"输入: {X[i].tolist()}, 预测: {pred}, 真实: {int(y[i].item())}")
```

## 神经网络的关键超参数

在构建神经网络时，需要手动设置的超参数对模型性能有重要影响：

| 超参数 | 说明 | 常见取值 |
|--------|------|----------|
| 网络层数 | 隐藏层的数量 | 1-100+，取决于任务复杂度 |
| 每层神经元数 | 每层的宽度 | 32-4096，通常是 2 的幂 |
| 学习率 | 参数更新的步长 | 0.001-0.1 |
| 激活函数 | 非线性函数选择 | ReLU 系列最常用 |
| 批大小（Batch Size） | 每次更新的样本数 | 32-256 |
| 训练轮数（Epochs） | 遍历整个数据集的次数 | 取决于收敛情况 |

💡 **提示**：超参数的选择没有万能公式。建议从经验值开始，然后通过实验逐步调优。可以使用网格搜索、随机搜索或贝叶斯优化等自动调参方法。

## 神经网络的能力与局限

### 优势

- **强大的表达能力**：可以学习任意复杂的非线性映射关系
- **端到端学习**：无需手动设计特征，直接从原始数据中学习
- **并行计算友好**：矩阵运算非常适合 GPU 加速
- **可扩展性强**：增加数据和模型规模通常能持续提升性能

### 局限

- **需要大量数据**：参数量大，容易在小数据集上过拟合
- **黑盒性质**：决策过程难以解释
- **训练成本高**：需要大量计算资源和时间
- **超参数敏感**：参数设置对结果影响较大

## 总结

神经网络通过模拟生物神经元的工作方式，构建了强大的数学模型来解决复杂的学习任务。本章核心要点：

1. 人工神经元是对生物神经元的数学抽象，核心操作是加权求和加激活函数
2. 多层感知机（MLP）由输入层、隐藏层和输出层组成
3. 非线性激活函数是神经网络强大表达能力的关键
4. 前向传播计算预测，反向传播更新参数
5. 可以使用 NumPy 从零实现，也可以使用 PyTorch 等框架快速搭建

掌握神经网络的基本原理后，下一步需要深入理解反向传播算法，它是神经网络训练的核心机制。

---

**下一篇**: [32. 反向传播算法](32-backpropagation.md)
