---
title: 机器学习中的微积分
icon: calculator
order: 5
---

# 机器学习中的微积分

## 引言

在前面的学习中，我们掌握了 [线性代数基础](./03-linear-algebra.md) 和 [概率与统计基础](./04-probability-statistics.md)。线性代数让我们能够高效地表示和运算数据，概率与统计让我们能够处理不确定性。然而，机器学习模型是如何"学习"的？答案在于优化——通过不断调整模型参数来最小化误差。而优化的核心数学工具就是微积分。

梯度下降是机器学习中最基础的优化算法，它完全建立在导数和梯度的概念之上。深度学习中的反向传播算法本质上是链式法则的高效应用。理解这些微积分概念，你将能够：

- 理解模型训练的数学原理
- 分析损失函数的形状与性质
- 调试训练过程中遇到的问题
- 设计和改进优化算法

本篇将系统性地介绍机器学习中最核心的微积分概念。

## 函数与极限

### 函数的基本概念

在机器学习中，函数无处不在：

| 函数 | 输入 | 输出 | 示例 |
|------|------|------|------|
| 模型 f(x) | 特征 x | 预测值 ŷ | 房价预测 |
| 损失函数 L(θ) | 参数 θ | 误差值 | MSE、交叉熵 |
| 激活函数 σ(z) | 加权和 z | 激活值 a | Sigmoid、ReLU |
| 概率函数 P(y\|x) | 特征 x | 概率 | 分类概率 |

### 极限

极限是微积分的基石。当 x 无限趋近于 a 时，f(x) 的趋近值就是极限。

```
lim(x→a) f(x) = L
```

在机器学习中，极限的概念主要体现在：
- 学习率趋近于 0 时的收敛行为
- 样本量趋近于无穷时 MLE 的一致性
- 数值计算中的稳定性（如 log-sum-exp 技巧）

## 导数

### 定义

导数描述了函数在某一点的变化率，即函数值对输入变化的敏感程度。

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

**几何意义**：导数等于函数曲线在该点的切线斜率。

```python
import numpy as np

# 数值导数近似
def numerical_derivative(f, x, h=1e-7):
    """用极限近似计算导数"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 示例1：f(x) = x² 在 x=3 处的导数
def f1(x):
    return x ** 2

print(f"f(x) = x²")
print(f"  f'(3) 精确值: {2 * 3}")
print(f"  f'(3) 数值近似: {numerical_derivative(f1, 3):.6f}")

# 示例2：f(x) = sin(x) 在 x=π/4 处的导数
def f2(x):
    return np.sin(x)

print(f"\nf(x) = sin(x)")
print(f"  f'(π/4) 精确值: {np.cos(np.pi/4):.6f}")
print(f"  f'(π/4) 数值近似: {numerical_derivative(f2, np.pi/4):.6f}")
```

### 基本求导法则

| 法则 | 公式 | 说明 |
|------|------|------|
| 常数法则 | d/dx[c] = 0 | 常数的导数为零 |
| 幂法则 | d/dx[x^n] = n·x^(n-1) | 最基础的求导公式 |
| 线性法则 | d/dx[af(x)] = a·f'(x) | 常数可以提出来 |
| 加法法则 | d/dx[f+g] = f' + g' | 导数可以逐项求 |
| 乘法法则 | d/dx[fg] = f'g + fg' | 乘积的导数 |
| 除法法则 | d/dx[f/g] = (f'g - fg')/g² | 商的导数 |

```python
# 导数在机器学习中的应用：分析损失函数
# 线性回归的 MSE 损失函数
def mse_loss(w, X, y):
    """均方误差损失"""
    predictions = X @ w
    errors = predictions - y
    return np.mean(errors ** 2)

def mse_gradient(w, X, y):
    """MSE 损失的梯度"""
    predictions = X @ w
    errors = predictions - y
    return 2 * X.T @ errors / len(y)

# 验证
np.random.seed(42)
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100) * 0.1
w = np.array([1.0, 2.0])

# 数值梯度
num_grad = np.array([
    numerical_derivative(lambda w_i: mse_loss(np.array([w_i, w[1]]), X, y), w[0]),
    numerical_derivative(lambda w_i: mse_loss(np.array([w[0], w_i]), X, y), w[1]),
])

analytical_grad = mse_gradient(w, X, y)

print(f"MSE 损失函数分析:")
print(f"  数值梯度: {num_grad}")
print(f"  解析梯度: {analytical_grad}")
print(f"  梯度是否一致: {np.allclose(num_grad, analytical_grad)}")
```

### 常见函数的导数

| 函数 f(x) | 导数 f'(x) |
|-----------|-----------|
| x^n | n·x^(n-1) |
| e^x | e^x |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |
| σ(x) = 1/(1+e^(-x)) | σ(x)(1-σ(x)) |
| ReLU(x) = max(0,x) | 1 (x>0), 0 (x<0) |

> 💡 **Sigmoid 导数的优美性质**：σ'(x) = σ(x)(1-σ(x))，这个性质使得 Sigmoid 的反向传播计算非常高效——只需要前向传播的输出值即可计算导数。

## 偏导数与梯度

### 偏导数

当函数有多个自变量时，对其中一个变量求导（其他变量视为常数），得到的就是偏导数。

```
∂f/∂x_i = lim(h→0) [f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_n)] / h
```

```python
# 示例：f(x, y) = x² + 2xy + y²
def f(x, y):
    return x**2 + 2*x*y + y**2

# 偏导数
def df_dx(x, y):
    return 2*x + 2*y  # ∂f/∂x

def df_dy(x, y):
    return 2*x + 2*y  # ∂f/∂y

# 数值验证
x0, y0 = 2.0, 3.0
print(f"f(x,y) = x² + 2xy + y²")
print(f"  ∂f/∂x at ({x0},{y0}): 精确={df_dx(x0,y0)}, 数值={numerical_derivative(lambda x: f(x, y0), x0)}")
print(f"  ∂f/∂y at ({x0},{y0}): 精确={df_dy(x0,y0)}, 数值={numerical_derivative(lambda y: f(x0, y), y0)}")
```

### 梯度（Gradient）

梯度是所有偏导数组成的向量，指向函数增长最快的方向。

```
∇f = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]
```

```python
def gradient(f, point, h=1e-7):
    """计算多元函数的数值梯度"""
    n = len(point)
    grad = np.zeros(n)
    for i in range(n):
        def f_along_i(x_i):
            p = point.copy()
            p[i] = x_i
            return f(p)
        grad[i] = numerical_derivative(f_along_i, point[i], h)
    return grad

# 示例：f(x, y, z) = x² + 2y² + 3z²
def f_3d(p):
    return p[0]**2 + 2*p[1]**2 + 3*p[2]**2

point = np.array([1.0, 2.0, 3.0])
grad = gradient(f_3d, point)

print(f"f(x,y,z) = x² + 2y² + 3z²")
print(f"  在点 {point} 处的梯度: {grad}")
print(f"  梯度方向 (函数增长最快的方向): {grad / np.linalg.norm(grad)}")
print(f"  梯度大小 (最大变化率): {np.linalg.norm(grad):.2f}")

# 梯度的重要性质：梯度指向函数增长最快的方向
# 梯度下降法就是沿着梯度的反方向更新参数
```

> 💡 **梯度的物理类比**：想象你在山上（损失函数的曲面），想要最快地下到山谷（最小化损失）。梯度指向的是山上最陡的上坡方向，所以你需要沿着梯度的反方向（下坡方向）走。这就是梯度下降的直觉。

## 链式法则

链式法则是微积分中最重要也最实用的求导法则，也是深度学习中反向传播算法的核心。

### 单变量链式法则

```
如果 y = f(g(x))，则 dy/dx = f'(g(x)) · g'(x)
```

```python
# 链式法则示例
# f(x) = sin(x²)
# 令 g(x) = x², f(u) = sin(u)
# 则 df/dx = cos(x²) · 2x

def chain_example(x):
    # 前向计算
    u = x ** 2          # g(x) = x²
    y = np.sin(u)       # f(u) = sin(u)

    # 反向计算（链式法则）
    dy_du = np.cos(u)   # f'(u) = cos(u)
    du_dx = 2 * x       # g'(x) = 2x
    dy_dx = dy_du * du_dx  # 链式法则

    return y, dy_dx

x = 2.0
y, dy_dx = chain_example(x)
print(f"f(x) = sin(x²)")
print(f"  f({x}) = {y:.6f}")
print(f"  f'({x}) = {dy_dx:.6f}")
print(f"  数值验证: {numerical_derivative(lambda x: np.sin(x**2), x):.6f}")
```

### 多变量链式法则（反向传播的核心）

```python
# 模拟神经网络中的一层
# z = Wx + b, a = σ(z), loss = (a - y)²

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(a):
    """Sigmoid 的导数，用输出 a 表示"""
    return a * (1 - a)

# 前向传播
np.random.seed(42)
x = np.array([1.0, 2.0])       # 输入
W = np.array([[0.5, -0.3],     # 权重
              [0.2, 0.8]])
b = np.array([0.1, -0.2])      # 偏置
y_true = np.array([1.0, 0.0])  # 目标

# 前向计算
z = W @ x + b                  # 线性变换
a = sigmoid(z)                 # 激活
loss = np.mean((a - y_true)**2)  # 损失

print("=== 前向传播 ===")
print(f"z = {z}")
print(f"a = σ(z) = {a}")
print(f"loss = {loss:.6f}")

# 反向传播（链式法则）
# d(loss)/d(a) = 2(a - y) / n
# d(a)/d(z) = a * (1 - a)
# d(z)/d(W) = x^T, d(z)/d(b) = 1

d_loss_da = 2 * (a - y_true) / len(a)     # 损失对激活的导数
d_a_dz = sigmoid_derivative(a)             # 激活对z的导数
d_loss_dz = d_loss_da * d_a_dz             # 链式法则

# 权重和偏置的梯度
d_loss_dW = np.outer(d_loss_dz, x)         # 外积
d_loss_db = d_loss_dz.copy()

print("\n=== 反向传播 ===")
print(f"d(loss)/dW =\n{d_loss_dW}")
print(f"d(loss)/db = {d_loss_db}")
```

> 💡 **反向传播的本质**：反向传播就是链式法则在计算图上的系统应用。从输出层开始，逐层向前计算每个参数的梯度。理解链式法则是理解深度学习的关键。

## Hessian 矩阵

Hessian 矩阵是二阶偏导数组成的矩阵，描述了函数的局部曲率。

```
H(f) = [∂²f/∂x_i∂x_j]
```

```python
def hessian(f, point, h=1e-5):
    """计算数值 Hessian 矩阵"""
    n = len(point)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                # 二阶纯偏导
                p_plus = point.copy()
                p_plus[i] += h
                p_minus = point.copy()
                p_minus[i] -= h
                H[i, j] = (f(p_plus) - 2*f(point) + f(p_minus)) / h**2
            else:
                # 二阶混合偏导
                p_pp = point.copy(); p_pp[i] += h; p_pp[j] += h
                p_pm = point.copy(); p_pm[i] += h; p_pm[j] -= h
                p_mp = point.copy(); p_mp[i] -= h; p_mp[j] += h
                p_mm = point.copy(); p_mm[i] -= h; p_mm[j] -= h
                H[i, j] = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4 * h**2)
    return H

# 示例：f(x, y) = x² + 2y²
def f_2d(p):
    return p[0]**2 + 2*p[1]**2

point = np.array([1.0, 1.0])
H = hessian(f_2d, point)
print(f"f(x,y) = x² + 2y²")
print(f"Hessian 矩阵:\n{H}")

# Hessian 的特征值决定了函数的局部形状
eigenvalues = np.linalg.eigvalsh(H)
print(f"Hessian 特征值: {eigenvalues}")
if all(e > 0 for e in eigenvalues):
    print("所有特征值为正 → 局部极小值点")
elif all(e < 0 for e in eigenvalues):
    print("所有特征值为负 → 局部极大值点")
else:
    print("特征值有正有负 → 鞍点")
```

> 💡 **Hessian 在优化中的意义**：
> - 正定 Hessian：函数在该点是局部凸的，适合用梯度下降
> - 负定 Hessian：函数在该点是局部凹的
> - 不定 Hessian：该点是鞍点，梯度下降可能停滞
> - 条件数大的 Hessian：损失函数呈狭长谷状，梯度下降收敛慢

## 微积分在机器学习中的应用

### 线性回归的梯度推导

线性回归的损失函数（MSE）：

```
L(w, b) = (1/n) Σ (y_i - (w·x_i + b))²
```

对 w 和 b 求偏导：

```
∂L/∂w = (-2/n) Σ x_i(y_i - (w·x_i + b))
∂L/∂b = (-2/n) Σ (y_i - (w·x_i + b))
```

```python
def linear_regression_gradient_descent(X, y, lr=0.01, epochs=1000):
    """用梯度下降法训练线性回归"""
    n_samples, n_features = X.shape

    # 添加偏置列
    X_b = np.c_[np.ones(n_samples), X]

    # 初始化参数
    theta = np.zeros(n_features + 1)

    # 记录损失历史
    loss_history = []

    for epoch in range(epochs):
        # 前向传播
        predictions = X_b @ theta

        # 计算损失
        errors = predictions - y
        loss = np.mean(errors ** 2)
        loss_history.append(loss)

        # 计算梯度
        gradients = 2 * X_b.T @ errors / n_samples

        # 更新参数
        theta -= lr * gradients

    return theta, loss_history

# 测试
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.ravel() + np.random.randn(100) * 0.5

theta, losses = linear_regression_gradient_descent(X, y, lr=0.1, epochs=500)
print(f"截距: {theta[0]:.4f} (真实值: 4)")
print(f"斜率: {theta[1]:.4f} (真实值: 3)")
print(f"最终损失: {losses[-1]:.6f}")
print(f"损失收敛: {losses[-1] < losses[0]}")
```

### 逻辑回归的梯度推导

逻辑回归使用交叉熵损失：

```
L(w) = -(1/n) Σ [y_i·log(σ(w·x_i)) + (1-y_i)·log(1-σ(w·x_i))]
```

梯度：

```
∂L/∂w = (1/n) Σ (σ(w·x_i) - y_i) · x_i
```

```python
def logistic_regression_gradient_descent(X, y, lr=0.1, epochs=1000):
    """用梯度下降法训练逻辑回归"""
    n_samples, n_features = X.shape
    X_b = np.c_[np.ones(n_samples), X]
    theta = np.zeros(n_features + 1)
    loss_history = []

    for epoch in range(epochs):
        # 前向传播
        z = X_b @ theta
        predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        # 交叉熵损失
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        loss_history.append(loss)

        # 梯度
        errors = predictions - y
        gradients = X_b.T @ errors / n_samples

        # 更新
        theta -= lr * gradients

    return theta, loss_history

# 测试：二分类
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)

theta, losses = logistic_regression_gradient_descent(X, y, lr=0.5, epochs=1000)
print(f"逻辑回归参数: {theta}")
print(f"最终损失: {losses[-1]:.6f}")

# 预测
def predict_prob(X, theta):
    X_b = np.c_[np.ones(len(X)), X]
    return 1 / (1 + np.exp(-X_b @ theta))

probs = predict_prob(X[:5], theta)
print(f"前5个样本的预测概率: {probs}")
```

### Softmax 与多分类

Softmax 函数将任意实数向量转化为概率分布：

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

```python
def softmax(z):
    """数值稳定的 Softmax"""
    z_shifted = z - np.max(z)  # 防止溢出
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

# 示例
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Softmax 概率: {probs}")
print(f"概率和: {probs.sum():.6f}")

# Softmax 的导数
def softmax_cross_entropy_gradient(logits, labels):
    """Softmax + 交叉熵的梯度"""
    probs = softmax(logits)
    # 梯度形式非常简洁：probs - labels
    gradient = probs - labels
    return gradient

labels = np.array([1.0, 0.0, 0.0])  # one-hot 编码
grad = softmax_cross_entropy_gradient(logits, labels)
print(f"梯度: {grad}")
```

> 💡 **Softmax + 交叉熵的美妙性质**：Softmax 输出与真实标签的差值就是梯度。这个简洁的形式使得多分类的反向传播非常高效。

## 常见激活函数的导数

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x)**2

def gelu(x):
    """GELU: x · Φ(x)，Φ 是标准正态 CDF"""
    from scipy.stats import norm
    return x * norm.cdf(x)

# 可视化导数
x = np.linspace(-3, 3, 100)
functions = [
    ("Sigmoid", sigmoid, sigmoid_grad),
    ("ReLU", relu, relu_grad),
    ("Tanh", tanh, tanh_grad),
]

for name, fn, gn in functions:
    values = fn(x)
    grads = gn(x)
    print(f"\n{name}:")
    print(f"  x=0: 函数值={fn(0):.4f}, 导数={gn(0):.4f}")
    print(f"  x=1: 函数值={fn(1):.4f}, 导数={gn(1):.4f}")
    print(f"  x=-1: 函数值={fn(-1):.4f}, 导数={gn(-1):.4f}")
```

## 总结

通过本篇的学习，你应该掌握了以下微积分核心知识：

1. **导数**：变化率的度量，分析函数的局部行为
2. **偏导数与梯度**：多元函数的导数，梯度指向函数增长最快的方向
3. **链式法则**：复合函数求导的核心法则，反向传播的数学基础
4. **Hessian 矩阵**：二阶导数，描述函数的局部曲率
5. **梯度下降**：沿梯度反方向更新参数以最小化损失
6. **常见激活函数的导数**：Sigmoid、ReLU、Tanh、GELU

微积分为理解模型优化提供了数学工具。在后续的 [梯度下降](./13-gradient-descent.md) 和 [优化算法](./14-optimization-algorithms.md) 中，我们将深入探讨如何高效地优化模型参数。

> [!NOTE] 下一篇
> [06 - Python 编程基础](./06-python-basics.md) —— 掌握 Python 编程基础，为后续的代码实践做好准备。
