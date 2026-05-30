---
title: 自动微分 Autograd 机制
icon: zap
order: 48
---

# 自动微分 Autograd 机制

自动微分（Automatic Differentiation）是深度学习框架的核心能力。PyTorch 的 autograd 引擎能够自动计算任意可微函数的梯度，是训练神经网络的基础。本文将深入理解 autograd 的工作原理、使用方法和最佳实践。

## 为什么需要自动微分

在神经网络训练中，我们需要计算损失函数对模型参数的梯度，以指导参数更新。手动推导和编写梯度公式不仅繁琐，而且容易出错。自动微分解决了这个问题。

### 数值微分 vs 符号微分 vs 自动微分

| 方法 | 原理 | 精度 | 效率 | 适用场景 |
|------|------|------|------|---------|
| 数值微分 | 有限差分近似 | 低（截断误差） | 极慢（O(n)次计算） | 梯度检查 |
| 符号微分 | 解析求导公式 | 高 | 表达式膨胀 | 简单函数 |
| 自动微分 | 链式法则+计算图 | 机器精度 | 高效 | 深度学习 |

```python
# 数值微分示例（仅作对比）
def numerical_gradient(f, x, eps=1e-5):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_flat = x.view(-1).clone()
        x_flat[i] += eps
        plus = f(x_flat.view_as(x))
        x_flat[i] -= 2 * eps
        minus = f(x_flat.view_as(x))
        grad.view(-1)[i] = (plus - minus) / (2 * eps)
    return grad

# 测试：f(x) = x^2
x = torch.tensor([1.0, 2.0, 3.0])
f = lambda x: (x ** 2).sum()

num_grad = numerical_gradient(f, x)
print(f"Numerical gradient: {num_grad}")  # 应接近 [2, 4, 6]
```

## 计算图（Computational Graph）

PyTorch 的 autograd 基于动态计算图。每次前向传播时，PyTorch 都会构建一个计算图来追踪所有操作。

### 计算图的概念

```python
import torch

# 创建需要梯度的 Tensor
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

# 构建计算图
z = x * y        # z = x * y
loss = z ** 2    # loss = (x*y)^2 = 36

print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")
print(f"loss: {loss}")

# 反向传播
loss.backward()

# 查看梯度
# d(loss)/dx = 2*(x*y)*y = 2*12*4 = 96
# d(loss)/dy = 2*(x*y)*x = 2*12*3 = 72
print(f"dz/dx = {x.grad}")  # 应为 2*x*y*y = 96... 不对
# d(loss)/dx = d(z^2)/dx = 2*z * dz/dx = 2*z * y = 2*12*4 = 96
# 实际 x.grad = d(loss)/dx = 2*z*y = 2*12*4 = 96

# 更清晰的例子
x = torch.tensor(2.0, requires_grad=True)
y = x + 2      # y = x + 2
z = y * y * 3  # z = 3*(x+2)^2
z.backward()

print(f"x = {x}")
print(f"dz/dx = 6*(x+2) = {6*(2+2)}")
print(f"x.grad = {x.grad}")
```

### 可视化计算图

```python
from torchviz import make_dot  # 需要 pip install torchviz graphviz

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = x * y
loss = z ** 2

# 可视化（需要安装 graphviz）
# make_dot(loss, params={"x": x, "y": y}).render("computational_graph", format="png")
```

## requires_grad 与梯度追踪

### 控制梯度计算

```python
# 方式 1：创建时指定
x = torch.tensor([1.0, 2.0], requires_grad=True)

# 方式 2：创建后设置
y = torch.tensor([3.0, 4.0])
y.requires_grad_(True)

# 方式 3：使用 context manager
z = torch.tensor([5.0, 6.0])
with torch.no_grad():
    z = z * 2  # 这个操作不追踪梯度

print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}")
print(f"z.requires_grad: {z.requires_grad}")
```

### 三种梯度控制模式

| 模式 | 语法 | 是否计算梯度 | 内存占用 | 典型场景 |
|------|------|------------|---------|---------|
| 追踪 | 默认 | 是 | 高 | 训练 |
| 不追踪 | `torch.no_grad()` | 否 | 低 | 推理/验证 |
| 仅推理 | `torch.inference_mode()` | 否 | 最低 | 生产部署 |

```python
# torch.no_grad()：不记录梯度，但允许原地操作
with torch.no_grad():
    x = torch.randn(3, 4)
    y = x * 2
    print(f"y.requires_grad: {y.requires_grad}")  # False

# torch.inference_mode()：更激进的优化，连版本计数器都不记录
with torch.inference_mode():
    x = torch.randn(3, 4)
    y = x * 2
    print(f"Inference mode: {y.requires_grad}")  # False

# 实际使用场景：训练 vs 评估
model = torch.nn.Linear(10, 5)

# 训练模式
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 评估模式
model.eval()
with torch.no_grad():
    output = model(input)
    accuracy = (output.argmax(dim=1) == target).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
```

## backward 与梯度累积

### 基本用法

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

z.backward()
print(f"Gradient: {x.grad}")  # [2, 4, 6]
```

### 梯度累积

默认情况下，调用 `backward()` 后梯度会累积而不是覆盖：

```python
x = torch.tensor([2.0], requires_grad=True)

# 第一次反向传播
y1 = x ** 2
y1.backward()
print(f"After 1st backward: {x.grad}")  # [4.0]

# 第二次反向传播（累积）
y2 = x ** 3
y2.backward()
print(f"After 2nd backward (accumulated): {x.grad}")  # [4.0 + 12.0] = [16.0]
```

💡 **提示**：每次训练迭代前必须调用 `optimizer.zero_grad()` 来清除累积的梯度，否则梯度会不断累加导致训练异常。

### 标量反向 vs 向量反向

```python
# 标量输出：直接 backward
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()  # OK，y 是标量
print(f"Scalar backward: {x.grad}")

# 非标量输出：需要提供 gradient 参数
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2  # y 不是标量
# y.backward()  # 会报错！
y.backward(gradient=torch.tensor([1.0, 1.0]))  # 需要指定 gradient
print(f"Vector backward: {x.grad}")
```

## 计算图的细节

### 保留的中间变量

```python
x = torch.tensor(2.0, requires_grad=True)
a = torch.tensor(3.0, requires_grad=True)

y = x + a        # y = x + a
z = y * y        # z = y^2
z.backward()

print(f"dz/dx = {x.grad}")  # 2*y = 2*(x+a) = 10
print(f"dz/da = {a.grad}")  # 2*y = 10
```

### 多次使用同一变量

```python
x = torch.tensor(2.0, requires_grad=True)

# x 被使用了 3 次
y = x + x + x ** 2  # y = 2x + x^2
y.backward()

# dy/dx = 2 + 2x = 2 + 4 = 6
print(f"Gradient: {x.grad}")  # 6
```

### 释放计算图

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

# 前向传播
y = x ** 2
z = y * 3
loss = z.sum()

# 反向传播时保留计算图
loss.backward(retain_graph=True)
print(f"First backward: {x.grad}")  # [6, 12]

# 因为保留了计算图，可以再次反向传播
x.grad.zero_()  # 清除梯度
loss.backward()
print(f"Second backward: {x.grad}")  # [6, 12]

# 通常不需要 retain_graph=True，除非特殊需求
```

## 自定义 Autograd Function

当内置操作不能满足需求时，可以自定义前向和反向传播：

```python
class MyReLU(torch.autograd.Function):
    """自定义 ReLU 激活函数"""

    @staticmethod
    def forward(ctx, input):
        # ctx 用于保存前向传播的信息，供反向传播使用
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # ReLU 的梯度：x > 0 时为 1，否则为 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 使用自定义函数
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
relu = MyReLU.apply
y = relu(x)
y.sum().backward()

print(f"Forward: {y}")
print(f"Gradient: {x.grad}")
```

### 自定义线性层

```python
class MyLinear(torch.autograd.Function):
    """自定义线性层 y = xW^T + b"""

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t()) + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias

# 使用
x = torch.randn(4, 10, requires_grad=True)
w = torch.randn(5, 10, requires_grad=True)
b = torch.randn(5, requires_grad=True)

y = MyLinear.apply(x, w, b)
loss = y.sum()
loss.backward()

print(f"x.grad shape: {x.grad.shape}")
print(f"w.grad shape: {w.grad.shape}")
print(f"b.grad shape: {b.grad.shape}")
```

## 梯度裁剪（Gradient Clipping）

梯度爆炸是训练 RNN 和深层网络时的常见问题，梯度裁剪可以缓解这个问题：

```python
model = torch.nn.Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(32, 10)
target = torch.randn(32, 5)

optimizer.zero_grad()
output = model(x)
loss = torch.nn.functional.mse_loss(output, target)
loss.backward()

# 方法 1：按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方法 2：按值裁剪
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

print(f"Max grad norm after clipping: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0):.4f}")
optimizer.step()
```

## 常见问题排查

### 梯度为 None

```python
x = torch.tensor(2.0, requires_grad=True)

# 问题：创建了新的 Tensor，脱离了计算图
y = x ** 2
y = y.detach()  # detach 后不再追踪
# y.backward()  # 报错：没有 grad_fn

# 正确做法
y = x ** 2
y.backward()
print(f"Gradient: {x.grad}")
```

### 梯度为 0

```python
# 问题：使用了 int 类型
x = torch.tensor([1, 2, 3], dtype=torch.int32, requires_grad=True)
y = x.sum()
# y.backward()  # 报错：int 类型不支持梯度

# 正确：使用浮点类型
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum()
y.backward()
print(f"Gradient: {x.grad}")
```

### 内存泄漏排查

```python
# 使用 torch.autograd.gradcheck 检查梯度
def check_gradient(f, x):
    """检查自定义梯度的正确性"""
    from torch.autograd import gradcheck
    x = x.double().requires_grad_(True)
    return gradcheck(f, (x,), eps=1e-6)

# 测试线性层
linear = torch.nn.Linear(10, 5).double()
input = torch.randn(4, 10, dtype=torch.double)
print(f"Gradient check: {check_gradient(linear, input)}")
```

## 总结

本文深入学习了 PyTorch 的 autograd 自动微分系统：

- **计算图**：动态构建，前向传播时记录操作，反向传播时应用链式法则
- **requires_grad**：控制哪些 Tensor 需要计算梯度
- **三种模式**：训练（默认）、推理（no_grad）、生产部署（inference_mode）
- **backward**：标量直接调用，非标量需要指定 gradient 参数
- **自定义 Function**：通过继承 `torch.autograd.Function` 实现自定义前向/反向传播
- **梯度裁剪**：防止梯度爆炸的实用技术

理解 autograd 是编写正确 PyTorch 代码的关键。在下一篇文章中，我们将学习 `torch.nn` 模块，了解如何使用 PyTorch 构建神经网络。

[上一篇：Tensor 高级操作](./47-tensor-operation.md) | [下一篇：神经网络模块 torch.nn →](./49-nn-module.md)
