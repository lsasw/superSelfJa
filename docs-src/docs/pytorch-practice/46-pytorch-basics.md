---
title: PyTorch 框架概述与基础
icon: fire
order: 46
---

# PyTorch 框架概述与基础

PyTorch 是由 Meta（原 Facebook）AI 研究院开源的深度学习框架，以其动态计算图、Pythonic 的 API 设计和强大的 GPU 加速能力，已成为学术界和工业界最主流的深度学习工具之一。本文将带你从零开始了解 PyTorch 的核心概念和基础操作。

## PyTorch 的发展历程

| 时间 | 版本 | 重要特性 |
|------|------|----------|
| 2016 | 0.1 | 初始发布，基于 Torch7 Lua 框架 |
| 2018 | 0.4 | 合并 Variable 与 Tensor，引入设备无关代码 |
| 2019 | 1.0 | 生产级稳定性，JIT 编译器 |
| 2020 | 1.5 | 新增 `torch.nn.GELU`，API 完善 |
| 2021 | 1.10 | `torch.compile` 前身，性能优化 |
| 2022 | 2.0 | 引入 `torch.compile`，2-4 倍训练加速 |
| 2024 | 2.4+ | 分布式训练增强，FlashAttention 原生支持 |

PyTorch 的成功源于以下几个关键设计理念：

1. **动态计算图（Define-by-Run）**：计算图在运行时动态构建，支持条件分支和循环
2. **Pythonic API**：与 NumPy 高度一致的接口风格，降低学习成本
3. **GPU 无缝加速**：`tensor.to('cuda')` 一行代码即可迁移到 GPU
4. **丰富的生态系统**：TorchVision、TorchText、TorchAudio 等配套库

## 安装 PyTorch

### 使用 pip 安装

```bash
# CPU 版本
pip install torch torchvision torchaudio

# GPU 版本（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（轻量安装）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 验证安装

```python
import torch

# 检查版本
print(f"PyTorch Version: {torch.__version__}")

# 检查 CUDA 可用性
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

## Tensor：PyTorch 的核心数据结构

Tensor（张量）是 PyTorch 中最基本的数据结构，本质上是一个多维数组。它与 NumPy 的 ndarray 非常相似，但有两个关键优势：可以在 GPU 上运行，以及支持自动微分。

### 创建 Tensor

```python
import torch
import numpy as np

# 1. 从 Python 列表创建
data = [[1, 2], [3, 4]]
x = torch.tensor(data)
print(f"From list:\n{x}")

# 2. 从 NumPy 数组创建
np_array = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(np_array)
print(f"From numpy:\n{y}")

# 3. 创建特殊 Tensor
zeros = torch.zeros(2, 3)        # 全零
ones = torch.ones(2, 3)          # 全一
rand = torch.rand(2, 3)          # 均匀分布随机 [0, 1)
randn = torch.randn(2, 3)        # 标准正态分布
eye = torch.eye(3)               # 单位矩阵
full = torch.full((2, 3), 5.0)   # 填充指定值

print(f"Zeros:\n{zeros}")
print(f"Random:\n{rand}")
print(f"Eye:\n{eye}")

# 4. 创建与已有 Tensor 形状相同的新 Tensor
x = torch.randn(3, 4)
y_like = torch.zeros_like(x)     # 与 x 同形状的全零 Tensor
rand_like = torch.rand_like(x)   # 与 x 同形状的随机 Tensor
print(f"zeros_like shape: {y_like.shape}")
```

### Tensor 的属性

```python
x = torch.randn(3, 4, 5)

print(f"Shape: {x.shape}")         # torch.Size([3, 4, 5])
print(f"Size: {x.size()}")         # 等价于 shape
print(f"Dimensions: {x.dim()}")    # 3
print(f"Data type: {x.dtype}")     # torch.float32
print(f"Device: {x.device}")       # cpu
print(f"Total elements: {x.numel()}")  # 60
print(f"Memory (bytes): {x.element_size() * x.numel()}")
```

### Tensor 与 NumPy 的互操作

```python
# Tensor -> NumPy
tensor_x = torch.randn(3, 4)
numpy_x = tensor_x.numpy()
print(f"NumPy array: {type(numpy_x)}")

# NumPy -> Tensor
array_y = np.random.randn(3, 4)
tensor_y = torch.from_numpy(array_y)
print(f"Tensor: {type(tensor_y)}")

# 注意：共享内存！修改一个会影响另一个
tensor_x[0, 0] = 999
print(f"Original tensor: {tensor_x[0, 0]}")
print(f"Corresponding numpy: {numpy_x[0, 0]}")
```

💡 **提示**：`torch.tensor()` 会拷贝数据，而 `torch.from_numpy()` 共享内存。在大数据场景下，共享内存可以节省大量内存开销。

## 数据类型与设备

### 常见数据类型

| 数据类型 | PyTorch 类型 | NumPy 类型 | 说明 |
|----------|-------------|------------|------|
| 32位浮点 | torch.float32 | np.float32 | 默认类型，深度学习常用 |
| 64位浮点 | torch.float64 | np.float64 | 双精度，科学计算 |
| 16位浮点 | torch.float16 | np.float16 | 半精度，加速训练 |
| BrainFloat16 | torch.bfloat16 | - | 8位指数+7位尾数 |
| 32位整数 | torch.int32 | np.int32 | 索引常用 |
| 64位整数 | torch.int64 | np.int64 | LongTensor，标签常用 |
| 8位无符号 | torch.uint8 | np.uint8 | 图像像素值 |
| 布尔 | torch.bool | np.bool_ | 掩码操作 |

```python
# 指定数据类型
x = torch.tensor([1, 2, 3], dtype=torch.float64)
print(f"Specified dtype: {x.dtype}")

# 类型转换
y = x.to(torch.int32)
print(f"Converted dtype: {y.dtype}")

# 快捷方法
z = x.long()      # 转为 int64
w = x.float()     # 转为 float32
print(f"Long dtype: {z.dtype}, Float dtype: {w.dtype}")
```

### 设备管理

```python
# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建在指定设备上的 Tensor
x_gpu = torch.randn(3, 4, device="cuda") if torch.cuda.is_available() else torch.randn(3, 4)

# 设备间迁移
if torch.cuda.is_available():
    x_cpu = torch.randn(3, 4)
    x_on_gpu = x_cpu.to("cuda")
    x_back = x_on_gpu.to("cpu")
    print(f"CUDA device: {x_on_gpu.device}")
    print(f"Back to CPU: {x_back.device}")
```

💡 **提示**：在深度学习中，通常使用 `bfloat16` 进行混合精度训练。相比 `float16`，它拥有更大的动态范围，不容易出现梯度溢出问题。

## Tensor 的索引与切片

PyTorch 支持所有 NumPy 风格的索引和切片操作：

```python
x = torch.arange(12).reshape(3, 4)
print(f"Original:\n{x}")

# 基本索引
print(f"Element [1,2]: {x[1, 2]}")
print(f"Row 0: {x[0]}")
print(f"Column 2: {x[:, 2]}")
print(f"Sub-matrix:\n{x[0:2, 1:3]}")

# 高级索引
indices = torch.tensor([0, 2])
print(f"Fancy indexing:\n{x[indices]}")

# 布尔索引
mask = x > 5
print(f"Elements > 5: {x[mask]}")

# 条件赋值
y = torch.where(x > 5, x, torch.tensor(0.0))
print(f"Where result:\n{y}")
```

## Tensor 的形状操作

```python
x = torch.arange(12)
print(f"Original: {x.shape}")

# 改变形状
x_reshaped = x.view(3, 4)        # 推荐用于 contiguous tensor
print(f"view(3,4): {x_reshaped.shape}")

x_reshaped2 = x.reshape(2, 6)   # 自动处理 contiguous 问题
print(f"reshape(2,6): {x_reshaped2.shape}")

# 使用 -1 自动推导维度
x_auto = x.view(3, -1)
print(f"view(3, -1): {x_auto.shape}")

# 压缩维度
x_squeezed = x.view(1, 12, 1)
print(f"Squeeze: {x_squeezed.squeeze().shape}")       # torch.Size([12])
print(f"Squeeze dim=0: {x_squeezed.squeeze(0).shape}") # torch.Size([12, 1])

# 增加维度
x_unsqueezed = x_squeezed.unsqueeze(0)
print(f"Unsqueeze: {x_unsqueezed.shape}")

# 转置
x_2d = x.view(3, 4)
print(f"Transpose:\n{x_2d.t()}")
print(f"Transpose (general):\n{x_2d.permute(1, 0).shape}")
```

💡 **提示**：`view()` 要求 Tensor 在内存中是连续的（contiguous）。如果不确定，使用 `reshape()` 更安全，它会自动处理连续性问题。

## 基本数学运算

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 逐元素运算
print(f"Add: {a + b}")
print(f"Multiply: {a * b}")
print(f"Divide: {a / b}")
print(f"Power: {a ** 2}")

# 矩阵乘法
print(f"Matmul:\n{a @ b}")
print(f"Matmul (function): {torch.matmul(a, b)}")

# 聚合操作
print(f"Sum: {a.sum()}")
print(f"Sum along dim 0: {a.sum(dim=0)}")
print(f"Mean: {a.mean()}")
print(f"Max value: {a.max()}, Max index: {a.argmax()}")
print(f"Min along dim 1: {a.min(dim=1)}")

# 数学函数
print(f"Exp: {torch.exp(a)}")
print(f"Log: {torch.log(a)}")
print(f"Sqrt: {torch.sqrt(torch.abs(a))}")
print(f"Abs: {torch.abs(a - 2.5)}")
```

## 广播机制（Broadcasting）

广播是 PyTorch 中非常重要的概念，允许不同形状的 Tensor 进行运算：

```python
# 标量广播
x = torch.ones(3, 4)
print(f"x + 5:\n{x + 5}")

# 向量广播到矩阵
row = torch.arange(4)       # shape: (4,)
print(f"Add row vector:\n{x + row}")  # 广播到 (3, 4)

col = torch.arange(3).view(3, 1)  # shape: (3, 1)
print(f"Add column vector:\n{x + col}")  # 广播到 (3, 4)

# 广播规则
a = torch.ones(4, 3, 2)
b = torch.ones(3, 2)
c = a + b  # b 广播到 (4, 3, 2)
print(f"Broadcast result shape: {c.shape}")
```

广播规则：从尾部维度开始比较，两个维度兼容当且仅当它们相等或其中一个为 1。

## 总结

本文介绍了 PyTorch 的核心基础概念：

- **Tensor 创建**：多种方式创建 Tensor，包括从列表、NumPy 数组以及特殊初始化方法
- **数据类型与设备**：掌握 dtype 和 device 管理，是后续 GPU 训练的基础
- **索引与切片**：与 NumPy 一致的索引语法，支持高级索引和布尔掩码
- **形状操作**：view、reshape、squeeze、unsqueeze 等核心操作
- **数学运算**：逐元素运算、矩阵乘法、聚合函数和数学函数
- **广播机制**：理解广播规则可以写出更简洁高效的代码

这些基础知识是后续所有 PyTorch 实战的基石。在下一篇文章中，我们将深入学习 Tensor 的高级操作，包括原地操作、内存视图、复杂索引等实用技巧。

[下一篇：Tensor 高级操作 →](./47-tensor-operation.md)
