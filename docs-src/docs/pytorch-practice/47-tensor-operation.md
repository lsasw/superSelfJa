---
title: Tensor 高级操作
icon: layers
order: 47
---

# Tensor 高级操作

在掌握了 Tensor 的基础创建和索引操作后，本文将深入探讨 Tensor 的高级操作技巧，包括原地操作、内存布局、复杂张量运算、以及实际深度学习场景中常用的操作模式。

## 原地操作（In-place Operations）

原地操作是指直接修改 Tensor 本身而不创建新副本的操作。在 PyTorch 中，原地操作以 `_` 后缀标识。

### 原地操作与常规操作的对比

| 特性 | 原地操作 | 常规操作 |
|------|---------|---------|
| 内存占用 | 不额外分配内存 | 创建新 Tensor |
| 性能 | 更快，减少内存分配开销 | 稍慢 |
| 计算图兼容性 | 可能破坏反向传播 | 安全 |
| 调试友好 | 修改原始数据，调试困难 | 保留原始数据 |
| 典型方法 | `add_`, `mul_`, `copy_` | `add`, `mul`, `clone` |

```python
import torch

# 常规操作：创建新 Tensor
x = torch.tensor([1.0, 2.0, 3.0])
y = x.add(1.0)        # y 是新 Tensor
print(f"Original x: {x}")       # tensor([1., 2., 3.])
print(f"Result y: {y}")         # tensor([2., 3., 4.])

# 原地操作：修改原 Tensor
x = torch.tensor([1.0, 2.0, 3.0])
x.add_(1.0)                     # 直接修改 x
print(f"After add_: {x}")       # tensor([2., 3., 4.])

# 常见的原地操作方法
x = torch.ones(2, 3)
x.zero_()                       # 全部置零
x.fill_(5.0)                    # 填充指定值
x.copy_(torch.ones(2, 3) * 3)   # 从另一个 Tensor 拷贝
```

💡 **提示**：在训练循环中谨慎使用原地操作。如果 Tensor 参与了计算图（需要梯度），原地修改可能导致反向传播失败。通常只在数据预处理或推理阶段使用原地操作。

## 内存布局与 Contiguous

PyTorch Tensor 的内存布局直接影响性能。理解 contiguous 概念对于写出高性能代码至关重要。

```python
# contiguous 检查
x = torch.randn(3, 4)
print(f"Is contiguous: {x.is_contiguous()}")  # True

# transpose 破坏 contiguous
x_t = x.t()
print(f"After transpose: {x_t.is_contiguous()}")  # False

# 恢复 contiguous
x_contig = x_t.contiguous()
print(f"After contiguous(): {x_contig.is_contiguous()}")  # True

# 内存查看
print(f"Stride: {x.stride()}")      # (4, 1) - 行主序
print(f"Transpose stride: {x_t.stride()}")  # (1, 4) - 步长交换
```

### Stride 的含义

Stride 表示在每个维度上移动一个位置时，需要跳过的元素个数：

```python
x = torch.arange(12).reshape(3, 4)
print(f"Tensor:\n{x}")
print(f"Stride: {x.stride()}")  # (4, 1)

# dim 0 的 stride = 4：移动到下一行需要跳过 4 个元素
# dim 1 的 stride = 1：移动到下一列只需跳过 1 个元素

# 转置后 stride 交换
print(f"Transposed stride: {x.t().stride()}")  # (1, 4)
```

## 拼接与分割

### 拼接操作

```python
a = torch.ones(2, 3)
b = torch.zeros(2, 3)

# 沿第一个维度拼接
cat_0 = torch.cat([a, b], dim=0)
print(f"Cat dim=0 shape: {cat_0.shape}")  # torch.Size([4, 3])

# 沿第二个维度拼接
cat_1 = torch.cat([a, b], dim=1)
print(f"Cat dim=1 shape: {cat_1.shape}")  # torch.Size([2, 6])

# stack 创建新维度
stacked = torch.stack([a, b], dim=0)
print(f"Stack shape: {stacked.shape}")  # torch.Size([2, 2, 3])

# 对比 cat 和 stack
print(f"Cat preserves dimensions: {cat_0.dim()}")    # 2
print(f"Stack adds dimension: {stacked.dim()}")       # 3
```

### 分割操作

```python
x = torch.arange(12).reshape(3, 4)

# 按指定数量分割
chunks = torch.chunk(x, 2, dim=1)
print(f"Chunks: {len(chunks)}")  # 2
for i, c in enumerate(chunks):
    print(f"Chunk {i}:\n{c}")

# 按指定大小分割
splits = torch.split(x, 2, dim=0)
print(f"Split sizes: {[s.shape for s in splits]}")

# 不等长分割
splits_unequal = torch.split(x, [1, 2], dim=0)
print(f"Unequal split shapes: {[s.shape for s in splits_unequal]}")

# tensor_split (自动计算分割点)
splits_auto = torch.tensor_split(x, 3, dim=0)
print(f"Auto split: {[s.shape for s in splits_auto]}")
```

## 聚合与归约操作

### 全局与维度级归约

```python
x = torch.arange(1, 13, dtype=torch.float).reshape(3, 4)
print(f"Tensor:\n{x}")

# 全局归约
print(f"Sum: {x.sum()}")          # 78
print(f"Mean: {x.mean()}")        # 6.5
print(f"Product: {x.prod()}")     # 全部元素相乘
print(f"Std: {x.std():.4f}")      # 标准差
print(f"Var: {x.var():.4f}")      # 方差

# 维度级归约
print(f"Sum dim=0: {x.sum(dim=0)}")   # 按列求和
print(f"Sum dim=1: {x.sum(dim=1)}")   # 按行求和
print(f"Mean dim=0: {x.mean(dim=0)}") # 按列均值

# 保持维度
print(f"Sum keepdim: {x.sum(dim=0, keepdim=True).shape}")  # torch.Size([1, 4])

# 最大值/最小值及其索引
max_val, max_idx = x.max(dim=1)
print(f"Max values: {max_val}, Indices: {max_idx}")

# 累积操作
print(f"Cumsum dim=0:\n{x.cumsum(dim=0)}")
print(f"Cumprod dim=1:\n{x.cumprod(dim=1)}")
```

### 常用归约函数对比

| 函数 | 功能 | 是否可导 | 典型用途 |
|------|------|---------|---------|
| sum | 求和 | 是 | 损失聚合 |
| mean | 均值 | 是 | 批量归一化 |
| max/min | 最大值/最小值 | 是 | 注意力机制 |
| argmax/argmin | 最大/小值索引 | 否 | 分类预测 |
| std/var | 标准差/方差 | 是 | 数据标准化 |
| norm | 范数计算 | 是 | 梯度裁剪 |
| prod | 连乘 | 是 | 概率计算 |

## 排序与搜索

```python
x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])

# 排序
sorted_vals, indices = torch.sort(x)
print(f"Sorted: {sorted_vals}")
print(f"Indices: {indices}")

# 降序排序
sorted_desc, _ = torch.sort(x, descending=True)
print(f"Descending: {sorted_desc}")

# Top-K
topk_vals, topk_idx = torch.topk(x, k=3)
print(f"Top-3: {topk_vals}, Indices: {topk_idx}")

# 条件搜索
mask = x > 4
indices = torch.nonzero(mask)
print(f"Indices where x > 4: {indices.squeeze()}")

# unique
unique_vals, counts = torch.unique(x, return_counts=True)
print(f"Unique: {unique_vals}, Counts: {counts}")

# 排序 2D Tensor
matrix = torch.randn(3, 4)
sorted_matrix, _ = torch.sort(matrix, dim=1)
print(f"Row-sorted:\n{sorted_matrix}")
```

## 比较与逻辑操作

```python
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([5, 4, 3, 2, 1])

# 逐元素比较
print(f"Equal: {torch.eq(a, b)}")
print(f"Greater: {torch.gt(a, b)}")
print(f"Less: {torch.lt(a, b)}")

# 逻辑操作
mask1 = a > 2
mask2 = b > 2
print(f"And: {torch.logical_and(mask1, mask2)}")
print(f"Or: {torch.logical_or(mask1, mask2)}")
print(f"Not: {torch.logical_not(mask1)}")

# 逐元素选择
result = torch.where(a > b, a, b)
print(f"Where (max): {result}")

# all 和 any
print(f"All > 0: {(a > 0).all()}")
print(f"Any > 4: {(a > 4).any()}")
```

## 采样与随机操作

```python
torch.manual_seed(42)  # 设置随机种子保证可复现

# 均匀分布
uniform = torch.rand(3, 4)
print(f"Uniform [0,1):\n{uniform}")

# 正态分布
normal = torch.randn(3, 4)
print(f"Standard normal:\n{normal}")

# 指定均值和方差的正态分布
normal_custom = torch.normal(mean=10.0, std=2.0, size=(3, 4))
print(f"Custom normal:\n{normal_custom}")

# 离散均匀分布
randint = torch.randint(0, 10, (3, 4))
print(f"Random integers [0,10):\n{randint}")

# 随机排列
perm = torch.randperm(10)
print(f"Random permutation: {perm}")

# 随机选择（有放回）
x = torch.tensor([10, 20, 30, 40, 50])
samples = x[torch.randint(0, len(x), (6,))]
print(f"Random samples: {samples}")

# 多项分布
probs = torch.tensor([0.1, 0.3, 0.6])
samples_multi = torch.multinomial(probs, 10, replacement=True)
print(f"Multinomial samples: {samples_multi}")
```

💡 **提示**：深度学习实验中务必设置随机种子（`torch.manual_seed()`），否则实验结果无法复现。建议在训练脚本开头同时设置 `torch.cuda.manual_seed_all()`。

## 高级索引操作

### gather 操作

`gather` 沿着指定维度收集值，是实现复杂索引模式的核心工具：

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 沿着 dim=1 收集
index = torch.tensor([[0], [1], [2]])
result = torch.gather(x, dim=1, index=index)
print(f"Gather result:\n{result}")

# 实际应用：根据类别索引收集概率
logits = torch.randn(4, 10)  # batch_size=4, num_classes=10
probs = torch.softmax(logits, dim=1)
labels = torch.tensor([3, 7, 1, 5])

# 收集每个样本对应标签的概率
selected_probs = probs.gather(1, labels.unsqueeze(1))
print(f"Selected probabilities: {selected_probs.squeeze()}")
```

### scatter_ 操作

`scatter_` 是 `gather` 的逆操作，将值写入指定位置：

```python
# One-hot 编码实现
num_classes = 5
labels = torch.tensor([0, 2, 1, 3])
one_hot = torch.zeros(4, num_classes)
one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
print(f"One-hot:\n{one_hot}")

# 值分散
src = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
result = torch.zeros(4, 5)
result.scatter_(1, labels.unsqueeze(1), src)
print(f"Scatter result:\n{result}")
```

## 实用工具函数

```python
# repeat 与 expand 的区别
x = torch.tensor([1, 2, 3])

# expand：创建视图，不拷贝数据（要求原始维度为 1）
expanded = x.expand(2, 3)
print(f"Expand shape: {expanded.shape}")
print(f"Expand shares memory: {expanded.data_ptr() == x.data_ptr()}")

# repeat：拷贝数据，实际重复
repeated = x.repeat(2, 1)
print(f"Repeat shape: {repeated.shape}")
print(f"Repeat shares memory: {repeated.data_ptr() == x.data_ptr()}")

# 翻转
x = torch.arange(12).reshape(3, 4)
print(f"Flip dim=0:\n{x.flip(0)}")
print(f"Flip dim=1:\n{x.flip(1)}")

# 滚动
print(f"Roll 1: {torch.roll(torch.arange(5), shifts=1)}")
print(f"Roll -2: {torch.roll(torch.arange(5), shifts=-2)}")

# 填充
x = torch.ones(3, 3)
padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
print(f"Padded:\n{padded}")
```

### repeat vs expand 对比

| 特性 | expand | repeat |
|------|--------|--------|
| 内存 | 创建视图，共享内存 | 拷贝数据，独立内存 |
| 速度 | 更快 | 较慢 |
| 限制 | 只能扩展维度为 1 的轴 | 无限制 |
| 原地修改 | 会影响原 Tensor | 不影响 |

## 实战：用 Tensor 操作实现常用功能

### 实现 Softmax

```python
def softmax(x, dim=-1):
    # 减去最大值保证数值稳定性
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

logits = torch.randn(4, 10)
my_softmax = softmax(logits)
torch_softmax = torch.softmax(logits, dim=-1)

# 验证
print(f"Max difference: {(my_softmax - torch_softmax).abs().max():.2e}")
```

### 实现 Layer Normalization

```python
def layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

x = torch.randn(4, 10)
result = layer_norm(x)
print(f"Layer norm mean (should be ~0): {result.mean():.2e}")
print(f"Layer norm std (should be ~1): {result.std():.4f}")
```

## 总结

本文深入探讨了 PyTorch Tensor 的高级操作：

- **原地操作**：以 `_` 后缀标识的方法，节省内存但需谨慎使用
- **内存布局**：理解 contiguous 和 stride 对性能优化的重要性
- **拼接与分割**：cat、stack、split、chunk 等核心操作
- **聚合归约**：sum、mean、max 等函数及其维度控制
- **排序与搜索**：sort、topk、unique 等实用函数
- **gather 与 scatter**：实现复杂索引模式的关键工具
- **expand 与 repeat**：高效广播和重复的策略选择

掌握这些高级操作，能够帮助你写出更高效、更简洁的 PyTorch 代码。在下一篇文章中，我们将学习 PyTorch 最核心的特性之一——自动微分系统（Autograd）。

[上一篇：PyTorch 框架概述与基础](./46-pytorch-basics.md) | [下一篇：自动微分 Autograd →](./48-autograd.md)
