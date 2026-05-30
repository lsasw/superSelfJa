---
title: 数据集与数据加载 Dataset 与 DataLoader
icon: database
order: 50
---

# 数据集与数据加载 Dataset 与 DataLoader

在深度学习中，数据加载是训练流程的第一步也是最重要的一环。PyTorch 提供了 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader` 两个核心类来管理数据。本文将学习如何自定义数据集、配置数据加载器、以及实现高效的数据管道。

## Dataset：数据源抽象

`Dataset` 是一个抽象类，用于表示数据集。继承 `Dataset` 需要实现两个方法：
- `__getitem__(index)`：返回第 index 个样本
- `__len__()`：返回数据集大小

### 自定义 Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    """自定义数据集示例"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        """加载文件列表"""
        # 实际项目中从目录读取文件列表
        return [f"sample_{i}.pt" for i in range(100)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 加载数据（示例中使用随机数据模拟）
        data = torch.randn(3, 224, 224)
        label = torch.randint(0, 10, (1,)).item()

        if self.transform:
            data = self.transform(data)

        return data, label

# 使用
dataset = CustomDataset(data_dir="./data")
print(f"Dataset size: {len(dataset)}")
data, label = dataset[0]
print(f"Sample shape: {data.shape}, Label: {label}")
```

### 从内存数据创建 Dataset

```python
class TensorDataset(Dataset):
    """从 Tensor 创建的简单数据集"""

    def __init__(self, features, labels):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 创建示例数据
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))

dataset = TensorDataset(X, y)
print(f"Features shape: {dataset[0][0].shape}")
print(f"Label: {dataset[0][1]}")

# 也可以使用 PyTorch 内置的 TensorDataset
from torch.utils.data import TensorDataset as PyTensorDataset
built_in = PyTensorDataset(X, y)
```

## 内置 Dataset

PyTorch 在 `torchvision.datasets`、`torchtext.datasets` 等包中提供了大量预构建数据集：

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

# MNIST 手写数字
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

print(f"MNIST train size: {len(train_data)}")
img, label = train_data[0]
print(f"Image shape: {img.shape}, Label: {label}")

# 其他常用数据集
# datasets.CIFAR10(root="./data", download=True, transform=transform)
# datasets.ImageFolder(root="./imagenet", transform=transform)
# datasets.CocoCaptions(root="./coco/images", annFile="./coco/annotations.json")
```

### ImageFolder：自定义图像数据集

```python
from torchvision.datasets import ImageFolder

# 目录结构要求：
# root/
#   class1/
#     img1.jpg
#     img2.jpg
#   class2/
#     img3.jpg
#     img4.jpg

# dataset = ImageFolder(
#     root="./data/imagenet",
#     transform=ToTensor()
# )
# print(f"Classes: {dataset.classes}")
# print(f"Class to index: {dataset.class_to_idx}")
```

## DataLoader：批量加载

`DataLoader` 将 Dataset 包装成可迭代的对象，支持批处理、打乱、多进程加载等功能。

### 基本用法

```python
# 创建 DataLoader
train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,          # 批大小
    shuffle=True,           # 训练时打乱
    num_workers=4,          # 数据加载进程数
    pin_memory=True,        # 锁页内存，加速 GPU 传输
    drop_last=True,         # 丢弃最后一个不完整的 batch
    prefetch_factor=2,      # 每个 worker 预取的 batch 数
    persistent_workers=True # 迭代间保持 worker 进程
)

print(f"Number of batches: {len(train_loader)}")

# 遍历数据
for batch_idx, (data, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx}: data={data.shape}, targets={targets.shape}")
    if batch_idx >= 2:
        break
```

### DataLoader 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| dataset | Dataset | 必需 | 数据源 |
| batch_size | int | 1 | 每个 batch 的样本数 |
| shuffle | bool | False | 是否每个 epoch 打乱数据 |
| sampler | Sampler | None | 自定义采样策略 |
| batch_sampler | BatchSampler | None | 自定义 batch 采样 |
| num_workers | int | 0 | 数据加载子进程数 |
| collate_fn | callable | None | 自定义 collate 函数 |
| pin_memory | bool | False | 是否使用锁页内存 |
| drop_last | bool | False | 是否丢弃最后一个不完整 batch |
| timeout | float | 0 | worker 超时时间 |
| worker_init_fn | callable | None | worker 初始化函数 |
| prefetch_factor | int | 2 | 预取因子 |
| persistent_workers | bool | False | 是否持久化 worker |

### num_workers 选择建议

```python
import os

# 推荐的 num_workers 选择策略
def get_num_workers():
    """根据系统资源推荐 worker 数量"""
    cpu_count = os.cpu_count() or 4

    # 经验法则：不超过 CPU 核心数，留 1-2 个给主进程
    recommended = min(cpu_count - 1, 8)
    return max(1, recommended)

print(f"CPU cores: {os.cpu_count()}")
print(f"Recommended workers: {get_num_workers()}")
```

💡 **提示**：`num_workers` 的设置对训练速度影响很大。过小会导致 GPU 等待数据，过大会占用过多内存。建议从小值（2-4）开始，观察 GPU 利用率逐步调大。使用 `pin_memory=True` 配合 GPU 训练可以加速 CPU 到 GPU 的数据传输。

## 数据变换 Transforms

数据变换是数据预处理的重要环节。`torchvision.transforms` 提供了丰富的变换操作。

### 常用图像变换

```python
from torchvision import transforms

# 单个变换
to_tensor = transforms.ToTensor()        # PIL Image -> Tensor [0,1]
normalize = transforms.Normalize(        # 标准化
    mean=[0.485, 0.456, 0.406],          # ImageNet 均值
    std=[0.229, 0.224, 0.225]            # ImageNet 标准差
)
resize = transforms.Resize((224, 224))   # 调整大小
center_crop = transforms.CenterCrop(224)  # 中心裁剪
random_crop = transforms.RandomCrop(224, padding=4)  # 随机裁剪
random_hflip = transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转
color_jitter = transforms.ColorJitter(   # 颜色抖动
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)

# 组合变换（Compose）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### 训练集 vs 验证集变换

| 变换类型 | 训练集 | 验证集 | 目的 |
|---------|--------|--------|------|
| 随机裁剪 | 使用 | 不使用 | 数据增强 |
| 随机翻转 | 使用 | 不使用 | 数据增强 |
| 颜色抖动 | 使用 | 不使用 | 数据增强 |
| 中心裁剪 | 不使用 | 使用 | 统一尺寸 |
| 标准化 | 使用 | 使用 | 统一分布 |

```python
# 完整的数据集创建
from torchvision.datasets import CIFAR10

# 训练集（带数据增强）
train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

# 验证集（无数据增强）
val_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=val_transform
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
```

## Collate Function

当数据需要特殊处理时（如变长序列、不同尺寸的图像），需要自定义 `collate_fn`：

```python
def custom_collate_fn(batch):
    """自定义 collate 函数，处理变长数据"""
    # batch 是一个列表，每个元素是 __getitem__ 的返回值
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 自定义处理逻辑
    # 例如：padding 变长序列
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    targets = torch.tensor(targets)

    return data_padded, targets

# 使用自定义 collate_fn
# loader = DataLoader(
#     dataset,
#     batch_size=32,
#     collate_fn=custom_collate_fn
# )

# 变长序列示例
class SequenceDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 变长序列
        seq_len = torch.randint(10, 50, (1,)).item()
        seq = torch.randn(seq_len, 64)  # seq_len x features
        label = torch.randint(0, 10, (1,)).item()
        return seq, label

dataset = SequenceDataset()

# 使用默认的 collate_fn 会失败（长度不一致）
# 需要自定义 collate_fn
def pad_collate_fn(batch):
    seqs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # padding
    seqs_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    labels = torch.tensor(labels)
    # 创建 mask
    lengths = torch.tensor([s.size(0) for s in seqs])
    mask = torch.arange(seqs_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return seqs_padded, labels, mask

loader = DataLoader(dataset, batch_size=8, collate_fn=pad_collate_fn)
for seqs, labels, mask in loader:
    print(f"Padded seq shape: {seqs.shape}")
    print(f"Mask shape: {mask.shape}")
    break
```

## 数据采样器（Sampler）

```python
from torch.utils.data import Sampler, WeightedRandomSampler

# WeightedRandomSampler：处理类别不平衡
# 计算每个样本的采样权重
targets = torch.randint(0, 10, (1000,))
class_counts = torch.bincount(targets)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[targets]

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=weighted_sampler  # 使用 sampler 时不能设置 shuffle=True
)

# 自定义 Sampler
class BalancedBatchSampler(Sampler):
    """确保每个 batch 包含所有类别的样本"""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # 按类别分组
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __iter__(self):
        # 从每个类别中采样
        for _ in range(len(self)):
            batch = []
            for label, indices in self.class_indices.items():
                selected = torch.randint(0, len(indices), (self.batch_size // len(self.class_indices),))
                batch.extend([indices[i] for i in selected])
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
```

## 分布式数据加载

```python
import torch.distributed as dist

def get_distributed_loader(dataset, batch_size, num_workers):
    """创建分布式数据加载器"""
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    return loader

# 使用
# loader = get_distributed_loader(train_dataset, batch_size=32, num_workers=4)
# for epoch in range(num_epochs):
#     sampler.set_epoch(epoch)  # 每个 epoch 重新打乱
#     for batch in loader:
#         ...
```

## 数据加载性能优化

### 性能分析

```python
import time

def profile_dataloader(loader, num_batches=10):
    """分析 DataLoader 性能"""
    times = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        start = time.time()
        # 模拟 GPU 传输
        if torch.cuda.is_available():
            batch_data = batch[0].cuda(non_blocking=True)
            torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Average batch loading time: {avg_time*1000:.2f} ms")
    print(f"Batches per second: {1/avg_time:.1f}")

# 比较不同 num_workers 的性能
for workers in [0, 2, 4, 8]:
    print(f"\nnum_workers={workers}")
    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=workers,
        pin_memory=(workers > 0)
    )
    profile_dataloader(loader)
```

### 优化建议

| 优化项 | 效果 | 适用场景 |
|--------|------|---------|
| `num_workers > 0` | 大幅提升 | CPU 密集型数据加载 |
| `pin_memory=True` | 加速 GPU 传输 | GPU 训练 |
| `prefetch_factor` | 减少等待 | 数据加载慢于模型训练 |
| 预缓存数据 | 避免 I/O | 小数据集 |
| 高效存储格式 | 减少解析 | 大型数据集 |

## 总结

本文深入学习了 PyTorch 的数据加载机制：

- **Dataset**：自定义数据集需要实现 `__getitem__` 和 `__len__` 方法
- **内置 Dataset**：MNIST、CIFAR、ImageFolder 等常用数据集开箱即用
- **DataLoader**：批处理、打乱、多进程加载的核心工具
- **Transforms**：数据增强和预处理的管道
- **collate_fn**：处理变长数据和复杂 batch 结构
- **Sampler**：控制数据采样策略，处理类别不平衡
- **性能优化**：合理设置 num_workers、pin_memory 等参数

高效的数据管道是模型训练的基础。在下一篇文章中，我们将学习如何编写完整的训练循环，将前面学到的知识整合起来。

[上一篇：神经网络模块 torch.nn](./49-nn-module.md) | [下一篇：训练循环 →](./51-training-loop.md)
