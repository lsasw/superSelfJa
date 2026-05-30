---
title: 分布式训练 Distributed Training
icon: network
order: 54
---

# 分布式训练 Distributed Training

当模型规模和数据量不断增长时，单 GPU 训练变得不再可行。分布式训练通过多台设备协同工作，可以大幅缩短训练时间并支持更大规模的模型。本文将系统学习 PyTorch 的分布式训练技术，从 DDP 基础到高级策略。

## 为什么需要分布式训练

| 场景 | 单 GPU 限制 | 分布式训练优势 |
|------|-----------|---------------|
| 大模型训练 | 显存不足 | 模型并行/ZeRO |
| 大数据集 | 训练时间过长 | 数据并行加速 |
| 超参数搜索 | 顺序执行 | 多实验并行 |
| 生产环境 | 资源浪费 | 高效利用集群 |

## 分布式训练策略对比

| 策略 | 原理 | 适用场景 | 通信开销 |
|------|------|---------|---------|
| 数据并行 (DP/DDP) | 每个 GPU 持有完整模型，处理不同数据 | 模型可放入单 GPU | 中 |
| 张量并行 (TP) | 模型张量切分到多 GPU | 超大单层 | 高 |
| 流水线并行 (PP) | 模型层切分到多 GPU | 深层模型 | 低 |
| ZeRO | 优化器状态/梯度/参数分片 | 超大规模模型 | 中 |

## DDP 核心概念

### 分布式环境初始化

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode():
    """初始化分布式环境"""
    # 从环境变量读取（torchrun 会自动设置）
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",          # GPU 使用 NCCL
            init_method="env://",    # 使用环境变量
            rank=rank,
            world_size=world_size
        )

    # 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device

def cleanup_distributed_mode():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
```

### 进程组（Process Group）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| backend | 通信后端 | `gloo`(CPU), `nccl`(GPU), `mpi` |
| init_method | 初始化方式 | `env://` |
| world_size | 总进程数 | 1 |
| rank | 当前进程排名 | 0 |

### 常用后端对比

| 后端 | GPU 支持 | 多机支持 | 性能 | 平台 |
|------|---------|---------|------|------|
| NCCL | 是 | 是 | 最优 | Linux |
| Gloo | 部分 | 是 | 中等 | 跨平台 |
| MPI | 部分 | 是 | 中等 | 需安装 |
| CPU-Gloo | 否 | 是 | 较慢 | 跨平台 |

## DDP 训练完整示例

### 单机多卡训练脚本

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

def train_one_epoch(model, train_loader, optimizer, criterion, epoch, rank):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.cuda(), targets.cuda()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0 and rank == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def ddp_worker(rank, world_size):
    """DDP worker 函数"""
    # 初始化
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 创建模型
    model = SimpleModel(input_size=784, num_classes=10).cuda()
    model = DDP(model, device_ids=[rank])

    # 创建数据集
    train_data = TensorDataset(
        torch.randn(10000, 784),
        torch.randint(0, 10, (10000,))
    )

    # 使用 DistributedSampler
    sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    num_epochs = 5
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 每个 epoch 重新打乱
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, rank)

        if rank == 0:
            print(f"Rank {rank} | Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    # 只保存一份模型
    if rank == 0:
        torch.save(model.module.state_dict(), 'ddp_model.pth')

    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

### 使用 torchrun 启动

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train_ddp.py

# 多机（2 台机器，每台 4 卡）
# 机器 0（主节点）
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
         --master_addr=192.168.1.100 --master_port=29500 \
         train_ddp.py

# 机器 1
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
         --master_addr=192.168.1.100 --master_port=29500 \
         train_ddp.py
```

## DDP 中的关键注意事项

### 同步 BatchNorm

```python
from torch.nn import SyncBatchNorm

# 标准 BatchNorm 在 DDP 中只统计单卡数据
# 使用 SyncBatchNorm 跨卡同步统计信息

model = MyModel()
model = SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model.cuda(), device_ids=[rank])
```

### 梯度同步与 no_sync

```python
# DDP 默认在每次 backward 后自动同步梯度
# 使用 no_sync 可以跳过梯度同步（用于梯度累积）

model = DDP(model)

for i, (data, targets) in enumerate(loader):
    if i % accumulation_steps != 0:
        with model.no_sync():
            outputs = model(data)
            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()
    else:
        outputs = model(data)
        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 分布式通信原语

```python
# 广播（从 rank 0 广播到所有进程）
tensor = torch.ones(10).cuda()
dist.broadcast(tensor, src=0)
print(f"Rank {rank}: {tensor}")

# 聚合（所有进程聚合到 rank 0）
tensor = torch.ones(10).cuda() * rank
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"Reduced tensor (sum): {tensor}")

# All-Reduce（所有进程都获得聚合结果）
tensor = torch.ones(10).cuda() * rank
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {rank} after all_reduce: {tensor}")

# All-Gather（所有进程收集所有 Tensor）
tensor_list = [torch.zeros(10).cuda() for _ in range(world_size)]
dist.all_gather(tensor_list, torch.ones(10).cuda() * rank)
print(f"Rank {rank} all_gather: {tensor_list}")

# Gather（仅目标进程收集）
if rank == 0:
    gather_list = [torch.zeros(10).cuda() for _ in range(world_size)]
else:
    gather_list = None
dist.gather(torch.ones(10).cuda() * rank, gather_list, dst=0)

# Barrier（同步点）
dist.barrier()  # 等待所有进程到达此点
```

## 分布式验证

```python
@torch.no_grad()
def distributed_evaluate(model, val_loader, device, rank, world_size):
    """分布式验证"""
    model.eval()
    total_correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)

    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().float()
        total_samples += targets.size(0)

    # All-Reduce 收集全局结果
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    accuracy = (total_correct / total_samples).item()

    if rank == 0:
        print(f"Validation accuracy: {accuracy:.4f} "
              f"({total_correct.item():.0f}/{total_samples.item():.0f})")

    return accuracy
```

## 模型并行（Model Parallelism）

当模型太大无法放入单张 GPU 时，可以使用模型并行：

```python
class ModelParallelModel(nn.Module):
    """模型并行：不同层分配到不同 GPU"""

    def __init__(self):
        super().__init__()
        # 前半部分在 GPU 0
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU()
        ).to('cuda:0')

        # 后半部分在 GPU 1
        self.features2 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10)
        ).to('cuda:1')

    def forward(self, x):
        x = self.features1(x)      # GPU 0
        x = x.to('cuda:1')         # 传输到 GPU 1
        return self.features2(x)   # GPU 1

# 使用
model = ModelParallelModel()
# 输入数据从 GPU 0 开始
x = torch.randn(32, 3, 32, 32, device='cuda:0')
output = model(x)
```

## 流水线并行（Pipeline Parallelism）

```python
from torch.distributed.pipeline.sync import Pipe

# 将模型切分为多个阶段
class PipelineStage1(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

class PipelineStage2(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10),
        )

# 构建流水线
# balance = [4, 5]  # 每个阶段的层数
# devices = [0, 1]   # 设备分配
# model = Pipe(nn.Sequential(PipelineStage1(), PipelineStage2()),
#              balance=balance, devices=devices)
```

## ZeRO 优化（DeepSpeed）

ZeRO（Zero Redundancy Optimizer）是 DeepSpeed 的核心技术，通过分片优化器状态来减少内存占用：

```python
# 安装 DeepSpeed
# pip install deepspeed

# 配置文件（ds_config.json）
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO-2: 分片优化器状态和梯度
        # "stage": 3  # ZeRO-3: 分片参数、梯度和优化器状态
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}

# 使用 DeepSpeed
# model_engine, optimizer, trainloader, _ = deepspeed.initialize(
#     args=ds_args,
#     model=model,
#     model_parameters=model.parameters(),
#     training_data=train_dataset
# )

# for epoch in range(num_epochs):
#     for batch in trainloader:
#         loss = model_engine(batch)
#         model_engine.backward(loss)
#         model_engine.step()
```

### ZeRO 阶段对比

| 阶段 | 分片内容 | 内存节省 | 通信开销 |
|------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | 2x | 低 |
| ZeRO-2 | 优化器状态 + 梯度 | 4x | 中 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | Nx | 高 |

## 总结

本文深入学习了分布式训练技术：

- **DDP 核心**：进程组、分布式采样器、模型包装
- **torchrun**：推荐的 DDP 启动方式
- **同步 BatchNorm**：跨卡同步统计信息
- **分布式通信**：broadcast、reduce、all_reduce、all_gather
- **模型并行**：将大模型切分到多张 GPU
- **流水线并行**：层级别的并行策略
- **ZeRO 优化**：DeepSpeed 的内存优化技术

在下一篇文章中，我们将学习 TorchVision 库，了解如何使用它进行计算机视觉任务的开发。

[上一篇：GPU 训练加速](./53-gpu-training.md) | [下一篇：TorchVision →](./55-torchvision.md)
