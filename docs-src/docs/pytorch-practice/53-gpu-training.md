---
title: GPU 训练加速
icon: monitor
order: 53
---

# GPU 训练加速

GPU 是深度学习训练的核心加速器。PyTorch 提供了从基础 GPU 使用到高级混合精度训练的完整支持。本文将学习如何高效地利用 GPU 进行模型训练，包括设备管理、多 GPU 策略和混合精度训练。

## GPU 基础：设备管理

### 检查 GPU 可用性

```python
import torch

# 基本检查
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (compiled with): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

# GPU 信息
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")

    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")

    # GPU 内存信息
    mem_allocated = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    print(f"Memory allocated: {mem_allocated / 1024**2:.2f} MB")
    print(f"Memory reserved: {mem_reserved / 1024**2:.2f} MB")
```

### Tensor 和模型的 GPU 迁移

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tensor 迁移到 GPU
x = torch.randn(3, 4)
x_gpu = x.to(device)
x_gpu2 = x.cuda()  # 等价写法

# 模型迁移到 GPU
model = torch.nn.Linear(10, 5)
model_gpu = model.to(device)

# 完整训练循环中的 GPU 使用
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)

for data, targets in train_loader:
    data = data.to(device)
    targets = targets.to(device)

    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### 便捷设备获取方式

```python
# 方式 1：传统写法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 方式 2：字符串直接
device = "cuda" if torch.cuda.is_available() else "cpu"

# 方式 3：指定具体 GPU
device = "cuda:0"  # 第一张 GPU
device = "cuda:1"  # 第二张 GPU

# 方式 4：使用环境变量
# CUDA_VISIBLE_DEVICES=0,1 python train.py
device = torch.device("cuda")  # 只使用可见的 GPU
```

## CUDA 内存管理

### 内存监控

```python
def get_gpu_memory_info():
    """获取 GPU 内存使用情况"""
    if not torch.cuda.is_available():
        return "No GPU available"

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory

    return {
        'allocated_mb': allocated / 1024**2,
        'reserved_mb': reserved / 1024**2,
        'total_mb': total / 1024**2,
        'allocated_pct': allocated / total * 100,
    }

# 训练前重置内存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 训练后查看峰值内存
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1024**2:.2f} MB")
```

### 内存优化技巧

```python
# 1. 使用 non_blocking 实现异步传输
data = data.to(device, non_blocking=True)

# 2. 及时释放不需要的中间变量
with torch.no_grad():
    output = model(input)
    predictions = output.argmax(dim=1)
    # del output  # 如果不再需要，显式删除
```

| 优化技巧 | 效果 | 适用场景 |
|----------|------|---------|
| `non_blocking=True` | 加速 CPU->GPU 传输 | 小 batch、频繁传输 |
| `torch.cuda.empty_cache()` | 释放缓存内存 | 多阶段训练 |
| `torch.cuda.amp` | 减少 50% 内存 | GPU 训练 |
| `gradient checkpointing` | 大幅减少内存 | 超大模型 |
| 减小 batch size | 直接减少内存 | 内存不足时 |

## 混合精度训练（AMP）

自动混合精度（Automatic Mixed Precision，AMP）使用 float16/bfloat16 进行前向和反向传播，同时在 float32 中维护 master weights，可以大幅提升训练速度并减少内存占用。

### AMP 基础用法

```python
from torch.cuda.amp import autocast, GradScaler

# 创建 scaler
scaler = GradScaler()

# 训练循环中的 AMP
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().cuda()

for epoch in range(num_epochs):
    model.train()
    for data, targets in train_loader:
        data, targets = data.cuda(), targets.cuda()

        optimizer.zero_grad(set_to_none=True)

        # 使用 autocast 上下文管理器
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        # 缩放梯度并反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪（需要先 unscale）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
```

### bfloat16 vs float16

| 特性 | float16 (FP16) | bfloat16 (BF16) |
|------|---------------|-----------------|
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 动态范围 | 较小 | 与 float32 相同 |
| 梯度溢出风险 | 较高 | 较低 |
| GPU 支持 | 所有 CUDA GPU | Ampere 架构及以上 |
| 推荐场景 | 老 GPU | RTX 30 系列及以上 |

```python
# 使用 bfloat16
with autocast(dtype=torch.bfloat16):
    outputs = model(data)
    loss = criterion(outputs, targets)

# 动态选择
if torch.cuda.is_bf16_supported():
    print("BF16 is supported!")
    amp_dtype = torch.bfloat16
else:
    amp_dtype = torch.float16
```

### AMP 性能对比

```python
import time

def benchmark_training(use_amp=False, dtype=torch.float16):
    """基准测试训练速度"""
    model = torch.nn.Linear(1024, 512).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler(enabled=use_amp)

    x = torch.randn(256, 1024, device='cuda')
    target = torch.randn(256, 512, device='cuda')

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(dtype=dtype):
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, target)
            scaler.scale(loss).backward()
        else:
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()

        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"{'AMP (FP16)' if use_amp else 'FP32'}: {elapsed:.2f}s "
          f"({elapsed/100*1000:.2f}ms/step)")

# benchmark_training(use_amp=False)
# benchmark_training(use_amp=True, dtype=torch.float16)
```

## 梯度检查点（Gradient Checkpointing）

梯度检查点通过重新计算中间激活值来减少内存使用，以计算时间换取内存空间：

```python
# 方式 1：对整个模型启用
model = MyModel()
model.gradient_checkpointing_enable()  # 部分模型支持

# 方式 2：使用 torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint

class CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)

    def forward(self, x):
        # 使用 checkpoint 包装需要重新计算的部分
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = self.layer3(x)
        return x
```

### 梯度检查点效果

| 指标 | 不使用检查点 | 使用检查点 |
|------|------------|-----------|
| 内存 | 100% | ~30-50% |
| 计算时间 | 100% | ~120-150% |
| 最大 batch size | 较小 | 可增大 2-3 倍 |

## DataParallel：单机多卡训练

```python
# DataParallel（DP）
# 简单但不推荐用于生产环境
model = MyModel()
model = nn.DataParallel(model)  # 使用所有可用 GPU
model = model.cuda()

# 或指定 GPU
# model = nn.DataParallel(model, device_ids=[0, 1, 2])

# 训练循环不需要修改
for data, targets in train_loader:
    data, targets = data.cuda(), targets.cuda()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### DataParallel 的局限性

| 特性 | DataParallel | DistributedDataParallel |
|------|-------------|----------------------|
| 并行方式 | 进程内多线程 | 多进程 |
| 多机支持 | 否 | 是 |
| Python GIL 影响 | 是 | 否 |
| 推荐场景 | 简单调试 | 生产环境 |
| 性能 | 较低 | 较高 |

💡 **提示**：`nn.DataParallel` 简单但性能有限。在生产环境中，建议使用 `DistributedDataParallel`（DDP）。`DataParallel` 适合快速原型验证和调试。

## 分布式训练入门（DDP）

```python
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    """初始化 DDP 进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理 DDP 进程组"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """DDP 训练函数"""
    setup_ddp(rank, world_size)

    # 创建模型并分配到指定 GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 使用 DistributedSampler
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10),
        torch.randint(0, 10, (1000,))
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=sampler
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # 每个 epoch 重新打乱
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(rank), targets.to(rank)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

    cleanup_ddp()

# 启动 DDP 训练
# torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

💡 **提示**：推荐使用 `torchrun` 命令行启动 DDP 训练：
```bash
torchrun --nproc_per_node=4 train.py
```
在代码中使用 `torch.distributed` API：
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

## 性能分析工具

### torch.cuda.Event 计时

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# 要测量的代码
output = model(input)
end.record()

torch.cuda.synchronize()  # 等待 GPU 操作完成
elapsed_ms = start.elapsed_time(end)
print(f"Elapsed time: {elapsed_ms:.2f} ms")
```

### torch.profiler 性能分析

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, targets) in enumerate(train_loader):
        with record_function("train_step"):
            data, targets = data.cuda(), targets.cuda()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        prof.step()

# 查看分析结果
# print(prof.key_averages().table(sort_by="cuda_time_total"))
# prof.export_chrome_trace("trace.json")  # 在 Chrome 中查看
```

## 总结

本文学习了 GPU 训练的核心技术：

- **设备管理**：GPU 可用性检查、Tensor 和模型迁移
- **内存管理**：监控、优化和异步传输
- **混合精度训练（AMP）**：大幅提升训练速度、减少内存占用
- **bfloat16 vs float16**：根据硬件选择合适的精度
- **梯度检查点**：以计算换内存的策略
- **DataParallel**：简单多卡训练方案
- **DDP 分布式训练**：生产环境推荐的多 GPU 方案
- **性能分析**：使用 Profiler 定位性能瓶颈

掌握了 GPU 训练技术后，在下一篇文章中，我们将深入探讨多机多卡的分布式训练策略。

[上一篇：模型保存与加载](./52-model-save-load.md) | [下一篇：分布式训练 →](./54-distributed-training.md)
