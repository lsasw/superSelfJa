---
title: 模型保存与加载
icon: save
order: 52
---

# 模型保存与加载

在深度学习开发中，模型的保存与加载是必不可少的操作。无论是中断训练后的恢复、模型部署、还是模型复用，都需要正确处理模型的持久化。本文将系统学习 PyTorch 提供的模型保存与加载方法。

## PyTorch 序列化机制

PyTorch 使用 Python 的 pickle 模块进行序列化，核心函数是 `torch.save()` 和 `torch.load()`。

```python
import torch
import torch.nn as nn

# 保存
torch.save(obj, filepath)

# 加载
obj = torch.load(filepath, map_location='cpu')
```

### map_location 参数

| 值 | 说明 | 适用场景 |
|----|------|---------|
| `'cpu'` | 加载到 CPU | CPU 推理、跨设备加载 |
| `'cuda'` | 加载到 GPU | GPU 推理 |
| `'cuda:0'` | 加载到指定 GPU | 多 GPU 环境 |
| `None` | 加载到原设备 | 原设备可用时 |
| `dict` | 设备映射 | 跨设备迁移 |

## 保存与加载 state_dict（推荐方式）

`state_dict` 是 PyTorch 模型参数的有序字典（OrderedDict），包含所有可学习参数和缓冲区。**这是 PyTorch 官方推荐的方式。**

### 保存模型参数

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 30 * 30, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# 创建模型
model = MyModel()

# 训练后保存 state_dict
torch.save(model.state_dict(), 'model_weights.pt')
print("Model weights saved!")

# 查看 state_dict 内容
state_dict = model.state_dict()
for key, value in state_dict.items():
    print(f"  {key}: {value.shape}")
```

### 加载模型参数

```python
# 步骤 1：创建模型实例（必须有相同的架构）
model = MyModel()

# 步骤 2：加载 state_dict
state_dict = torch.load('model_weights.pt', map_location='cpu')
model.load_state_dict(state_dict)

# 步骤 3：设置评估模式
model.eval()

print("Model loaded successfully!")
```

### 严格模式加载

```python
# strict=True（默认）：key 必须完全匹配
model.load_state_dict(state_dict, strict=True)

# strict=False：允许部分匹配（迁移学习场景）
# 用于微调时，加载预训练权重到新架构
model.load_state_dict(state_dict, strict=False)

# 检查哪些 key 缺失/多余
result = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {result.missing_keys}")
print(f"Unexpected keys: {result.unexpected_keys}")
```

## 保存完整的 Checkpoint

保存训练所需的全部信息，以便恢复训练：

```python
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, filepath):
    """保存完整检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'amp_scaler_state': None,  # 如果使用 AMP
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler=None, device='cpu'):
    """加载检查点并恢复训练状态"""
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint['val_loss']

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Previous val_loss: {val_loss:.4f}")

    return start_epoch

# 使用示例
# save_checkpoint(model, optimizer, scheduler, epoch, val_loss, 'checkpoint.pth')
# start_epoch = load_checkpoint('checkpoint.pth', model, optimizer, scheduler)
```

## 模型部署：TorchScript 保存

```python
# 方式 1：通过 torch.jit.script 保存
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_scripted.pt')

# 方式 2：通过 torch.jit.trace 保存
sample_input = torch.randn(1, 3, 32, 32)
traced_model = torch.jit.trace(model, sample_input)
torch.jit.save(traced_model, 'model_traced.pt')

# 加载 TorchScript 模型
loaded_model = torch.jit.load('model_scripted.pt')
output = loaded_model(torch.randn(1, 3, 32, 32))
```

## ONNX 格式导出

ONNX（Open Neural Network Exchange）是跨框架的模型交换格式：

```python
import torch.onnx

# 导出为 ONNX
sample_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,                           # 模型
    sample_input,                    # 示例输入
    'model.onnx',                    # 输出文件
    export_params=True,              # 包含权重
    opset_version=14,                # ONNX 版本
    do_constant_folding=True,        # 常量折叠优化
    input_names=['input'],           # 输入名称
    output_names=['output'],         # 输出名称
    dynamic_axes={                   # 动态维度
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'}
    }
)

# 验证 ONNX 模型
import onnx
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model validation passed!")
```

### ONNX opset 版本选择

| opset 版本 | PyTorch 支持 | 典型场景 |
|-----------|-------------|---------|
| 11-13 | 较老 | 兼容旧版推理引擎 |
| 14-16 | 主流 | TensorRT、ONNX Runtime |
| 17-18 | 最新 | 最新算子支持 |

## 跨设备加载

```python
# GPU 训练保存，CPU 推理加载
device = torch.device('cpu')
model = MyModel()
model.load_state_dict(torch.load('model_weights.pt', map_location=device))
model.eval()

# CPU 训练保存，GPU 推理加载
device = torch.device('cuda')
model = MyModel().to(device)
model.load_state_dict(torch.load('model_weights.pt', map_location=device))
model.eval()

# 多 GPU 保存，单 GPU 加载
# 使用 DataParallel 训练
# model = nn.DataParallel(model)
# torch.save(model.module.state_dict(), 'model_dp.pt')

# 单 GPU 加载
# model = MyModel()
# model.load_state_dict(torch.load('model_dp.pt'))
```

💡 **提示**：使用 `nn.DataParallel` 训练时，state_dict 的 key 会加上 `module.` 前缀。保存时建议使用 `model.module.state_dict()`，或在加载时去掉前缀。

## 模型权重格式对比

| 格式 | 扩展名 | 跨框架 | 优化 | 典型用途 |
|------|--------|--------|------|---------|
| state_dict | .pt/.pth | 否 | 否 | PyTorch 内部 |
| TorchScript | .pt | 否 | 是 | PyTorch 部署 |
| ONNX | .onnx | 是 | 是 | 跨框架交换 |
| SavedModel | - | 否 | 是 | TensorFlow |
| GGUF | .gguf | 部分 | 是 | 量化推理 |

## 迁移学习中的权重加载

```python
def load_pretrained_weights(model, pretrained_path, strict=False):
    """加载预训练权重（迁移学习）"""
    pretrained_state = torch.load(pretrained_path, map_location='cpu')

    # 处理可能的不同 key 格式
    if 'model_state_dict' in pretrained_state:
        pretrained_state = pretrained_state['model_state_dict']
    elif 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']

    # 过滤不需要的 key
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in pretrained_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state[k] = v
        else:
            print(f"Skipping {k}: shape mismatch or key not found")

    # 加载
    model_state.update(filtered_state)
    model.load_state_dict(model_state)

    loaded_count = len(filtered_state)
    total_count = len(model_state)
    print(f"Loaded {loaded_count}/{total_count} layers from pretrained")

    return model

# 使用 torchvision 预训练模型
from torchvision import models

# 方式 1：直接加载预训练权重
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 方式 2：手动加载
resnet = models.resnet18(weights=None)
# load_pretrained_weights(resnet, 'resnet18_pretrained.pth')
```

## 微调时冻结层

```python
def freeze_layers(model, freeze_until=None):
    """冻结模型的指定层"""
    for name, param in model.named_parameters():
        if freeze_until is None:
            param.requires_grad = False
        elif freeze_until in name:
            # 冻结 freeze_until 之前的所有层
            break
        param.requires_grad = (freeze_until not in name)

# 冻结 ResNet 的卷积部分，只训练分类头
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 冻结所有参数
for param in resnet.parameters():
    param.requires_grad = False

# 只训练最后的 FC 层
resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # 替换分类头
# FC 层的参数默认 requires_grad=True

# 验证
for name, param in resnet.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
```

## 模型压缩与量化

### 权重量化

```python
# 动态量化（适合 CPU 推理）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},  # 需要量化的层类型
    dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), 'model_quantized.pt')

# 量化前后对比
def get_model_size(filepath):
    import os
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"

# 量化前
torch.save(model.state_dict(), 'model_fp32.pt')
print(f"FP32 size: {get_model_size('model_fp32.pt')}")

# 量化后
torch.save(quantized_model.state_dict(), 'model_int8.pt')
print(f"INT8 size: {get_model_size('model_int8.pt')}")
```

## 自动保存最佳模型

```python
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.best_state = None

    def __call__(self, value, model):
        if self.mode == 'min':
            is_improved = value < (self.best_value - self.min_delta)
        else:
            is_improved = value > (self.best_value + self.min_delta)

        if is_improved:
            self.best_value = value
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

# 使用
# early_stopper = EarlyStopping(patience=10, mode='min')
# for epoch in range(num_epochs):
#     val_loss = train_one_epoch(...)
#     if early_stopper(val_loss, model):
#         print("Early stopping!")
#         break
# model.load_state_dict(early_stopper.best_state)
```

## 总结

本文学习了 PyTorch 模型保存与加载的完整知识：

- **state_dict**：官方推荐的模型参数保存方式
- **完整 Checkpoint**：保存训练状态以便恢复
- **TorchScript**：用于部署的序列化格式
- **ONNX 导出**：跨框架模型交换的标准格式
- **跨设备加载**：使用 `map_location` 处理不同设备
- **迁移学习**：预训练权重的加载与微调策略
- **模型量化**：减小模型体积、加速 CPU 推理

在下一篇文章中，我们将学习如何在 GPU 上进行高效训练，包括多 GPU 策略和混合精度训练。

[上一篇：训练循环 Training Loop](./51-training-loop.md) | [下一篇：GPU 训练 →](./53-gpu-training.md)
