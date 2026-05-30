---
title: 模型量化技术详解
icon: compress-arrows-alt
order: 4
---

# 79. 模型量化技术详解

在前一篇文档中，我们学习了 TensorRT 高性能推理优化，其中 INT8 量化是 TensorRT 最重要的加速手段之一。本文将深入探讨模型量化的完整技术体系，从理论基础到工程实践，涵盖 PTQ、QAT 等核心量化方法，以及在 PyTorch、ONNX Runtime 和 TensorRT 中的具体实现。

## 量化基础

### 什么是模型量化

模型量化是将深度学习模型中的浮点权重和激活值转换为低精度表示（如 INT8、INT4）的技术。量化后的模型具有以下优势：

| 维度 | FP32 | FP16 | INT8 | INT4 |
|------|------|------|------|------|
| 存储大小 | 4 bytes/参数 | 2 bytes/参数 | 1 byte/参数 | 0.5 bytes/参数 |
| 显存带宽 | 基准 | 2x | 4x | 8x |
| Tensor Core 支持 | 是 | 是 | 是（较新型号） | 有限 |
| 精度损失 | 无 | 极小 | 可控（1-3%） | 较大 |

### 量化原理

量化的核心思想是建立浮点值与整数之间的线性映射关系：

```
量化公式: q = round(r / S) + Z
反量化公式: r = S * (q - Z)

其中:
- r: 浮点值 (real value)
- q: 量化后的整数值 (quantized value)
- S: 缩放因子 (scale)
- Z: 零点 (zero point)
```

```python
import numpy as np

def quantize_symmetric(tensor, num_bits=8):
    """对称量化"""
    max_val = np.abs(tensor).max()
    scale = max_val / (2 ** (num_bits - 1) - 1)
    
    # 量化
    q_tensor = np.round(tensor / scale)
    q_tensor = np.clip(q_tensor, -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    
    return q_tensor.astype(np.int8), scale

def quantize_asymmetric(tensor, num_bits=8):
    """非对称量化"""
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val == min_val:
        return np.zeros_like(tensor, dtype=np.int8), 1.0, 0
    
    scale = (max_val - min_val) / (2 ** num_bits - 1)
    zero_point = np.round(-min_val / scale)
    zero_point = int(np.clip(zero_point, 0, 2 ** num_bits - 1))
    
    # 量化
    q_tensor = np.round(tensor / scale) + zero_point
    q_tensor = np.clip(q_tensor, 0, 2 ** num_bits - 1)
    
    return q_tensor.astype(np.uint8), scale, zero_point

def dequantize_symmetric(q_tensor, scale):
    """反量化"""
    return q_tensor.astype(np.float32) * scale

def dequantize_asymmetric(q_tensor, scale, zero_point):
    """反量化"""
    return scale * (q_tensor.astype(np.float32) - zero_point)

# 示例
weights = np.random.randn(64, 64).astype(np.float32)

# 对称量化
q_weights, scale = quantize_symmetric(weights)
recovered_weights = dequantize_symmetric(q_weights, scale)
print(f"对称量化误差: {np.abs(weights - recovered_weights).mean():.6f}")

# 非对称量化
q_weights, scale, zp = quantize_asymmetric(weights)
recovered_weights = dequantize_asymmetric(q_weights, scale, zp)
print(f"非对称量化误差: {np.abs(weights - recovered_weights).mean():.6f}")
```

### 量化粒度

| 量化粒度 | 说明 | 精度 | 开销 |
|----------|------|------|------|
| 逐层量化（Layer-wise） | 整层共享一个 scale | 较低 | 最小 |
| 逐通道量化（Channel-wise） | 每个输出通道独立 scale | 较高 | 适中 |
| 逐张量化（Tensor-wise） | 每个张量独立 scale | 最高 | 最大 |

## PyTorch 量化

### 动态量化（Dynamic Quantization）

动态量化在推理时对激活值进行动态量化，适用于 RNN 和 Linear 层：

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    """文本分类模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        out = torch.relu(self.fc1(hidden))
        out = self.fc2(out)
        return out

# 创建模型
model = TextClassifier(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    num_classes=10
)

# 查看原始模型大小
def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 * 1024)  # FP32 每参数 4 bytes

original_size = get_model_size(model)
print(f"原始模型大小: {original_size:.2f} MB")

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},  # 要量化的层
    dtype=torch.qint8
)

quantized_size = get_model_size(quantized_model)
print(f"量化后模型大小: {quantized_size:.2f} MB")
print(f"压缩比: {original_size / quantized_size:.1f}x")

# 推理测试
input_ids = torch.randint(0, 10000, (2, 128))
with torch.no_grad():
    output_original = model(input_ids)
    output_quantized = quantized_model(input_ids)

# 检查精度损失
mse = torch.nn.functional.mse_loss(output_original.float(), output_quantized.float())
print(f"MSE 误差: {mse.item():.6f}")
```

### 静态量化（Post-Training Quantization, PTQ）

静态量化在量化过程中需要校准数据集来确定激活值的量化参数：

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 定义量化配置
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# 准备量化
torch.quantization.prepare(model, inplace=True)

# 校准阶段
def calibrate(model, calibration_loader, num_batches=10):
    """使用校准数据集确定量化参数"""
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            model(images)
            print(f"校准批次 {i+1}/{num_batches}")

# 假设 calibration_loader 是一个 DataLoader
# calibrate(model, calibration_loader)

# 转换量化模型
torch.quantization.convert(model, inplace=True)

# 保存量化模型
torch.save(model.state_dict(), "resnet18_quantized.pt")
```

### 感知训练量化（Quantization-Aware Training, QAT）

QAT 在训练过程中模拟量化效果，可以最大程度减少精度损失：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class QATResNet:
    """QAT 训练器"""
    
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prepare_qat(self):
        """准备 QAT"""
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(self.model, inplace=True)
        self.model.to(self.device)
    
    def train_qat(self, train_loader, val_loader, num_epochs=10, lr=1e-4):
        """执行 QAT 训练"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
        
        # 转换为量化模型
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model, best_acc

# 使用示例
model = models.resnet18(pretrained=True)
model.eval()

qat_trainer = QATResNet(model)
qat_trainer.prepare_qat()
# quantized_model, best_acc = qat_trainer.train_qat(train_loader, val_loader)
```

### 自定义 QConfig

```python
import torch

# 自定义量化配置
custom_qconfig = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=True
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )
)

# 应用到模型
model.qconfig = custom_qconfig
```

## ONNX Runtime 量化

```python
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    quantize_qat,
    QuantType,
    CalibrationDataReader
)
import onnx

class ImageClassificationDataReader(CalibrationDataReader):
    """校准数据读取器"""
    
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.enum_data = None
    
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = self._load_data()
        return next(self.enum_data, None)
    
    def _load_data(self):
        import os
        import cv2
        import numpy as np
        
        image_files = os.listdir(self.data_dir)[:1000]
        batches = len(image_files) // self.batch_size
        
        for i in range(batches):
            batch_files = image_files[i*self.batch_size:(i+1)*self.batch_size]
            images = []
            
            for f in batch_files:
                img = cv2.imread(os.path.join(self.data_dir, f))
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
            
            yield {"input": np.array(images)}

# 动态量化（无需校准数据）
quantize_dynamic(
    model_input="resnet18.onnx",
    model_output="resnet18_dynamic_quant.onnx",
    weight_type=QuantType.QUInt8,
    optimize_model=True,
    per_channel=True
)

# 静态量化（需要校准数据）
dr = ImageClassificationDataReader("/path/to/calibration/images")
quantize_static(
    model_input="resnet18.onnx",
    model_output="resnet18_static_quant.onnx",
    calibration_data_reader=dr,
    quant_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    per_channel=True,
    reduce_range=False
)

# QAT 量化
quantize_qat(
    model_input="resnet18_qat.onnx",
    model_output="resnet18_qat_quant.onnx",
    weight_type=QuantType.QUInt8
)
```

## LLM 量化方案

### 4-bit 量化（QLoRA 式量化）

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # 双重量化
    bnb_4bit_quant_type="nf4",         # NF4 量化类型
    bnb_4bit_compute_dtype=torch.float16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 检查显存占用
for name, param in model.named_parameters():
    if param.device.type == "cuda":
        print(f"{name}: {param.shape}, dtype={param.dtype}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params / 1e9:.2f}B")
```

### GPTQ 量化

```python
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GPTQ 配置
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="wikitext2",
    desc_act=False,
)

# 量化模型
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=gptq_config
)

# 保存量化模型
quantized_model.save_pretrained("./llama2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama2-7b-gptq-4bit")

# 加载量化模型
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "./llama2-7b-gptq-4bit",
    device="cuda:0",
    use_triton=True
)
```

### AWQ 量化

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"

# 加载模型和分词器
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 执行量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized("./llama2-7b-awq-4bit")
tokenizer.save_pretrained("./llama2-7b-awq-4bit")
```

## 量化效果评估

### 精度对比

```python
def evaluate_quantization(original_model, quantized_model, test_loader):
    """评估量化前后的精度差异"""
    def evaluate_model(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    original_acc = evaluate_model(original_model, test_loader)
    quantized_acc = evaluate_model(quantized_model, test_loader)
    
    print(f"原始模型准确率: {original_acc:.4f}")
    print(f"量化模型准确率: {quantized_acc:.4f}")
    print(f"精度损失: {(original_acc - quantized_acc) * 100:.2f}%")
    
    return original_acc, quantized_acc
```

### 性能对比

```python
import time
import numpy as np

def benchmark_inference(model, input_data, num_runs=100):
    """推理性能基准测试"""
    model.eval()
    latencies = []
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # 测量
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    return {
        "mean": latencies.mean(),
        "median": np.median(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "throughput": 1000 / latencies.mean()
    }

# 对比测试
original_results = benchmark_inference(original_model, test_input)
quantized_results = benchmark_inference(quantized_model, test_input)

print("原始模型性能:")
for metric, value in original_results.items():
    print(f"  {metric}: {value:.2f}")

print("量化模型性能:")
for metric, value in quantized_results.items():
    print(f"  {metric}: {value:.2f}")

print(f"加速比: {quantized_results['throughput'] / original_results['throughput']:.2f}x")
```

## 量化策略选择指南

| 场景 | 推荐方案 | 精度损失 | 加速比 |
|------|----------|----------|--------|
| RNN/Linear 模型 | 动态量化 | <1% | 2-3x |
| CNN 分类模型 | PTQ 静态量化 | 1-3% | 3-4x |
| 对精度要求高 | QAT | <1% | 3-4x |
| LLM 部署 | GPTQ/AWQ 4-bit | 2-5% | 4-6x |
| 边缘设备 | INT8 + 剪枝 | 3-8% | 5-10x |

## 总结

模型量化是实现高效 AI 部署的核心技术。本文系统介绍了：

- 量化的数学原理：对称量化、非对称量化、量化粒度
- PyTorch 中的三种量化方式：动态量化、静态 PTQ、QAT
- ONNX Runtime 量化工具的使用方法
- LLM 专用量化方案：4-bit 量化、GPTQ、AWQ
- 量化效果评估方法和策略选择指南

量化技术的核心是在精度和效率之间寻找最佳平衡点。在下一篇文档中，我们将学习另一种重要的模型压缩技术——模型剪枝，了解如何通过移除冗余参数来进一步减小模型体积。

[下一篇：80-模型剪枝技术实战](80-model-pruning.md)
