---
title: 模型剪枝技术实战
icon: cut
order: 5
---

# 80. 模型剪枝技术实战

在上一篇文档中，我们深入学习了模型量化技术。量化通过降低数值精度来压缩模型，而模型剪枝则通过移除冗余参数来实现模型压缩。两者结合使用，可以显著减小模型体积并提升推理速度。本文将全面介绍模型剪枝的理论基础、主流方法和工程实践。

## 剪枝基础

### 什么是模型剪枝

模型剪枝（Model Pruning）是一种通过移除神经网络中不重要的连接、神经元或整个层来压缩模型的技术。剪枝后的模型通常具有更少的参数、更低的计算量和更快的推理速度。

### 剪枝粒度对比

| 剪枝粒度 | 说明 | 压缩率 | 硬件友好度 |
|----------|------|--------|------------|
| 非结构化剪枝 | 移除单个权重 | 最高 | 差（需要稀疏矩阵支持） |
| 结构化剪枝 | 移除滤波器/通道 | 中等 | 好（直接加速） |
| 层剪枝 | 移除整个层 | 低 | 好 |

### 剪枝流程

```
训练模型 → 评估重要性 → 剪枝 → 微调 → 重复 → 部署
     │         │         │       │
     ▼         ▼         ▼       ▼
  全精度模型   重要性评分  稀疏模型  恢复精度
```

## PyTorch 非结构化剪枝

### 基础剪枝

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 查看原始模型的参数数量和稀疏度
def print_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if "weight" in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params * 100 if total_params > 0 else 0
    print(f"总参数量: {total_params:,}")
    print(f"零参数量: {zero_params:,}")
    print(f"稀疏度: {sparsity:.2f}%")

print("剪枝前:")
print_sparsity(model)

# 对特定层应用 L1 非结构化剪枝
parameters_to_prune = [
    (model.layer1[0].conv1, "weight"),
    (model.layer1[0].conv2, "weight"),
    (model.layer2[0].conv1, "weight"),
    (model.layer2[0].conv2, "weight"),
    (model.layer3[0].conv1, "weight"),
    (model.layer3[0].conv2, "weight"),
    (model.layer4[0].conv1, "weight"),
    (model.layer4[0].conv2, "weight"),
]

# 应用 L1 非结构化剪枝，剪枝率 30%
for module, param in parameters_to_prune:
    prune.l1_unstructured(module, name=param, amount=0.3)

print("\n剪枝后:")
print_sparsity(model)

# 剪枝后需要移除 reparameterization
for module, param in parameters_to_prune:
    prune.remove(module, param)
```

### 全局剪枝

```python
# 全局 L1 剪枝（在整个模型范围内选择最小的权重）
parameters_to_prune = [
    (model.layer1[0].conv1, "weight"),
    (model.layer1[0].conv2, "weight"),
    (model.layer2[0].conv1, "weight"),
    (model.layer2[0].conv2, "weight"),
    (model.layer3[0].conv1, "weight"),
    (model.layer3[0].conv2, "weight"),
    (model.layer4[0].conv1, "weight"),
    (model.layer4[0].conv2, "weight"),
]

# 全局剪枝 40%
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4,
)

# 检查每层的稀疏度
for module, param in parameters_to_prune:
    total_params = getattr(module, param).numel()
    zero_params = (getattr(module, param) == 0).sum().item()
    sparsity = zero_params / total_params * 100
    print(f"{module.__class__.__name__}: 稀疏度 {sparsity:.2f}%")

# 移除 reparameterization
for module, param in parameters_to_prune:
    prune.remove(module, param)
```

### 迭代剪枝

```python
import copy

class IterativePruner:
    """迭代剪枝器"""
    
    def __init__(self, model, parameters_to_prune, train_loader, val_loader, device):
        self.model = model
        self.parameters_to_prune = parameters_to_prune
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def evaluate(self):
        """评估模型准确率"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    def train_finetune(self, epochs=5, lr=1e-4):
        """微调剪枝后的模型"""
        optimizer = torch.optim.Adam(
            [p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
    def iterative_prune(
        self,
        initial_amount=0.1,
        amount_increment=0.1,
        max_amount=0.6,
        finetune_epochs=5
    ):
        """执行迭代剪枝"""
        current_amount = initial_amount
        results = []
        
        while current_amount <= max_amount:
            print(f"\n--- 剪枝率: {current_amount:.1%} ---")
            
            # 剪枝
            prune.global_unstructured(
                self.parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=current_amount,
            )
            
            # 评估
            accuracy = self.evaluate()
            print(f"剪枝后准确率: {accuracy:.4f}")
            
            # 微调
            self.train_finetune(epochs=finetune_epochs)
            accuracy_after = self.evaluate()
            print(f"微调后准确率: {accuracy_after:.4f}")
            
            # 计算稀疏度
            total_params = sum(
                getattr(module, param).numel()
                for module, param in self.parameters_to_prune
            )
            zero_params = sum(
                (getattr(module, param) == 0).sum().item()
                for module, param in self.parameters_to_prune
            )
            sparsity = zero_params / total_params * 100
            
            results.append({
                "pruning_rate": current_amount,
                "accuracy": accuracy_after,
                "sparsity": sparsity
            })
            
            # 移除 reparameterization 以便下一次剪枝
            for module, param in self.parameters_to_prune:
                prune.remove(module, param)
            
            current_amount += amount_increment
        
        return results

# 使用示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)

parameters_to_prune = [
    (model.layer1[0].conv1, "weight"),
    (model.layer2[0].conv1, "weight"),
    (model.layer3[0].conv1, "weight"),
    (model.layer4[0].conv1, "weight"),
]

# pruner = IterativePruner(model, parameters_to_prune, train_loader, val_loader, device)
# results = pruner.iterative_prune()
```

## 结构化剪枝

### 通道剪枝

```python
import torch
import torch.nn as nn
import numpy as np

class ChannelPruner:
    """通道级结构化剪枝"""
    
    def __init__(self, model):
        self.model = model
    
    def get_bn_weights(self):
        """获取 BatchNorm 层的 gamma 权重"""
        bn_weights = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_weights.append(module.weight.data.clone())
        return bn_weights
    
    def prune_channels(self, prune_ratio=0.3):
        """根据 BN gamma 值剪枝通道"""
        all_bn_weights = self.get_bn_weights()
        all_weights = torch.cat([w.abs() for w in all_bn_weights])
        
        # 确定剪枝阈值
        threshold = torch.kthvalue(
            all_weights,
            int(len(all_weights) * prune_ratio)
        )[0]
        
        print(f"剪枝阈值: {threshold.item():.4f}")
        
        # 对每个 BN 层应用剪枝 mask
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                weight_copy = module.weight.data.clone()
                mask = weight_copy.abs().gt(threshold).float()
                
                # 检查是否所有通道都被剪枝（至少保留一个）
                if mask.sum() == 0:
                    print("Warning: 所有通道都被剪枝，保留最大值通道")
                    mask[weight_copy.abs().argmax()] = 1.0
                
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
    
    def apply_pruning_to_conv(self):
        """将 BN 层的剪枝 mask 应用到对应的 Conv 层"""
        prev_bn = None
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if prev_bn is not None:
                    # 获取 mask
                    mask = prev_bn.weight.data.abs().gt(1e-6).float()
                    
                    # 应用到前一个 conv 的输出通道
                    if hasattr(self, "prev_conv"):
                        self.prev_conv.weight.data *= mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        if self.prev_conv.bias is not None:
                            self.prev_conv.bias.data *= mask
                
                prev_bn = module
                self.prev_conv = None
            
            elif isinstance(module, nn.Conv2d):
                self.prev_conv = module

# 使用示例
model = models.resnet18(pretrained=True)
pruner = ChannelPruner(model)

# 剪枝前评估
# accuracy_before = evaluate(model, test_loader)
# print(f"剪枝前准确率: {accuracy_before:.4f}")

# 执行剪枝
pruner.prune_channels(prune_ratio=0.3)

# 剪枝后微调
# fine_tune(model, train_loader, epochs=20)

# 剪枝后评估
# accuracy_after = evaluate(model, test_loader)
# print(f"剪枝后准确率: {accuracy_after:.4f}")
```

### 基于 L1 正则化的结构化剪枝

```python
class L1RegularizedTrainer:
    """L1 正则化训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, sparsity_target=0.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.sparsity_target = sparsity_target
    
    def compute_bn_l1_loss(self):
        """计算 BN 层的 L1 正则化损失"""
        l1_loss = 0.0
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                l1_loss += module.weight.abs().sum()
        return l1_loss
    
    def train_with_l1(self, epochs=50, lr=1e-3, l1_lambda=1e-4):
        """带 L1 正则化的训练"""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                cls_loss = criterion(outputs, labels)
                l1_loss = self.compute_bn_l1_loss()
                
                loss = cls_loss + l1_lambda * l1_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 动态调整 L1 lambda
            if (epoch + 1) % 10 == 0:
                current_sparsity = self.get_sparsity()
                if current_sparsity < self.sparsity_target:
                    l1_lambda *= 1.5
                    print(f"  增加 L1 lambda 至: {l1_lambda:.2e}")
    
    def get_sparsity(self):
        """计算 BN 层权重的稀疏度"""
        total = 0
        zeros = 0
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                total += module.weight.numel()
                zeros += (module.weight.abs() < 1e-6).sum().item()
        return zeros / total if total > 0 else 0

# 使用示例
# trainer = L1RegularizedTrainer(model, train_loader, val_loader, device)
# trainer.train_with_l1(epochs=50)
# 剪枝
# pruner = ChannelPruner(model)
# pruner.prune_channels()
```

## Torch-Pruning 库实战

```bash
pip install torch-pruning
```

```python
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.models as models

class PruningWithTP:
    """使用 torch-pruning 库进行剪枝"""
    
    def __init__(self, model):
        self.model = model
        self.pruner = None
    
    def prune_model(
        self,
        example_input,
        pruning_ratio=0.5,
        pruning_method=tp.methods.MagnitudePruner
    ):
        """使用 torch-pruning 进行结构化剪枝"""
        # 创建重要性评估器
        importance = tp.importance.MagnitudeImportance(p=2)
        
        # 创建剪枝器
        self.pruner = tp.pruner.MetaPruner(
            model=self.model,
            example_inputs=example_input,
            importance=importance,
            pruning_ratio=pruning_ratio,
            ignored_layers=[],
            round_to=None,
        )
        
        # 查看剪枝前的模型信息
        print("剪枝前:")
        self.print_model_info()
        
        # 执行剪枝
        self.pruner.step()
        
        # 查看剪枝后的模型信息
        print("\n剪枝后:")
        self.print_model_info()
        
        return self.model
    
    def iterative_pruning(
        self,
        example_input,
        target_ratio=0.5,
        step_ratio=0.1,
        finetune_fn=None
    ):
        """迭代剪枝"""
        current_ratio = 0.0
        
        while current_ratio < target_ratio:
            print(f"\n--- 当前剪枝率: {current_ratio:.1%} ---")
            
            importance = tp.importance.MagnitudeImportance(p=2)
            pruner = tp.pruner.MetaPruner(
                model=self.model,
                example_inputs=example_input,
                importance=importance,
                pruning_ratio=step_ratio,
            )
            
            pruner.step()
            current_ratio += step_ratio
            
            # 微调
            if finetune_fn is not None:
                finetune_fn(self.model)
        
        return self.model
    
    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")

# 使用示例
model = models.resnet18(pretrained=True)
model.eval()

example_input = torch.randn(1, 3, 224, 224)

pruner = PruningWithTP(model)
pruned_model = pruner.prune_model(
    example_input=example_input,
    pruning_ratio=0.3
)

# 剪枝后需要微调
# def finetune(model):
#     # 微调逻辑
#     pass
# pruner.iterative_pruning(example_input, target_ratio=0.5, finetune_fn=finetune)
```

## LLM 剪枝

### ShortGPT 方法

```python
import torch
from transformers import AutoModelForCausalLM

def compute_layer_importance(model, calibration_data):
    """计算层重要性评分"""
    importance_scores = {}
    
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # 计算每层输出的 L1 范数
            for i, hs in enumerate(hidden_states):
                if i not in importance_scores:
                    importance_scores[i] = 0.0
                importance_scores[i] += hs.abs().mean().item()
    
    # 平均
    for k in importance_scores:
        importance_scores[k] /= len(calibration_data)
    
    return importance_scores

def prune_llm_layers(model, importance_scores, num_layers_to_prune):
    """剪枝 LLM 的层"""
    # 按重要性排序
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
    layers_to_prune = [idx for idx, _ in sorted_layers[:num_layers_to_prune]]
    
    print(f"将剪枝的层: {layers_to_prune}")
    
    # 修改模型配置
    config = model.config
    num_layers = config.num_hidden_layers
    
    # 创建新的层列表（移除被剪枝的层）
    new_layers = []
    for i, layer in enumerate(model.model.layers):
        if i not in layers_to_prune:
            new_layers.append(layer)
    
    model.model.layers = nn.ModuleList(new_layers)
    config.num_hidden_layers = len(new_layers)
    
    return model

# 使用示例
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# calibration_data = [{"input_ids": torch.randint(0, 32000, (1, 128))} for _ in range(100)]
# importance = compute_layer_importance(model, calibration_data)
# pruned_model = prune_llm_layers(model, importance, num_layers_to_prune=4)
```

## 剪枝效果评估与部署

### 模型压缩效果

```python
def analyze_pruning_results(original_model, pruned_model):
    """分析剪枝效果"""
    original_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    original_size = original_params * 4 / (1024 * 1024)  # FP32
    pruned_size = pruned_params * 4 / (1024 * 1024)
    
    print(f"原始模型: {original_params:,} 参数, {original_size:.2f} MB")
    print(f"剪枝模型: {pruned_params:,} 参数, {pruned_size:.2f} MB")
    print(f"参数减少: {(1 - pruned_params/original_params)*100:.1f}%")
    print(f"体积减少: {(1 - pruned_size/original_size)*100:.1f}%")
    
    return {
        "original_params": original_params,
        "pruned_params": pruned_params,
        "compression_ratio": original_params / pruned_params
    }

def benchmark_pruned_model(pruned_model, test_input, num_runs=100):
    """剪枝模型性能基准测试"""
    import time
    import numpy as np
    
    pruned_model.eval()
    latencies = []
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = pruned_model(test_input)
    
    # 测量
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = pruned_model(test_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    print(f"\n性能指标:")
    print(f"  平均延迟: {latencies.mean():.2f} ms")
    print(f"  P99 延迟: {np.percentile(latencies, 99):.2f} ms")
    print(f"  吞吐量: {1000/latencies.mean():.2f} 次/秒")
    
    return latencies
```

### 保存和加载剪枝模型

```python
# 保存剪枝后的模型
def save_pruned_model(model, path):
    """保存剪枝模型"""
    # 确保所有 prune 操作已被移除
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config if hasattr(model, "config") else None,
        "sparsity_info": get_sparsity_info(model)
    }, path)

def get_sparsity_info(model):
    """获取模型稀疏度信息"""
    info = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            total = param.numel()
            zeros = (param == 0).sum().item()
            info[name] = zeros / total if total > 0 else 0
    return info

# 加载剪枝模型
def load_pruned_model(model_class, path, **kwargs):
    """加载剪枝模型"""
    checkpoint = torch.load(path)
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
```

## 剪枝与量化的联合优化

```python
class PruneAndQuantizePipeline:
    """剪枝与量化联合优化流水线"""
    
    def __init__(self, model):
        self.model = model
    
    def run(
        self,
        train_loader,
        val_loader,
        prune_ratio=0.5,
        quant_bits=8,
        finetune_epochs=10
    ):
        """执行剪枝 + 量化联合优化"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 第一步：剪枝
        print("=== 第一步：模型剪枝 ===")
        pruner = ChannelPruner(self.model)
        pruner.prune_channels(prune_ratio=prune_ratio)
        
        # 第二步：微调
        print("\n=== 第二步：微调恢复精度 ===")
        # fine_tune(self.model, train_loader, epochs=finetune_epochs)
        
        # 第三步：量化
        print("\n=== 第三步：模型量化 ===")
        self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准
        # calibrate(self.model, train_loader)
        
        torch.quantization.convert(self.model, inplace=True)
        
        # 最终评估
        print("\n=== 最终模型信息 ===")
        print(f"参数压缩比: {1/(1-prune_ratio):.1f}x")
        print(f"量化精度: {quant_bits} bit")
        print(f"总压缩比: {1/(1-prune_ratio) * 32/quant_bits:.1f}x")
        
        return self.model

# 使用示例
# model = models.resnet18(pretrained=True)
# pipeline = PruneAndQuantizePipeline(model)
# optimized_model = pipeline.run(
#     train_loader=train_loader,
#     val_loader=val_loader,
#     prune_ratio=0.4,
#     quant_bits=8
# )
```

## 总结

模型剪枝是深度学习模型压缩的重要手段。本文涵盖了：

- 剪枝的基本概念：非结构化剪枝、结构化剪枝、迭代剪枝
- PyTorch 内置剪枝 API 的使用方法
- 基于 L1 正则化的结构化通道剪枝
- torch-pruning 库的实战应用
- LLM 剪枝方法和层重要性评估
- 剪枝与量化的联合优化流水线

剪枝和量化是互补的技术：剪枝减少了参数数量，量化降低了每个参数的存储开销。在实际应用中，通常先进行剪枝，再进行量化，可以获得最佳的压缩效果。在下一篇文档中，我们将学习知识蒸馏技术，了解如何通过教师-学生模型框架进一步压缩模型。

[下一篇：82-知识蒸馏技术实战](82-knowledge-distillation.md)
