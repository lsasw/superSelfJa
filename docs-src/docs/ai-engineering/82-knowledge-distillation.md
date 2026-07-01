---
title: 知识蒸馏技术实战
icon: lightbulb
order: 6
---

# 82. 知识蒸馏技术实战

在前面的文档中，我们学习了模型量化和模型剪枝两种模型压缩技术。本文将介绍第三种重要的压缩方法——知识蒸馏（Knowledge Distillation）。与前两种直接修改模型参数的方法不同，知识蒸馏通过教师-学生模型框架，让小型学生模型学习大型教师模型的"知识"，在保持较高精度的同时显著减小模型体积。

## 知识蒸馏基础

### 什么是知识蒸馏

知识蒸馏由 Hinton 等人在 2015 年提出，其核心思想是让一个小模型（学生）学习一个大模型（教师）的输出行为，而不仅仅是学习原始标签。教师模型产生的"软标签"（soft labels）包含了丰富的暗知识（dark knowledge），这些信息有助于学生模型更好地学习。

### 蒸馏原理

```
教师模型（大模型）                    学生模型（小模型）
     │                                    │
     ├─── 硬标签 (Ground Truth) ──────────▶│
     │                                    │
     ├─── 软标签 (Soft Labels) ───────────▶│
     │        (带温度参数 T)               │ 蒸馏损失
     │                                    │
     └─── 中间层特征 (可选) ──────────────▶│ 特征匹配损失
                                          │
                                   最终损失 = α * 蒸馏损失 + β * 硬标签损失 + γ * 特征损失
```

### 损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=5.0,
    alpha=0.5
):
    """
    知识蒸馏损失
    
    Args:
        student_logits: 学生模型的输出 logits
        teacher_logits: 教师模型的输出 logits
        labels: 真实标签
        temperature: 温度参数，控制软标签的平滑程度
        alpha: 蒸馏损失的权重
    
    Returns:
        总损失
    """
    # 硬标签损失（交叉熵）
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 软标签损失（KL 散度）
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    soft_loss = F.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean"
    ) * (temperature ** 2)
    
    # 总损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss
```

### 温度参数的作用

| 温度 T | 效果 | 软标签分布 | 适用场景 |
|--------|------|------------|----------|
| T=1 | 无平滑 | 接近原始输出 | 硬蒸馏 |
| T=2-5 | 适度平滑 | 保留类别间关系 | 常规蒸馏 |
| T=10-20 | 强平滑 | 接近均匀分布 | 细粒度任务 |
| T>50 | 过度平滑 | 信息损失 | 不推荐 |

```python
import numpy as np

def visualize_temperature_effect():
    """可视化温度参数对软标签的影响"""
    logits = torch.tensor([10.0, 5.0, 1.0, 0.5])
    
    temperatures = [1, 2, 5, 10, 20]
    
    print("原始 logits:", logits.tolist())
    print("原始 softmax:", F.softmax(logits, dim=0).tolist())
    print()
    
    for T in temperatures:
        soft = F.softmax(logits / T, dim=0)
        print(f"T={T:2d}: {soft.tolist()}")

visualize_temperature_effect()
```

## PyTorch 知识蒸馏实现

### 基础蒸馏框架

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        device,
        temperature=5.0,
        alpha=0.5,
        lr=1e-4
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr
        
        # 教师模型不需要梯度
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def train(self, num_epochs=50):
        """执行知识蒸馏训练"""
        optimizer = optim.AdamW(self.student.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion_ce = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        history = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.student.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 学生模型前向传播
                student_logits = self.student(images)
                
                # 教师模型前向传播（不计算梯度）
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                
                # 计算蒸馏损失
                loss = self._compute_distillation_loss(
                    student_logits, teacher_logits, labels
                )
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(student_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = correct / total
            avg_loss = train_loss / len(self.train_loader)
            
            # 验证阶段
            val_acc = self._evaluate()
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            })
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.student.state_dict(), "best_student.pth")
                print(f"  保存最佳模型，Val Acc: {val_acc:.4f}")
        
        return history
    
    def _compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """计算蒸馏损失"""
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 软标签损失
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    
    def _evaluate(self):
        """评估学生模型"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total

# 使用示例
# teacher_model = models.resnet50(pretrained=True).to(device)
# student_model = models.resnet18().to(device)
# trainer = DistillationTrainer(teacher_model, student_model, train_loader, val_loader, device)
# history = trainer.train(num_epochs=50)
```

### 特征蒸馏

```python
class FeatureDistillationTrainer(DistillationTrainer):
    """特征蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        device,
        temperature=5.0,
        alpha=0.5,
        beta=0.5,  # 特征蒸馏权重
        lr=1e-4
    ):
        super().__init__(
            teacher_model, student_model, train_loader, val_loader,
            device, temperature, alpha, lr
        )
        self.beta = beta
        
        # 特征投影层（对齐教师和学生的特征维度）
        self.feature_projections = nn.ModuleDict()
    
    def register_feature_projection(self, name, teacher_dim, student_dim):
        """注册特征投影层"""
        self.feature_projections[name] = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim)
        )
    
    def _compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        labels,
        student_features=None,
        teacher_features=None
    ):
        """计算蒸馏损失（包含特征蒸馏）"""
        # 输出蒸馏损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)
        
        output_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        # 特征蒸馏损失
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            for name in student_features:
                if name in self.feature_projections:
                    projected_student = self.feature_projections[name](student_features[name])
                    feature_loss += F.mse_loss(projected_student, teacher_features[name])
        
        return output_loss + self.beta * feature_loss

# 使用示例
class IntermediateFeatureExtractor(nn.Module):
    """中间特征提取器"""
    
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        
        # 注册 hook
        self._register_hooks()
    
    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.features[name] = output[0]
                else:
                    self.features[name] = output
            return hook
        
        # 假设我们想提取特定层的输出
        # 这里需要根据具体模型结构调整
        pass
    
    def forward(self, x):
        self.features.clear()
        output = self.model(x)
        return output, self.features
```

## 在线知识蒸馏

```python
class OnlineDistillationTrainer:
    """在线知识蒸馏（多个学生模型互相学习）"""
    
    def __init__(self, student_models, train_loader, val_loader, device, lr=1e-4):
        self.students = student_models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.num_students = len(student_models)
    
    def train(self, num_epochs=50):
        """在线蒸馏训练"""
        optimizers = [
            optim.AdamW(model.parameters(), lr=self.lr)
            for model in self.students
        ]
        
        for epoch in range(num_epochs):
            for student_idx, student in enumerate(self.students):
                student.train()
                optimizer = optimizers[student_idx]
                
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # 当前学生的输出
                    current_logits = student(images)
                    
                    # 其他学生的平均输出作为"教师"
                    with torch.no_grad():
                        other_logits = []
                        for i, other_student in enumerate(self.students):
                            if i != student_idx:
                                other_logits.append(other_student(images))
                        teacher_logits = torch.stack(other_logits).mean(dim=0)
                    
                    # 计算蒸馏损失
                    loss = self._compute_distillation_loss(
                        current_logits, teacher_logits, labels
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # 验证
            for idx, student in enumerate(self.students):
                acc = self._evaluate_student(student)
                print(f"Student {idx+1} Val Acc: {acc:.4f}")
    
    def _compute_distillation_loss(self, student_logits, teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)
        soft_student = F.log_softmax(student_logits / 5.0, dim=1)
        soft_teacher = F.softmax(teacher_logits / 5.0, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * 25
        return 0.5 * soft_loss + 0.5 * hard_loss
    
    def _evaluate_student(self, student):
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = student(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

# 使用示例
# student_models = [
#     models.resnet18().to(device),
#     models.resnet18().to(device),
#     models.resnet18().to(device)
# ]
# online_trainer = OnlineDistillationTrainer(student_models, train_loader, val_loader, device)
# online_trainer.train(num_epochs=50)
```

## 针对 LLM 的知识蒸馏

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

class LLMKDTrainer:
    """LLM 知识蒸馏训练器"""
    
    def __init__(
        self,
        teacher_model_name,
        student_model_name,
        dataset_name,
        output_dir="./distilled_model"
    ):
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.teacher.eval()
        
        self.student = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.dataset_name = dataset_name
        self.output_dir = output_dir
    
    def prepare_dataset(self, max_samples=10000):
        """准备蒸馏数据集"""
        from datasets import load_dataset
        
        dataset = load_dataset(self.dataset_name, split="train")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, num_epochs=3, batch_size=4):
        """执行 LLM 知识蒸馏"""
        dataset = self.prepare_dataset()
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            fp16=True,
            remove_unused_columns=False,
        )
        
        class KDCallback:
            def __init__(self, teacher, temperature=2.0, alpha=0.5):
                self.teacher = teacher
                self.temperature = temperature
                self.alpha = alpha
            
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
                
                # 学生输出
                student_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                student_loss = student_outputs.loss
                
                # 教师输出
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # 计算蒸馏损失
                student_logits = student_outputs.logits
                teacher_logits = teacher_outputs.logits
                
                soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
                
                kd_loss = F.kl_div(
                    soft_student.view(-1, soft_student.size(-1)),
                    soft_teacher.view(-1, soft_teacher.size(-1)),
                    reduction="batchmean"
                ) * (self.temperature ** 2)
                
                total_loss = self.alpha * kd_loss + (1 - self.alpha) * student_loss
                
                return (total_loss, student_outputs) if return_outputs else total_loss
        
        trainer = Trainer(
            model=self.student,
            args=training_args,
            train_dataset=dataset,
        )
        
        # 替换损失计算
        kd_callback = KDCallback(self.teacher)
        
        trainer.train()
        self.student.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

# 使用示例
# kd_trainer = LLMKDTrainer(
#     teacher_model_name="meta-llama/Llama-2-70b-hf",
#     student_model_name="meta-llama/Llama-2-7b-hf",
#     dataset_name="wikitext",
#     output_dir="./llama2-7b-distilled"
# )
# kd_trainer.train(num_epochs=3)
```

## TinyLlama 蒸馏方案

```python
# TinyLlama 是一个通过知识蒸馏从 LLaMA 压缩而来的小型模型
# 参数: 1.1B vs 7B/13B

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 加载 TinyLlama
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 创建推理 pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9
)

# 推理示例
messages = [
    {"role": "system", "content": "你是一个有用的 AI 助手。"},
    {"role": "user", "content": "请解释一下什么是知识蒸馏？"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False)
result = pipe(prompt)
print(result[0]["generated_text"])

# 性能测试
import time

def benchmark(model, tokenizer, prompt, num_runs=10):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=50)
        latencies.append(time.time() - start)
    
    print(f"平均延迟: {np.mean(latencies):.3f}s")
    print(f"吞吐量: {1/np.mean(latencies):.2f} 次/秒")

benchmark(model, tokenizer, "解释一下机器学习模型部署的流程。")
```

## 蒸馏效果对比

| 教师模型 | 学生模型 | 参数量比 | 精度损失 | 加速比 |
|----------|----------|----------|----------|--------|
| ResNet-152 | ResNet-18 | 25:1 | 1.2% | 4.2x |
| BERT-Large | BERT-Base | 4:1 | 0.5% | 3.8x |
| BERT-Base | DistilBERT | 2:1 | 2.0% | 1.6x |
| BERT-Base | TinyBERT | 7:1 | 3.0% | 3.1x |
| LLaMA-2 70B | LLaMA-2 7B | 10:1 | 5-8% | 8.5x |

## 总结

知识蒸馏是一种高效的模型压缩方法，通过教师模型向学生模型传递"暗知识"来实现精度与效率的平衡。本文介绍了：

- 知识蒸馏的基本原理和温度参数的作用
- PyTorch 实现的基础蒸馏框架
- 特征蒸馏和在线蒸馏的高级方法
- LLM 知识蒸馏的具体实现
- 主流蒸馏模型的效果对比

知识蒸馏与量化、剪枝可以结合使用，形成完整的模型压缩流水线。在下一篇文档中，我们将学习 vLLM，这是一个专为 LLM 高并发推理设计的推理框架，通过创新的 PagedAttention 技术实现显著的吞吐量提升。

[下一篇：83-vLLM 高并发推理框架](83-vllm.md)
