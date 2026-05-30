---
title: 训练循环 Training Loop
icon: refresh-cw
order: 51
---

# 训练循环 Training Loop

训练循环是深度学习代码的核心骨架，它将数据加载、前向传播、损失计算、反向传播和参数更新有机地串联起来。本文将学习如何编写高效、健壮的训练循环，并掌握训练过程中的最佳实践。

## 标准训练循环模板

一个标准的训练循环包含以下几个步骤：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

class TrainingManager:
    """训练管理器，封装完整的训练流程"""

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_one_epoch(self, optimizer, criterion, epoch):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # 1. 数据迁移到设备
            data, targets = data.to(self.device), targets.to(self.device)

            # 2. 清零梯度
            optimizer.zero_grad(set_to_none=True)

            # 3. 前向传播
            outputs = self.model(data)
            loss = criterion(outputs, targets)

            # 4. 反向传播
            loss.backward()

            # 5. 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 6. 参数更新
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印进度
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        self.history['train_loss'].append(avg_loss)
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, criterion, epoch):
        """验证/评估"""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for data, targets in self.val_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            outputs = self.model(data)
            loss = criterion(outputs, targets)

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = epoch_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        return avg_loss, accuracy

    def train(self, num_epochs, optimizer, criterion, scheduler=None):
        """完整训练流程"""
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)

            # 训练
            train_loss, train_acc = self.train_one_epoch(optimizer, criterion, epoch)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

            # 验证
            val_loss, val_acc = self.validate(criterion, epoch)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # 学习率调度
            if scheduler is not None:
                scheduler.step(val_loss)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_model.pt')
                print(f"  Saved best model (val_acc: {val_acc:.2f}%)")

        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
        return self.history
```

## 完整使用示例

```python
# 创建示例数据和模型
def create_sample_data():
    X_train = torch.randn(5000, 784)
    y_train = torch.randint(0, 10, (5000,))
    X_val = torch.randn(1000, 784)
    y_val = torch.randint(0, 10, (1000,))
    return X_train, y_train, X_val, y_val

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# 准备数据
X_train, y_train, X_val, y_val = create_sample_data()

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# 开始训练
manager = TrainingManager(model, train_loader, val_loader, device)
history = manager.train(
    num_epochs=10,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler
)
```

## 训练步骤详解

### 1. 模型模式切换

```python
# 训练模式：启用 Dropout 和 BatchNorm 的训练行为
model.train()

# 评估模式：禁用 Dropout，BatchNorm 使用滑动统计
model.eval()
```

| 模式 | Dropout | BatchNorm | 梯度计算 | 典型场景 |
|------|---------|-----------|---------|---------|
| train() | 随机置零 | 使用 batch 统计 | 启用 | 训练 |
| eval() | 恒等映射 | 使用 running 统计 | 启用 | 验证 |
| no_grad() + eval() | 恒等映射 | 使用 running 统计 | 禁用 | 推理 |

### 2. 梯度清零

```python
# 方式 1：标准清零
optimizer.zero_grad()

# 方式 2：清零并释放梯度引用（推荐，节省内存）
optimizer.zero_grad(set_to_none=True)
```

💡 **提示**：`set_to_none=True` 比默认的 `set_to_none=False` 更节省内存，因为它将梯度设为 `None` 而不是全零 Tensor。推荐在所有训练循环中使用。

### 3. 前向传播与损失计算

```python
# 分类任务
logits = model(images)              # [batch, num_classes]
loss = criterion(logits, labels)    # CrossEntropyLoss

# 回归任务
predictions = model(features)       # [batch, output_dim]
loss = criterion(predictions, targets)  # MSELoss

# 多任务学习
outputs1 = model1(x)
outputs2 = model2(x)
loss = alpha * criterion1(outputs1, targets1) + beta * criterion2(outputs2, targets2)
```

### 4. 反向传播

```python
# 标准反向传播
loss.backward()

# 保留计算图（通常不需要）
# loss.backward(retain_graph=True)

# 累积梯度（梯度累积）
loss = loss / accumulation_steps
loss.backward()
# 不立即调用 optimizer.step()，等累积足够步数后再更新
```

### 5. 参数更新

```python
# 标准更新
optimizer.step()

# 学习率调度（在 optimizer.step() 之后调用）
scheduler.step()        # 每个 epoch
# 或
scheduler.step(loss)    # 基于验证损失（ReduceLROnPlateau）
```

## 学习率调度器

PyTorch 提供了多种学习率调度策略：

```python
# 1. StepLR：每 step_size 个 epoch 衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 2. MultiStepLR：在指定 epoch 衰减
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

# 3. CosineAnnealingLR：余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 4. CosineAnnealingWarmRestarts：余弦退火+热重启
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# 5. ReduceLROnPlateau：基于验证指标
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)

# 6. OneCycleLR：One Cycle 学习率策略
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=10
)

# 7. LambdaLR：自定义衰减函数
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: 0.95 ** epoch
)

# 学习率调度器对比
schedulers = {
    'StepLR': '固定步长衰减',
    'MultiStepLR': '多阶段衰减',
    'CosineAnnealingLR': '平滑衰减到最小值',
    'ReduceLROnPlateau': '自适应衰减',
    'OneCycleLR': '先升后降，适合快速训练',
}

for name, desc in schedulers.items():
    print(f"{name}: {desc}")
```

### 调度器调用时机

| 调度器 | 调用时机 | 参数 |
|--------|---------|------|
| StepLR, MultiStepLR | 每个 epoch 末尾 | `scheduler.step()` |
| CosineAnnealingLR | 每个 epoch 或 step | `scheduler.step()` |
| ReduceLROnPlateau | 每个 epoch 末尾 | `scheduler.step(val_loss)` |
| OneCycleLR | 每个 step | `scheduler.step()` |

## 梯度累积（Gradient Accumulation）

当 GPU 内存不足以使用大 batch size 时，可以使用梯度累积：

```python
def train_with_accumulation(model, train_loader, optimizer, criterion,
                            num_epochs, accumulation_steps=4):
    """使用梯度累积的训练"""
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)

            # 损失缩放（补偿累积步数）
            loss = loss / accumulation_steps

            # 反向传播
            loss.backward()

            epoch_loss += loss.item()

            # 累积足够步数后更新参数
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f}")
```

💡 **提示**：梯度累积的等效 batch_size = `DataLoader batch_size × accumulation_steps`。如果 DataLoader 的 batch_size 是 16，accumulation_steps 是 4，那么等效 batch_size 为 64。

## 训练过程可视化

### 使用 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/experiment_1')

# 记录标量
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

# 记录图像
writer.add_images('Training Batch', data_grid, epoch)

# 记录模型结构
writer.add_graph(model, sample_input)

# 记录权重直方图
for name, param in model.named_parameters():
    writer.add_histogram(f'Weights/{name}', param, epoch)

writer.close()

# 启动 TensorBoard
# tensorboard --logdir runs/
```

### 使用 Matplotlib

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history.get('train_acc', []), label='Train Acc', marker='o')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

# plot_training_history(history)
```

## 训练过程中的常见陷阱

### 1. 忘记 model.train() / model.eval()

```python
# 错误：验证时没有切换到 eval 模式
for data, targets in val_loader:
    outputs = model(data)  # Dropout 仍在生效！

# 正确
model.eval()
with torch.no_grad():
    for data, targets in val_loader:
        outputs = model(data)
```

### 2. 忘记 optimizer.zero_grad()

```python
# 错误：梯度会累积
for batch in train_loader:
    outputs = model(batch[0])
    loss = criterion(outputs, batch[1])
    loss.backward()
    optimizer.step()  # 没有 zero_grad()！

# 正确
for batch in train_loader:
    optimizer.zero_grad()  # 必须先清零
    outputs = model(batch[0])
    loss = criterion(outputs, batch[1])
    loss.backward()
    optimizer.step()
```

### 3. 忘记 tensor.to(device)

```python
# 错误：Tensor 在 CPU 上，模型在 GPU 上
for data, targets in train_loader:
    outputs = model(data)  # 报错！

# 正确
for data, targets in train_loader:
    data, targets = data.to(device), targets.to(device)
    outputs = model(data)
```

### 4. 在验证时计算梯度

```python
# 错误：浪费内存和时间
for data, targets in val_loader:
    outputs = model(data)  # 会记录计算图
    loss = criterion(outputs, targets)  # 也会记录
```

## 训练检查点管理

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler=None):
    """加载训练检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
    return start_epoch

# 使用示例
# save_checkpoint(model, optimizer, scheduler, epoch, loss, 'checkpoint.pt')
# start_epoch = load_checkpoint('checkpoint.pt', model, optimizer, scheduler)
```

## 总结

本文详细学习了训练循环的编写：

- **标准模板**：train_one_epoch 和 validate 的完整实现
- **五步流程**：数据迁移 → 梯度清零 → 前向传播 → 反向传播 → 参数更新
- **学习率调度**：多种调度器及其适用场景
- **梯度累积**：在有限内存下模拟大 batch 训练
- **可视化**：TensorBoard 和 Matplotlib 的使用方法
- **常见陷阱**：忘记模式切换、梯度清零、设备迁移等
- **检查点管理**：保存和恢复训练状态

掌握了训练循环的编写后，我们已经能够训练大多数深度学习模型。在下一篇文章中，我们将学习如何保存和加载模型，确保训练成果可以持久化使用。

[上一篇：数据集与数据加载 Dataset 与 DataLoader](./50-dataloader-dataset.md) | [下一篇：模型保存与加载 →](./52-model-save-load.md)
