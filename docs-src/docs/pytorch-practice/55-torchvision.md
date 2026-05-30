---
title: TorchVision 计算机视觉实战
icon: image
order: 55
---

# TorchVision 计算机视觉实战

TorchVision 是 PyTorch 官方计算机视觉库，提供了常用数据集、预训练模型、图像变换和实用工具。它是计算机视觉开发的首选工具。本文将学习如何使用 TorchVision 构建完整的图像识别系统。

## TorchVision 核心组件

| 模块 | 功能 | 说明 |
|------|------|------|
| `torchvision.datasets` | 数据集 | MNIST、CIFAR、ImageNet 等 |
| `torchvision.models` | 模型 | ResNet、ViT、EfficientNet 等 |
| `torchvision.transforms` | 变换 | 数据增强、预处理 |
| `torchvision.ops` | 算子 | NMS、ROI Align 等 |
| `torchvision.utils` | 工具 | 网格可视化、保存图像 |

## 图像变换系统

### transforms v2（新版 API）

PyTorch 2.0 引入了 `transforms.v2`，提供了更一致的 API：

```python
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# 新版 transforms v2 API
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToImage(),           # 新版：替代 ToTensor
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 传统 API 仍然广泛使用
train_transform_v1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 常用变换操作

```python
from PIL import Image

# 几何变换
transforms.Resize(256)                         # 调整大小
transforms.CenterCrop(224)                     # 中心裁剪
transforms.RandomCrop(224, padding=4)          # 随机裁剪
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))  # 随机缩放裁剪
transforms.RandomHorizontalFlip(p=0.5)         # 随机水平翻转
transforms.RandomVerticalFlip(p=0.5)           # 随机垂直翻转
transforms.RandomRotation(30)                  # 随机旋转
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # 随机仿射变换

# 颜色变换
transforms.ColorJitter(brightness=0.2)         # 亮度
transforms.ColorJitter(contrast=0.2)           # 对比度
transforms.ColorJitter(saturation=0.2)         # 饱和度
transforms.ColorJitter(hue=0.1)                # 色调
transforms.Grayscale(num_output_channels=3)    # 灰度化（3 通道）

# 正则化变换
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))  # 随机擦除
transforms.GaussianBlur(kernel_size=3)         # 高斯模糊

# 转换操作
transforms.ToTensor()                          # PIL -> Tensor [0, 1]
transforms.Normalize(mean, std)                # 标准化
transforms.Lambda(lambda x: x.convert('RGB'))  # 自定义 lambda
```

## 预训练模型

### 可用的模型架构

```python
from torchvision import models

# 列出所有可用模型
# print(dir(models))

# 查看模型权重
# print(dir(models.resnet18))
```

### 主流架构对比

| 架构 | 参数量 | Top-1 Acc | 特点 | 典型用途 |
|------|--------|-----------|------|---------|
| ResNet-18 | 11M | 69.8% | 浅而快 | 基础任务 |
| ResNet-50 | 25M | 76.1% | 经典架构 | 通用视觉 |
| ResNet-101 | 44M | 77.4% | 更深 | 高精度任务 |
| EfficientNet-B0 | 5M | 77.1% | 高效 | 移动端 |
| EfficientNet-B4 | 19M | 83.0% | 缩放策略 | 高精度/高效平衡 |
| ViT-B/16 | 86M | 78.4% | Transformer | 大规模数据 |
| Swin-T | 28M | 81.3% | 层级 Transformer | 通用视觉 |
| ConvNeXt-T | 28M | 82.1% | 现代化 CNN | 通用视觉 |

### 加载预训练模型

```python
from torchvision.models import ResNet50_Weights

# 方式 1：推荐方式（使用 Weights 枚举）
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet50.eval()

# 方式 2：直接指定
# resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 方式 3：不加载预训练权重
# resnet50 = models.resnet50(weights=None)

# 查看权重元数据
weights = ResNet50_Weights.IMAGENET1K_V2
print(f"Categories: {weights.meta['categories']}")
print(f"Input size: {weights.meta['min_size']}")
print(f"Preprocessing: {weights.transforms()}")

# 使用内置预处理
preprocess = weights.transforms()
```

### 使用预训练模型进行推理

```python
from PIL import Image
import torch

# 加载模型和预处理
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

# 加载图像
# image = Image.open("dog.jpg").convert("RGB")

# 预处理
# batch = preprocess(image).unsqueeze(0)  # 添加 batch 维度

# 推理
# with torch.no_grad():
#     output = model(batch)
#     probabilities = torch.softmax(output, dim=1)[0]
#     top5_prob, top5_idx = torch.topk(probabilities, 5)

# 获取类别名称
# categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
# for prob, idx in zip(top5_prob, top5_idx):
#     print(f"{categories[idx]}: {prob*100:.2f}%")
```

### 修改预训练模型进行微调

```python
def create_fine_tuned_model(model_name, num_classes, pretrained=True):
    """创建微调模型"""

    if model_name == 'resnet50':
        model = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        # 替换最后的分类层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'vit':
        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

# 使用
# model = create_fine_tuned_model('resnet50', num_classes=10, pretrained=True)
# print(f"Model output shape: {model(torch.randn(4, 3, 224, 224)).shape}")
```

### 分阶段微调策略

```python
def setup_fine_tuning(model, strategy='full'):
    """设置不同微调策略的优化器"""

    if strategy == 'head_only':
        # 只训练分类头
        for param in model.parameters():
            param.requires_grad = False
        # 分类头参数默认 requires_grad=True
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(params, lr=1e-3)

    elif strategy == 'last_layer':
        # 冻结前面层，训练最后几层
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(params, lr=1e-4)

    elif strategy == 'full':
        # 全部训练
        return torch.optim.Adam(model.parameters(), lr=1e-4)

    elif strategy == 'two_stage':
        # 两阶段：先小学习率训练头部，再大学习率全量训练
        return None  # 返回 None，由训练循环处理

# 两阶段微调
def two_stage_fine_tuning(model, train_loader, val_loader, num_epochs_head=5, num_epochs_full=20):
    """两阶段微调：先训分类头，再全量微调"""

    criterion = nn.CrossEntropyLoss()

    # 阶段 1：只训练分类头
    print("=== Stage 1: Training classification head ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    train_epoch(model, train_loader, optimizer, criterion, num_epochs_head)

    # 阶段 2：全量微调
    print("=== Stage 2: Full model fine-tuning ===")
    for param in model.parameters():
        param.requires_grad = True

    # 使用不同学习率
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3},           # 分类头：大学习率
        {'params': model.layer4.parameters(), 'lr': 1e-4},       # 最后阶段：中学习率
        {'params': model.layer3.parameters(), 'lr': 5e-5},       # 中间层：小学习率
        {'params': model.layer2.parameters(), 'lr': 1e-5},       # 前层：更小学习率
        {'params': model.layer1.parameters(), 'lr': 1e-5},       # 最前层：最小学习率
    ])

    train_epoch(model, train_loader, optimizer, criterion, num_epochs_full)
```

## 目标检测

```python
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn
)

# 加载预训练的目标检测模型
detector = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
)
detector.eval()

# 准备输入（需要 Tensor 列表）
# images = [preprocess(Image.open("image.jpg")) for _ in range(2)]

# 推理
# with torch.no_grad():
#     predictions = detector(images)

# 输出包含 boxes, labels, scores, masks
# for pred in predictions:
#     print(f"Boxes: {pred['boxes'].shape}")
#     print(f"Labels: {pred['labels']}")
#     print(f"Scores: {pred['scores']}")
```

### 目标检测模型对比

| 模型 | 速度 | 精度 | 特点 | 适用场景 |
|------|------|------|------|---------|
| Faster R-CNN | 中等 | 高 | 两阶段检测 | 通用目标检测 |
| RetinaNet | 快 | 中高 | Focal Loss | 实时检测 |
| SSD | 快 | 中 | 单阶段 | 移动端 |
| FCOS | 中等 | 高 | Anchor-Free | 通用检测 |

## 图像分割

```python
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
    fcn_resnet50,
    FCN_ResNet50_Weights
)

# DeepLabV3 语义分割
segmenter = deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
)
segmenter.eval()

# 使用
# with torch.no_grad():
#     output = segmenter(input_tensor)['out']
#     predictions = output.argmax(dim=1)
```

## 实用工具

### 图像网格可视化

```python
from torchvision.utils import make_grid, save_image

# 创建图像网格
def visualize_batch(images, labels, class_names, num_rows=4):
    """可视化训练 batch"""
    grid = make_grid(images[:num_rows*4], nrow=4, normalize=True, padding=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Training Batch')
    plt.show()

# 保存图像
# save_image(images[:16], 'batch_grid.png', nrow=4, normalize=True)
```

### 使用 torchvision.ops

```python
from torchvision.ops import (
    nms,                         # 非极大值抑制
    batched_nms,                 # 批量 NMS
    roi_align,                   # ROI 对齐
    roi_pool,                    # ROI 池化
    box_iou,                     # IoU 计算
    generalized_box_iou,         # GIoU
    complete_box_iou,            # CIoU
    boxes,                       # 框工具
    feature_pyramid_network,     # FPN
    sigmoid_focal_loss,          # Focal Loss
)

# NMS 示例
boxes = torch.tensor([
    [0, 0, 100, 100],
    [10, 10, 110, 110],
    [200, 200, 300, 300],
])
scores = torch.tensor([0.9, 0.8, 0.7])

keep = nms(boxes, scores, iou_threshold=0.5)
print(f"Keep indices: {keep}")  # [0, 2]

# IoU 计算
boxes1 = torch.tensor([[0, 0, 100, 100]])
boxes2 = torch.tensor([[50, 50, 150, 150]])
iou = box_iou(boxes1, boxes2)
print(f"IoU: {iou}")
```

## 完整项目示例：图像分类

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

class ImageClassificationTrainer:
    """图像分类训练器"""

    def __init__(self, model_name='resnet50', num_classes=10, device='cuda'):
        self.device = torch.device(device)
        self.model = self._create_model(model_name, num_classes)
        self.model = self.model.to(self.device)

    def _create_model(self, model_name, num_classes):
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return model

    def get_transforms(self):
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return train_tf, val_tf

    def get_datasets(self, data_dir='./data'):
        train_tf, val_tf = self.get_transforms()
        train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=train_tf)
        val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=val_tf)
        return train_dataset, val_dataset

    def train(self, data_dir, num_epochs=25, batch_size=32, lr=1e-3):
        train_dataset, val_dataset = self.get_datasets(data_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        for epoch in range(num_epochs):
            # 训练
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 验证
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = correct / total
            print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {accuracy:.4f}")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')

            scheduler.step()

        print(f"Training complete. Best accuracy: {best_acc:.4f}")
        return self.model

# 使用
# trainer = ImageClassificationTrainer(model_name='resnet50', num_classes=10)
# model = trainer.train('./data', num_epochs=25, batch_size=32)
```

## 总结

本文系统学习了 TorchVision 的核心功能：

- **transforms**：数据增强和预处理的完整工具集
- **models**：主流预训练模型（ResNet、EfficientNet、ViT 等）
- **迁移学习**：加载预训练模型、修改分类头、分阶段微调
- **目标检测**：Faster R-CNN 等检测模型
- **图像分割**：DeepLabV3、FCN 等分割模型
- **实用工具**：NMS、IoU、图像网格等
- **完整项目**：图像分类训练器的实现

在下一篇文章中，我们将学习 TorchText 库，了解如何使用它进行自然语言处理任务的开发。

[上一篇：分布式训练 Distributed Training](./54-distributed-training.md) | [下一篇：TorchText →](./56-torchtext.md)
