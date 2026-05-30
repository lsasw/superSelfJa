---
title: 生成对抗网络（GAN）
icon: mask
order: 40
---

# 生成对抗网络（GAN）

生成对抗网络（Generative Adversarial Network，GAN）由 Ian Goodfellow 等人在 2014 年提出，是深度学习中最具创意的架构之一。GAN 通过两个网络的对抗博弈来生成逼真的数据，在图像生成、风格迁移、数据增强等领域取得了革命性的成果。

## GAN 的核心思想

GAN 的灵感来自博弈论中的二人零和博弈。它包含两个网络：

1. **生成器（Generator）**：从随机噪声中生成假数据，目标是"骗过"判别器
2. **判别器（Discriminator）**：区分真实数据和生成器产生的假数据

```
随机噪声 z -> [生成器 G] -> 假数据 G(z)
真实数据 x -------------> 一起送入 [判别器 D] -> 判断真假
```

两个网络在训练中不断竞争：生成器试图产生更逼真的假数据，判别器试图更好地区分真假。最终，生成器学会产生与真实数据无法区分的样本。

## GAN 的数学原理

### 目标函数

GAN 的训练可以表示为一个极小极大博弈（Minimax Game）：

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

其中：
- $D(\mathbf{x})$ 是判别器判断真实数据为真的概率
- $D(G(\mathbf{z}))$ 是判别器判断生成数据为真的概率
- 理想情况下，训练收敛时 $D(G(\mathbf{z})) = 0.5$，即判别器无法区分

### 训练过程

GAN 的训练交替进行两步：

**步骤 1：更新判别器（最大化判别能力）**

$$\max_D \mathbb{E}_{\mathbf{x}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]$$

**步骤 2：更新生成器（最小化判别器判断正确的概率）**

原始论文使用的是：

$$\min_G \mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]$$

但实际中更常用**非饱和目标**：

$$\max_G \mathbb{E}_{\mathbf{z}}[\log D(G(\mathbf{z}))]$$

这是因为原始目标在训练初期会导致生成器梯度消失（当判别器很强时，$D(G(\mathbf{z})) \approx 0$，$\log(1 - D(G(\mathbf{z}))) \approx 0$，梯度很小）。

## GAN 的基础实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    """生成器：从噪声生成图像"""

    def __init__(self, noise_dim=100, output_dim=784):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # 输出 [-1, 1]
        )

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    """判别器：判断图像真假"""

    def __init__(self, input_dim=784):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        return self.network(x)


class SimpleGAN:
    """简单 GAN 训练器"""

    def __init__(self, noise_dim=100, image_dim=784, lr=0.0002, beta1=0.5):
        self.noise_dim = noise_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        self.G = Generator(noise_dim, image_dim).to(self.device)
        self.D = Discriminator(image_dim).to(self.device)

        # 损失函数
        self.criterion = nn.BCELoss()

        # 优化器
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))

    def train_step(self, real_images):
        """一个训练步骤"""
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(self.device)

        # 标签
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # === 训练判别器 ===
        self.optimizer_D.zero_grad()

        # 真实数据
        d_real = self.D(real_images)
        d_real_loss = self.criterion(d_real, real_labels)

        # 假数据
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_images = self.G(noise).detach()  # 不更新生成器
        d_fake = self.D(fake_images)
        d_fake_loss = self.criterion(d_fake, fake_labels)

        # 判别器总损失
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.optimizer_D.step()

        # === 训练生成器 ===
        self.optimizer_G.zero_grad()

        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_images = self.G(noise)
        d_fake_for_g = self.D(fake_images)
        g_loss = self.criterion(d_fake_for_g, real_labels)  # 希望被判别为真

        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item()

    def train(self, dataloader, epochs=50):
        """训练循环"""
        for epoch in range(epochs):
            for i, (images, _) in enumerate(dataloader):
                d_loss, g_loss = self.train_step(images)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")

    def generate(self, n_samples=16):
        """生成样本"""
        self.G.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim).to(self.device)
            fake_images = self.G(noise)
        return fake_images.cpu()


# 在 MNIST 上训练
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])

    dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    gan = SimpleGAN(noise_dim=100, image_dim=784, lr=0.0002, beta1=0.5)
    gan.train(dataloader, epochs=50)

    # 生成并可视化
    fake_images = gan.generate(16)
    fake_images = fake_images.view(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i].numpy(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated.png', dpi=150)
```

## GAN 训练的难点

GAN 是出了名的难以训练。以下是常见问题及解决方案：

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 模式崩溃（Mode Collapse） | 生成器只产生少量相似的样本 | Mini-batch Discrimination, Unrolled GAN |
| 训练不稳定 | 损失值剧烈震荡 | 降低学习率，使用 WGAN |
| 梯度消失 | 生成器无法学习 | 使用非饱和目标，WGAN-GP |
| 判别器过强 | 判别器完美区分，生成器无梯度 | 减少判别器训练次数 |
| 判别器过弱 | 生成器轻松骗过判别器 | 增加判别器训练次数 |

## DCGAN（深度卷积 GAN）

DCGAN 将卷积网络引入 GAN，是最成功的 GAN 变体之一。

### 设计原则

| 组件 | 规则 |
|------|------|
| 用步长卷积替代池化 | 生成器用转置卷积，判别器用步长卷积 |
| 批量归一化 | 两个网络都使用 BN |
| 移除全连接隐藏层 | 使用全局平均池化 |
| 激活函数 | 生成器输出用 Tanh，其他用 ReLU（G）/ Leaky ReLU（D） |

```python
class DCGAN_Generator(nn.Module):
    """DCGAN 生成器"""

    def __init__(self, noise_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            # 输入: noise (100, 1, 1)
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 输出: (512, 4, 4)

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 输出: (256, 8, 8)

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 输出: (128, 16, 16)

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 输出: (64, 32, 32)

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (3, 64, 64) RGB 图像
        )

    def forward(self, z):
        # z: (batch, noise_dim) -> (batch, noise_dim, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        return self.network(z)


class DCGAN_Discriminator(nn.Module):
    """DCGAN 判别器"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 输入: (3, 64, 64)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (64, 32, 32)

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (128, 16, 16)

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (256, 8, 8)

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (512, 4, 4)

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: (1, 1, 1)
        )

    def forward(self, x):
        x = self.network(x)
        return x.view(-1, 1)
```

## WGAN（Wasserstein GAN）

WGAN 使用 Wasserstein 距离（Earth-Mover 距离）替代 JS 散度，从根本上解决了 GAN 训练不稳定的问题。

### 核心改进

**新的目标函数**：

$$W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_r}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$$

其中 $\|f\|_L \leq 1$ 要求判别器（现在称为"评论家"Critic）是 1-Lipschitz 函数。

### WGAN 的关键修改

| 修改 | 原始 GAN | WGAN |
|------|----------|------|
| 判别器最后一层 | Sigmoid + BCE | 无激活，线性输出 |
| 损失函数 | 交叉熵 | Wasserstein 距离 |
| 权重约束 | 无 | 权重裁剪（WGAN）或梯度惩罚（WGAN-GP） |
| 优化器 | Adam | RMSProp |

```python
class WGANGP(nn.Module):
    """带梯度惩罚的 WGAN"""

    def __init__(self, noise_dim=100):
        super().__init__()
        self.G = DCGAN_Generator(noise_dim)
        self.D = DCGAN_Discriminator()
        # 移除最后的 Sigmoid
        self.D.network[-1] = nn.Identity()

        self.lambda_gp = 10  # 梯度惩罚系数

    def gradient_penalty(self, real_data, fake_data):
        """WGAN-GP 梯度惩罚"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        d_interpolated = self.D(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def train_step(self, real_data):
        batch_size = real_data.size(0)

        # 生成假数据
        noise = torch.randn(batch_size, 100, 1, 1).to(real_data.device)
        fake_data = self.G(noise).detach()

        # 判别器损失
        d_real = self.D(real_data).view(-1)
        d_fake = self.D(fake_data).view(-1)

        gp = self.gradient_penalty(real_data, fake_data)
        d_loss = -(d_real.mean() - d_fake.mean()) + self.lambda_gp * gp

        # 生成器损失
        self.optimizer_G.zero_grad()
        fake_data_for_g = self.G(torch.randn(batch_size, 100, 1, 1).to(real_data.device))
        g_loss = -self.D(fake_data_for_g).view(-1).mean()
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item()
```

## GAN 的主要变体

| 变体 | 年份 | 核心贡献 |
|------|------|----------|
| DCGAN | 2015 | 卷积架构，奠定了 GAN 的基本设计 |
| cGAN | 2014 | 条件生成，控制生成类别 |
| WGAN | 2017 | Wasserstein 距离，训练稳定 |
| WGAN-GP | 2017 | 梯度惩罚替代权重裁剪 |
| CycleGAN | 2017 | 无配对数据的风格迁移 |
| StyleGAN | 2018 | 风格控制，人脸生成质量极高 |
| StyleGAN2 | 2019 | 消除水滴伪影 |
| StyleGAN3 | 2021 | 等变信号处理 |
| Pix2Pix | 2016 | 配对数据的图像翻译 |

## GAN 的应用

| 应用 | 使用技术 | 说明 |
|------|----------|------|
| 图像生成 | StyleGAN | 生成逼真的人脸、风景等 |
| 风格迁移 | CycleGAN | 马变斑马、白天变夜晚 |
| 图像超分辨率 | SRGAN | 低分辨率图像增强 |
| 图像修复 | Context Encoder | 填充缺失的图像区域 |
| 数据增强 | GAN | 生成额外训练数据 |
| 文本到图像 | DALL-E, VQGAN | 根据文字描述生成图像 |
| 视频生成 | MoCoGAN | 生成连续视频帧 |

## 评估 GAN 的指标

| 指标 | 含义 | 好坏判断 |
|------|------|----------|
| IS（Inception Score） | 生成图像的清晰度和多样性 | 越高越好 |
| FID（Fréchet Inception Distance） | 生成分布与真实分布的距离 | 越低越好 |
| KID（Kernel Inception Distance） | 无偏估计的分布距离 | 越低越好 |
| 人工评估 | 人类判断逼真程度 | 主观但可靠 |

## 总结

GAN 通过对抗博弈学习数据分布，是生成模型的重要支柱。本章核心要点：

1. GAN 由生成器和判别器组成，通过对抗训练生成逼真数据
2. 原始 GAN 使用极小极大目标函数，但训练不稳定
3. DCGAN 引入卷积架构，是图像生成的标准选择
4. WGAN 使用 Wasserstein 距离解决了训练不稳定性
5. 模式崩溃是 GAN 的固有问题，需要精心调参
6. StyleGAN 系列代表了当前图像生成的最高水平

GAN 通过对抗学习数据分布，而另一种思路是直接对序列建立依赖关系。接下来我们将学习迁移学习，它让模型能够将在一个任务上学到的知识迁移到新任务。

---

**上一篇**: [39. 自编码器](39-autoencoder.md)
**下一篇**: [41. 迁移学习](41-transfer-learning.md)
