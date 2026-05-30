---
title: 自编码器
icon: compress-arrows-alt
order: 39
---

# 自编码器

自编码器（Autoencoder）是一种无监督学习的神经网络架构，通过学习数据的紧凑表示（编码）来实现降维、去噪和生成等任务。本章将系统讲解自编码器的原理、变体以及实际应用。

## 自编码器的基本结构

自编码器由两部分组成：

1. **编码器（Encoder）**：将输入数据压缩为低维潜在表示（Latent Representation）
2. **解码器（Decoder）**：从潜在表示中重建原始输入

```
输入 x -> [编码器] -> 潜在编码 z -> [解码器] -> 重建输出 x'
```

目标是最小化重建误差：

$$\mathcal{L} = \| \mathbf{x} - \mathbf{x}' \|^2$$

### 为什么需要自编码器

| 与 PCA 对比 | PCA | 自编码器 |
|-------------|-----|----------|
| 映射类型 | 线性变换 | 非线性变换 |
| 表达能力 | 只能捕捉线性关系 | 可以学习复杂的非线性流形 |
| 深度 | 单层 | 可以任意深 |
| 适用场景 | 简单的降维任务 | 复杂的表征学习 |

💡 **提示**：如果自编码器的编码器和解码器都是线性的，且使用 MSE 损失，那么它学习到的潜在子空间与 PCA 相同。非线性激活函数才是自编码器超越 PCA 的关键。

### 基本实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleAutoencoder(nn.Module):
    """简单自编码器"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 假设输入在 [0, 1] 范围
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def encode(self, x):
        """仅编码"""
        return self.encoder(x)

    def decode(self, z):
        """仅解码"""
        return self.decoder(z)


# 训练示例
def train_autoencoder(model, data, epochs=100, learning_rate=1e-3):
    """训练自编码器"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data_tensor = torch.FloatTensor(data)

    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model


# 使用示例
if __name__ == "__main__":
    # 生成简单的模拟数据
    np.random.seed(42)
    data = np.random.rand(1000, 784).astype(np.float32)  # 模拟 28x28 图像

    model = SimpleAutoencoder(input_dim=784, hidden_dim=256, latent_dim=32)
    model = train_autoencoder(model, data, epochs=100)

    # 编码数据
    encoded = model.encode(torch.FloatTensor(data))
    print(f"编码后维度: {encoded.shape}")  # (1000, 32)
```

## 欠完备自编码器

当潜在维度小于输入维度时，称为欠完备（Undercomplete）自编码器。这是最常见的情况，迫使网络学习数据的最重要特征。

### 潜在空间大小的选择

| 输入类型 | 输入维度 | 建议潜在维度 | 压缩率 |
|----------|----------|-------------|--------|
| MNIST 图像 | 784 | 16-64 | 12x-49x |
| CIFAR-10 图像 | 3072 | 64-256 | 12x-48x |
| 文本嵌入 | 300 | 32-128 | 2x-9x |
| 音频频谱 | 1024 | 64-256 | 4x-16x |

## 正则化自编码器

为了防止自编码器简单地学习恒等映射（即直接复制输入到输出），需要对其施加约束。

### 稀疏自编码器（Sparse Autoencoder）

在损失函数中添加稀疏性惩罚，迫使潜在表示中大部分值为零。

$$\mathcal{L} = \| \mathbf{x} - \mathbf{x}' \|^2 + \lambda \|\mathbf{z}\|_1$$

```python
class SparseAutoencoder(nn.Module):
    """稀疏自编码器"""

    def __init__(self, input_dim, hidden_dim, latent_dim, sparsity_weight=0.001):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        # 稀疏性惩罚（L1 正则化潜在表示）
        sparsity_loss = self.sparsity_weight * torch.mean(torch.abs(z))

        reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x)
        total_loss = reconstruction_loss + sparsity_loss

        return x_reconstructed, total_loss, reconstruction_loss
```

### 去噪自编码器（Denoising Autoencoder）

在输入中加入噪声，训练网络从噪声中恢复原始数据。这迫使网络学习数据的稳健表示。

```python
class DenoisingAutoencoder(nn.Module):
    """去噪自编码器"""

    def __init__(self, input_dim, hidden_dim, latent_dim, noise_level=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.noise_level = noise_level

    def add_noise(self, x):
        """添加高斯噪声"""
        noise = torch.randn_like(x) * self.noise_level
        return torch.clamp(x + noise, 0, 1)

    def forward(self, x):
        # 输入加噪声
        x_noisy = self.add_noise(x)

        z = self.encoder(x_noisy)
        x_reconstructed = self.decoder(z)

        # 与干净输入计算损失
        loss = nn.functional.mse_loss(x_reconstructed, x)

        return x_reconstructed, loss
```

### 合同自编码器（Contractive Autoencoder）

通过惩罚编码器对输入的雅可比矩阵的 Frobenius 范数，使潜在表示对输入扰动具有鲁棒性。

$$\mathcal{L} = \| \mathbf{x} - \mathbf{x}' \|^2 + \lambda \| J_f(\mathbf{x}) \|_F^2$$

## 变分自编码器（VAE）

VAE 由 Kingma 和 Welling 在 2013 年提出，是自编码器最重要的变体，兼具表征学习和数据生成的能力。

### 核心思想

VAE 不再将输入映射到固定的潜在向量，而是映射到潜在空间中的一个概率分布。假设潜在变量服从高斯分布：

$$q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}) \mathbf{I})$$

编码器输出均值 $\boldsymbol{\mu}$ 和方差 $\boldsymbol{\sigma}^2$，然后通过**重参数化技巧（Reparameterization Trick）**进行采样：

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

### 损失函数（ELBO）

VAE 的优化目标是证据下界（Evidence Lower Bound, ELBO）：

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

其中：
- 第一项是**重建项**：解码器从潜在变量重建输入的能力
- 第二项是 **KL 散度**：潜在分布接近标准正态分布的程度

### VAE 实现

```python
class VariationalAutoencoder(nn.Module):
    """变分自编码器（VAE）"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # 编码器：输出均值和方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """编码，返回均值和对数方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码"""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar


def vae_loss(x_reconstructed, x, mu, logvar):
    """VAE 损失 = 重建损失 + KL 散度"""
    # 重建损失（BCE 适用于 [0,1] 范围的输入）
    reconstruction_loss = nn.functional.binary_cross_entropy(
        x_reconstructed, x, reduction='sum'
    )

    # KL 散度：N(mu, sigma^2) || N(0, 1)
    # 解析形式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_divergence


# 训练 VAE
def train_vae(model, data, epochs=50, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_tensor = torch.FloatTensor(data)

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_reconstructed, mu, logvar = model(data_tensor)
        loss = vae_loss(x_reconstructed, data_tensor, mu, logvar)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.2f}")

    return model


# 生成新样本
def generate_samples(model, n_samples, latent_dim):
    """从 VAE 生成新样本"""
    model.eval()
    with torch.no_grad():
        # 从先验分布 N(0, I) 采样
        z = torch.randn(n_samples, latent_dim)
        samples = model.decode(z)
    return samples
```

### VAE 的关键特性

| 特性 | 说明 |
|------|------|
| 潜在空间连续性 | 相似的输入映射到相近的潜在向量 |
| 潜在空间完整性 | 潜在空间中的随机点都能解码为有意义的输出 |
| 可控生成 | 通过修改潜在向量可以控制生成结果 |
| 正则化分布 | KL 散度迫使潜在分布接近标准正态分布 |

## 自编码器变体对比

| 类型 | 核心改进 | 应用场景 |
|------|----------|----------|
| 标准自编码器 | 基本压缩重建 | 降维、特征提取 |
| 稀疏自编码器 | L1 正则化潜在表示 | 学习可解释的特征 |
| 去噪自编码器 | 从噪声中恢复原始数据 | 图像去噪、鲁棒特征 |
| 变分自编码器 | 概率化潜在空间 | 数据生成、表征学习 |
| 合同自编码器 | 对输入扰动鲁棒 | 学习稳定的特征 |
| 卷积自编码器 | 卷积层替代全连接 | 图像处理 |

## 卷积自编码器

对于图像数据，使用卷积层替代全连接层效果更好。

```python
class ConvolutionalAutoencoder(nn.Module):
    """卷积自编码器"""

    def __init__(self):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.ReLU(),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


# 在 MNIST 上训练
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )

    model = ConvolutionalAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        for images, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = nn.functional.mse_loss(reconstructed, images)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
```

## 自编码器的应用

| 应用 | 方法 | 使用的自编码器类型 |
|------|------|--------------------|
| 图像压缩 | 编码器压缩，解码器重建 | 标准/卷积自编码器 |
| 异常检测 | 正常数据重建好，异常数据重建差 | 去噪自编码器 |
| 数据去噪 | 从噪声输入重建干净输出 | 去噪自编码器 |
| 数据生成 | 从潜在空间采样解码 | VAE |
| 降维可视化 | 潜在表示作为低维嵌入 | 标准自编码器 |
| 半监督学习 | 预训练编码器作为特征提取器 | 标准自编码器 |

## 总结

自编码器通过学习数据的紧凑表示，在无监督学习中发挥着重要作用。本章核心要点：

1. 自编码器由编码器（压缩）和解码器（重建）组成，目标是最小化重建误差
2. 非线性激活使自编码器超越了 PCA 的线性降维能力
3. 正则化自编码器（稀疏、去噪、合同）通过不同约束学习更有意义的表示
4. VAE 将潜在空间概率化，兼具表征学习和数据生成能力
5. 重参数化技巧使 VAE 的采样操作可导
6. 卷积自编码器比全连接自编码器更适合图像处理

自编码器学习了数据的潜在表示，但它的生成能力有限。接下来我们将学习更强大的生成模型：生成对抗网络（GAN）。

---

**上一篇**: [38. RNN、LSTM 与 GRU](38-rnn-lstm-gru.md)
**下一篇**: [40. 生成对抗网络（GAN）](40-gan.md)
