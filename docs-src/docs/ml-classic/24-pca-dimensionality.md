---
title: PCA 降维算法详解
icon: compress-alt
order: 24
---

# PCA 降维算法详解

主成分分析（Principal Component Analysis, PCA）是最经典、最基础的降维算法。它通过线性变换将高维数据投影到低维空间，在尽可能保留原始信息的同时减少特征维度。PCA 不仅是无监督学习的重要工具，更是机器学习流程中不可或缺的数据预处理步骤。

---

## 一、为什么需要降维

### 1.1 高维数据的挑战

| 问题 | 说明 | 后果 |
|------|------|------|
| 维度灾难 | 高维空间稀疏，距离度量失效 | KNN 等算法失效 |
| 计算成本 | 训练时间与维度相关 | 训练慢、内存大 |
| 过拟合 | 参数多、样本少 | 模型泛化能力差 |
| 多重共线性 | 特征之间高度相关 | 回归系数不稳定 |
| 可视化困难 | 人类只能理解 2~3 维 | 难以直观分析 |

### 1.2 降维的两种策略

| 策略 | 原理 | 代表算法 | 特点 |
|------|------|----------|------|
| 特征选择 | 选择原始特征的子集 | 过滤法、包装法、嵌入法 | 可解释性强 |
| 特征提取 | 构建新的组合特征 | PCA、LDA、t-SNE | 信息保留更好 |

### 1.3 PCA 的应用场景

| 场景 | 说明 | 效果 |
|------|------|------|
| 数据可视化 | 降到 2D/3D 可视化 | 直观理解数据结构 |
| 特征压缩 | 减少训练特征数 | 加速训练、减少过拟合 |
| 噪声去除 | 去除低方差主成分 | 提高信号质量 |
| 图像处理 | 特征脸（Eigenfaces） | 人脸识别 |
| 金融分析 | 风险因子提取 | 发现市场主导因素 |

---

## 二、PCA 的数学原理

### 2.1 核心思想

PCA 寻找一个投影方向，使得数据在该方向上的方差最大。

**第一主成分**：最大化投影方差的方向

$$\max_{w_1} \frac{1}{n}\sum_{i=1}^{n}(w_1^T x_i)^2, \quad \|w_1\| = 1$$

**第二主成分**：与第一主成分正交且方差最大的方向

$$\max_{w_2} \frac{1}{n}\sum_{i=1}^{n}(w_2^T x_i)^2, \quad \|w_2\| = 1, \quad w_2^T w_1 = 0$$

### 2.2 求解过程

```
输入: 数据矩阵 X (n×d), 目标维度 k
输出: 投影后的数据 Z (n×k)

1. 中心化: X_centered = X - mean(X)
2. 计算协方差矩阵: Σ = X_centered^T · X_centered / (n-1)
3. 特征值分解: Σ = VΛV^T
4. 选择前 k 个最大特征值对应的特征向量
5. 投影: Z = X_centered · V_k
```

---

## 三、从零实现 PCA

```python
class PCA:
    """从零实现 PCA 降维"""
    
    def __init__(self, n_components=None, explained_variance_ratio=None):
        self.n_components = n_components
        self.explained_variance_ratio_threshold = explained_variance_ratio
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.n_components_ = None
    
    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值从大到小排序
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # 计算方差解释比例
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / total_variance
        
        # 确定主成分数量
        if self.explained_variance_ratio_threshold is not None:
            cumulative = np.cumsum(self.explained_variance_ratio)
            self.n_components_ = np.searchsorted(
                cumulative, self.explained_variance_ratio_threshold
            ) + 1
        elif self.n_components is not None:
            self.n_components_ = self.n_components
        else:
            self.n_components_ = X.shape[1]
        
        # 选择主成分
        self.components = eigenvectors[:, :self.n_components_]
        
        return self
    
    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean
        return X_centered.dot(self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        return Z.dot(self.components.T) + self.mean

# 测试
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"原始维度: {X.shape[1]}")
print(f"降维后维度: {X_pca.shape[1]}")
print(f"方差解释比例: {pca.explained_variance_ratio}")
print(f"累计方差解释: {np.sum(pca.explained_variance_ratio)*100:.1f}%")
```

---

## 四、sklearn PCA 实战

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 1. 加载手写数字数据集
digits = load_digits()
X = digits.data  # 1797 样本 × 64 特征（8×8 像素）
y = digits.target

# 2. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA 降维到 2D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# 4. 可视化
plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y == digit
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
               label=str(digit), alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Digits Dataset (2D)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 五、选择主成分数量

### 5.1 累计方差解释比例

```python
# 研究不同主成分数量保留的信息量
pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 阈值')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% 阈值')
plt.xlabel('主成分数量')
plt.ylabel('累计方差解释比例')
plt.title('PCA: 主成分数量与信息保留')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 找到保留 95% 信息所需的主成分数
n_95 = np.searchsorted(cumulative_variance, 0.95) + 1
n_99 = np.searchsorted(cumulative_variance, 0.99) + 1
print(f"保留 95% 信息需要 {n_95} 个主成分 (原始 {X.shape[1]} 维)")
print(f"保留 99% 信息需要 {n_99} 个主成分")
print(f"压缩比 (95%): {X.shape[1]/n_95:.1f}x")
```

### 5.2 自动选择方法

```python
# 方法 1: 指定方差解释比例
pca_auto = PCA(n_components=0.95)  # 保留 95% 方差
X_pca_auto = pca_auto.fit_transform(X_scaled)
print(f"自动选择: {pca_auto.n_components_} 个主成分")

# 方法 2: MLE 方法（自动估计最优维度）
pca_mle = PCA(n_components='mle')
X_pca_mle = pca_mle.fit_transform(X_scaled)
print(f"MLE 估计: {pca_mle.n_components_} 个主成分")
```

---

## 六、主成分的解释

### 6.1 查看主成分载荷

```python
# 使用鸢尾花数据解释主成分
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
pca.fit(X_scaled)

# 主成分载荷（特征在主成分上的权重）
print("=== 主成分载荷矩阵 ===")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=iris.feature_names
)
print(loadings.round(3))

print(f"\n=== 方差解释比例 ===")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio*100:.1f}%")
```

### 6.2 碎石图（Scree Plot）

```python
# 碎石图
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(pca_full.explained_variance_ratio_)+1),
        pca_full.explained_variance_ratio_, alpha=0.7)
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
         cumulative_variance, 'ro-')
plt.xlabel('主成分')
plt.ylabel('方差解释比例')
plt.title('PCA 碎石图')
plt.axhline(y=0.05, color='g', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 七、PCA 在分类任务中的应用

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 加载数据
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# 对比不同维度的效果
print(f"{'维度':<6} {'训练时间':<10} {'准确率':<8} {'压缩比':<6}")
print("-" * 45)

for n_comp in [64, 50, 30, 20, 10, 5]:
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    import time
    start = time.time()
    rf.fit(X_train_pca, y_train)
    train_time = time.time() - start
    
    acc = rf.score(X_test_pca, y_test)
    
    print(f"{n_comp:<6} {train_time*1000:<10.1f} {acc:<8.4f} {64/n_comp:<6.1f}x")
```

---

## 八、其他降维方法对比

| 算法 | 类型 | 原理 | 适用场景 |
|------|------|------|----------|
| PCA | 线性 | 最大化方差 | 通用降维、可视化 |
| LDA | 线性 | 最大化类间/类内方差比 | 有监督降维 |
| t-SNE | 非线性 | 保持局部结构 | 高维可视化 |
| UMAP | 非线性 | 保持拓扑结构 | 大规模可视化 |
| NMF | 线性 | 非负矩阵分解 | 文本、图像 |
| SVD | 线性 | 奇异值分解 | 推荐系统、NLP |
| 自编码器 | 非线性 | 神经网络压缩 | 复杂非线性降维 |

```python
# 对比 PCA、t-SNE、UMAP
from sklearn.manifold import TSNE
import umap

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, X_embed, title in zip(
    axes, [X_pca, X_tsne, X_umap],
    ['PCA', 't-SNE', 'UMAP']
):
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_embed[mask, 0], X_embed[mask, 1],
                  label=str(digit), alpha=0.6, s=20)
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

---

## 九、实战建议

1. **先标准化**：PCA 对特征尺度极度敏感，必须先做标准化
2. **方差解释比例**：选择保留 95%~99% 信息的主成分数
3. **可视化**：降到 2D/3D 后观察数据分布
4. **预处理步骤**：将 PCA 作为 Pipeline 的第一步
5. **非线性数据**：如果数据有非线性结构，考虑 t-SNE 或 UMAP
6. **逆变换**：可以用 inverse_transform 将降维数据还原到原始空间

> 💡 **提示**：PCA 是线性降维的经典方法，通过保留最大方差方向来压缩数据。理解了 PCA 后，下一篇的朴素贝叶斯将带你回到分类领域，学习基于概率的分类方法。

---

## 十、总结

PCA 是最基础、最重要的降维算法，通过线性变换将数据投影到方差最大的方向。核心要点包括：

- **核心思想**：找到最大化投影方差的正交方向
- **求解方法**：对协方差矩阵做特征值分解
- **主成分选择**：累计方差解释比例 ≥ 95%
- **应用场景**：数据可视化、特征压缩、噪声去除
- **可视化方法**：碎石图、散点图、载荷矩阵
- **与其他方法对比**：PCA 线性、t-SNE/UMAP 非线性
- **注意事项**：必须先标准化，结果可解释性有限

---

📖 **上一篇**：[K-Means 聚类](./23-kmeans-clustering.md) | 📖 **下一篇**：[朴素贝叶斯](./25-naive-bayes.md) — 学习基于贝叶斯定理的概率分类方法。
