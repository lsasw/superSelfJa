---
title: K-Means 聚类算法详解
icon: circle-nodes
order: 23
---

# K-Means 聚类算法详解

K-Means 聚类是最经典、最广泛使用的无监督学习算法。与前文学习的所有监督学习算法（线性回归、逻辑回归、SVM、决策树、随机森林、XGBoost、KNN）不同，K-Means 不需要标签信息，而是直接从数据本身的内在结构中发现自然的分组。这是进入无监督学习世界的第一步。

---

## 一、从监督学习到无监督学习

### 1.1 两类学习范式的对比

| 对比维度 | 监督学习 | 无监督学习 |
|----------|----------|-----------|
| 数据形式 | {(x_1, y_1), ..., (x_n, y_n)} | {x_1, ..., x_n} |
| 目标 | 学习 x → y 的映射 | 发现数据的内在结构 |
| 评估 | 准确率、MSE 等 | 轮廓系数、DBI 等 |
| 典型任务 | 分类、回归 | 聚类、降维 |
| 先验知识 | 需要标签 | 不需要标签 |

### 1.2 K-Means 的应用场景

| 场景 | 说明 | 为什么适合 K-Means |
|------|------|-------------------|
| 客户分群 | 按消费行为分群 | 发现自然客户群体 |
| 图像压缩 | 颜色量化 | 减少颜色数量 |
| 文档主题 | 文本聚类 | 发现文档主题簇 |
| 异常检测 | 远离所有中心的点 | 离群点天然异常 |
| 预处理 | 为监督学习生成特征 | 聚类簇可作为新特征 |

---

## 二、K-Means 算法原理

### 2.1 算法流程

```
输入: 数据集 X = {x1, x2, ..., xn}, 聚类数 K, 最大迭代次数 max_iter
输出: K 个聚类中心 {μ1, μ2, ..., μK}, 每个样本的簇分配

1. 初始化: 随机选择 K 个点作为初始聚类中心

2. 重复以下步骤直到收敛或达到 max_iter:
    a. 分配步骤: 将每个样本分配到最近的聚类中心
       c_i = argmin_k ||x_i - μ_k||²
    
    b. 更新步骤: 重新计算每个聚类的中心
       μ_k = mean({x_i | c_i = k})

3. 返回聚类中心和簇分配
```

### 2.2 目标函数

K-Means 最小化的目标函数是簇内平方和（WCSS，Within-Cluster Sum of Squares）：

$$J = \sum_{k=1}^{K}\sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

---

## 三、从零实现 K-Means

```python
class KMeans:
    """从零实现 K-Means 聚类算法"""
    
    def __init__(self, k=3, max_iter=300, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter_ = None
    
    def _init_centroids(self, X):
        """随机初始化聚类中心"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n = X.shape[0]
        indices = np.random.choice(n, self.k, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X):
        """分配每个样本到最近的聚类中心"""
        # 高效计算距离矩阵
        # ||x - μ||² = ||x||² - 2x·μ + ||μ||²
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        C_sq = np.sum(self.centroids ** 2, axis=1, keepdims=True).T
        cross = X.dot(self.centroids.T)
        dists_sq = X_sq - 2 * cross + C_sq
        
        return np.argmin(dists_sq, axis=1)
    
    def _update_centroids(self, X, labels):
        """更新聚类中心"""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.k):
            members = X[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids
    
    def fit(self, X):
        X = np.array(X)
        self.centroids = self._init_centroids(X)
        
        for i in range(self.max_iter):
            # 分配簇
            labels = self._assign_clusters(X)
            
            # 更新中心
            new_centroids = self._update_centroids(X, labels)
            
            # 检查收敛
            shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            self.centroids = new_centroids
            
            if shift < self.tol:
                self.n_iter_ = i + 1
                break
        
        self.labels = self._assign_clusters(X)
        self.inertia = self._compute_inertia(X)
        return self
    
    def _compute_inertia(self, X):
        """计算簇内平方和"""
        inertia = 0
        for k in range(self.k):
            members = X[self.labels == k]
            if len(members) > 0:
                inertia += np.sum((members - self.centroids[k]) ** 2)
        return inertia
    
    def predict(self, X):
        """预测新样本所属的簇"""
        return self._assign_clusters(np.array(X))

# 测试
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)

kmeans = KMeans(k=4, random_state=42)
kmeans.fit(X)

print(f"聚类中心数: {kmeans.k}")
print(f"迭代次数: {kmeans.n_iter_}")
print(f"WCSS (Inertia): {kmeans.inertia:.2f}")
print(f"轮廓系数: {silhouette_score(X, kmeans.labels):.4f}")
```

---

## 四、sklearn K-Means 实战

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# 1. 训练模型
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',      # 智能初始化
    n_init=10,             # 运行 10 次取最优
    max_iter=300,
    tol=1e-4,
    random_state=42
)
kmeans.fit(X)

print("=== K-Means 聚类结果 ===")
print(f"聚类中心数: {kmeans.n_clusters}")
print(f"迭代次数: {kmeans.n_iter_}")
print(f"WCSS (Inertia): {kmeans.inertia_:.2f}")

# 2. 聚类评估
sil_score = silhouette_score(X, kmeans.labels_)
db_score = davies_bouldin_score(X, kmeans.labels_)
print(f"轮廓系数: {sil_score:.4f}")
print(f"DBI 指数: {db_score:.4f}")

# 3. 可视化
plt.figure(figsize=(12, 5))

# 聚类结果
plt.subplot(1, 2, 1)
colors = plt.cm.Spectral(kmeans.labels_ / (kmeans.n_clusters - 1))
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3, label='聚类中心')
plt.title(f'K-Means 聚类 (K={kmeans.n_clusters})')
plt.legend()

# 真实标签（如果有）
plt.subplot(1, 2, 2)
colors_true = plt.cm.Spectral(y_true / (len(np.unique(y_true)) - 1))
plt.scatter(X[:, 0], X[:, 1], c=colors_true, s=50, alpha=0.7)
plt.title('真实标签')

plt.tight_layout()
plt.show()
```

---

## 五、确定最优 K 值

### 5.1 肘部法则（Elbow Method）

```python
# 肘部法则
k_range = range(1, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('聚类数 K')
plt.ylabel('WCSS (Inertia)')
plt.title('肘部法则')
plt.grid(True, alpha=0.3)
plt.show()

# K=1 时 WCSS 最大，随 K 增大而减小
# "肘部"位置是最优 K 值
for k, wcss in zip(k_range, inertias):
    print(f"K={k}: WCSS={wcss:.2f}")
```

### 5.2 轮廓系数法

```python
# 轮廓系数法
k_range = range(2, 11)
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sil_scores, 'ro-')
best_k = k_range[np.argmax(sil_scores)]
plt.axvline(x=best_k, color='g', linestyle='--', label=f'最优 K={best_k}')
plt.xlabel('聚类数 K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法选择最优 K')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"最优 K 值: {best_k}")
for k, score in zip(k_range, sil_scores):
    print(f"K={k}: 轮廓系数={score:.4f}")
```

### 5.3 评估指标对比

| 指标 | 范围 | 最优值 | 说明 |
|------|------|--------|------|
| WCSS (Inertia) | [0, +∞) | 越小越好 | 肘部法则 |
| 轮廓系数 | [-1, 1] | 越大越好 | 凝聚度与分离度 |
| DBI | [0, +∞) | 越小越好 | 簇间相似性与簇内散布度 |
| CH 指数 | [0, +∞) | 越大越好 | 簇间方差与簇内方差比 |

```python
# 多指标综合评估
from sklearn.metrics import calinski_harabasz_score

print(f"{'K':<4} {'WCSS':<12} {'轮廓系数':<8} {'DBI':<8} {'CH指数':<8}")
print("-" * 50)

for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    
    wcss = km.inertia_
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    print(f"{k:<4} {wcss:<12.2f} {sil:<8.4f} {dbi:<8.4f} {ch:<8.2f}")
```

---

## 六、K-Means++ 初始化

### 6.1 为什么需要 K-Means++

传统随机初始化可能导致：
- 聚类中心靠得太近
- 收敛到局部最优
- 需要多次运行取最优

### 6.2 K-Means++ 流程

```
1. 随机选择第一个中心 μ_1
2. 对每个样本 x_i，计算到最近中心的距离 D(x_i)
3. 按概率 D(x_i)² / ΣD(x_j)² 选择下一个中心
4. 重复步骤 2-3 直到选出 K 个中心
```

```python
def kmeans_plus_plus_init(X, k, random_state=None):
    """K-Means++ 初始化"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n, d = X.shape
    centroids = np.empty((k, d))
    
    # 1. 随机选择第一个中心
    idx = np.random.randint(n)
    centroids[0] = X[idx]
    
    for c in range(1, k):
        # 2. 计算每个点到最近中心的距离平方
        dists = np.array([
            np.min([np.sum((X[i] - centroids[j])**2) for j in range(c)])
            for i in range(n)
        ])
        
        # 3. 按概率选择下一个中心
        probs = dists / dists.sum()
        cumprobs = np.cumsum(probs)
        idx = np.searchsorted(cumprobs, np.random.random())
        centroids[c] = X[idx]
    
    return centroids
```

---

## 七、实战项目：图像颜色量化

```python
from sklearn.cluster import MiniBatchKMeans
from skimage import io

# 加载图像（示例：使用 numpy 生成模拟图像）
# 实际应用中：image = io.imread('photo.jpg')
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# 将图像数据重塑为 (n_pixels, 3)
h, w, c = image.shape
pixels = image.reshape(-1, 3).astype(np.float64)

# 使用 MiniBatchKMeans 压缩颜色
n_colors = 16

# 原始颜色
print(f"原始颜色数: {len(np.unique(pixels, axis=0))}")

# 聚类压缩
kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=1000)
kmeans.fit(pixels)

# 用聚类中心替换所有像素
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = compressed_pixels.reshape(h, w, c).astype(np.uint8)

print(f"压缩后颜色数: {n_colors}")
print(f"压缩比: {len(np.unique(pixels, axis=0)) / n_colors:.1f}x")
```

---

## 八、K-Means 的优缺点

### 8.1 优点

| 优点 | 说明 |
|------|------|
| 简单高效 | 算法简单，计算复杂度 O(n·K·d·T) |
| 可扩展 | MiniBatchKMeans 处理大数据集 |
| 收敛快 | 通常只需少量迭代 |
| 结果可解释 | 每个簇有明确的中心 |

### 8.2 缺点

| 缺点 | 说明 | 缓解方案 |
|------|------|----------|
| 需指定 K | 不知道最优簇数 | 肘部法、轮廓系数 |
| 球形假设 | 假设簇为凸球形 | DBSCAN、谱聚类 |
| 对异常值敏感 | 均值受极端值影响 | K-Medians、K-Medoids |
| 局部最优 | 依赖初始化 | K-Means++、多次运行 |
| 尺度敏感 | 大尺度特征主导 | 特征标准化 |
| 不适用密度不同 | 各簇大小差异大时效果差 | GMM、DBSCAN |

---

## 九、变体算法

| 变体 | 改进点 | 适用场景 |
|------|--------|----------|
| K-Means++ | 更好的初始化 | 通用 |
| MiniBatch K-Means | 小批量采样加速 | 大数据集 |
| K-Medoids | 用中心点替代均值 | 有异常值 |
| K-Medians | 用中位数替代均值 | 异常值鲁棒 |
| Fuzzy C-Means | 软聚类（成员度） | 模糊边界场景 |
| Bisecting K-Means | 层次化分裂 | 层次聚类 |

```python
# MiniBatch K-Means 对比
from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(
    n_clusters=4,
    batch_size=100,
    random_state=42
)
mbk.fit(X)

print(f"标准 K-Means Inertia: {kmeans.inertia_:.2f}")
print(f"MiniBatch Inertia: {mbk.inertia_:.2f}")
print(f"差异: {abs(kmeans.inertia_ - mbk.inertia)/kmeans.inertia*100:.1f}%")
```

---

## 十、实战建议

1. **必须标准化**：K-Means 对特征尺度极度敏感
2. **多次运行**：n_init=10 或更高，取最优结果
3. **选择最优 K**：结合肘部法则和轮廓系数
4. **大数据用 MiniBatch**：n>10000 时使用 MiniBatchKMeans
5. **异常值处理**：先检测并移除异常值，或用 K-Medoids
6. **可视化验证**：2D 数据用散点图，高维数据用 PCA 降维后观察

> 💡 **提示**：K-Means 是聚类入门必学算法，但它假设簇为球形。当数据具有任意形状的簇时，下一篇的 PCA 降维可以为数据提供更好的表示，而后面的 DBSCAN 等密度聚类方法可以处理非球形簇。

---

## 十一、总结

K-Means 是最经典、最常用的聚类算法，通过迭代优化簇内平方和来实现数据分组。核心要点包括：

- **算法流程**：初始化中心 → 分配样本 → 更新中心 → 收敛
- **目标函数**：最小化 WCSS（簇内平方和）
- **初始化策略**：K-Means++ 显著优于随机初始化
- **K 值选择**：肘部法则、轮廓系数、DBI 指数
- **变体**：MiniBatch K-Means 处理大数据集
- **局限性**：球形假设、对异常值敏感、需指定 K 值
- **应用场景**：客户分群、图像压缩、文档聚类、异常检测

---

📖 **上一篇**：[KNN 近邻算法](./22-knn.md) | 📖 **下一篇**：[PCA 降维](./24-pca-dimensionality.md) — 学习如何在保留信息的前提下降低数据维度。
