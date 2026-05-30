---
title: KNN 近邻算法详解
icon: dots-hexagon
order: 22
---

# KNN 近邻算法详解

K 近邻算法（K-Nearest Neighbors, KNN）是机器学习中最简单、最直观的算法之一。它的核心思想可以用一句话概括："近朱者赤，近墨者黑"。与前面学习的线性回归、SVM、决策树、随机森林和 XGBoost 不同，KNN 是一种"懒惰学习"方法——它没有显式的训练过程，所有计算都延迟到预测时进行。

---

## 一、KNN 的核心思想

### 1.1 算法原理

```
输入: 训练集 D = {(x1, y1), (x2, y2), ..., (xn, yn)}, 待预测样本 x, 近邻数 K
输出: 样本 x 的预测值

1. 计算 x 与 D 中每个样本的距离
2. 按距离从小到大排序
3. 选择前 K 个最近的样本
4. 分类: 多数投票; 回归: 取平均值
```

### 1.2 KNN 与其他算法的对比

| 对比维度 | 参数模型 | KNN（非参数模型） |
|----------|----------|-------------------|
| 训练阶段 | 学习参数，速度快 | 仅存储数据 |
| 预测阶段 | 计算简单，速度快 | 计算距离，速度慢 |
| 模型形式 | 显式公式 | 隐式（训练数据本身） |
| 假设 | 有（如线性、树形） | 无 |
| 适用场景 | 数据量大、维度适中 | 数据量适中、维度低 |

### 1.3 应用场景

| 场景 | 说明 | 为什么适合 KNN |
|------|------|---------------|
| 推荐系统 | 根据相似用户的行为推荐 | 用户相似度天然可计算 |
| 异常检测 | 与最近邻的距离过大则异常 | 局部密度信息直接可用 |
| 缺失值填充 | 用相似样本的值填充 | 相似样本的特征近似 |
| 图像分类 | 像素级相似度计算 | 小规模数据集效果好 |
| 文本分类 | 文档向量余弦相似度 | TF-IDF 向量空间天然适用 |

---

## 二、距离度量方法

### 2.1 常用距离公式

| 距离 | 公式 | 特点 |
|------|------|------|
| 欧氏距离 | $\sqrt{\sum(x_i - y_i)^2}$ | 最常用，各维度权重相等 |
| 曼哈顿距离 | $\sum\|x_i - y_i\|$ | 适合网格状空间 |
| 闵可夫斯基距离 | $(\sum\|x_i - y_i\|^p)^{1/p}$ | 欧氏和曼哈顿的推广 |
| 切比雪夫距离 | $\max\|x_i - y_i\|$ | 只关注最大差异维度 |
| 余弦相似度 | $\frac{x \cdot y}{\|x\|\|y\|}$ | 关注方向而非大小 |

```python
def minkowski_distance(x1, x2, p=2):
    """闵可夫斯基距离"""
    return np.sum(np.abs(x1 - x2) ** p) ** (1/p)

def euclidean_distance(x1, x2):
    """欧氏距离"""
    return minkowski_distance(x1, x2, p=2)

def manhattan_distance(x1, x2):
    """曼哈顿距离"""
    return minkowski_distance(x1, x2, p=1)

def cosine_similarity(x1, x2):
    """余弦相似度"""
    dot_product = np.dot(x1, x2)
    norm_product = np.linalg.norm(x1) * np.linalg.norm(x2)
    return dot_product / (norm_product + 1e-10)

def cosine_distance(x1, x2):
    """余弦距离（1 - 相似度）"""
    return 1 - cosine_similarity(x1, x2)

# 测试不同距离
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

print(f"欧氏距离: {euclidean_distance(x1, x2):.4f}")
print(f"曼哈顿距离: {manhattan_distance(x1, x2):.4f}")
print(f"p=3 闵可夫斯基距离: {minkowski_distance(x1, x2, p=3):.4f}")
print(f"余弦距离: {cosine_distance(x1, x2):.4f}")
```

### 2.2 距离度量对比

| 距离 | 对量纲敏感 | 对方向敏感 | 适用数据类型 |
|------|-----------|-----------|-------------|
| 欧氏距离 | 是 | 否 | 数值型（标准化后） |
| 曼哈顿距离 | 是 | 否 | 高维稀疏数据 |
| 余弦距离 | 否 | 是 | 文本、图像 |
| 马氏距离 | 否（考虑协方差） | 否 | 相关性强的特征 |

---

## 三、从零实现 KNN

```python
from collections import Counter

class KNN:
    """从零实现 KNN 算法"""
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights  # 'uniform' 或 'distance'
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """KNN 的 fit 只存储数据"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_distances(self, X):
        """计算待预测样本与所有训练样本的距离"""
        if self.distance_metric == 'euclidean':
            # 高效计算：(a-b)² = a² - 2ab + b²
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
            cross = X.dot(self.X_train.T)
            dists = np.sqrt(np.maximum(X_sq - 2 * cross + train_sq, 0))
            return dists
        else:
            # 通用实现
            n_test = X.shape[0]
            n_train = self.X_train.shape[0]
            dists = np.zeros((n_test, n_train))
            for i in range(n_test):
                for j in range(n_train):
                    dists[i, j] = euclidean_distance(X[i], self.X_train[j])
            return dists
    
    def predict(self, X):
        """预测类别"""
        X = np.array(X)
        dists = self._compute_distances(X)
        predictions = []
        
        for i in range(X.shape[0]):
            # 获取最近 K 个邻居的索引
            k_indices = np.argsort(dists[i])[:self.k]
            k_labels = self.y_train[k_indices]
            k_distances = dists[i][k_indices]
            
            if self.weights == 'uniform':
                # 等权重投票
                vote = Counter(k_labels).most_common(1)[0][0]
            else:
                # 距离加权投票
                weighted_votes = {}
                for label, dist in zip(k_labels, k_distances):
                    weight = 1 / (dist + 1e-10)
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                vote = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(vote)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """预测概率（各邻居类别占比）"""
        X = np.array(X)
        dists = self._compute_distances(X)
        classes = np.unique(self.y_train)
        all_probas = []
        
        for i in range(X.shape[0]):
            k_indices = np.argsort(dists[i])[:self.k]
            k_labels = self.y_train[k_indices]
            counts = Counter(k_labels)
            proba = [counts.get(c, 0) / self.k for c in classes]
            all_probas.append(proba)
        
        return np.array(all_probas)

# 测试
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

knn = KNN(k=5, distance_metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"KNN 准确率: {accuracy_score(y_test, predictions):.4f}")
```

---

## 四、sklearn KNN 实战

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. 标准化（KNN 对特征尺度极度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 训练分类器
knn = KNeighborsClassifier(
    n_neighbors=5,           # K 值
    weights='distance',      # 权重：'uniform' 或 'distance'
    algorithm='auto',        # 搜索算法：'auto', 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',      # 距离度量
    p=2,                     # 闵可夫斯基参数 (p=2 为欧氏距离)
    n_jobs=-1               # 并行
)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print("=== KNN 分类评估 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 五、K 值的选择

### 5.1 K 值对模型的影响

| K 值 | 决策边界 | 过拟合 | 欠拟合 | 计算成本 |
|------|----------|--------|--------|----------|
| 小（1~3） | 复杂、不规则 | 容易 | 不易 | 低（局部） |
| 中（5~15） | 平滑、合理 | 平衡 | 平衡 | 中 |
| 大（>20） | 平滑、简单 | 不易 | 容易 | 高（全局） |

```python
# 不同 K 值的性能对比
k_range = range(1, 31)
train_scores = []
test_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 5))
plt.plot(k_range, train_scores, 'b-o', label='训练集')
plt.plot(k_range, test_scores, 'r-o', label='测试集')
plt.axvline(x=k_range[np.argmax(test_scores)], color='g', linestyle='--', 
            label=f'最优 K={k_range[np.argmax(test_scores)]}')
plt.xlabel('K 值')
plt.ylabel('准确率')
plt.title('K 值对模型性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.2 K 值选择原则

| 原则 | 说明 |
|------|------|
| 避免偶数 | 二分类时防止投票平局 |
| 经验法则 | K ≈ √n（n 为样本数） |
| 交叉验证 | 最可靠的方法 |
| 样本分布不均 | 使用距离加权投票 |

---

## 六、加速搜索算法

### 6.1 搜索算法对比

| 算法 | 原理 | 构建复杂度 | 查询复杂度 | 适用场景 |
|------|------|-----------|-----------|----------|
| Brute Force | 暴力计算所有距离 | O(1) | O(n·d) | 小样本、高维 |
| KD-Tree | 二叉树空间分割 | O(dn log n) | O(d log n) | 低维（d < 20） |
| Ball Tree | 超球体空间分割 | O(dn log n) | O(d log n) | 中等维度 |

```python
# 不同搜索算法的性能对比
from sklearn.neighbors import KNeighborsClassifier
import time

for algorithm in ['brute', 'kd_tree', 'ball_tree', 'auto']:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algorithm, n_jobs=-1)
    
    start = time.time()
    knn.fit(X_train_scaled, y_train)
    fit_time = time.time() - start
    
    start = time.time()
    knn.predict(X_test_scaled)
    predict_time = time.time() - start
    
    score = knn.score(X_test_scaled, y_test)
    
    print(f"{algorithm:<12} | 训练: {fit_time*1000:.1f}ms | 预测: {predict_time*1000:.1f}ms | 准确率: {score:.4f}")
```

---

## 七、KNN 回归

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# 加载回归数据
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练 KNN 回归器
knn_reg = KNeighborsRegressor(
    n_neighbors=10,
    weights='distance',
    metric='minkowski',
    p=2,
    n_jobs=-1
)
knn_reg.fit(X_train_scaled, y_train)

y_pred = knn_reg.predict(X_test_scaled)
print(f"KNN 回归 R²: {r2_score(y_test, y_pred):.4f}")
print(f"KNN 回归 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

## 八、KNN 的优缺点

### 8.1 优点

| 优点 | 说明 |
|------|------|
| 简单直观 | 原理容易理解，实现简单 |
| 无训练过程 | 新增数据无需重新训练 |
| 无需假设 | 不对数据分布做任何假设 |
| 多分类友好 | 天然支持多分类 |
| 非线性建模 | 自动学习复杂决策边界 |

### 8.2 缺点

| 缺点 | 说明 | 缓解方案 |
|------|------|----------|
| 预测慢 | 每次预测需计算所有距离 | 使用 KD-Tree/Ball Tree |
| 内存大 | 需存储全部训练数据 | 使用原型选择/提取 |
| 维度灾难 | 高维下距离度量失效 | 降维（PCA） |
| 对量纲敏感 | 大尺度特征主导距离 | 特征标准化 |
| 不平衡数据 | 多数类主导投票 | 距离加权投票 |

---

## 九、维度灾难详解

当特征维度增加时，数据点之间的距离趋于相似，使得"近邻"的概念变得模糊。

```python
# 维度灾难演示
np.random.seed(42)

dimensions = [2, 5, 10, 50, 100, 500, 1000]
n_samples = 1000

for dim in dimensions:
    # 在 [0,1]^d 中随机采样
    X = np.random.uniform(0, 1, size=(n_samples, dim))
    
    # 计算所有点到第一个点的距离
    distances = np.linalg.norm(X - X[0], axis=1)
    
    min_dist = np.min(distances[1:])  # 排除自身
    max_dist = np.max(distances)
    mean_dist = np.mean(distances[1:])
    
    ratio = (max_dist - min_dist) / mean_dist
    
    print(f"维度={dim:<4} | 最小={min_dist:.4f} | 最大={max_dist:.4f} | "
          f"平均={mean_dist:.4f} | 散布率={ratio:.4f}")
```

| 维度 | 现象 | 对策 |
|------|------|------|
| 低维（d<10） | 距离区分明显 | KNN 效果良好 |
| 中维（10~50） | 距离开始模糊 | 特征选择、标准化 |
| 高维（d>100） | 距离区分几乎消失 | PCA 降维、特征选择 |

---

## 十、实战建议

1. **必须标准化**：KNN 对特征尺度极度敏感，标准化是必须的预处理步骤
2. **交叉验证选 K**：使用交叉验证确定最优 K 值，不要凭感觉
3. **距离加权**：样本不平衡时使用 weights='distance'
4. **维度控制**：高维数据先做 PCA 或特征选择，再用 KNN
5. **搜索算法**：低维用 kd_tree，高维用 brute，不确定用 auto
6. **样本裁剪**：大样本可以考虑 Edited NN 等算法裁剪冗余样本

> 💡 **提示**：KNN 作为基于实例的学习方法，与之前所有参数化模型形成鲜明对比。它的"懒惰学习"特性使其在数据频繁变化的场景中特别有用。接下来我们将学习第一类无监督学习算法——K-Means 聚类。

---

## 十一、总结

KNN 是一种简单而强大的非参数学习算法，通过计算样本间距离进行预测。核心要点包括：

- **核心思想**："近朱者赤"，根据最近邻的标签进行预测
- **距离度量**：欧氏距离最常用，文本数据常用余弦相似度
- **K 值选择**：通过交叉验证确定，通常 √n 附近
- **加权策略**：uniform 等权重，distance 距离加权
- **加速方法**：KD-Tree、Ball Tree 加速近邻搜索
- **维度灾难**：高维下距离度量失效，需先降维
- **应用场景**：推荐系统、异常检测、缺失值填充、小样本分类

---

📖 **上一篇**：[GBM 与 XGBoost](./21-gbm-xgboost.md) | 📖 **下一篇**：[K-Means 聚类](./23-kmeans-clustering.md) — 进入无监督学习领域，学习最经典的聚类算法。
