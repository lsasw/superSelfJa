---
title: 支持向量机算法详解
icon: vector-square
order: 18
---

# 支持向量机（SVM）算法详解

支持向量机（Support Vector Machine, SVM）是机器学习中最强大的分类算法之一。它通过寻找最优分类超平面，使得两类样本之间的间隔最大化。SVM 不仅在理论上有严格的数学保证，在实践中也展现出了卓越的性能，尤其是在小样本、高维数据的场景下。

---

## 一、SVM 的核心思想

### 1.1 什么是支持向量机

SVM 的基本目标是找到一个超平面，使得：
1. 该超平面能够正确分开两类样本
2. 该超平面到最近样本点的距离（间隔）最大

**超平面方程：**

$$w^Tx + b = 0$$

其中 w 是法向量，决定了超平面的方向；b 是偏置，决定了超平面的位置。

### 1.2 为什么间隔最大化很重要

| 间隔大小 | 训练误差 | 泛化能力 | 鲁棒性 |
|----------|----------|----------|--------|
| 小间隔 | 可能为零 | 差 | 对新样本敏感 |
| 大间隔 | 可能为零 | 好 | 对新样本鲁棒 |

> **提示**：SVM 只依赖距离超平面最近的几个样本点（支持向量），这使得它对远离边界的样本不敏感，具有良好的鲁棒性。

---

## 二、线性可分 SVM 的数学推导

### 2.1 间隔的定义

样本点 (x_i, y_i) 到超平面的距离为：

$$d_i = \frac{y_i(w^Tx_i + b)}{\|w\|}$$

函数间隔为 γ_i = y_i(w^T x_i + b)，几何间隔为 γ_i / ||w||。

### 2.2 原始优化问题

$$\min_{w,b} \frac{1}{2}\|w\|^2$$

$$\text{s.t. } y_i(w^Tx_i + b) \geq 1, \quad i = 1, 2, ..., n$$

### 2.3 对偶问题

通过拉格朗日乘子法，将原始问题转化为对偶问题：

$$\max_{\alpha} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j y_i y_j x_i^T x_j$$

$$\text{s.t. } \alpha_i \geq 0, \quad \sum_{i=1}^{n}\alpha_i y_i = 0$$

**对偶问题的关键洞察：**
- 只有支持向量对应的 α_i > 0
- 决策函数只依赖支持向量
- 可以通过核函数处理非线性问题

---

## 三、软间隔 SVM

### 3.1 为什么需要软间隔

现实数据很少是完美线性可分的。软间隔 SVM 允许部分样本被误分类，通过引入松弛变量 ξ_i 和惩罚参数 C 来平衡间隔大小与分类错误。

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i$$

$$\text{s.t. } y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### 3.2 参数 C 的影响

| C 值 | 正则化强度 | 间隔大小 | 误分类容忍 | 适用场景 |
|------|-----------|----------|-----------|----------|
| 很小 | 强 | 大 | 高 | 噪声多、数据重叠 |
| 适中 | 中 | 中 | 中 | 一般场景 |
| 很大 | 弱 | 小 | 低 | 数据干净、可分 |

---

## 四、核技巧：处理非线性问题

### 4.1 核函数的作用

当数据线性不可分时，SVM 通过核函数将数据映射到高维空间，在高维空间中寻找线性超平面。

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

### 4.2 常用核函数

| 核函数 | 公式 | 参数 | 适用场景 |
|--------|------|------|----------|
| 线性核 | x_i^T x_j | 无 | 特征维度高、样本量大 |
| 多项式核 | (γx_i^T x_j + r)^d | degree, gamma, coef0 | 结构化数据 |
| RBF 核 | exp(-γ||x_i - x_j||²) | gamma | 通用场景，默认选择 |
| Sigmoid 核 | tanh(γx_i^T x_j + r) | gamma, coef0 | 类似神经网络 |

### 4.3 RBF 核的直观理解

RBF（径向基函数）核是最常用的核函数，它将每个样本点视为一个"中心"，通过高斯函数衡量样本间的相似度。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# 生成非线性可分数据
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X = StandardScaler().fit_transform(X)

# 对比不同核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, kernel in zip(axes, kernels):
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X, y)
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    ax.set_title(f'Kernel: {kernel}')

plt.tight_layout()
plt.show()
```

---

## 五、从零实现线性 SVM

```python
class SVM:
    """从零实现线性 SVM（使用梯度下降近似）"""
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 将 y 从 {0, 1} 转换为 {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 测试
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=2, n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
predictions = np.where(predictions == -1, 0, 1)  # 转换回 {0, 1}

print(f"SVM 准确率: {accuracy_score(y_test, predictions):.4f}")
```

---

## 六、sklearn SVM 实战

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# 1. 加载数据
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 网格搜索最优参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid = GridSearchCV(
    SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid.fit(X_train_scaled, y_train)

print(f"最优参数: {grid.best_params_}")
print(f"最优交叉验证准确率: {grid.best_score_:.4f}")

# 4. 使用最优模型评估
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. 查看支持向量
print(f"支持向量数量: {len(best_svm.support_vectors_)}")
print(f"支持向量占比: {len(best_svm.support_vectors_)/len(X_train)*100:.1f}%")
```

---

## 七、SVM 多分类策略

### 7.1 One-vs-One（OvO）

| 特性 | 说明 |
|------|------|
| 原理 | 每两个类别之间训练一个分类器 |
| 分类器数量 | K(K-1)/2 |
| 训练速度 | 每个分类器只用两个类别的数据 |
| 预测方式 | 投票机制 |

### 7.2 One-vs-Rest（OvR）

| 特性 | 说明 |
|------|------|
| 原理 | 每个类别 vs 其余所有类别 |
| 分类器数量 | K |
| 训练速度 | 每个分类器使用全部数据 |
| 预测方式 | 选得分最高的类别 |

```python
# OvO vs OvR 对比
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

ovo_svm = OneVsOneClassifier(SVC(kernel='rbf', random_state=42))
ovr_svm = OneVsRestClassifier(SVC(kernel='rbf', random_state=42))

ovo_svm.fit(X_train_s, y_train)
ovr_svm.fit(X_train_s, y_train)

print(f"OvO 准确率: {accuracy_score(y_test, ovo_svm.predict(X_test_s)):.4f}")
print(f"OvR 准确率: {accuracy_score(y_test, ovr_svm.predict(X_test_s)):.4f}")
```

---

## 八、SVM 与其他分类器的对比

| 对比维度 | SVM | 逻辑回归 | 决策树 |
|----------|-----|----------|--------|
| 决策边界 | 最大间隔超平面 | 概率阈值超平面 | 轴平行分割 |
| 核技巧 | 支持 | 不支持 | 不支持 |
| 输出 | 类别标签/距离 | 概率 | 类别标签 |
| 异常值敏感 | 中等（软间隔缓解） | 较敏感 | 不敏感 |
| 小样本表现 | 优秀 | 一般 | 容易过拟合 |
| 高维表现 | 优秀 | 好 | 容易过拟合 |
| 可解释性 | 差 | 好 | 很好 |
| 训练速度 | O(n²)~O(n³) | 快 | 快 |

---

## 九、实战建议

1. **数据标准化**：SVM 对特征尺度非常敏感，训练前务必标准化
2. **核函数选择**：RBF 核是默认且通用的选择，线性核适合高维稀疏数据
3. **参数调优**：C 和 gamma 是最重要的两个参数，使用网格搜索或随机搜索
4. **大样本慎用**：SVM 训练时间复杂度较高，样本量超过 10 万时考虑其他方法
5. **概率输出**：SVM 不直接输出概率，需要调用 probability=True 使用 Platt Scaling

> 💡 **提示**：SVM 的核技巧思想深刻影响了后续算法的发展。核函数本质上是一种隐式的特征映射，这一思想在核 PCA、核 K-Means 等算法中都有应用。理解核函数是掌握非线性机器学习方法的关键。

---

## 十、总结

支持向量机是一种基于间隔最大化的强大分类算法。核心要点包括：

- **核心思想**：寻找使间隔最大化的最优分类超平面
- **数学基础**：凸二次规划问题，通过拉格朗日对偶求解
- **软间隔**：通过参数 C 控制间隔与误分类之间的权衡
- **核技巧**：通过核函数隐式映射到高维空间，解决非线性分类问题
- **支持向量**：只有边界样本影响决策边界，模型具有稀疏性
- **应用范围**：文本分类、图像识别、生物信息学等领域表现优异

---

📖 **上一篇**：[逻辑回归](./17-logistic-regression.md) | 📖 **下一篇**：[决策树](./19-decision-trees.md) — 学习基于树结构的分类与回归方法，理解信息增益与基尼不纯度。
