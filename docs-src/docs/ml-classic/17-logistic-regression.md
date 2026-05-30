---
title: 逻辑回归算法详解
icon: code-branch
order: 17
---

# 逻辑回归算法详解

逻辑回归（Logistic Regression）虽然名字中带有"回归"二字，但它实际上是一种经典的分类算法。它在线性回归的基础上引入 Sigmoid 函数，将线性模型的输出映射到 (0, 1) 区间，从而实现二分类任务。理解逻辑回归是深入掌握分类算法、支持向量机乃至神经网络的关键桥梁。

---

## 一、从线性回归到逻辑回归

### 1.1 为什么需要逻辑回归

线性回归的输出范围是 (-∞, +∞)，而分类任务的标签是离散的（如 0 和 1）。直接用线性回归做分类存在以下问题：

| 问题 | 说明 |
|------|------|
| 输出范围不匹配 | 线性回归输出可能远大于1或远小于0 |
| 对异常值敏感 | 极端样本会大幅拉偏决策边界 |
| 概率解释缺失 | 无法给出样本属于某类的概率 |

### 1.2 Sigmoid 函数

逻辑回归通过 Sigmoid 函数将线性输出压缩到 (0, 1) 区间：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

其中 z = Xw + b。

Sigmoid 函数的关键性质：

| 性质 | 值 |
|------|-----|
| 值域 | (0, 1) |
| σ(0) | 0.5 |
| σ(+∞) | 趋近于 1 |
| σ(-∞) | 趋近于 0 |
| 导数 | σ'(z) = σ(z)(1 - σ(z)) |

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid 函数，数值稳定实现"""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

# 可视化 Sigmoid 函数
z = np.linspace(-10, 10, 200)
y = sigmoid(z)

plt.figure(figsize=(8, 4))
plt.plot(z, y, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='g', linestyle='--', alpha=0.7)
plt.xlabel('z = Xw + b')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 二、逻辑回归的数学推导

### 2.1 假设函数

$$h_w(x) = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}$$

h_w(x) 表示样本属于正类（y=1）的概率。

### 2.2 决策规则

| 条件 | 预测 | 含义 |
|------|------|------|
| h_w(x) ≥ 0.5 | y = 1 | 属于正类 |
| h_w(x) < 0.5 | y = 0 | 属于负类 |

等价于：
- Xw + b ≥ 0 → y = 1
- Xw + b < 0 → y = 0

### 2.3 损失函数：交叉熵

逻辑回归使用对数损失（Log Loss），也称为二元交叉熵：

$$J(w) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(h_w(x_i)) + (1-y_i)\log(1-h_w(x_i))]$$

```python
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """计算二元交叉熵损失"""
    # 防止 log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

def log_loss_gradient(X, y, w, b):
    """计算逻辑回归的梯度"""
    n = X.shape[0]
    z = X.dot(w) + b
    y_pred = sigmoid(z)
    error = y_pred - y
    
    dw = (1 / n) * X.T.dot(error)
    db = (1 / n) * np.sum(error)
    return dw, db
```

### 2.4 为什么不用 MSE

| 损失函数 | 凸性 | 梯度特性 | 适用场景 |
|----------|------|----------|----------|
| MSE | 非凸（对逻辑回归） | 存在多个局部最优 | 线性回归 |
| 交叉熵 | 凸函数 | 梯度单调，无局部最优 | 逻辑回归 |

---

## 三、从零实现逻辑回归

```python
class LogisticRegression:
    """从零实现逻辑回归分类器"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # 前向传播
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)
            
            # 记录损失
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw, db = log_loss_gradient(X, y, self.weights, self.bias)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

# 使用示例：生成二分类数据
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2,
    n_redundant=0, random_state=42
)

model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
model.fit(X, y)

# 查看训练损失变化
print(f"最终损失: {model.loss_history[-1]:.4f}")
print(f"权重: {model.weights}")
print(f"偏置: {model.bias:.4f}")
```

---

## 四、sklearn 逻辑回归实战

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.datasets import make_classification

# 1. 加载数据
X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=8,
    n_redundant=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 特征标准化（逻辑回归对特征尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 训练模型
lr = LogisticRegression(
    penalty='l2',          # L2 正则化
    C=1.0,                 # 正则化强度的倒数
    solver='lbfgs',        # 优化算法
    max_iter=1000,         # 最大迭代次数
    random_state=42
)
lr.fit(X_train_scaled, y_train)

# 4. 预测与评估
y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("=== 模型评估 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率: {precision_score(y_test, y_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
print(f"AUC:    {roc_auc_score(y_test, y_proba):.4f}")

print("\n=== 混淆矩阵 ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred))
```

---

## 五、正则化与超参数

### 5.1 正则化参数 C

参数 C 是正则化强度的倒数：C 越小，正则化越强。

```python
import matplotlib.pyplot as plt

# 不同 C 值的影响
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
results = []

for C in C_values:
    lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, lr.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
    n_nonzero = np.sum(lr.coef_ != 0)
    
    results.append({
        "C": C,
        "训练集准确率": round(train_acc, 4),
        "测试集准确率": round(test_acc, 4),
        "非零系数": n_nonzero
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

### 5.2 不同正则化方法

| penalty | 说明 | 支持的 solver | 适用场景 |
|---------|------|---------------|----------|
| l2 | L2 正则化 | lbfgs, liblinear, sag, saga | 通用场景 |
| l1 | L1 正则化 | liblinear, saga | 特征选择 |
| elasticnet | 弹性网络 | saga | 特征多且相关 |
| None | 无正则化 | lbfgs, newton-cg, sag, saga | 数据量大且无过拟合 |

### 5.3 优化器选择

| solver | 说明 | 适用场景 |
|--------|------|----------|
| lbfgs | 拟牛顿法 | 默认选择，中小型数据集 |
| liblinear | 坐标下降 | 支持 L1 正则化，小数据集 |
| saga | 随机平均梯度 | 大型数据集，支持所有正则化 |
| newton-cg | 牛顿法 | 中小型数据集 |

---

## 六、多分类扩展：Softmax 回归

### 6.1 Softmax 函数

逻辑回归可以通过 Softmax 扩展到多分类：

$$P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

```python
def softmax(z):
    """数值稳定的 Softmax 函数"""
    # 减去最大值防止溢出
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 使用 sklearn 的 LogisticRegression 做多分类
lr_multi = LogisticRegression(
    multi_class='multinomial',  # 使用 Softmax
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

from sklearn.datasets import load_iris
iris = load_iris()
lr_multi.fit(iris.data, iris.target)

# 预测
predictions = lr_multi.predict(iris.data)
probabilities = lr_multi.predict_proba(iris.data)

print(f"类别: {iris.target_names}")
print(f"前5个样本的预测概率:")
print(np.round(probabilities[:5], 4))
```

### 6.2 One-vs-Rest 策略

| 策略 | 原理 | 训练分类器数量 | 适用场景 |
|------|------|---------------|----------|
| OvR | 每个类别 vs 其余所有 | K | 默认策略 |
| OvO | 每对类别之间训练 | K(K-1)/2 | 训练集很大时 |
| Multinomial | 直接使用 Softmax | 1 | 推荐用于多分类 |

---

## 七、实战项目：信用卡欺诈检测

```python
import pandas as pd
from sklearn.datasets import make_classification

# 1. 生成不平衡数据（模拟欺诈检测场景）
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=12,
    n_redundant=5,
    weights=[0.97, 0.03],  # 3% 的欺诈率
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 训练模型（注意处理类别不平衡）
lr = LogisticRegression(
    C=0.1,
    class_weight='balanced',  # 自动调整类别权重
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)

# 4. 在不同阈值下评估
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("=== 不同阈值下的评估结果 ===")
print(f"{'阈值':<6} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8}")
print("-" * 50)

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{threshold:<6.1f} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f}")

# 5. ROC 曲线分析
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
```

---

## 八、分类评估指标详解

### 8.1 混淆矩阵

|  | 预测正例 | 预测负例 |
|---|--------|--------|
| 实际正例 | TP（真正例） | FN（假负例） |
| 实际负例 | FP（假正例） | TN（真负例） |

### 8.2 核心指标

| 指标 | 公式 | 含义 | 适用场景 |
|------|------|------|----------|
| 准确率 | (TP+TN)/(TP+TN+FP+FN) | 整体预测正确率 | 类别平衡时 |
| 精确率 | TP/(TP+FP) | 预测为正的样本中有多少是真的 | 关注误报时 |
| 召回率 | TP/(TP+FN) | 真实正例中有多少被找出 | 关注漏报时 |
| F1 分数 | 2×P×R/(P+R) | 精确率和召回率的调和平均 | 综合评估 |
| AUC | ROC 曲线下面积 | 模型区分能力 | 类别不平衡时 |

---

## 九、实战建议

1. **特征标准化**：逻辑回归对特征尺度敏感，训练前务必做标准化
2. **处理不平衡**：使用 class_weight='balanced' 或调整分类阈值
3. **正则化选择**：默认 L2 正则化，需要特征选择时用 L1
4. **概率校准**：当需要准确的概率输出时，使用 Platt Scaling 或 Isotonic Regression
5. **决策阈值**：根据业务需求（更关注精确率还是召回率）调整阈值

> 💡 **提示**：逻辑回归虽然是一个线性分类器，但通过特征工程（如多项式特征、特征交叉）可以处理非线性问题。同时，它是神经网络的基础——可以将神经元看作带有激活函数的逻辑回归单元。

---

## 十、总结

逻辑回归将线性回归扩展到分类领域，是机器学习中最重要的分类算法之一。核心要点包括：

- **核心思想**：通过 Sigmoid 函数将线性输出映射到 (0, 1)，表示属于正类的概率
- **损失函数**：使用交叉熵（对数损失），保证了凸优化性质
- **正则化**：L1 正则化用于特征选择，L2 正则化防止过拟合
- **多分类扩展**：通过 Softmax 或 One-vs-Rest 策略实现
- **评估体系**：精确率、召回率、F1、AUC 是核心评估指标
- **工业应用**：广告点击率预测、信用评分、欺诈检测等领域广泛应用

---

📖 **上一篇**：[线性回归](./16-linear-regression.md) | 📖 **下一篇**：[支持向量机](./18-svm.md) — 学习如何找到最优分类超平面，理解间隔最大化与核技巧。
