---
title: 随机森林算法详解
icon: tree
order: 20
---

# 随机森林算法详解

随机森林（Random Forest）是基于决策树的集成学习算法，通过构建多棵决策树并综合它们的预测结果，有效克服了单棵决策树容易过拟合的缺点。随机森林是机器学习实践中最实用的"开箱即用"算法之一，在分类和回归任务中都表现出色。

---

## 一、从决策树到随机森林

### 1.1 单棵决策树的局限性

| 问题 | 说明 | 影响 |
|------|------|------|
| 过拟合 | 树太深时记住训练数据的噪声 | 泛化能力差 |
| 不稳定性 | 数据微小变化导致不同树结构 | 结果不可靠 |
| 高方差 | 对训练集扰动敏感 | 测试表现波动大 |

### 1.2 集成学习的思路

集成学习（Ensemble Learning）的核心思想：多个弱学习器组合成一个强学习器。

| 策略 | 原理 | 代表算法 |
|------|------|----------|
| Bagging | 并行训练，投票/平均 | 随机森林 |
| Boosting | 串行训练，关注错题 | AdaBoost, GBDT, XGBoost |
| Stacking | 分层训练，元学习器 | Stacked Generalization |

### 1.3 随机森林的两个"随机"

随机森林的"随机"体现在两个层面：

| 随机性 | 说明 | 作用 |
|--------|------|------|
| 样本随机（Bootstrap） | 有放回抽样，每棵树用不同的训练子集 | 减少树之间的相关性 |
| 特征随机 | 每个分裂节点只考虑随机子集的特征 | 增加多样性，防止过拟合 |

```python
import numpy as np

def bootstrap_sample(X, y):
    """Bootstrap 采样"""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

# 演示 Bootstrap 采样的特点
n = 1000
original = np.arange(n)
bootstrap = np.random.choice(n, size=n, replace=True)

unique_in_bootstrap = len(np.unique(bootstrap))
print(f"原始样本数: {n}")
print(f"Bootstrap 中唯一样本数: {unique_in_bootstrap}")
print(f"占比: {unique_in_bootstrap/n*100:.1f}%")
print(f"未出现的样本（OOB）: {n - unique_in_bootstrap} ({(n-unique_in_bootstrap)/n*100:.1f}%)")
```

---

## 二、随机森林算法流程

```
输入: 训练集 D = {(x1,y1), ..., (xn,yn)}, 树的数量 T
输出: 随机森林模型

for t = 1 to T:
    1. Bootstrap 采样: 从 D 中有放回地抽取 n 个样本
    2. 构建树:
        for 每个节点:
            a. 随机选择 m 个特征 (m << 总特征数)
            b. 在 m 个特征中选最优分裂
            c. 分裂直到满足停止条件（不剪枝）
    3. 保存树 T_t

预测:
    分类: 多数投票  ŷ = mode{T_t(x)}
    回归: 平均      ŷ = mean{T_t(x)}
```

---

## 三、从零实现随机森林

```python
from collections import Counter

class RandomForest:
    """从零实现随机森林"""
    
    def __init__(self, n_trees=100, max_depth=None, 
                 min_samples_split=2, max_features='sqrt',
                 random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.oob_predictions = None
    
    def _bootstrap_sample(self, X, y):
        """Bootstrap 采样"""
        n = X.shape[0]
        indices = np.random.choice(n, size=n, replace=True)
        return X[indices], y[indices]
    
    def _get_n_features(self, n_total):
        """计算每棵树考虑的特征数"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total))
        elif self.max_features == 'log2':
            return int(np.log2(n_total))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_total
    
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.trees = []
        n_features = self._get_n_features(X.shape[1])
        
        for i in range(self.n_trees):
            # Bootstrap 采样
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # 训练决策树（使用 sklearn 的树简化实现）
            from sklearn.tree import DecisionTreeClassifier
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=n_features,
                random_state=np.random.randint(0, 10000)
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """预测类别（多数投票）"""
        # 收集所有树的预测
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # 多数投票
        predictions = []
        for i in range(X.shape[0]):
            votes = Counter(all_predictions[:, i])
            predictions.append(votes.most_common(1)[0][0])
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """预测概率（各树预测的平均）"""
        all_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_probas, axis=0)
    
    def feature_importances(self):
        """计算特征重要性"""
        importances = np.zeros(self.trees[0].n_features_in_)
        for tree in self.trees:
            importances += tree.feature_importances_
        return importances / len(self.trees)

# 测试
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForest(n_trees=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(f"随机森林准确率: {accuracy_score(y_test, predictions):.4f}")
```

---

## 四、sklearn 随机森林实战

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score

# 1. 分类任务
rf_clf = RandomForestClassifier(
    n_estimators=100,        # 树的数量
    max_depth=10,            # 最大深度
    min_samples_split=5,     # 分裂最小样本
    min_samples_leaf=2,      # 叶子最小样本
    max_features='sqrt',     # 每次分裂考虑的特征数
    bootstrap=True,          # 使用 Bootstrap 采样
    oob_score=True,          # 计算 OOB 分数
    n_jobs=-1,               # 并行处理
    random_state=42
)
rf_clf.fit(X_train, y_train)

# OOB 分数（无需单独验证集的评估）
print(f"OOB 分数: {rf_clf.oob_score_:.4f}")
print(f"训练集准确率: {rf_clf.score(X_train, y_train):.4f}")
print(f"测试集准确率: {rf_clf.score(X_test, y_test):.4f}")

# 2. 特征重要性
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n=== 特征重要性排名 ===")
print(f"{'排名':<4} {'特征重要性':<10} {'累计重要性':<10}")
cumulative = 0
for i, idx in enumerate(indices[:10]):
    cumulative += importances[idx]
    print(f"{i+1:<4} {importances[idx]:<10.4f} {cumulative:<10.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), indices[:10])
plt.xlabel('特征索引')
plt.ylabel('重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()
```

---

## 五、OOB 评估

### 5.1 什么是 OOB

OOB（Out-of-Bag）样本是指在 Bootstrap 采样中未被选中的样本，约占原始数据的 36.8%。这些样本可以用于评估模型，无需单独划分验证集。

### 5.2 OOB 的优势

| 对比 | OOB 评估 | 交叉验证 |
|------|----------|----------|
| 数据利用率 | 100% | 每次只用部分数据 |
| 计算成本 | 低（免费获得） | 高（需要重新训练） |
| 评估可靠性 | 中等 | 高 |
| 适用场景 | 随机森林自带 | 所有模型 |

```python
# OOB 分数与交叉验证的对比
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# OOB 分数
oob_score = rf.oob_score_

# 5折交叉验证
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()

print(f"OOB 分数: {oob_score:.4f}")
print(f"CV 平均分数: {cv_mean:.4f} (+/- {cv_scores.std()*2:.4f})")
```

---

## 六、超参数调优

### 6.1 关键超参数

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|----------|
| n_estimators | 树的数量 | 100 | 越多越好，注意边际效益递减 |
| max_depth | 树的最大深度 | None | 5~20，防止过拟合 |
| min_samples_split | 分裂最小样本数 | 2 | 5~20 |
| min_samples_leaf | 叶子最小样本数 | 1 | 1~5 |
| max_features | 分裂考虑的特征数 | sqrt | sqrt/log2/None/具体数字 |
| bootstrap | 是否Bootstrap采样 | True | 一般保持True |

### 6.2 网格搜索调优

```python
from sklearn.model_selection import RandomizedSearchCV

# 随机搜索（比网格搜索更快）
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"最优参数: {random_search.best_params_}")
print(f"最优交叉验证准确率: {random_search.best_score_:.4f}")
```

### 6.3 n_estimators 的影响

```python
# 研究树的数量对模型性能的影响
n_trees_range = [10, 20, 50, 100, 200, 500]
train_scores = []
test_scores = []

for n in n_trees_range:
    rf = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(n_trees_range, train_scores, 'b-o', label='训练集')
plt.plot(n_trees_range, test_scores, 'r-o', label='测试集')
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('树的数量对模型性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 七、随机森林回归

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# 加载数据
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# 训练回归森林
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf_reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = rf_reg.predict(X_test)
print(f"R² 分数: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

## 八、随机森林的优势与局限

### 8.1 核心优势

| 优势 | 说明 |
|------|------|
| 准确率高 | 集成多棵树，通常表现优异 |
| 抗过拟合 | Bagging + 特征随机，有效控制方差 |
| 无需标准化 | 对特征尺度不敏感 |
| 特征重要性 | 自动评估各特征的贡献度 |
| 鲁棒性强 | 对异常值和缺失值有较好的容忍度 |
| 并行训练 | 树之间相互独立，可并行加速 |
| OOB 评估 | 无需单独验证集即可评估 |

### 8.2 局限性

| 局限 | 说明 | 缓解方案 |
|------|------|----------|
| 模型体积大 | 100棵树可能占用数百MB | 减少树数或使用 ExtraTrees |
| 预测速度慢 | 需要遍历所有树 | 减少树数或限制深度 |
| 不如 Boosting | 精度略低于 XGBoost/LightGBM | 使用梯度提升方法 |
| 外推能力差 | 无法预测训练范围外的值 | 结合线性模型 |
| 可解释性中等 | 不如单棵树直观 | 使用特征重要性或 SHAP 值 |

---

## 九、实战建议

1. **默认参数即可**：随机森林通常使用默认参数就有不错的表现
2. **增加树的数量**：更多的树不会导致过拟合，只会增加计算成本
3. **控制树深度**：max_depth 是防止过拟合最有效的参数
4. **利用 OOB**：开启 oob_score=True 获取免费的性能评估
5. **特征工程**：随机森林能自动处理非线性关系，但仍需关注缺失值处理

> 💡 **提示**：随机森林属于 Bagging 策略的代表，通过并行训练多棵树并投票来提高稳定性。下一篇我们将学习 Boosting 策略，通过串行训练、逐步纠错来进一步提升模型精度。

---

## 十、总结

随机森林是基于 Bagging 的集成学习算法，通过组合多棵随机决策树实现强大的预测能力。核心要点包括：

- **两个随机性**：Bootstrap 样本随机 + 分裂特征随机
- **集成方式**：分类用多数投票，回归用平均值
- **OOB 评估**：利用未被采样的样本进行免费评估
- **特征重要性**：天然支持特征重要性分析
- **调参策略**：优先调整 max_depth 和 min_samples_split
- **应用广泛**：分类、回归、特征选择、异常检测均可使用

---

📖 **上一篇**：[决策树](./19-decision-trees.md) | 📖 **下一篇**：[GBM 与 XGBoost](./21-gbm-xgboost.md) — 学习梯度提升算法，理解如何通过串行纠错构建更强的模型。
