---
title: 决策树算法详解
icon: project-diagram
order: 19
---

# 决策树算法详解

决策树（Decision Tree）是最直观、最易于理解的机器学习算法之一。它通过树状结构对数据进行逐层划分，每一步都基于某个特征做出决策，最终到达叶子节点得到预测结果。决策树不仅是独立的强大算法，更是随机森林、XGBoost 等集成学习方法的基石。

---

## 一、决策树的核心思想

### 1.1 什么是决策树

决策树是一种树形结构，其中：
- **内部节点**：表示对某个特征的判断
- **分支**：表示判断的结果
- **叶子节点**：表示最终的预测结果

```
              [年龄 >= 30?]
              /          \
           是 /            \ 否
            /              \
      [收入 >= 5万?]      [学历 = 本科?]
        /        \          /        \
      是/          \否     是/          \否
      /            \       /            \
   [购买]      [信用分?] [购买]       [不购买]
                 /   \
               是/     \否
               /       \
           [购买]    [不购买]
```

### 1.2 决策树的应用场景

| 场景 | 特征 | 目标 | 优势 |
|------|------|------|------|
| 信用评估 | 年龄、收入、信用记录 | 是否放贷 | 规则可解释 |
| 医疗诊断 | 症状、化验结果 | 疾病类型 | 符合医生思维 |
| 客户流失 | 使用频率、投诉次数 | 是否流失 | 可视化分析 |
| 风控决策 | 交易金额、时间、地点 | 是否欺诈 | 规则透明 |
| 用户画像 | 行为数据、人口统计 | 用户类别 | 特征重要性清晰 |

---

## 二、特征选择：分裂准则

决策树构建的核心问题：在每一步，应该选择哪个特征进行分裂？

### 2.1 信息增益（ID3 算法）

信息增益基于信息论中的熵概念：

$$H(D) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

$$Gain(D, A) = H(D) - \sum_{v=1}^{V}\frac{|D_v|}{|D|}H(D_v)$$

```python
def entropy(y):
    """计算信息熵"""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain(y, X, feature_idx):
    """计算信息增益"""
    parent_entropy = entropy(y)
    
    values = X[:, feature_idx]
    unique_values = np.unique(values)
    
    weighted_child_entropy = 0
    for v in unique_values:
        mask = values == v
        child_entropy = entropy(y[mask])
        weighted_child_entropy += (np.sum(mask) / len(y)) * child_entropy
    
    return parent_entropy - weighted_child_entropy
```

### 2.2 增益率（C4.5 算法）

信息增益偏向于选择取值多的特征，增益率对此进行了修正：

$$SplitInfo(A) = -\sum_{v=1}^{V}\frac{|D_v|}{|D|}\log_2\frac{|D_v|}{|D|}$$

$$GainRatio(D, A) = \frac{Gain(D, A)}{SplitInfo(A)}$$

### 2.3 基尼不纯度（CART 算法）

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

```python
def gini_impurity(y):
    """计算基尼不纯度"""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)
```

### 2.4 分裂准则对比

| 准则 | 算法 | 处理类型 | 优缺点 |
|------|------|----------|--------|
| 信息增益 | ID3 | 离散特征 | 偏向多值特征 |
| 增益率 | C4.5 | 离散+连续 | 修正了多值偏向 |
| 基尼不纯度 | CART | 离散+连续 | 计算效率高，默认选择 |

---

## 三、从零实现决策树

```python
class TreeNode:
    """决策树节点"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx    # 分裂特征索引
        self.threshold = threshold         # 分裂阈值（连续特征）
        self.left = left                   # 左子树
        self.right = right                 # 右子树
        self.value = value                 # 叶子节点预测值


class DecisionTree:
    """从零实现 CART 决策树"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # 终止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return TreeNode(value=self._most_common_label(y))
        
        # 寻找最佳分裂
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            return TreeNode(value=self._most_common_label(y))
        
        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 检查子节点样本数
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            return TreeNode(value=self._most_common_label(y))
        
        # 递归构建子树
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(feature_idx=best_feature, threshold=best_threshold,
                       left=left, right=right)
    
    def _best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        parent_gini = gini_impurity(y)
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 计算基尼增益
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_gini = (n_left / n_samples) * gini_impurity(y[left_mask]) + \
                                (n_right / n_samples) * gini_impurity(y[right_mask])
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
    
    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

# 测试
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

tree = DecisionTree(max_depth=4, min_samples_split=5)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print(f"自定义决策树准确率: {accuracy_score(y_test, predictions):.4f}")
```

---

## 四、sklearn 决策树实战

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report

# 1. 分类树
clf = DecisionTreeClassifier(
    criterion='gini',        # 分裂准则
    max_depth=4,             # 最大深度
    min_samples_split=5,     # 分裂所需最小样本数
    min_samples_leaf=2,      # 叶子节点最小样本数
    max_features=None,       # 考虑的最大特征数
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("=== 分类树评估 ===")
print(f"训练集准确率: {clf.score(X_train, y_train):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 2. 特征重要性
print("=== 特征重要性 ===")
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

# 3. 可视化决策树
plt.figure(figsize=(16, 8))
plot_tree(clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.show()
```

---

## 五、回归树

决策树同样可以用于回归任务，分裂准则变为最小化方差。

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 加载回归数据
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# 训练回归树
reg_tree = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
reg_tree.fit(X_train, y_train)

y_pred = reg_tree.predict(X_test)
print(f"回归树 R²: {r2_score(y_test, y_pred):.4f}")
print(f"回归树 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 不同深度的影响
print("\n=== 不同深度的影响 ===")
print(f"{'深度':<4} {'训练R²':<10} {'测试R²':<10} {'过拟合':<6}")
print("-" * 40)

for depth in [1, 2, 3, 5, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_r2 = tree.score(X_train, y_train)
    test_r2 = tree.score(X_test, y_test)
    overfit = "是" if train_r2 - test_r2 > 0.1 else "否"
    print(f"{str(depth):<4} {train_r2:<10.4f} {test_r2:<10.4f} {overfit:<6}")
```

---

## 六、剪枝技术

### 6.1 预剪枝（Pre-pruning）

在树构建过程中提前终止分裂。

| 参数 | 说明 | 默认值 | 建议 |
|------|------|--------|------|
| max_depth | 最大深度 | None | 3~10 |
| min_samples_split | 分裂所需最小样本 | 2 | 5~20 |
| min_samples_leaf | 叶子节点最小样本 | 1 | 1~5 |
| max_features | 最大特征数 | None | sqrt(n_features) |

### 6.2 后剪枝（Post-pruning）

先构建完整树，再从下往上剪枝。

```python
from sklearn.tree import cost_complexity_pruning_path

# 获取剪枝路径
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)

path = cost_complexity_pruning_path(X_train, y_train, clf_full)
ccp_alphas = path.ccp_alphas

# 不同剪枝强度的效果
print(f"{'Alpha':<12} {'深度':<4} {'叶子节点':<6} {'训练准确率':<8} {'测试准确率':<8}")
print("-" * 55)

for alpha in ccp_alphas[:10]:
    clf_pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf_pruned.fit(X_train, y_train)
    train_acc = clf_pruned.score(X_train, y_train)
    test_acc = clf_pruned.score(X_test, y_test)
    print(f"{alpha:<12.4f} {clf_pruned.get_depth():<4} {clf_pruned.get_n_leaves():<6} {train_acc:<8.4f} {test_acc:<8.4f}")
```

### 6.3 预剪枝 vs 后剪枝

| 对比 | 预剪枝 | 后剪枝 |
|------|--------|--------|
| 时机 | 构建时 | 构建后 |
| 速度 | 快 | 慢 |
| 欠拟合风险 | 有 | 较小 |
| 效果 | 一般 | 通常更好 |
| sklearn 支持 | 通过超参数 | 通过 ccp_alpha |

---

## 七、决策树的优缺点

### 7.1 优点

| 优点 | 说明 |
|------|------|
| 可解释性强 | 规则可视化，业务人员可理解 |
| 无需特征缩放 | 不受特征量纲影响 |
| 处理混合类型 | 同时支持数值和类别特征 |
| 非线性建模 | 自动学习非线性关系 |
| 特征选择 | 自动忽略不重要的特征 |

### 7.2 缺点

| 缺点 | 说明 | 解决方案 |
|------|------|----------|
| 容易过拟合 | 树太深会记住训练数据 | 剪枝、限制深度 |
| 不稳定 | 数据微小变化导致树结构大变化 | 使用集成方法 |
| 偏向主导特征 | 取值多的特征更容易被选中 | 使用随机森林 |
| 不擅长外推 | 无法预测训练范围之外的值 | 使用线性模型或集成 |
| XOR 问题 | 需要多层才能学习异或关系 | 使用集成方法 |

---

## 八、实战建议

1. **控制深度**：max_depth 是最重要的调参参数，通常 3~10 为宜
2. **最小样本数**：增大 min_samples_split 和 min_samples_leaf 可防止过拟合
3. **可视化分析**：使用 plot_tree 观察树的分裂逻辑，理解模型决策
4. **集成优先**：单棵决策树通常不够稳定，建议直接使用随机森林或 XGBoost
5. **特征重要性**：利用 tree.feature_importances_ 进行特征选择

> 💡 **提示**：决策树虽然直观，但单棵树往往容易过拟合。下一篇我们将学习随机森林，通过"群体智慧"将多棵决策树组合起来，大幅提升模型的稳定性和泛化能力。

---

## 九、总结

决策树是一种直观、可解释的监督学习算法，通过树状结构对数据进行逐层划分。核心要点包括：

- **分裂准则**：信息增益（ID3）、增益率（C4.5）、基尼不纯度（CART）
- **构建过程**：贪心策略，每一步选择最优分裂
- **剪枝技术**：预剪枝（控制超参数）和后剪枝（ccp_alpha）是防止过拟合的关键
- **适用场景**：分类和回归均可，特别适合需要可解释性的场景
- **局限性**：容易过拟合、不稳定，通常作为集成方法的基础组件

---

📖 **上一篇**：[支持向量机](./18-svm.md) | 📖 **下一篇**：[随机森林](./20-random-forest.md) — 学习如何通过 Bagging 集成多棵决策树，构建更强大的模型。
