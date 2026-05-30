---
title: GBM 与 XGBoost 算法详解
icon: chart-bar
order: 21
---

# GBM 与 XGBoost 算法详解

梯度提升机（Gradient Boosting Machine, GBM）及其高效实现 XGBoost 是机器学习中精度最高的传统算法之一。它们通过串行训练多个弱学习器，每个学习器专注于修正前一个模型的错误，逐步提升预测精度。在深度学习普及之前，XGBoost 几乎统治了所有结构化数据的机器学习竞赛。

---

## 一、从 Bagging 到 Boosting

### 1.1 两种集成策略的对比

| 对比维度 | Bagging（随机森林） | Boosting（GBM/XGBoost） |
|----------|---------------------|------------------------|
| 训练方式 | 并行 | 串行 |
| 样本采样 | Bootstrap 随机抽样 | 关注上一轮的错误样本 |
| 模型权重 | 平等投票 | 加权求和 |
| 偏差-方差 | 主要降低方差 | 主要降低偏差 |
| 过拟合风险 | 低 | 较高 |
| 训练速度 | 快（可并行） | 慢（必须串行） |
| 精度 | 高 | 通常更高 |

### 1.2 Boosting 的核心思想

```
初始模型: F_0(x) = 0

第 1 轮: 训练弱学习器 h_1 拟合 y
         F_1(x) = F_0(x) + η · h_1(x)

第 2 轮: 计算残差 r_2 = y - F_1(x)
         训练 h_2 拟合 r_2
         F_2(x) = F_1(x) + η · h_2(x)

...

第 m 轮: F_m(x) = F_{m-1}(x) + η · h_m(x)
```

其中 η 是学习率，控制每棵树的贡献。

---

## 二、梯度提升机（GBM）原理

### 2.1 算法框架

```
输入: 训练集 {(xi, yi)}, 损失函数 L(y, F), 学习率 η, 树的数量 M
输出: 最终模型 F_M(x)

1. 初始化: F_0(x) = argmin_γ Σ L(yi, γ)

2. for m = 1 to M:
    a. 计算伪残差: r_im = -[∂L(yi, F(xi))/∂F(xi)]|_{F=F_{m-1}}
    b. 用回归树拟合 r_im，得到树结构 R_mj (j=1,...,J)
    c. 计算每个叶子节点的最优值 γ_mj
    d. 更新: F_m(x) = F_{m-1}(x) + η · Σ γ_mj · I(x ∈ R_mj)

3. 返回: F_M(x)
```

### 2.2 损失函数选择

| 损失函数 | 适用任务 | 说明 |
|----------|----------|------|
| squared_error | 回归 | 最小二乘，对异常值敏感 |
| absolute_error | 回归 | 最小绝对偏差，鲁棒性强 |
| huber | 回归 | 结合 MSE 和 MAE |
| log_loss | 二分类 | 对数损失，等价于逻辑回归 |
| multi_log_loss | 多分类 | 多项对数损失 |

### 2.3 从零实现简单 GBM

```python
import numpy as np

class SimpleGBM:
    """简化版梯度提升机（回归）"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None
    
    def fit(self, X, y):
        from sklearn.tree import DecisionTreeRegressor
        
        # 初始化：用均值作为初始预测
        self.init_prediction = np.mean(y)
        F = np.full(len(y), self.init_prediction)
        self.trees = []
        
        for m in range(self.n_estimators):
            # 计算负梯度（伪残差）
            residuals = y - F
            
            # 训练新树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=m)
            tree.fit(X, residuals)
            
            # 更新预测
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            
            if (m + 1) % 20 == 0:
                mse = np.mean((y - F) ** 2)
                print(f"迭代 {m+1}/{self.n_estimators}, MSE: {mse:.4f}")
        
        return self
    
    def predict(self, X):
        F = np.full(X.shape[0], self.init_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

# 测试
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = SimpleGBM(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

## 三、XGBoost：高效的梯度提升

### 3.1 XGBoost 的创新

| 创新 | 说明 | 效果 |
|------|------|------|
| 二阶泰勒展开 | 同时使用一阶和二阶导数 | 更快的收敛 |
| 正则化 | L1 + L2 正则化树复杂度 | 减少过拟合 |
| 缺失值处理 | 自动学习缺失值分裂方向 | 无需预处理 |
| 并行计算 | 特征级别的并行 | 大幅提升速度 |
| 缓存优化 | 高效的数据结构 | 内存友好 |
| 交叉验证 | 内置 CV，支持早停 | 防止过拟合 |

### 3.2 目标函数

$$Obj = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

其中 T 是叶子节点数，w 是叶子节点权重。

---

## 四、XGBoost 实战

### 4.1 分类任务

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. 加载数据
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=12,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 转换为 DMatrix（XGBoost 的高效数据结构）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 3. 配置参数
params = {
    'objective': 'binary:logistic',   # 二分类
    'eval_metric': 'logloss',         # 评估指标
    'eta': 0.1,                       # 学习率
    'max_depth': 6,                   # 最大深度
    'min_child_weight': 1,            # 子节点最小权重和
    'gamma': 0,                       # 叶子节点最小损失减少
    'subsample': 0.8,                 # 样本采样比例
    'colsample_bytree': 0.8,          # 特征采样比例
    'lambda': 1,                      # L2 正则化
    'alpha': 0,                       # L1 正则化
    'seed': 42
}

# 4. 训练（带早停）
evals = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=50
)

# 5. 预测
y_pred_proba = bst.predict(dtest)
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"XGBoost 准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))
```

### 4.2 回归任务

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

bst = xgb.train(
    params, dtrain, num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=20,
    verbose_eval=50
)

y_pred = bst.predict(dtest)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### 4.3 sklearn 风格接口

```python
from xgboost import XGBClassifier, XGBRegressor

# sklearn 风格更简洁
xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
print(f"XGBClassifier 准确率: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 五、特征重要性分析

```python
import matplotlib.pyplot as plt

# 1. 权重（特征被用作分裂的次数）
xgb.plot_importance(bst, importance_type='weight', max_num_features=10)
plt.title('Feature Importance (Weight)')
plt.show()

# 2. 增益（特征带来的平均增益）
xgb.plot_importance(bst, importance_type='gain', max_num_features=10)
plt.title('Feature Importance (Gain)')
plt.show()

# 3. 覆盖（特征影响的样本数）
xgb.plot_importance(bst, importance_type='cover', max_num_features=10)
plt.title('Feature Importance (Cover)')
plt.show()

# 4. 获取重要性数值
importance_dict = bst.get_score(importance_type='gain')
print("特征增益重要性:")
for feat, gain in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  f{feat}: {gain:.4f}")
```

| 重要性类型 | 说明 | 适用场景 |
|-----------|------|----------|
| weight | 特征被用作分裂的次数 | 快速了解哪些特征活跃 |
| gain | 特征带来的平均损失减少 | 最有信息量的特征 |
| cover | 特征影响的样本数量 | 特征的影响范围 |

---

## 六、关键超参数调优

### 6.1 参数分类

| 类别 | 参数 | 说明 |
|------|------|------|
| 通用 | booster | 基学习器类型（gbtree/gblinear/dart） |
| 通用 | n_estimators | 树的数量 |
| 通用 | learning_rate | 学习率（eta），越小越稳定 |
| 树参数 | max_depth | 树的最大深度 |
| 树参数 | min_child_weight | 子节点最小权重和 |
| 树参数 | gamma | 分裂所需的最小损失减少 |
| 采样参数 | subsample | 样本采样比例 |
| 采样参数 | colsample_bytree | 每棵树特征采样比例 |
| 采样参数 | colsample_bylevel | 每个层级特征采样比例 |
| 正则化 | lambda (reg_lambda) | L2 正则化系数 |
| 正则化 | alpha (reg_alpha) | L1 正则化系数 |

### 6.2 调参策略

```python
from sklearn.model_selection import GridSearchCV

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)

# 网格搜索
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1, 2, 5]
}

# 推荐使用随机搜索（网格搜索太慢）
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=50,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"最优参数: {random_search.best_params_}")
print(f"最优准确率: {random_search.best_score_:.4f}")
```

### 6.3 推荐调参顺序

| 步骤 | 参数 | 推荐范围 |
|------|------|----------|
| 1 | max_depth | 3~10 |
| 2 | min_child_weight | 1~10 |
| 3 | gamma | 0~5 |
| 4 | subsample | 0.6~1.0 |
| 5 | colsample_bytree | 0.6~1.0 |
| 6 | learning_rate | 0.01~0.2 |
| 7 | n_estimators | 100~2000（配合学习率） |

---

## 七、早停机制

```python
# XGBoost 内置早停
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,  # 20 轮不改善则停止
    verbose=True
)

# 实际使用的树的数量
print(f"最佳迭代数: {xgb_model.best_iteration}")
print(f"最佳验证集准确率: {xgb_model.best_score:.4f}")
```

---

## 八、XGBoost vs 随机森林

| 对比维度 | 随机森林 | XGBoost |
|----------|----------|---------|
| 训练策略 | 并行 Bagging | 串行 Boosting |
| 优化目标 | 降低方差 | 降低偏差 |
| 训练速度 | 快 | 慢 |
| 预测速度 | 较慢（多树） | 快 |
| 过拟合 | 不易 | 需要调参 |
| 精度 | 高 | 通常更高 |
| 参数敏感度 | 低 | 高 |
| 缺失值 | 需要处理 | 自动处理 |
| 适用数据量 | 中等到大型 | 中等 |

---

## 九、实战建议

1. **学习率与树数搭配**：低学习率（0.01~0.1）配多树，高学习率配少树
2. **早停机制**：始终开启早停，设置合理的 patience（20~50）
3. **防过拟合**：控制 max_depth、增大 min_child_weight、使用正则化
4. **数据预处理**：XGBoost 能处理缺失值，但类别特征需要编码
5. **特征工程**：XGBoost 能自动学习非线性关系，但领域特征仍有帮助
6. **大规模数据**：考虑使用 LightGBM（更快）或 Dask XGBoost（分布式）

> 💡 **提示**：XGBoost 是基于 Boosting 策略的代表算法，通过串行训练、逐步纠错达到极高精度。理解了 Bagging（随机森林）和 Boosting（XGBoost）后，下一篇的 KNN 将带你了解另一种完全不同的学习范式——基于实例的学习。

---

## 十、总结

GBM 和 XGBoost 是基于 Boosting 策略的集成学习算法，通过串行训练多个弱学习器逐步提升预测精度。核心要点包括：

- **核心思想**：每棵树拟合前一模型的残差（负梯度方向）
- **二阶优化**：XGBoost 使用二阶泰勒展开，收敛更快
- **正则化**：树复杂度控制 + 叶子权重正则化
- **特征重要性**：weight、gain、cover 三种衡量方式
- **超参数**：max_depth、learning_rate、subsample 是最关键的参数
- **工业应用**：广告点击率预测、风控评分、推荐排序等领域广泛使用

---

📖 **上一篇**：[随机森林](./20-random-forest.md) | 📖 **下一篇**：[KNN 近邻算法](./22-knn.md) — 学习基于实例的懒惰学习方法，理解"近朱者赤"的朴素思想。
