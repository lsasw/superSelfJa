---
title: 线性回归算法详解
icon: chart-line
order: 16
---

# 线性回归算法详解

线性回归（Linear Regression）是机器学习中最基础、最经典的监督学习算法。它通过建立特征与目标变量之间的线性关系模型，实现对连续值的预测。理解线性回归不仅是掌握监督学习的起点，更是后续学习逻辑回归、神经网络等高级算法的基石。

---

## 一、线性回归的核心思想

### 1.1 什么是线性回归

线性回归假设目标变量 y 与输入特征 X 之间存在线性关系，通过找到最优的权重参数，使得预测值尽可能接近真实值。

**一元线性回归方程：**

$$y = w \cdot x + b$$

其中：
- y 是目标变量（因变量）
- x 是特征变量（自变量）
- w 是权重（斜率）
- b 是偏置（截距）

**多元线性回归方程：**

$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

用矩阵形式表示为：

$$y = X \cdot w + b$$

### 1.2 应用场景

| 场景 | 特征 | 目标 |
|------|------|------|
| 房价预测 | 面积、房龄、地段评分 | 房价（万元） |
| 销售预测 | 广告投入、季节因子、促销活动 | 销售额（万元） |
| 温度预测 | 湿度、气压、风速 | 温度（摄氏度） |
| 用户生命周期价值 | 注册时间、活跃度、消费频次 | LTV（元） |
| 股票趋势分析 | 历史价格、成交量、MACD | 未来价格 |

>  **提示**：线性回归适用于目标变量为连续值的场景。如果目标是分类问题，请参考下一篇「逻辑回归」。

---

## 二、损失函数与优化目标

### 2.1 均方误差（MSE）

线性回归最常用的损失函数是均方误差（Mean Squared Error）：

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

其中：
- n 是样本数量
- y_i 是第 i 个样本的真实值
- ŷ_i 是第 i 个样本的预测值

### 2.2 为什么使用 MSE

| 损失函数 | 优点 | 缺点 |
|----------|------|------|
| MSE（均方误差） | 可导、对异常值敏感（有助于关注大问题） | 对极端异常值过于敏感 |
| MAE（平均绝对误差） | 对异常值鲁棒 | 在零点不可导 |
| Huber Loss | 结合 MSE 和 MAE 的优点 | 需要调节 delta 参数 |

```python
import numpy as np

def mse_loss(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """计算 Huber Loss"""
    residual = y_true - y_pred
    condition = np.abs(residual) <= delta
    return np.mean(
        np.where(condition, 0.5 * residual**2, delta * np.abs(residual) - 0.5 * delta**2)
    )
```

---

## 三、参数求解方法

### 3.1 正规方程（闭式解）

对于线性回归，存在解析解：

$$w = (X^TX)^{-1}X^Ty$$

```python
import numpy as np

class LinearRegressionNormalEquation:
    """使用正规方程求解线性回归"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # 添加偏置列（全1列）
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # 正规方程求解
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# 使用示例
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 3.9, 6.2, 8.1, 9.8])

model = LinearRegressionNormalEquation()
model.fit(X, y)
print(f"权重: {model.weights}, 偏置: {model.bias:.2f}")
```

**正规方程的优缺点：**

| 优点 | 缺点 |
|------|------|
| 无需设置学习率 | 时间复杂度 O(n^3) |
| 无需迭代，一次求解 | 特征数量大时计算缓慢 |
| 结果精确 | 要求 X^TX 可逆 |
| 无需特征缩放 | 不适用于非线性模型 |

### 3.2 梯度下降法

梯度下降是更通用的优化方法，适用于大规模数据集。

```python
class LinearRegressionGD:
    """使用梯度下降求解线性回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = X.dot(self.weights) + self.bias
            
            # 计算损失
            loss = mse_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (2 / n_samples) * X.T.dot(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# 使用示例
model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
print(f"权重: {model.weights}, 偏置: {model.bias:.2f}")
```

### 3.3 梯度下降的变体

| 类型 | 每次迭代样本数 | 优点 | 缺点 |
|------|---------------|------|------|
| 批量梯度下降（BGD） | 全部样本 | 收敛稳定 | 速度慢 |
| 随机梯度下降（SGD） | 1个样本 | 速度快 | 震荡大 |
| 小批量梯度下降（MBGD） | 32~256个样本 | 平衡速度与稳定性 | 需要调节batch size |

```python
class LinearRegressionSGD:
    """使用小批量随机梯度下降求解"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # 随机采样一个batch
            indices = np.random.choice(n_samples, self.batch_size)
            X_batch, y_batch = X[indices], y[indices]
            
            # 计算梯度
            y_pred = X_batch.dot(self.weights) + self.bias
            error = y_pred - y_batch
            dw = (2 / self.batch_size) * X_batch.T.dot(error)
            db = (2 / self.batch_size) * np.sum(error)
            
            # 学习率衰减
            lr = self.learning_rate / (1 + i * 0.001)
            self.weights -= lr * dw
            self.bias -= lr * db
        
        return self
```

---

## 四、正则化线性回归

### 4.1 过拟合问题

当特征过多或数据量不足时，模型容易过拟合，在训练集上表现好但在测试集上泛化能力差。

### 4.2 Ridge 回归（L2 正则化）

$$Loss = MSE + \alpha \sum_{j=1}^{n} w_j^2$$

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Ridge 回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
predictions = ridge.predict(X_test)

print(f"Ridge 系数: {ridge.coef_}")
print(f"Ridge 截距: {ridge.intercept_:.2f}")
```

### 4.3 Lasso 回归（L1 正则化）

$$Loss = MSE + \alpha \sum_{j=1}^{n} |w_j|$$

```python
from sklearn.linear_model import Lasso

# Lasso 回归 - 具有特征选择能力
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 查看哪些特征被选中（系数不为0）
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"选中的特征索引: {selected_features}")
print(f"系数: {lasso.coef_}")
```

### 4.4 Elastic Net（弹性网络）

结合 L1 和 L2 正则化：

$$Loss = MSE + \alpha \cdot \rho \sum |w_j| + \frac{\alpha(1-\rho)}{2} \sum w_j^2$$

```python
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
predictions = elastic.predict(X_test)
```

### 4.5 正则化方法对比

| 方法 | 正则项 | 特征选择 | 适用场景 |
|------|--------|----------|----------|
| 普通线性回归 | 无 | 无 | 特征少、无共线性 |
| Ridge | L2 | 无 | 特征间存在共线性 |
| Lasso | L1 | 有 | 高维稀疏数据 |
| Elastic Net | L1+L2 | 有 | 特征多且相关性强 |

---

## 五、完整实战项目：波士顿房价预测

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. 加载数据
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"数据集形状: {X.shape}")
print(f"特征列表: {list(X.columns)}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 训练多个模型并比较
models = {
    "普通线性回归": LinearRegression(),
    "Ridge(alpha=1.0)": Ridge(alpha=1.0),
    "Lasso(alpha=0.1)": Lasso(alpha=0.1),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "模型": name,
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "非零系数": np.sum(model.coef_ != 0)
    })
    
    print(f"\n--- {name} ---")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# 5. 结果对比表
results_df = pd.DataFrame(results)
print("\n模型对比:")
print(results_df.to_string(index=False))

# 6. 特征重要性分析（Ridge 模型）
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({
    "特征": housing.feature_names,
    "系数": ridge_model.coef_
}).sort_values("系数", key=abs, ascending=False)
print("\n特征重要性:")
print(feature_importance.to_string(index=False))
```

---

## 六、模型评估指标

### 6.1 常用评估指标

| 指标 | 公式 | 含义 | 理想值 |
|------|------|------|--------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 均方误差 | 越小越好 |
| RMSE | $\sqrt{MSE}$ | 均方根误差 | 越小越好 |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | 平均绝对误差 | 越小越好 |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 决定系数 | 越接近1越好 |

```python
def evaluate_regression(y_true, y_pred):
    """计算回归模型的所有评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4)
    }

# 使用示例
metrics = evaluate_regression(y_test, y_pred)
for metric, value in metrics.items():
    print(f"{metric}: {value}")
```

### 6.2 R² 的理解

- R² = 1：完美拟合
- R² = 0：模型预测能力等同于预测均值
- R² < 0：模型比预测均值还差

---

## 七、线性回归的假设与诊断

### 7.1 基本假设

| 假设 | 说明 | 违反后果 |
|------|------|----------|
| 线性关系 | 特征与目标存在线性关系 | 模型拟合不足 |
| 独立性 | 样本之间相互独立 | 标准误估计不准确 |
| 同方差性 | 残差方差恒定 | 置信区间不准确 |
| 正态性 | 残差服从正态分布 | 假设检验失效 |
| 无多重共线性 | 特征之间不高度相关 | 系数估计不稳定 |

### 7.2 残差分析

```python
import matplotlib.pyplot as plt

# 计算残差
residuals = y_test - y_pred

# 残差图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 残差 vs 预测值
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('预测值')
axes[0].set_ylabel('残差')
axes[0].set_title('残差 vs 预测值')

# 残差分布直方图
axes[1].hist(residuals, bins=30, edgecolor='black')
axes[1].set_xlabel('残差')
axes[1].set_ylabel('频数')
axes[1].set_title('残差分布')

plt.tight_layout()
plt.show()
```

---

## 八、实战建议

1. **数据预处理**：特征缩放对梯度下降法至关重要，对正规方程也建议做
2. **异常值处理**：线性回归对异常值敏感，务必在训练前进行清洗
3. **特征工程**：多项式特征可以扩展线性回归的非线性拟合能力
4. **正则化选择**：高维数据优先使用 Lasso，共线性数据使用 Ridge
5. **模型诊断**：务必检查残差图，验证线性回归的基本假设

> 💡 **提示**：线性回归虽然简单，但它是一切回归算法的起点。后续的逻辑回归将在线性回归的基础上引入 Sigmoid 函数，将回归问题转化为分类问题。

---

## 九、总结

线性回归是机器学习中最基础也最重要的算法之一。它通过建立特征与目标变量之间的线性关系，实现连续值预测。核心要点包括：

- **数学基础**：假设 y = Xw + b，通过最小化 MSE 来求解参数
- **求解方法**：正规方程适合小规模数据，梯度下降适合大规模数据
- **正则化**：Ridge、Lasso、Elastic Net 有效控制过拟合
- **评估指标**：MSE、MAE、R² 是常用的回归评估指标
- **诊断方法**：残差分析是验证模型假设的重要手段

---

📖 **下一篇**：[逻辑回归](./17-logistic-regression.md) — 学习如何将线性回归扩展到二分类问题，理解 Sigmoid 函数与交叉熵损失。
