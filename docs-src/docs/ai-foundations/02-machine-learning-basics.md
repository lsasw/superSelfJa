---
title: 机器学习基础
icon: brain
order: 2
---

# 机器学习基础

机器学习（Machine Learning, ML）是人工智能的核心分支，让计算机能够从数据中学习并做出预测或决策，而无需显式编程。

---

## 一、什么是机器学习

### 1.1 定义

机器学习是**研究如何让计算机从数据中自动学习规律**的科学。与传统编程不同：

```
传统编程：
  输入数据 + 程序规则 → 输出结果

机器学习：
  输入数据 + 输出结果 → 学习程序规则
```

### 1.2 核心要素

| 要素 | 说明 | 示例 |
|------|------|------|
| 数据 (Data) | 用于学习的样本 | 图片、文本、数值 |
| 模型 (Model) | 从数据中学到的规律 | 线性函数、神经网络 |
| 损失函数 (Loss) | 衡量预测好坏的标准 | MSE、交叉熵 |
| 优化算法 (Optimizer) | 调整模型参数的方法 | 梯度下降 |

---

## 二、机器学习分类

### 2.1 监督学习 (Supervised Learning)

**定义**：使用**带标签的数据**进行训练，学习输入到输出的映射关系。

```
训练数据：(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
目标：学习函数 f，使得 f(x) ≈ y
```

**典型算法**：

| 算法 | 类型 | 应用场景 |
|------|------|----------|
| 线性回归 | 回归 | 房价预测、销量预测 |
| 逻辑回归 | 分类 | 垃圾邮件检测、疾病诊断 |
| 决策树 | 分类/回归 | 信用评分、客户分类 |
| 支持向量机 | 分类 | 图像分类、文本分类 |
| 随机森林 | 分类/回归 | 特征重要性分析 |
| XGBoost | 分类/回归 | Kaggle竞赛首选 |

**代码示例**：

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 准备数据
X = [[1], [2], [3], [4], [5]]  # 特征
y = [2, 4, 5, 4, 5]            # 标签

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. 创建模型
model = LinearRegression()

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测与评估
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"均方误差: {mse}")
```

### 2.2 无监督学习 (Unsupervised Learning)

**定义**：使用**无标签的数据**进行训练，发现数据内在的结构和模式。

**典型算法**：

| 算法 | 用途 | 应用场景 |
|------|------|----------|
| K-Means | 聚类 | 客户分群、图像压缩 |
| 层次聚类 | 聚类 | 生物分类、文档聚类 |
| PCA | 降维 | 数据可视化、特征提取 |
| t-SNE | 降维 | 高维数据可视化 |
| DBSCAN | 聚类 | 异常检测、空间聚类 |

**代码示例**：

```python
from sklearn.cluster import KMeans
import numpy as np

# 生成示例数据
X = np.random.rand(100, 2)

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print(f"聚类中心:\n{centers}")
```

### 2.3 强化学习 (Reinforcement Learning)

**定义**：智能体通过与环境交互，学习最优策略以最大化累积奖励。

```
智能体 (Agent) → 动作 (Action) → 环境 (Environment)
     ↑                              │
     └────── 奖励 (Reward) ←────────┘
```

**核心概念**：

| 概念 | 说明 |
|------|------|
| 状态 (State) | 环境的当前情况 |
| 动作 (Action) | 智能体可以采取的行为 |
| 奖励 (Reward) | 执行动作后获得的反馈 |
| 策略 (Policy) | 状态到动作的映射 |
| 价值函数 (Value) | 状态的长期期望奖励 |

**典型算法**：
- Q-Learning
- DQN (Deep Q-Network)
- Policy Gradient
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

---

## 三、机器学习工作流程

### 3.1 标准流程

```
1. 问题定义 → 2. 数据收集 → 3. 数据预处理 → 4. 特征工程
       ↓                                              ↓
8. 模型部署 ← 7. 模型评估 ← 6. 模型训练 ← 5. 模型选择
```

### 3.2 详细步骤

**步骤 1：问题定义**
- 明确业务目标
- 确定是分类、回归还是聚类问题
- 定义评估指标

**步骤 2：数据收集**
- 确定数据来源
- 收集足够多的样本
- 确保数据质量

**步骤 3：数据预处理**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 加载数据
df = pd.read_csv('data.csv')

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df))

# 标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_filled))
```

**步骤 4：特征工程**
- 特征选择：选择最有用的特征
- 特征提取：PCA、LDA 等
- 特征构造：组合现有特征

**步骤 5：模型选择**
- 根据问题类型选择算法
- 考虑数据量和特征维度
- 平衡复杂度与性能

**步骤 6：模型训练**
```python
from sklearn.model_selection import cross_val_score

# 交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f"交叉验证得分: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

**步骤 7：模型评估**

| 任务类型 | 评估指标 |
|----------|----------|
| 分类 | 准确率、精确率、召回率、F1、AUC |
| 回归 | MSE、RMSE、MAE、R² |
| 聚类 | 轮廓系数、Calinski-Harabasz |

**步骤 8：模型部署**
- 模型序列化
- API 封装
- 监控与更新

---

## 四、过拟合与欠拟合

### 4.1 概念对比

| 类型 | 训练集表现 | 测试集表现 | 原因 |
|------|-----------|-----------|------|
| 欠拟合 | 差 | 差 | 模型太简单 |
| 正常 | 好 | 好 | 模型适中 |
| 过拟合 | 好 | 差 | 模型太复杂 |

### 4.2 解决方案

**欠拟合解决**：
- 增加模型复杂度
- 增加特征
- 减少正则化

**过拟合解决**：
- 增加训练数据
- 减少模型复杂度
- 增加正则化
- 使用 Dropout
- 早停法 (Early Stopping)

---

## 五、偏差-方差权衡

### 5.1 误差分解

```
总误差 = 偏差² + 方差 + 不可约误差

偏差 (Bias)：模型预测的系统性错误
方差 (Variance)：模型对训练数据的敏感程度
```

### 5.2 权衡关系

```
高偏差 ←───────── 最佳点 ─────────→ 高方差
 (欠拟合)        (平衡点)        (过拟合)
```

---

## 六、Python 机器学习生态

### 6.1 核心库

| 库 | 用途 | 特点 |
|----|------|------|
| NumPy | 数值计算 | 数组操作、数学函数 |
| Pandas | 数据处理 | DataFrame、数据清洗 |
| Scikit-learn | 机器学习 | 经典算法、工具齐全 |
| Matplotlib | 数据可视化 | 静态图表 |
| Seaborn | 统计可视化 | 美观的统计图表 |

### 6.2 学习资源

- **官方文档**：scikit-learn.org
- **书籍**：《Python 机器学习基础教程》
- **课程**：吴恩达 Machine Learning
- **实践**：Kaggle 入门竞赛

---

## 总结

机器学习是让计算机从数据中学习的科学。掌握监督学习、无监督学习和强化学习的基本概念，理解完整的工作流程，是成为 AI 工程师的基础。

---

>  **上一篇**：[AI 概述](./01-ai-overview.md)  
>  **下一篇**：[线性代数基础](./03-linear-algebra.md)
