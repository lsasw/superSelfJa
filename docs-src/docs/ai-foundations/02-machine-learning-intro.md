---
title: 机器学习入门
icon: brain
order: 2
---

# 机器学习入门

## 引言

在上一篇 [AI 全景概览](./01-ai-overview.md) 中，我们已经了解到人工智能是一个广阔的技术领域，而机器学习是其核心子集。本篇将聚焦机器学习这一关键方向，带你深入理解它的基本概念、核心范式、典型工作流程以及与传统编程的本质区别。通过本篇的学习，你将建立起对机器学习的系统性理解，为后续深入各个具体算法和数学基础做好准备。

## 什么是机器学习

### 经典定义

机器学习领域最被广泛引用的定义来自 Arthur Samuel（1959）：

> "机器学习是赋予计算机无需显式编程而具有学习能力的研究领域。"

Tom Mitchell（1997）给出了更形式化的定义：

> "一个计算机程序被认为从经验 E 中学习某个任务 T 和性能度量 P，如果它在 T 上的性能（由 P 度量）随经验 E 而提高。"

### 形式化理解

让我们用一个具体的例子来理解这个定义：

| 要素 | 示例：下棋程序 |
|------|--------------|
| 任务 T | 下国际象棋 |
| 经验 E | 与自己或对手对弈的历史记录 |
| 性能 P | 赢得比赛的百分比 |

随着对弈经验 E 的增加，程序在任务 T 上的胜率 P 不断提高，这就是学习的过程。

### 机器学习与传统编程的本质区别

这是理解机器学习最核心的认知转变：

```
传统编程：
    输入数据 + 规则（程序） → 输出结果

机器学习：
    输入数据 + 输出结果 → 规则（模型）
```

用一个对比表格来说明：

| 维度 | 传统编程 | 机器学习 |
|------|---------|---------|
| 输入 | 数据 + 人类编写的规则 | 数据 + 期望输出 |
| 输出 | 计算结果 | 学习到的规则（模型） |
| 规则来源 | 人类专家手动编写 | 算法从数据中自动学习 |
| 适用场景 | 规则明确、逻辑清晰 | 规则复杂、难以手动编写 |
| 维护方式 | 修改代码 | 更新数据、重新训练 |
| 泛化能力 | 只能处理规则覆盖的情况 | 可以处理未见过的数据 |

### 为什么需要机器学习？

有些问题人类很难用规则来描述，但计算机可以通过数据来学习。经典案例：

| 问题 | 传统编程难点 | 机器学习方法 |
|------|-------------|-------------|
| 垃圾邮件识别 | 垃圾邮件的特征千变万化 | 从标注邮件中学习模式 |
| 手写数字识别 | 每个人的书写风格不同 | 从大量手写样本中学习 |
| 语音识别 | 口音、语速、环境噪声差异大 | 从海量语音数据中学习 |
| 推荐系统 | 用户兴趣难以用规则定义 | 从用户行为数据中挖掘偏好 |
| 自动驾驶 | 道路情况极其复杂 | 从驾驶经验中学习决策 |

> 💡 **核心理解**：机器学习的本质不是让计算机"记住"规则，而是让计算机从数据中"发现"规律。这种从数据到规律的映射关系，就是机器学习的核心价值。

## 机器学习的三大范式

### 监督学习（Supervised Learning）

**核心思想**：给定带有标签（正确答案）的训练数据，学习从输入到输出的映射函数。

**工作流程**：
1. 收集带有标签的训练数据集
2. 选择合适的算法
3. 训练模型，使其能够预测标签
4. 在未见过的测试数据上评估性能

**数学表达**：给定输入 X 和标签 y，学习函数 f，使得 f(X) ≈ y

#### 回归问题（Regression）

预测连续数值输出。

**典型算法**：
- 线性回归（Linear Regression）
- 多项式回归（Polynomial Regression）
- 支持向量回归（SVR）
- 决策树回归

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# 示例：房屋面积与价格的关系
X = np.array([[50], [60], [70], [80], [90], [100]])  # 房屋面积（平方米）
y = np.array([150, 180, 210, 240, 270, 300])           # 房价（万元）

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
X_new = np.array([[75], [95]])
predictions = model.predict(X_new)
print(f"75平方米预测价格: {predictions[0]:.2f} 万元")
print(f"95平方米预测价格: {predictions[1]:.2f} 万元")

# 查看模型参数
print(f"斜率: {model.coef_[0]:.2f}")
print(f"截距: {model.intercept_:.2f}")
```

**应用场景**：房价预测、股票价格预测、销售额预估

#### 分类问题（Classification）

预测离散类别标签。

**典型算法**：
- 逻辑回归（Logistic Regression）
- 支持向量机（SVM）
- 决策树（Decision Tree）
- 随机森林（Random Forest）
- K 近邻（KNN）

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测并评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
```

**应用场景**：垃圾邮件分类、疾病诊断、图像分类

### 无监督学习（Unsupervised Learning）

**核心思想**：给定没有标签的数据，发现数据内在的结构和模式。

**与监督学习的关键区别**：没有"正确答案"可供参考，算法需要自己发现数据的规律。

#### 聚类（Clustering）

将数据分成不同的组，使得同一组内的数据相似度高，不同组之间的相似度低。

**典型算法**：
- K-Means
- DBSCAN
- 层次聚类（Hierarchical Clustering）
- 高斯混合模型（GMM）

```python
from sklearn.cluster import KMeans
import numpy as np

# 示例：客户分群
# 假设有客户的消费金额和访问频率数据
customers = np.array([
    [5000, 20],   # 高消费、高频访问
    [4500, 18],
    [5200, 22],
    [1000, 5],    # 低消费、低频访问
    [800, 3],
    [1200, 6],
    [3000, 12],   # 中等消费、中频访问
    [2800, 10],
    [3200, 14],
])

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(customers)

# 输出聚类结果
for i, label in enumerate(labels):
    print(f"客户 {i+1}: 消费={customers[i][0]}, 频率={customers[i][1]}, 群组={label}")

print(f"\n聚类中心:\n{kmeans.cluster_centers_}")
```

**应用场景**：客户分群、文档分类、异常检测

#### 降维（Dimensionality Reduction）

将高维数据映射到低维空间，同时保留最重要的信息。

**典型算法**：
- 主成分分析（PCA）
- t-SNE
- UMAP
- 自编码器（Autoencoder）

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X = iris.data  # 原始4维特征

# PCA 降维到2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"原始维度: {X.shape[1]}")
print(f"降维后维度: {X_reduced.shape[1]}")
print(f"保留的方差比例: {pca.explained_variance_ratio_.sum():.2%}")
print(f"\n前5个样本降维结果:\n{X_reduced[:5]}")
```

**应用场景**：数据可视化、特征压缩、噪声过滤

### 强化学习（Reinforcement Learning）

**核心思想**：智能体（Agent）通过与环境（Environment）交互，根据奖励信号（Reward）学习最优策略。

**关键概念**：

| 概念 | 说明 | 示例（下棋） |
|------|------|-------------|
| 状态（State） | 环境的当前情况 | 棋盘上棋子的布局 |
| 动作（Action） | 智能体可以执行的操作 | 下一步棋走哪里 |
| 奖励（Reward） | 执行动作后的反馈 | 赢棋+1，输棋-1 |
| 策略（Policy） | 状态到动作的映射 | 看到某种棋局该怎么走 |
| 价值函数（Value） | 状态的长期预期回报 | 当前棋局最终获胜的概率 |

**学习过程**：
1. 智能体观察当前状态
2. 根据策略选择动作
3. 环境返回新状态和奖励
4. 智能体根据奖励更新策略
5. 重复上述过程，不断优化

```python
# 简化版 Q-Learning 示例：迷宫寻路
import numpy as np
import random

# 定义迷宫环境（4x4 网格）
# 0: 可通行, 1: 墙壁, 2: 终点
maze = [
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 2, 0],
]

# Q-Learning 参数
alpha = 0.1      # 学习率
gamma = 0.9      # 折扣因子
epsilon = 0.1    # 探索率
episodes = 1000  # 训练轮数

# 初始化 Q 表
q_table = np.zeros((4, 4, 4))  # (行, 列, 动作) 动作: 上、下、左、右

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

def get_valid_actions(state):
    """获取当前位置的合法动作"""
    valid = []
    for i, (dr, dc) in enumerate(actions):
        nr, nc = state[0] + dr, state[1] + dc
        if 0 <= nr < 4 and 0 <= nc < 4 and maze[nr][nc] != 1:
            valid.append(i)
    return valid

# Q-Learning 训练
for episode in range(episodes):
    state = (0, 0)  # 起点
    while maze[state[0]][state[1]] != 2:  # 未到达终点
        if random.random() < epsilon:
            action = random.choice(get_valid_actions(state))
        else:
            valid = get_valid_actions(state)
            action = max(valid, key=lambda a: q_table[state[0], state[1], a])

        # 执行动作
        dr, dc = actions[action]
        next_state = (state[0] + dr, state[1] + dc)

        # 计算奖励
        reward = 100 if maze[next_state[0]][next_state[1]] == 2 else -1

        # 更新 Q 值
        valid_next = get_valid_actions(next_state)
        max_q_next = max([q_table[next_state[0], next_state[1], a] for a in valid_next]) if valid_next else 0
        q_table[state[0], state[1], action] += alpha * (reward + gamma * max_q_next - q_table[state[0], state[1], action])

        state = next_state

print("Q-Learning 训练完成！")
print(f"从起点(0,0)出发，各方向Q值: {q_table[0, 0]}")
```

**应用场景**：游戏 AI（AlphaGo）、机器人控制、资源调度、自动驾驶决策

## 机器学习的典型工作流程

一个完整的机器学习项目通常包含以下步骤：

```
数据收集 → 数据探索 → 数据预处理 → 特征工程 → 模型选择 →
模型训练 → 模型评估 → 超参数调优 → 模型部署 → 持续监控
```

### 详细步骤说明

| 步骤 | 内容 | 关键问题 | 常用工具 |
|------|------|---------|---------|
| 1. 数据收集 | 获取原始数据 | 数据从哪里来？质量如何？ | 数据库、API、爬虫 |
| 2. 数据探索 | 了解数据特征和分布 | 数据有什么规律？ | pandas、matplotlib |
| 3. 数据预处理 | 清洗和转换数据 | 缺失值、异常值如何处理？ | pandas、sklearn |
| 4. 特征工程 | 构造和优化特征 | 哪些特征对预测有用？ | pandas、sklearn |
| 5. 模型选择 | 选择合适的算法 | 什么算法适合这个问题？ | scikit-learn |
| 6. 模型训练 | 用训练数据拟合模型 | 模型参数如何优化？ | scikit-learn、PyTorch |
| 7. 模型评估 | 在测试集上评估 | 模型表现如何？ | sklearn.metrics |
| 8. 超参数调优 | 优化模型超参数 | 如何找到最佳参数？ | GridSearchCV、Optuna |
| 9. 模型部署 | 将模型投入生产 | 如何服务化？ | Flask、FastAPI、Docker |
| 10. 持续监控 | 监控模型表现 | 模型是否需要重新训练？ | MLflow、Prometheus |

### 完整示例：鸢尾花分类全流程

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 第一步：数据收集与探索
# ============================================================
print("=" * 50)
print("第一步：数据收集与探索")
print("=" * 50)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"数据集形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"\n数据统计描述:")
df = pd.DataFrame(X, columns=feature_names)
print(df.describe())

# ============================================================
# 第二步：数据预处理
# ============================================================
print("\n" + "=" * 50)
print("第二步：数据预处理")
print("=" * 50)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"标准化后训练集均值: {X_train_scaled.mean():.6f}")
print(f"标准化后训练集标准差: {X_train_scaled.std():.6f}")

# ============================================================
# 第三步：模型训练
# ============================================================
print("\n" + "=" * 50)
print("第三步：模型训练")
print("=" * 50)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("模型训练完成！")

# ============================================================
# 第四步：模型评估
# ============================================================
print("\n" + "=" * 50)
print("第四步：模型评估")
print("=" * 50)

# 训练集评估
y_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"训练集准确率: {train_accuracy:.4f}")

# 测试集评估
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 交叉验证
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\n5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 混淆矩阵
print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_test_pred)}")

# 分类报告
print(f"\n详细分类报告:\n{classification_report(y_test, y_test_pred, target_names=target_names)}")

# ============================================================
# 第五步：特征重要性分析
# ============================================================
print("=" * 50)
print("第五步：特征重要性分析")
print("=" * 50)

importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp:.4f}")

print("\n完整机器学习流程执行完毕！")
```

## 数据集的组成与划分

### 数据集的三个部分

| 集合 | 用途 | 占比 | 类比 |
|------|------|------|------|
| 训练集（Training Set） | 训练模型参数 | 60-80% | 学生的学习材料 |
| 验证集（Validation Set） | 调整超参数、选择模型 | 10-20% | 模拟考试 |
| 测试集（Test Set） | 最终评估模型性能 | 10-20% | 正式考试 |

> 💡 **重要原则**：测试集只能在最终评估时使用一次。如果在训练过程中反复使用测试集来调整模型，就相当于"作弊"，会导致对模型真实高估。

### 数据划分方法

```python
from sklearn.model_selection import train_test_split

# 简单划分：训练集 80% + 测试集 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 三分法：训练集 70% + 验证集 15% + 测试集 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

## 过拟合与欠拟合初探

### 概念理解

| 问题 | 表现 | 原因 | 比喻 |
|------|------|------|------|
| 欠拟合（Underfitting） | 训练集和测试集表现都差 | 模型太简单，无法捕捉数据规律 | 学生没学到知识点 |
| 过拟合（Overfitting） | 训练集表现好，测试集表现差 | 模型太复杂，记住了训练数据 | 学生死记硬背，不会举一反三 |
| 适度拟合 | 训练集和测试集表现都好且接近 | 模型复杂度与问题匹配 | 学生真正理解了知识 |

### 简单判断方法

```python
# 判断过拟合/欠拟合
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
gap = train_accuracy - test_accuracy

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"差距: {gap:.4f}")

if train_accuracy < 0.7:
    print("判断: 可能欠拟合 - 模型太简单")
elif gap > 0.1:
    print("判断: 可能过拟合 - 需要正则化或更多数据")
else:
    print("判断: 拟合适度 - 模型表现良好")
```

> 💡 **提示**：关于过拟合和欠拟合的深入分析，请参考后续文档 [过拟合与欠拟合](./10-overfitting-underfitting.md)。

## 如何选择机器学习算法

### 算法选择决策树

```
你的问题是什么类型？
├── 预测连续值 → 回归问题
│   ├── 数据量小、特征少 → 线性回归
│   ├── 特征之间有交互 → 决策树回归
│   └── 追求高精度 → 梯度提升树（XGBoost/LightGBM）
│
├── 预测离散类别 → 分类问题
│   ├── 二分类 → 逻辑回归、SVM
│   ├── 多分类 → 随机森林、XGBoost
│   └── 文本分类 → 朴素贝叶斯、深度学习
│
├── 发现数据分组 → 聚类问题
│   ├── 知道分组数量 → K-Means
│   ├── 不知道分组数量 → DBSCAN
│   └── 层次关系重要 → 层次聚类
│
├── 降低数据维度 → 降维问题
│   ├── 线性关系 → PCA
│   └── 非线性关系 → t-SNE、UMAP
│
└── 序列决策 → 强化学习
    ├── 离散动作空间 → Q-Learning
    └── 连续动作空间 → PPO、SAC
```

### 实用建议

| 场景 | 推荐策略 |
|------|---------|
| 入门项目 | 先用简单模型（线性回归、逻辑回归）建立基线 |
| 表格数据 | 随机森林或梯度提升树通常是最佳选择 |
| 图像数据 | 卷积神经网络（CNN） |
| 文本数据 | Transformer 或预训练语言模型 |
| 时间序列 | LSTM、Transformer 或专门的时序模型 |
| 小数据集 | 简单模型 + 交叉验证，避免过拟合 |

## 总结

通过本篇的学习，你应该掌握了以下核心知识：

1. **机器学习的定义**：让计算机从数据中自动学习规律，而非手动编写规则
2. **三大学习范式**：监督学习（回归、分类）、无监督学习（聚类、降维）、强化学习
3. **典型工作流程**：从数据收集到模型部署的完整流程
4. **数据集划分**：训练集、验证集、测试集的作用与区别
5. **过拟合与欠拟合**：初步理解模型泛化的核心问题
6. **算法选择**：根据问题类型和数据特点选择合适的算法

机器学习是一个理论与实践并重的领域。理解概念只是第一步，更重要的是动手实践。在后续的学习中，我们将深入探讨数学基础、数据处理技巧、模型评估方法等关键主题。

> [!NOTE] 下一篇
> [03 - 线性代数](./03-linear-algebra.md) —— 掌握机器学习中不可或缺的向量与矩阵运算知识。
