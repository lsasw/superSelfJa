---
title: 机器学习基础概念
icon: brain
order: 2
---

# 机器学习基础概念

机器学习（Machine Learning, ML）是人工智能的核心技术，它使计算机能够从数据中自动学习并改进性能，而无需显式编程。本文将介绍机器学习的基本概念、学习范式、核心术语以及完整的工作流程。

## 1. 什么是机器学习

### 1.1 定义

汤姆·米歇尔（Tom Mitchell）给出了机器学习的经典定义：

> 对于某类任务T和性能度量P，如果一个计算机程序在T上以P衡量的性能随着经验E的提高而提高，那么我们说这个程序从经验E中进行了学习。

通俗地说，机器学习就是**从数据中发现模式**，并利用这些模式对未知数据做出预测或决策。

### 1.2 机器学习 vs 传统编程

| 维度 | 传统编程 | 机器学习 |
|------|----------|----------|
| 输入 | 规则 + 数据 | 数据 + 答案（标签） |
| 输出 | 结果 | 规则（模型） |
| 规则来源 | 人类编写 | 数据中学习 |
| 适用场景 | 规则明确、逻辑确定 | 规则模糊、模式复杂 |
| 可维护性 | 规则复杂时难以维护 | 模型随数据更新而进化 |
| 典型例子 | 计算器、排序算法 | 垃圾邮件过滤、图像识别 |

```python
# 传统编程 vs 机器学习的对比示例

# 传统编程：规则明确
def classify_spam_traditional(email_text: str) -> bool:
    """基于明确规则判断垃圾邮件"""
    spam_keywords = ["中奖", "免费", "限时抢购", "点击领取"]
    return any(keyword in email_text for keyword in spam_keywords)

# 机器学习：从数据中学习
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpamClassifier:
    """基于机器学习的垃圾邮件分类器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
    
    def train(self, texts: list, labels: list):
        """从标注数据中学习规则"""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
    
    def predict(self, text: str) -> int:
        """预测新邮件是否为垃圾邮件"""
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

# 训练数据
train_texts = [
    "恭喜您中奖了，点击链接领取奖品",
    "明天的会议在上午9点",
    "限时抢购，全场五折优惠",
    "项目周报已发送到您的邮箱"
]
train_labels = [1, 0, 1, 0]  # 1表示垃圾邮件，0表示正常邮件

classifier = SpamClassifier()
classifier.train(train_texts, train_labels)
print(f"预测结果: {classifier.predict('免费试用，先到先得')}")  # 输出: 1
```

### 1.3 为什么需要机器学习？

许多现实世界的问题无法用明确的规则来描述：

- **图像识别**：如何用规则描述"猫"的样子？
- **语音识别**：不同口音、语速、背景噪声下如何识别？
- **自然语言理解**：同一句话在不同语境下含义不同。
- **推荐系统**：用户兴趣随时间变化，规则难以穷尽。

这些问题中，**输入和输出之间的关系高度非线性且复杂**，人类难以手动编写规则，但可以通过大量样本让机器自动学习规律。

## 2. 机器学习的学习范式

### 2.1 监督学习（Supervised Learning）

监督学习是最常见的机器学习范式。训练数据包含**输入特征**和**对应的标签**，模型学习从输入到输出的映射关系。

| 子类型 | 任务描述 | 典型算法 | 应用场景 |
|--------|----------|----------|----------|
| 分类（Classification） | 预测离散类别 | 逻辑回归、SVM、决策树、随机森林 | 垃圾邮件检测、疾病诊断 |
| 回归（Regression） | 预测连续数值 | 线性回归、岭回归、SVR | 房价预测、销量预测 |

```python
# 监督学习示例：房价预测（回归任务）
from sklearn.linear_model import LinearRegression
import numpy as np

# 特征：面积(平米)、房间数、距市中心距离(公里)
X_train = np.array([
    [80, 2, 5],
    [120, 3, 10],
    [60, 1, 3],
    [150, 4, 15],
    [90, 2, 8],
])
y_train = np.array([320, 480, 240, 600, 360])  # 房价(万元)

model = LinearRegression()
model.fit(X_train, y_train)

# 预测新数据
X_new = np.array([[100, 3, 7]])
predicted_price = model.predict(X_new)
print(f"预测房价: {predicted_price[0]:.1f} 万元")

# 分析特征权重
for i, feature in enumerate(["面积", "房间数", "距离"]):
    print(f"{feature}的权重: {model.coef_[i]:.2f}")
```

### 2.2 无监督学习（Unsupervised Learning）

无监督学习的训练数据**没有标签**，模型需要自行发现数据中的结构和模式。

| 子类型 | 任务描述 | 典型算法 | 应用场景 |
|--------|----------|----------|----------|
| 聚类（Clustering） | 将数据分为若干组 | K-Means、DBSCAN、层次聚类 | 用户分群、异常检测 |
| 降维（Dimensionality Reduction） | 减少特征数量 | PCA、t-SNE、UMAP | 数据可视化、特征压缩 |
| 关联规则（Association） | 发现特征间的关联 | Apriori、FP-Growth | 购物篮分析 |

```python
# 无监督学习示例：客户分群（聚类任务）
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成模拟数据
X, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)

# K-Means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 输出每个簇的中心
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"簇 {i+1} 中心: ({center[0]:.2f}, {center[1]:.2f})")

# 预测新样本所属的簇
new_customer = [[2.5, 3.0]]
cluster = kmeans.predict(new_customer)
print(f"新客户属于簇: {cluster[0] + 1}")
```

### 2.3 半监督学习（Semi-supervised Learning）

半监督学习结合了少量标注数据和大量未标注数据。标注成本高而数据量大的场景特别适合此方法。

| 特点 | 描述 |
|------|------|
| 数据组成 | 少量有标签数据 + 大量无标签数据 |
| 核心思想 | 利用无标签数据的分布信息辅助学习 |
| 适用场景 | 医疗影像（标注成本极高）、文本分类 |

### 2.4 强化学习（Reinforcement Learning）

强化学习中，**智能体（Agent）**通过与**环境（Environment）**交互来学习策略，目标是在长期累积奖励最大化。

| 要素 | 描述 |
|------|------|
| 状态（State） | 环境在某一时刻的描述 |
| 动作（Action） | 智能体可执行的操作 |
| 奖励（Reward） | 环境对动作的反馈信号 |
| 策略（Policy） | 从状态到动作的映射规则 |

典型应用：AlphaGo、机器人控制、自动驾驶、游戏AI。

```python
# 强化学习的简化概念示例
class SimpleAgent:
    """简化版强化学习智能体"""
    
    def __init__(self, n_states: int, n_actions: int):
        self.q_table = {}  # Q值表
        self.n_actions = n_actions
        self.learning_rate = 0.1
        self.discount_factor = 0.9
    
    def choose_action(self, state: tuple, epsilon: float = 0.1) -> int:
        """epsilon-greedy策略：大部分时间选最优动作，偶尔随机探索"""
        import random
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        state_key = str(state)
        if state_key in self.q_table:
            return self.q_table[state_key].index(max(self.q_table[state_key]))
        return random.randint(0, self.n_actions - 1)
    
    def update(self, state, action, reward, next_state):
        """更新Q值（Q-Learning核心公式）"""
        state_key, next_key = str(state), str(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0] * self.n_actions
        
        # Q(s,a) = Q(s,a) + α * [r + γ*max(Q(s')) - Q(s,a)]
        old_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_key])
        self.q_table[state_key][action] = old_q + self.learning_rate * \
            (reward + self.discount_factor * max_next_q - old_q)
```

## 3. 核心术语

### 3.1 数据集术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 特征（Feature） | Feature | 描述数据的属性，如房屋面积、房间数 |
| 标签（Label） | Label | 预测目标，如房价、是否垃圾邮件 |
| 样本（Sample） | Sample/Instance | 一条数据记录 |
| 训练集 | Training Set | 用于训练模型的数据 |
| 验证集 | Validation Set | 用于调参和选择模型的数据 |
| 测试集 | Test Set | 用于最终评估模型性能的数据 |

### 3.2 模型术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 模型（Model） | Model | 从数据中学习到的函数或规则 |
| 参数（Parameter） | Parameter | 模型内部可学习的变量（如权重、偏置） |
| 超参数（Hyperparameter） | Hyperparameter | 训练前需手动设置的参数（如学习率、树深度） |
| 假设空间 | Hypothesis Space | 模型可能学到的所有函数的集合 |

### 3.3 学习过程术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 训练（Training） | Training | 用数据优化模型参数的过程 |
| 预测（Prediction/Inference） | Inference | 使用训练好的模型对新数据做出判断 |
| 损失函数（Loss Function） | Loss Function | 衡量模型预测与真实值差距的函数 |
| 优化器（Optimizer） | Optimizer | 调整参数以最小化损失函数的算法 |

## 4. 机器学习工作流程

### 4.1 完整流程

```
数据收集 → 数据探索 → 数据预处理 → 特征工程
    ↓
模型选择 → 模型训练 → 模型评估 → 超参数调优
    ↓
模型部署 → 模型监控 → 模型更新（持续迭代）
```

### 4.2 各阶段说明

| 阶段 | 关键活动 | 常用工具 |
|------|----------|----------|
| 数据收集 | 确定数据源、采集数据、数据清洗 | SQL、爬虫、API |
| 数据探索 | 统计分析、可视化、发现规律 | Pandas、Matplotlib |
| 数据预处理 | 缺失值处理、异常值处理、编码 | scikit-learn |
| 特征工程 | 特征选择、特征提取、特征构造 | scikit-learn、Featuretools |
| 模型训练 | 选择算法、训练模型、调参 | scikit-learn、PyTorch |
| 模型评估 | 交叉验证、指标计算、对比分析 | scikit-learn |
| 模型部署 | 服务化、API封装、监控 | Flask、FastAPI、ONNX |

```python
# 完整的机器学习工作流程示例（Iris数据集分类）
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 数据收集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 数据探索
print(f"数据集形状: {X.shape}")  # (150, 4)
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 3. 数据拆分（训练集/测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 数据预处理（标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. 模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n准确率: {accuracy:.4f}")
print(f"\n详细评估报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. 特征重要性分析
importances = model.feature_importances_
for feature, importance in zip(iris.feature_names, importances):
    print(f"{feature}: {importance:.4f}")
```

> 💡 **提示**：机器学习项目中，数据预处理和特征工程往往占据了60%-80%的时间。不要急于训练模型，充分理解和处理数据是成功的关键。

## 5. 常见算法一览

### 5.1 经典算法分类

| 算法 | 类型 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| 线性回归 | 监督-回归 | 连续值预测 | 简单、可解释 | 假设线性关系 |
| 逻辑回归 | 监督-分类 | 二分类 | 简单、可解释、概率输出 | 线性决策边界 |
| 决策树 | 监督-分类/回归 | 可解释性要求高 | 可解释、非线性 | 容易过拟合 |
| 随机森林 | 监督-分类/回归 | 大多数场景 | 鲁棒、精度高 | 模型较大、慢 |
| SVM | 监督-分类 | 高维数据、小样本 | 高维表现好 | 大数据慢 |
| KNN | 监督-分类/回归 | 简单基线 | 无需训练、简单 | 预测慢、维度灾难 |
| K-Means | 无监督-聚类 | 数据分群 | 简单、快速 | 需指定K值 |
| PCA | 无监督-降维 | 特征压缩 | 保留方差 | 线性降维 |
| XGBoost | 监督-分类/回归 | 表格数据竞赛 | 精度高、快速 | 调参复杂 |

### 5.2 如何选择算法

选择算法时考虑以下因素：

| 因素 | 推荐算法 |
|------|----------|
| 数据量小（<1000） | SVM、朴素贝叶斯、KNN |
| 数据量大（>10万） | 随机森林、XGBoost、深度学习 |
| 需要可解释性 | 逻辑回归、决策树 |
| 追求最高精度 | XGBoost、LightGBM、神经网络 |
| 快速原型 | 逻辑回归、随机森林 |
| 高维稀疏数据（文本） | 朴素贝叶斯、SVM |

## 6. 总结

机器学习是一门让计算机从数据中自动学习的科学。核心要点包括：

- **三大范式**：监督学习（有标签）、无监督学习（无标签）、强化学习（交互奖励）。
- **核心术语**：特征、标签、模型、参数、超参数、损失函数等构成了交流基础。
- **工作流程**：数据收集→预处理→特征工程→模型训练→评估→部署，形成闭环。
- **算法选择**：没有万能算法（No Free Lunch定理），应根据数据特点、任务需求和资源约束综合选择。

掌握了这些基础概念后，下一节我们将进入数学基础的学习——线性代数。它是理解机器学习中矩阵运算、向量空间、特征变换等核心概念的必要前提。

---

**上一篇**：[AI概述与发展历程](01-ai-overview.md) | **下一篇**：[线性代数基础](03-linear-algebra.md)
