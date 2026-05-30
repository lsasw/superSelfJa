---
title: 朴素贝叶斯算法详解
icon: envelope
order: 25
---

# 朴素贝叶斯算法详解

朴素贝叶斯（Naive Bayes）是基于贝叶斯定理的概率分类算法。尽管其"朴素"假设（特征之间相互独立）在现实中很少成立，但它却在文本分类、垃圾邮件过滤等实际应用中表现出色，甚至常常超过更复杂的算法。

---

## 一、贝叶斯定理

### 1.1 从直觉理解贝叶斯

贝叶斯定理描述了在获得新证据后，如何更新对某个假设的信念：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| 术语 | 含义 | 说明 |
|------|------|------|
| P(A) | 先验概率 | 看到证据前，A 发生的概率 |
| P(B\|A) | 似然 | 在 A 发生的条件下，B 出现的概率 |
| P(B) | 边际概率 | B 发生的总概率 |
| P(A\|B) | 后验概率 | 看到 B 后，A 发生的概率 |

### 1.2 医疗诊断示例

```
已知:
- 某疾病的患病率: P(Disease) = 0.01
- 检测的灵敏度: P(Pos|Disease) = 0.95
- 检测的假阳性率: P(Pos|No Disease) = 0.05

问题: 检测结果为阳性，实际患病的概率是多少？

解答:
P(Disease|Pos) = P(Pos|Disease) × P(Disease) / P(Pos)
               = 0.95 × 0.01 / (0.95×0.01 + 0.05×0.99)
               = 0.0095 / 0.059
               ≈ 16.1%
```

> **直觉**：即使检测准确率高达 95%，由于疾病本身很罕见，阳性结果中仍有约 84% 是假阳性。这就是贝叶斯思维的力量。

---

## 二、朴素贝叶斯分类器

### 2.1 贝叶斯分类公式

对于分类问题，我们需要计算：

$$P(y|x_1, x_2, ..., x_n) = \frac{P(x_1, x_2, ..., x_n|y) \cdot P(y)}{P(x_1, x_2, ..., x_n)}$$

**朴素假设**：给定类别 y，特征 x_1, x_2, ..., x_n 相互独立

$$P(x_1, x_2, ..., x_n|y) = \prod_{i=1}^{n} P(x_i|y)$$

因此：

$$P(y|x_1, ..., x_n) \propto P(y) \cdot \prod_{i=1}^{n} P(x_i|y)$$

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i|y)$$

### 2.2 算法流程

```
训练阶段:
1. 计算每个类别的先验概率 P(y)
2. 对每个特征计算条件概率 P(x_i|y)

预测阶段:
1. 对每个类别计算 P(y) × Π P(x_i|y)
2. 选择概率最大的类别
```

---

## 三、三种朴素贝叶斯变体

### 3.1 高斯朴素贝叶斯

假设特征服从正态分布。

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right)$$

```python
class GaussianNaiveBayes:
    """从零实现高斯朴素贝叶斯"""
    
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.mean = None
        self.var = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.class_prior = np.zeros(n_classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[i] = len(X_c) / len(X)
            self.mean[i] = X_c.mean(axis=0)
            self.var[i] = X_c.var(axis=0)
        
        return self
    
    def _likelihood(self, X, idx):
        """计算似然 P(x|y)"""
        return -0.5 * np.sum(
            np.log(2 * np.pi * self.var[idx]) +
            (X - self.mean[idx]) ** 2 / self.var[idx],
            axis=1
        )
    
    def predict(self, X):
        # 对数概率防止下溢
        log_posteriors = np.zeros((X.shape[0], len(self.classes)))
        
        for i in range(len(self.classes)):
            log_posteriors[:, i] = (
                np.log(self.class_prior[i]) +
                self._likelihood(X, i)
            )
        
        return self.classes[np.argmax(log_posteriors, axis=1)]
    
    def predict_proba(self, X):
        log_posteriors = np.zeros((X.shape[0], len(self.classes)))
        
        for i in range(len(self.classes)):
            log_posteriors[:, i] = (
                np.log(self.class_prior[i]) +
                self._likelihood(X, i)
            )
        
        # Softmax 归一化
        log_posteriors -= np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors)
        return posteriors / posteriors.sum(axis=1, keepdims=True)

# 测试
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
print(f"高斯朴素贝叶斯准确率: {accuracy_score(y_test, predictions):.4f}")
```

### 3.2 多项式朴素贝叶斯

适用于离散计数数据（如文本 TF-IDF 特征）。

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 文本分类示例
corpus = [
    'I love machine learning',
    'I love deep learning',
    'Python is great for programming',
    'I love programming in Python',
    'deep learning is powerful',
]
labels = [1, 1, 0, 0, 1]  # 1: ML, 0: 非 ML

# 特征提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# 训练
mnb = MultinomialNB(alpha=1.0)  # alpha 是平滑参数
mnb.fit(X, labels)

# 预测新文本
new_texts = ['I love AI and machine learning', 'Java programming is fun']
X_new = vectorizer.transform(new_texts)
predictions = mnb.predict(X_new)
probas = mnb.predict_proba(X_new)

for text, pred, proba in zip(new_texts, predictions, probas):
    print(f"'{text}' -> {'ML' if pred==1 else '非ML'} (置信度: {max(proba):.2f})")
```

### 3.3 伯努利朴素贝叶斯

适用于二值特征（存在/不存在）。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# 伯努利朴素贝叶斯关注特征是否出现，而不关注出现次数
bnb = BernoulliNB(alpha=1.0)

# 使用二元特征
vectorizer = CountVectorizer(binary=True)
X_binary = vectorizer.fit_transform(corpus)

bnb.fit(X_binary, labels)
```

### 3.4 三种变体对比

| 变体 | 数据分布 | 特征类型 | 典型场景 |
|------|----------|----------|----------|
| 高斯 NB | 正态分布 | 连续数值 | Iris、房价分类 |
| 多项式 NB | 多项分布 | 离散计数 | 文本分类 |
| 伯努利 NB | 伯努利分布 | 二值（0/1） | 文本二值特征 |

---

## 四、sklearn 朴素贝叶斯实战

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification

# 1. 高斯 NB
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=15,
    n_classes=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("=== 高斯朴素贝叶斯 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"分类报告:")
print(classification_report(y_test, y_pred))

# 2. 查看概率输出
y_proba = gnb.predict_proba(X_test)
print("前 5 个样本的预测概率:")
print(np.round(y_proba[:5], 4))

# 3. 类的先验和条件参数
print(f"\n类先验概率: {gnb.class_prior_}")
print(f"类均值形状: {gnb.theta_.shape}")
print(f"类方差形状: {gnb.sigma_.shape}")
```

---

## 五、实战项目：垃圾邮件分类

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 使用 20 Newsgroups 数据集
categories = ['comp.graphics', 'sci.space', 'rec.sport.baseball', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(
    subset='train', categories=categories, remove=('headers', 'footers', 'quotes')
)

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
)

# 构建 Pipeline：TF-IDF + 朴素贝叶斯
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=== 20 Newsgroups 分类 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 测试新邮件
test_emails = [
    'The Hubble telescope discovered a new galaxy in deep space',
    'The baseball team won the championship game last night',
    'New graphics card with ray tracing performance is amazing',
    'The senate voted on the new policy bill yesterday',
]

for email in test_emails:
    pred = pipeline.predict([email])[0]
    print(f"'{email[:50]}...' -> {newsgroups.target_names[pred]}")
```

---

## 六、拉普拉斯平滑

### 6.1 零概率问题

当某个特征值在训练集中从未与某个类别同时出现时，条件概率为 0，导致整个后验概率为 0。

### 6.2 平滑方法

| 方法 | 公式 | 说明 |
|------|------|------|
| 拉普拉斯平滑 | P(x|y) = (N_xc + α) / (N_c + α·V) | α=1 |
| 利德斯通平滑 | 同上 | α < 1 |

```python
# 拉普拉斯平滑的效果
alphas = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
train_scores = []
test_scores = []

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train_tfidf, y_train)
    train_scores.append(mnb.score(X_train_tfidf, y_train))
    test_scores.append(mnb.score(X_test_tfidf, y_test))

plt.figure(figsize=(10, 5))
plt.plot(alphas, train_scores, 'b-o', label='训练集')
plt.plot(alphas, test_scores, 'r-o', label='测试集')
plt.xlabel('平滑参数 α')
plt.xscale('log')
plt.ylabel('准确率')
plt.title('平滑参数对模型性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 七、朴素贝叶斯的优缺点

### 7.1 优点

| 优点 | 说明 |
|------|------|
| 训练极快 | 只需计算统计量，O(n·d) |
| 预测极快 | 只需查表计算概率 |
| 小样本友好 | 不需要大量训练数据 |
| 多分类友好 | 天然支持多分类 |
| 增量学习 | partial_fit 支持在线更新 |
| 可解释性强 | 能看到每个特征的贡献 |

### 7.2 缺点

| 缺点 | 说明 | 缓解方案 |
|------|------|----------|
| 独立性假设不成立 | 特征通常相关 | 特征选择、PCA |
| 概率校准差 | 输出概率可能不准确 | Platt Scaling |
| 零频率问题 | 未出现的特征概率为 0 | 拉普拉斯平滑 |
| 特征权重相等 | 不区分特征重要性 | 使用 TF-IDF |

---

## 八、实战建议

1. **基线模型**：朴素贝叶斯是很好的基线模型，先用它建立基准性能
2. **文本分类首选**：多项式 NB + TF-IDF 是文本分类的经典组合
3. **调参简单**：主要调 alpha（平滑参数），通常在 0.01~1.0 之间
4. **在线学习**：使用 partial_fit 支持流式数据训练
5. **特征工程**：朴素贝叶斯受益于好的特征选择，如 TF-IDF
6. **组合使用**：与其他模型组合（如 Stacking）可以提升性能

> 💡 **提示**：朴素贝叶斯虽然基于"朴素"的独立假设，但在实践中常常出奇制胜。它的概率思维——从先验出发，用证据更新后验——是理解更复杂的隐马尔可夫模型（HMM）的基础。

---

## 九、总结

朴素贝叶斯是基于贝叶斯定理的概率分类算法，通过朴素独立性假设简化计算。核心要点包括：

- **理论基础**：贝叶斯定理，P(y|x) ∝ P(y) · Π P(x_i|y)
- **核心假设**：给定类别，特征之间相互独立
- **三种变体**：高斯 NB（连续）、多项式 NB（计数）、伯努利 NB（二值）
- **平滑技术**：拉普拉斯平滑解决零概率问题
- **优势**：训练和预测极快，小样本友好，文本分类首选
- **局限**：独立性假设很少成立，概率输出可能不准确
- **应用场景**：垃圾邮件过滤、情感分析、文档分类、医疗诊断

---

📖 **上一篇**：[PCA 降维](./24-pca-dimensionality.md) | 📖 **下一篇**：[隐马尔可夫模型](./26-hmm.md) — 学习处理序列数据的概率模型。
