---
title: 概率与统计基础
icon: chart-bar
order: 4
---

# 概率与统计基础

## 引言

在上一篇 [线性代数基础](./03-linear-algebra.md) 中，我们掌握了向量、矩阵和矩阵分解等线性代数核心知识。然而，现实世界中的数据充满了不确定性——传感器的测量误差、用户行为的随机波动、市场价格的不可预测变化。概率论与统计学正是处理这种不确定性的数学工具。

在机器学习中，概率与统计无处不在：朴素贝叶斯分类器直接基于概率理论，损失函数（如交叉熵）源于信息论，模型评估依赖统计检验，正则化方法可以从贝叶斯角度理解。本篇将系统性地介绍概率论与统计学的核心概念，重点关注它们在 AI 和机器学习中的实际应用。

## 概率基础

### 基本概念

概率是描述随机事件发生可能性的数值，取值范围为 [0, 1]。

| 概念 | 定义 | 示例 |
|------|------|------|
| 随机试验 | 结果不确定的实验 | 掷骰子 |
| 样本空间 | 所有可能结果的集合 | {1, 2, 3, 4, 5, 6} |
| 事件 | 样本空间的子集 | "掷出偶数" = {2, 4, 6} |
| 概率 | 事件发生的可能性 | P(偶数) = 3/6 = 0.5 |

### 概率公理（柯尔莫哥洛夫公理）

1. **非负性**：对于任何事件 A，P(A) >= 0
2. **规范性**：样本空间 S 的概率为 1，P(S) = 1
3. **可加性**：互斥事件的概率等于各事件概率之和

### 条件概率与独立性

**条件概率**：在事件 B 发生的条件下，事件 A 发生的概率。

```
P(A|B) = P(A ∩ B) / P(B)
```

```python
# 示例：疾病诊断中的条件概率
# 已知：
# - 某疾病的发病率 P(D) = 0.01
# - 检测的灵敏度 P(+|D) = 0.95（患病时检测为阳性的概率）
# - 检测的特异度 P(-|非D) = 0.90（未患病时检测为阴性的概率）
# 求：检测为阳性时实际患病的概率 P(D|+)

P_D = 0.01        # 先验概率
P_positive_given_D = 0.95    # 灵敏度
P_negative_given_not_D = 0.90  # 特异度

P_not_D = 1 - P_D
P_positive_given_not_D = 1 - P_negative_given_not_D  # 假阳性率 = 0.10

# 全概率公式：P(+) = P(+|D)P(D) + P(+|非D)P(非D)
P_positive = P_positive_given_D * P_D + P_positive_given_not_D * P_not_D

# 贝叶斯公式：P(D|+) = P(+|D)P(D) / P(+)
P_D_given_positive = P_positive_given_D * P_D / P_positive

print(f"检测为阳性时实际患病的概率: {P_D_given_positive:.4f} ({P_D_given_positive:.2%})")
print(f"这个结果远低于直觉预期，说明先验概率的重要性")
```

> 💡 **贝叶斯悖论**：即使检测准确率高达 95%，对于罕见病（发病率 1%），阳性结果中真正患病的比例可能只有不到 9%。这就是为什么医生通常会对阳性结果进行复诊确认。

**独立性**：如果 P(A ∩ B) = P(A) × P(B)，则事件 A 和 B 相互独立。

### 随机变量

随机变量是将随机试验的结果映射到数值的函数。

| 类型 | 说明 | 示例 |
|------|------|------|
| 离散随机变量 | 取值为有限或可列个数值 | 掷骰子的结果 |
| 连续随机变量 | 取值在某个区间内连续 | 身高、温度 |

### 期望与方差

```
期望 E[X] = Σ x_i · P(x_i)        (离散)
期望 E[X] = ∫ x · f(x) dx         (连续)

方差 Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
标准差 σ = √Var(X)
```

```python
import numpy as np

# 离散随机变量示例：掷骰子的期望和方差
outcomes = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6] * 6)

# 期望
expected_value = np.sum(outcomes * probabilities)
print(f"掷骰子的期望: {expected_value:.2f}")

# 方差
variance = np.sum((outcomes - expected_value) ** 2 * probabilities)
std_dev = np.sqrt(variance)
print(f"方差: {variance:.4f}")
print(f"标准差: {std_dev:.4f}")

# 方差的意义：衡量随机变量取值的离散程度
# 标准差越大，取值越分散；标准差越小，取值越集中
```

## 常见概率分布

### 离散分布

#### 伯努利分布（Bernoulli Distribution）

描述单次二元随机试验（成功/失败）的分布。

```
P(X=1) = p, P(X=0) = 1-p
E[X] = p, Var(X) = p(1-p)
```

```python
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# 伯努利分布：抛硬币（正面概率 0.5）
p = 0.5
bernoulli = stats.bernoulli(p)

x = np.array([0, 1])
pmf = bernoulli.pmf(x)
print(f"伯努利分布 PMF: P(X=0)={pmf[0]:.2f}, P(X=1)={pmf[1]:.2f}")
print(f"期望: {bernoulli.mean():.2f}, 方差: {bernoulli.var():.2f}")
```

#### 二项分布（Binomial Distribution）

描述 n 次独立伯努利试验中成功次数的分布。

```
P(X=k) = C(n,k) · p^k · (1-p)^(n-k)
E[X] = np, Var(X) = np(1-p)
```

```python
# 二项分布：抛10次硬币，正面出现次数的分布
n, p = 10, 0.5
binomial = stats.binom(n, p)

x = np.arange(0, n+1)
pmf = binomial.pmf(x)
print(f"二项分布 (n={n}, p={p}):")
for k, prob in zip(x, pmf):
    print(f"  P(X={k}) = {prob:.4f}")
print(f"期望: {binomial.mean():.2f}, 方差: {binomial.var():.2f}")
```

#### 泊松分布（Poisson Distribution）

描述单位时间内随机事件发生次数的分布。

```
P(X=k) = λ^k · e^(-λ) / k!
E[X] = λ, Var(X) = λ
```

```python
# 泊松分布：平均每小时收到 3 封邮件
lam = 3
poisson = stats.poisson(lam)

x = np.arange(0, 10)
pmf = poisson.pmf(x)
print(f"泊松分布 (λ={lam}):")
for k, prob in zip(x, pmf):
    print(f"  P(X={k}) = {prob:.4f}")
```

### 连续分布

#### 正态分布（高斯分布）

最重要的概率分布，自然界和人类社会中大量现象都近似服从正态分布。

```
f(x) = (1 / (σ√(2π))) · exp(-(x-μ)² / (2σ²))
E[X] = μ, Var(X) = σ²
```

```python
# 正态分布
mu, sigma = 0, 1
normal = stats.norm(mu, sigma)

# 68-95-99.7 规则
print(f"正态分布 N(μ={mu}, σ={sigma}):")
print(f"  P(μ-σ ≤ X ≤ μ+σ) = {normal.cdf(1) - normal.cdf(-1):.4f}  (≈68%)")
print(f"  P(μ-2σ ≤ X ≤ μ+2σ) = {normal.cdf(2) - normal.cdf(-2):.4f}  (≈95%)")
print(f"  P(μ-3σ ≤ X ≤ μ+3σ) = {normal.cdf(3) - normal.cdf(-3):.4f}  (≈99.7%)")

# 生成正态分布样本
samples = np.random.normal(mu, sigma, 10000)
print(f"\n10000个样本的统计量:")
print(f"  样本均值: {samples.mean():.4f}")
print(f"  样本标准差: {samples.std():.4f}")
```

> 💡 **中心极限定理**：无论原始总体分布如何，当样本量足够大时，样本均值的分布近似服从正态分布。这是统计推断的理论基石。

#### 均匀分布

在区间 [a, b] 内所有值等概率出现。

```python
# 均匀分布
a, b = 0, 1
uniform = stats.uniform(a, b - a)
print(f"均匀分布 U({a}, {b}):")
print(f"  期望: {uniform.mean():.2f}")
print(f"  方差: {uniform.var():.4f}")
```

#### 指数分布

描述泊松过程中事件间隔时间的分布。

```python
# 指数分布：公交车平均10分钟来一班
lam = 1/10  # 率参数
exponential = stats.expon(scale=10)  # scale = 1/λ

print(f"指数分布 (平均间隔10分钟):")
print(f"  P(等待时间 ≤ 5分钟) = {exponential.cdf(5):.4f}")
print(f"  P(等待时间 ≤ 10分钟) = {exponential.cdf(10):.4f}")
print(f"  P(等待时间 > 15分钟) = {1 - exponential.cdf(15):.4f}")
```

## 联合分布、边缘分布与条件分布

### 联合分布

多个随机变量同时取某些值的概率。

```python
# 示例：天气和交通方式的联合分布
#        晴天  雨天
# 开车    0.3   0.2
# 公交    0.2   0.15
# 步行    0.1   0.05

joint = np.array([
    [0.30, 0.20],   # 开车
    [0.20, 0.15],   # 公交
    [0.10, 0.05],   # 步行
])
transport = ['开车', '公交', '步行']
weather = ['晴天', '雨天']

# 验证：所有联合概率之和为 1
print(f"联合概率之和: {joint.sum():.2f}")
```

### 边缘分布

从联合分布中"消去"某些变量得到的分布。

```python
# 交通方式的边缘分布（对天气求和）
P_transport = joint.sum(axis=1)
for t, p in zip(transport, P_transport):
    print(f"P({t}) = {p:.2f}")

# 天气的边缘分布（对交通方式求和）
P_weather = joint.sum(axis=0)
for w, p in zip(weather, P_weather):
    print(f"P({w}) = {p:.2f}")
```

### 条件分布

```python
# 已知雨天的条件下，交通方式的条件分布
rain_idx = 1
P_transport_given_rain = joint[:, rain_idx] / P_weather[rain_idx]
for t, p in zip(transport, P_transport_given_rain):
    print(f"P({t}|雨天) = {p:.4f}")
```

## 贝叶斯定理

贝叶斯定理是概率论中最重要的公式之一，也是机器学习中贝叶斯方法的理论基础。

```
P(A|B) = P(B|A) · P(A) / P(B)

其中：
- P(A|B)：后验概率（Posterior）—— 看到证据 B 后 A 的概率
- P(B|A)：似然（Likelihood）—— A 成立时 B 出现的概率
- P(A)：先验概率（Prior）—— 看到证据前 A 的概率
- P(B)：证据（Evidence）—— 归一化常数
```

```python
# 贝叶斯定理在垃圾邮件分类中的应用
# 假设：
# - 垃圾邮件的先验概率 P(Spam) = 0.3
# - 垃圾邮件中出现"免费"的概率 P("免费"|Spam) = 0.4
# - 正常邮件中出现"免费"的概率 P("免费"|Ham) = 0.05

P_spam = 0.3
P_ham = 0.7
P_word_given_spam = 0.4
P_word_given_ham = 0.05

# P("免费") = P("免费"|Spam)P(Spam) + P("免费"|Ham)P(Ham)
P_word = P_word_given_spam * P_spam + P_word_given_ham * P_ham

# P(Spam|"免费") = P("免费"|Spam)P(Spam) / P("免费")
P_spam_given_word = P_word_given_spam * P_spam / P_word

print(f"先验概率 P(Spam) = {P_spam:.2f}")
print(f"看到'免费'后，后验概率 P(Spam|'免费') = {P_spam_given_word:.4f}")
print(f"邮件是垃圾邮件的概率从 {P_spam:.0%} 上升到了 {P_spam_given_word:.1%}")
```

> 💡 **朴素贝叶斯分类器**：假设特征之间条件独立，利用贝叶斯定理进行分类。尽管"条件独立"假设在现实中很少成立，但朴素贝叶斯在文本分类等任务中表现优异。

## 统计学基础

### 参数估计

#### 点估计

用样本统计量来估计总体参数。

```python
# 点估计示例
np.random.seed(42)
population = np.random.normal(loc=50, scale=10, size=100000)

# 抽取样本
sample = np.random.choice(population, size=100, replace=False)

# 点估计
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)  # ddof=1 使用无偏估计
sample_var = sample.var(ddof=1)

print(f"总体均值: {population.mean():.4f}")
print(f"样本均值估计: {sample_mean:.4f}")
print(f"总体标准差: {population.std():.4f}")
print(f"样本标准差估计: {sample_std:.4f}")
```

#### 区间估计（置信区间）

给出参数的估计范围，而非单一点值。

```python
from scipy import stats as st
import numpy as np

# 计算 95% 置信区间
np.random.seed(42)
sample = np.random.normal(loc=50, scale=10, size=100)

n = len(sample)
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
standard_error = sample_std / np.sqrt(n)

# t 分布的临界值（95% 置信水平）
t_critical = st.t.ppf(0.975, df=n-1)

# 置信区间
ci_lower = sample_mean - t_critical * standard_error
ci_upper = sample_mean + t_critical * standard_error

print(f"样本均值: {sample_mean:.4f}")
print(f"95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"总体真实均值 (50) 是否在区间内: {ci_lower <= 50 <= ci_upper}")
```

### 假设检验

假设检验用于判断样本数据是否支持某个关于总体的假设。

#### 基本步骤

1. **提出假设**：原假设 H0 和备择假设 H1
2. **选择检验统计量**
3. **计算 p 值**
4. **做出决策**：如果 p 值 < 显著性水平 α，拒绝 H0

```python
# 示例：A/B 测试
# 原版本转化率 10%，新版本有 12% 的转化率
# 问：新版本的提升是否统计显著？

from scipy import stats

# 模拟数据
np.random.seed(42)
n_a, n_b = 10000, 10000
conversions_a = np.random.binomial(1, 0.10, n_a)
conversions_b = np.random.binomial(1, 0.12, n_b)

# 双样本比例 z 检验
from statsmodels.stats.proportion import proportions_ztest

count = np.array([conversions_b.sum(), conversions_a.sum()])
nobs = np.array([n_b, n_a])

z_stat, p_value = proportions_ztest(count, nobs)

print(f"A 版本转化率: {conversions_a.mean():.4f}")
print(f"B 版本转化率: {conversions_b.mean():.4f}")
print(f"z 统计量: {z_stat:.4f}")
print(f"p 值: {p_value:.6f}")
print(f"在 α=0.05 水平下 {'拒绝' if p_value < 0.05 else '不拒绝'} 原假设")
```

#### 常见检验方法

| 检验方法 | 用途 | 前提条件 |
|---------|------|---------|
| t 检验 | 比较两组均值差异 | 正态分布、方差齐性 |
| 卡方检验 | 检验分类变量独立性 | 期望频数 >= 5 |
| ANOVA | 比较多组均值差异 | 正态分布、方差齐性 |
| Mann-Whitney U | 非参数两组比较 | 无分布假设 |

## 信息论基础

### 熵（Entropy）

熵度量了随机变量的不确定性。熵越大，不确定性越高。

```
H(X) = -Σ P(x) · log₂ P(x)
```

```python
def entropy(probs):
    """计算离散分布的熵"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # 过滤零概率
    return -np.sum(probs * np.log2(probs))

# 示例1：公平硬币（最大不确定性）
fair_coin = [0.5, 0.5]
print(f"公平硬币的熵: {entropy(fair_coin):.4f} bits")

# 示例2： biased 硬币（较小不确定性）
biased_coin = [0.9, 0.1]
print(f"biased 硬币的熵: {entropy(biased_coin):.4f} bits")

# 示例3：确定性事件（零不确定性）
certain = [1.0, 0.0]
print(f"确定性事件的熵: {entropy(certain):.4f} bits")
```

> 💡 **熵的意义**：熵可以理解为编码随机变量取值所需的最少平均比特数。公平硬币的熵为 1 bit，意味着每次抛掷至少需要 1 bit 来编码结果。

### 交叉熵（Cross-Entropy）

交叉熵衡量用一个分布 q 来编码另一个分布 p 所需的平均比特数。

```
H(p, q) = -Σ p(x) · log q(x)
```

```python
def cross_entropy(p, q):
    """计算交叉熵"""
    p, q = np.array(p), np.array(q)
    return -np.sum(p * np.log(q + 1e-15))

# 真实分布
p = [0.5, 0.3, 0.2]

# 两个预测分布
q1 = [0.5, 0.3, 0.2]  # 完美预测
q2 = [0.1, 0.2, 0.7]  # 差的预测

print(f"真实分布 p: {p}")
print(f"p 的自熵: {entropy(p):.4f}")
print(f"交叉熵 H(p, q1): {cross_entropy(p, q1):.4f}")
print(f"交叉熵 H(p, q2): {cross_entropy(p, q2):.4f}")
```

> 💡 **交叉熵损失**：在分类任务中，交叉熵是最常用的损失函数。当预测分布 q 接近真实分布 p 时，交叉熵最小。

### KL 散度（Kullback-Leibler Divergence）

KL 散度衡量两个概率分布之间的差异。

```
KL(p || q) = Σ p(x) · log(p(x) / q(x)) = H(p, q) - H(p)
```

```python
def kl_divergence(p, q):
    """计算 KL 散度"""
    p, q = np.array(p), np.array(q)
    return np.sum(p * np.log((p + 1e-15) / (q + 1e-15)))

p = [0.5, 0.3, 0.2]
q1 = [0.5, 0.3, 0.2]
q2 = [0.1, 0.2, 0.7]

print(f"KL(p || q1): {kl_divergence(p, q1):.4f} (相同分布，KL=0)")
print(f"KL(p || q2): {kl_divergence(p, q2):.4f}")

# KL 散度的性质：
# 1. KL(p || q) >= 0（非负性）
# 2. KL(p || q) = 0 当且仅当 p = q
# 3. KL(p || q) ≠ KL(q || p)（非对称性）
print(f"KL(q2 || p): {kl_divergence(q2, p):.4f} (非对称)")
```

### 互信息（Mutual Information）

互信息衡量两个随机变量之间的相互依赖程度。

```
I(X; Y) = KL(P(X,Y) || P(X)P(Y)) = H(X) - H(X|Y)
```

```python
def mutual_information(joint_probs):
    """计算互信息"""
    joint = np.array(joint_probs)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    
    mi = 0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j]))
    return mi

# 示例：高度相关的两个变量
#        Y=0  Y=1
# X=0    0.4  0.1
# X=1    0.1  0.4
joint_high = [[0.4, 0.1], [0.1, 0.4]]

# 示例：独立的两个变量
#        Y=0  Y=1
# X=0   0.25 0.25
# X=1   0.25 0.25
joint_indep = [[0.25, 0.25], [0.25, 0.25]]

print(f"高度相关变量的互信息: {mutual_information(joint_high):.4f} bits")
print(f"独立变量的互信息: {mutual_information(joint_indep):.4f} bits")
```

> 💡 **互信息在机器学习中的应用**：
> - 特征选择：选择与目标变量互信息高的特征
> - 决策树：信息增益 = 互信息，用于选择最佳分割点
> - 信息瓶颈理论：深度学习的理论基础之一

## 概率分布在机器学习中的应用

### 最大似然估计（MLE）

```python
# MLE 示例：估计硬币的正面概率
np.random.seed(42)
true_p = 0.7
flips = np.random.binomial(1, true_p, 1000)

# MLE 估计
mle_p = flips.mean()
print(f"真实概率: {true_p}")
print(f"MLE 估计: {mle_p:.4f}")

# 似然函数
def likelihood(p, data):
    """伯努利分布的似然函数"""
    n_heads = data.sum()
    n_tails = len(data) - n_heads
    return p ** n_heads * (1 - p) ** n_tails

# 绘制似然函数
p_range = np.linspace(0.01, 0.99, 100)
likelihoods = [likelihood(p, flips) for p in p_range]
# 为避免数值问题，通常使用对数似然
log_likelihoods = [np.log(likelihood(p, flips) + 1e-300) for p in p_range]

mle_idx = np.argmax(log_likelihoods)
print(f"对数似然最大化估计: {p_range[mle_idx]:.4f}")
```

### 贝叶斯推断

```python
# 贝叶斯推断示例：Beta-Bernoulli 共轭先验
# 先验：Beta(α=2, β=2)
# 观测数据：100次抛硬币，70次正面

from scipy.stats import beta

# 先验参数
alpha_prior, beta_prior = 2, 2

# 观测数据
n_heads = 70
n_tails = 30

# 后验参数（Beta-Bernoulli 共轭）
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + n_tails

# 可视化
x = np.linspace(0, 1, 200)
prior = beta.pdf(x, alpha_prior, beta_prior)
posterior = beta.pdf(x, alpha_post, beta_post)
mle_estimate = n_heads / (n_heads + n_tails)

print(f"先验期望: {alpha_prior / (alpha_prior + beta_prior):.2f}")
print(f"MLE 估计: {mle_estimate:.4f}")
print(f"后验期望 (贝叶斯估计): {alpha_post / (alpha_post + beta_post):.4f}")
print(f"后验众数 (MAP): {(alpha_post - 1) / (alpha_post + beta_post - 2):.4f}")
```

## 总结

通过本篇的学习，你应该掌握了以下概率与统计核心知识：

1. **概率基础**：条件概率、独立性、随机变量、期望与方差
2. **常见分布**：伯努利、二项、泊松、正态、均匀、指数分布
3. **贝叶斯定理**：先验、似然、后验的关系，在分类问题中的应用
4. **统计推断**：点估计、区间估计、假设检验
5. **信息论**：熵、交叉熵、KL 散度、互信息
6. **机器学习应用**：最大似然估计、贝叶斯推断、交叉熵损失

概率与统计为机器学习提供了处理不确定性的理论框架。下一篇我们将学习微积分——理解模型优化过程的数学基础。

> [!NOTE] 下一篇
> [05 - 机器学习中的微积分](./05-calculus-for-ml.md) —— 掌握导数、梯度与链式法则，理解模型训练和优化的数学基础。
