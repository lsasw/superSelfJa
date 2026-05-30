---
title: 线性代数基础
icon: calculator
order: 3
---

# 线性代数基础

## 引言

在上一篇 [机器学习入门](./02-machine-learning-intro.md) 中，我们了解了机器学习的基本概念和工作流程。你可能注意到了，无论是数据表示、模型运算还是优化过程，都频繁涉及向量、矩阵等数学对象。事实上，线性代数是机器学习最重要的数学基础之一——数据以向量和矩阵的形式存储，模型的运算本质上是矩阵运算，深度学习中的反向传播依赖于矩阵微积分。

本篇将系统性地介绍线性代数的核心概念，重点关注它们在机器学习中的实际应用。通过本篇学习，你将掌握标量、向量、矩阵和张量的运算规则，理解矩阵分解的意义，并能够使用 Python 进行高效的线性代数计算。

## 为什么机器学习需要线性代数

### 数据表示

在机器学习中，所有数据都需要转化为数字形式。线性代数提供了统一的数据表示方式：

| 数据类型 | 线性代数对象 | 示例 |
|---------|-------------|------|
| 单个数值 | 标量（Scalar） | 温度、价格、年龄 |
| 一维列表 | 向量（Vector） | 用户的特征列表 |
| 二维表格 | 矩阵（Matrix） | 数据集（样本 × 特征） |
| 多维数组 | 张量（Tensor） | 彩色图像（高 × 宽 × 通道） |

```python
import numpy as np

# 标量：一个房子的面积
scalar = 120.5
print(f"标量: {scalar}")

# 向量：一个房子的多个特征 [面积, 房间数, 楼层]
vector = np.array([120.5, 3, 2])
print(f"向量: {vector}")
print(f"向量形状: {vector.shape}")

# 矩阵：多个房子的数据集（每行一个样本，每列一个特征）
matrix = np.array([
    [120.5, 3, 2, 150],   # 房子1
    [85.0, 2, 1, 95],     # 房子2
    [200.0, 4, 3, 280],   # 房子3
    [95.5, 2, 2, 110],    # 房子4
])
print(f"矩阵形状: {matrix.shape}")  # (4个样本, 4个特征)
print(f"矩阵:\n{matrix}")

# 张量：一张 28×28 的灰度图像
image = np.random.rand(28, 28)
print(f"张量形状: {image.shape}")

# 张量：一张 28×28 的彩色图像（RGB 三通道）
color_image = np.random.rand(28, 28, 3)
print(f"彩色图像张量形状: {color_image.shape}")
```

### 计算效率

使用矩阵运算可以大幅加速计算。对比以下两种实现方式：

```python
import time

# 数据准备
np.random.seed(42)
size = 500
A = np.random.rand(size, size)
B = np.random.rand(size, size)

# 方法一：纯 Python 循环（慢）
start = time.time()
C_python = [[sum(A[i][k] * B[k][j] for k in range(size)) for j in range(size)] for i in range(size)]
python_time = time.time() - start
print(f"纯 Python 耗时: {python_time:.4f} 秒")

# 方法二：NumPy 矩阵运算（快）
start = time.time()
C_numpy = np.dot(A, B)
numpy_time = time.time() - start
print(f"NumPy 耗时: {numpy_time:.4f} 秒")
print(f"加速比: {python_time / numpy_time:.1f}x")
```

> 💡 **关键理解**：NumPy 底层使用 C 语言实现，并且利用了 BLAS/LAPACK 等高度优化的线性代数库。矩阵运算比 Python 循环快数十到数百倍。

## 标量、向量、矩阵与张量

### 标量（Scalar）

标量是最简单的数学对象，只是一个数字。

```python
# Python 原生标量
a = 3.14
b = -7

# NumPy 标量
scalar = np.float64(3.14)
```

### 向量（Vector）

向量是一维数组，有大小（维度）和方向。

```python
import numpy as np

# 创建向量
v = np.array([1, 2, 3])
print(f"向量 v: {v}")
print(f"维度: {v.shape[0]}")

# 向量的基本运算
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 向量加法
print(f"v1 + v2 = {v1 + v2}")

# 向量数乘
print(f"2 * v1 = {2 * v1}")

# 点积（内积）
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")

# 向量的模（长度）
norm = np.linalg.norm(v1)
print(f"||v1|| = {norm:.4f}")

# 单位向量（归一化）
unit_v1 = v1 / np.linalg.norm(v1)
print(f"v1 的单位向量: {unit_v1}")
print(f"单位向量的模: {np.linalg.norm(unit_v1):.4f}")
```

#### 向量点积的几何意义

两个向量的点积可以衡量它们的"相似度"：

- 点积 > 0：向量方向大致相同（夹角 < 90°）
- 点积 = 0：向量正交（夹角 = 90°）
- 点积 < 0：向量方向大致相反（夹角 > 90°）

```python
def vector_angle(v1, v2):
    """计算两个向量之间的夹角（度）"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

# 示例
a = np.array([1, 0])
b = np.array([0, 1])
c = np.array([1, 1])

print(f"a 和 b 的夹角: {vector_angle(a, b):.1f}°")  # 90° (正交)
print(f"a 和 c 的夹角: {vector_angle(a, c):.1f}°")  # 45°
print(f"a 和 -a 的夹角: {vector_angle(a, -a):.1f}°")  # 180°
```

### 矩阵（Matrix）

矩阵是二维数组，可以看作是多行多列的数字表格。

```python
# 创建矩阵
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(f"矩阵 A 的形状: {A.shape}")  # (2, 3) — 2行3列
print(f"矩阵 A:\n{A}")

# 矩阵的基本属性
print(f"行数: {A.shape[0]}")
print(f"列数: {A.shape[1]}")
print(f"转置:\n{A.T}")

# 矩阵加法
B = np.array([
    [7, 8, 9],
    [10, 11, 12],
])
print(f"A + B:\n{A + B}")

# 矩阵数乘
print(f"2 * A:\n{2 * A}")

# 矩阵乘法
C = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
print(f"A (2×3) × C (3×2) =\n{np.dot(A, C)}")  # 结果形状 (2, 2)
```

> 💡 **矩阵乘法规则**：矩阵 A（m×n）乘以矩阵 B（n×p）得到矩阵 C（m×p）。A 的列数必须等于 B 的行数。结果 C 的每个元素 c_ij 是 A 的第 i 行与 B 的第 j 列的点积。

### 张量（Tensor）

张量是向量和矩阵的推广，可以有任意维度。

```python
# 0阶张量 = 标量
t0 = np.array(5)

# 1阶张量 = 向量
t1 = np.array([1, 2, 3])

# 2阶张量 = 矩阵
t2 = np.array([[1, 2], [3, 4]])

# 3阶张量 = 三维数组（如彩色图像）
t3 = np.random.rand(3, 4, 2)  # 3个4×2的矩阵

# 4阶张量 = 四维数组（如批量图像：batch_size × height × width × channels）
t4 = np.random.rand(32, 28, 28, 3)  # 32张28×28的彩色图像

print(f"0阶张量形状: {t0.shape}")
print(f"1阶张量形状: {t1.shape}")
print(f"2阶张量形状: {t2.shape}")
print(f"3阶张量形状: {t3.shape}")
print(f"4阶张量形状: {t4.shape}")
```

## 特殊矩阵

### 单位矩阵

对角线为 1，其余为 0 的方阵，相当于矩阵乘法中的"1"。

```python
# 3×3 单位矩阵
I = np.eye(3)
print(f"3×3 单位矩阵:\n{I}")

# 验证：A × I = A
A = np.array([[1, 2], [3, 4]])
I_2 = np.eye(2)
print(f"A × I =\n{np.dot(A, I_2)}")
```

### 对角矩阵

只有对角线上有非零元素的方阵。

```python
# 从对角线元素创建对角矩阵
d = np.array([1, 2, 3])
D = np.diag(d)
print(f"对角矩阵:\n{D}")

# 提取矩阵的对角线
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diagonal = np.diag(A)
print(f"对角线元素: {diagonal}")
```

### 对称矩阵

满足 A = A^T 的方阵。

```python
A = np.array([
    [1, 2, 3],
    [2, 5, 6],
    [3, 6, 9],
])
print(f"矩阵 A:\n{A}")
print(f"A 的转置:\n{A.T}")
print(f"A 是否对称: {np.array_equal(A, A.T)}")
```

### 正定矩阵

对于任意非零向量 x，都有 x^T A x > 0。在机器学习中，正定矩阵保证了优化问题的良好性质。

```python
def is_positive_definite(A):
    """检查矩阵是否正定"""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

A_pos = np.array([[4, 2], [2, 3]])
A_neg = np.array([[1, 2], [2, 1]])

print(f"A_pos 是否正定: {is_positive_definite(A_pos)}")
print(f"A_neg 是否正定: {is_positive_definite(A_neg)}")
```

## 矩阵运算

### 转置（Transpose）

交换矩阵的行和列。

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
print(f"A:\n{A}")
print(f"A 的转置:\n{A.T}")

# 性质：(AB)^T = B^T A^T
B = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
print(f"(AB)^T:\n{np.dot(A, B).T}")
print(f"B^T A^T:\n{np.dot(B.T, A.T)}")
```

### 逆矩阵（Inverse）

方阵 A 的逆矩阵 A^{-1} 满足 A × A^{-1} = I。

```python
A = np.array([
    [1, 2],
    [3, 4],
])
A_inv = np.linalg.inv(A)
print(f"A 的逆矩阵:\n{A_inv}")

# 验证
I_computed = np.dot(A, A_inv)
print(f"A × A^(-1) (应接近单位矩阵):\n{np.round(I_computed, 10)}")

# 奇异矩阵（不可逆）
B = np.array([
    [1, 2],
    [2, 4],  # 第二行是第一行的2倍，线性相关
])
try:
    B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print(f"奇异矩阵不可逆: {e}")
```

> 💡 **伪逆矩阵**：当矩阵不可逆时（如列数大于行数），可以使用伪逆矩阵（Moore-Penrose 伪逆）来求解。在 NumPy 中使用 `np.linalg.pinv()` 计算。伪逆在线性回归的最小二乘解法中有重要应用。

### 行列式（Determinant）

方阵的行列式是一个标量，反映了矩阵所代表的线性变换的"缩放因子"。

```python
A = np.array([
    [1, 2],
    [3, 4],
])
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.4f}")

# 行列式的意义：
# |det(A)| > 1: 变换扩大了面积
# |det(A)| < 1: 变换缩小了面积
# det(A) = 0: 变换将空间压缩到了更低维度（矩阵不可逆）

print(f"det(I) = {np.linalg.det(np.eye(3)):.4f}")
```

## 线性方程组

### 矩阵形式表示

线性方程组可以用矩阵方程 Ax = b 表示：

```
2x + 3y = 8        [2  3] [x]   [8]
4x -  y = 2   →    [4 -1] [y] = [2]
```

```python
A = np.array([[2, 3], [4, -1]])
b = np.array([8, 2])

# 解法1：直接使用求解器（推荐）
x = np.linalg.solve(A, b)
print(f"解: x = {x[0]:.4f}, y = {x[1]:.4f}")

# 验证
print(f"验证: Ax = {np.dot(A, x)}")
print(f"应该等于 b = {b}")

# 解法2：使用逆矩阵（不推荐，数值不稳定）
x_inv = np.dot(np.linalg.inv(A), b)
print(f"使用逆矩阵求解: x = {x_inv[0]:.4f}, y = {x_inv[1]:.4f}")
```

## 特征值与特征向量

### 定义

对于方阵 A，如果存在非零向量 v 和标量 λ，使得：

```
A × v = λ × v
```

则 λ 称为 A 的特征值（eigenvalue），v 称为对应的特征向量（eigenvector）。

**几何意义**：特征向量是在矩阵变换下方向不变的向量，特征值表示变换在该方向上的缩放倍数。

```python
A = np.array([
    [4, 2],
    [1, 3],
])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"特征值: {eigenvalues}")
print(f"特征向量（每列对应一个特征向量）:\n{eigenvectors}")

# 验证 A × v = λ × v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = np.dot(A, v)
    lam_v = lam * v
    print(f"验证: A·v{i+1} = {Av}, λ{i+1}·v{i+1} = {lam_v}")
    print(f"是否相等: {np.allclose(Av, lam_v)}")
```

### 特征分解

如果方阵 A 有 n 个线性无关的特征向量，则可以进行特征分解：

```
A = QΛQ^{-1}
```

其中 Q 的列是 A 的特征向量，Λ 是对角矩阵，对角线元素是特征值。

```python
A = np.array([
    [4, 2],
    [1, 3],
])

eigenvalues, eigenvectors = np.linalg.eig(A)
Q = eigenvectors
Lambda = np.diag(eigenvalues)
Q_inv = np.linalg.inv(Q)

# 验证 A = QΛQ^{-1}
A_reconstructed = np.dot(Q, np.dot(Lambda, Q_inv))
print(f"原始矩阵:\n{A}")
print(f"重建矩阵:\n{np.round(A_reconstructed, 10)}")
print(f"是否相等: {np.allclose(A, A_reconstructed)}")
```

### 特征值在机器学习中的应用

| 应用 | 说明 |
|------|------|
| 主成分分析（PCA） | 协方差矩阵的特征向量确定主成分方向 |
| 谱聚类 | 利用图拉普拉斯矩阵的特征向量进行聚类 |
| 马尔可夫链 | 稳态分布对应转移矩阵的特征值为 1 的特征向量 |
| 页面排序（PageRank） | Google 链接矩阵的主特征向量 |

## 矩阵分解

### SVD 分解（奇异值分解）

SVD 是最重要的矩阵分解方法之一，适用于任意形状的矩阵。

```
A = UΣV^T
```

其中 U 和 V 是正交矩阵，Σ 是对角矩阵（对角线元素为奇异值）。

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
Sigma = np.diag(S)

print(f"原始矩阵 A:\n{A}")
print(f"\nU:\n{np.round(U, 4)}")
print(f"\n奇异值 Σ: {S}")
print(f"\nV^T:\n{np.round(Vt, 4)}")

# 验证
A_reconstructed = np.dot(U, np.dot(Sigma, Vt))
print(f"\n重建矩阵:\n{np.round(A_reconstructed, 4)}")
print(f"是否相等: {np.allclose(A, A_reconstructed)}")
```

#### SVD 的应用：数据压缩

```python
# 用 SVD 进行低秩近似（数据压缩）
image = np.random.rand(100, 100)  # 模拟一张 100×100 的图像

# 完整 SVD
U, S, Vt = np.linalg.svd(image, full_matrices=False)

# 保留前 k 个奇异值
k = 10
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

# 低秩近似
compressed = np.dot(U_k, np.dot(S_k, Vt_k))

# 压缩率
original_size = image.size
compressed_size = U_k.size + k + Vt_k.size
print(f"原始大小: {original_size}")
print(f"压缩后大小: {compressed_size}")
print(f"压缩率: {compressed_size / original_size * 100:.1f}%")

# 重建误差
error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
print(f"相对误差: {error:.4f}")
```

### QR 分解

将矩阵分解为正交矩阵 Q 和上三角矩阵 R 的乘积。

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])

Q, R = np.linalg.qr(A)
print(f"原始矩阵 A:\n{A}")
print(f"正交矩阵 Q:\n{np.round(Q, 4)}")
print(f"上三角矩阵 R:\n{np.round(R, 4)}")
print(f"验证 Q × R:\n{np.round(np.dot(Q, R), 4)}")
```

### Cholesky 分解

适用于对称正定矩阵，分解为下三角矩阵与其转置的乘积。

```python
# 对称正定矩阵
A = np.array([
    [4, 2],
    [2, 3],
])

L = np.linalg.cholesky(A)
print(f"原始矩阵:\n{A}")
print(f"下三角矩阵 L:\n{L}")
print(f"验证 L × L^T:\n{np.dot(L, L.T)}")
```

## 线性代数在机器学习中的应用实例

### 线性回归的矩阵解法

线性回归的解析解（正规方程）为：

```
θ = (X^T X)^{-1} X^T y
```

```python
# 准备数据
np.random.seed(42)
X_raw = 2 * np.random.rand(100, 1)  # 100个样本，1个特征
y = 4 + 3 * X_raw + np.random.randn(100, 1)  # y = 4 + 3x + noise

# 添加偏置列（x0 = 1）
X_b = np.c_[np.ones((100, 1)), X_raw]  # 100×2

# 正规方程求解
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"截距: {theta_best[0][0]:.4f} (真实值: 4)")
print(f"斜率: {theta_best[1][0]:.4f} (真实值: 3)")

# 使用新数据预测
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(f"\n预测: x=0 → y={y_predict[0][0]:.2f}, x=2 → y={y_predict[1][0]:.2f}")
```

### 主成分分析（PCA）

PCA 的核心是协方差矩阵的特征分解。

```python
# 模拟数据
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 4)  # 4维特征

# 标准化
X_mean = X - X.mean(axis=0)

# 计算协方差矩阵
cov_matrix = np.dot(X_mean.T, X_mean) / (n_samples - 1)
print(f"协方差矩阵形状: {cov_matrix.shape}")

# 特征分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 按特征值从大到小排序
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

print(f"\n特征值: {eigenvalues}")
print(f"各主成分解释的方差比例: {eigenvalues / eigenvalues.sum()}")

# 投影到前2个主成分
W = eigenvectors[:, :2]  # 取前2个特征向量
X_pca = np.dot(X_mean, W)
print(f"\nPCA 后数据形状: {X_pca.shape}")
print(f"前5个样本:\n{X_pca[:5]}")
```

### 余弦相似度计算

在推荐系统和信息检索中，余弦相似度用于衡量两个向量的相似度。

```python
def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 模拟三个用户的评分向量
user1 = np.array([5, 3, 4, 2, 1])
user2 = np.array([4, 4, 3, 3, 2])
user3 = np.array([1, 2, 5, 4, 5])

print(f"user1 与 user2 的相似度: {cosine_similarity(user1, user2):.4f}")
print(f"user1 与 user3 的相似度: {cosine_similarity(user1, user3):.4f}")
print(f"user2 与 user3 的相似度: {cosine_similarity(user2, user3):.4f}")
```

## 常用 NumPy 线性代数操作速查

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# === 基本运算 ===
A.T                    # 转置
np.dot(A, B)           # 矩阵乘法
A @ B                  # 矩阵乘法（Python 3.5+ 推荐写法）
A * B                  # 逐元素乘法（非矩阵乘法！）
np.linalg.inv(A)       # 逆矩阵
np.linalg.det(A)       # 行列式
np.linalg.trace(A)     # 迹（对角线元素之和）

# === 特征分解 ===
eigenvalues, eigenvectors = np.linalg.eig(A)

# === SVD 分解 ===
U, S, Vt = np.linalg.svd(A)

# === 解线性方程组 ===
x = np.linalg.solve(A, b)

# === 其他 ===
np.eye(3)              # 3×3 单位矩阵
np.diag([1, 2, 3])     # 对角矩阵
np.linalg.norm(v)      # 向量模/范数
np.linalg.pinv(A)      # 伪逆矩阵
np.linalg.cholesky(A)  # Cholesky 分解
```

## 总结

通过本篇的学习，你应该掌握了以下线性代数核心知识：

1. **基本对象**：标量、向量、矩阵、张量的概念与 NumPy 表示
2. **向量运算**：加法、数乘、点积、模、单位向量及其几何意义
3. **矩阵运算**：加法、数乘、乘法、转置、逆矩阵、行列式
4. **特殊矩阵**：单位矩阵、对角矩阵、对称矩阵、正定矩阵
5. **特征值与特征向量**：定义、计算、几何意义与机器学习应用
6. **矩阵分解**：SVD 分解、QR 分解、Cholesky 分解
7. **机器学习应用**：线性回归正规方程、PCA、余弦相似度

线性代数是理解机器学习算法的数学语言。在后续的学习中，你将看到这些概念如何贯穿整个机器学习流程。下一篇我们将学习概率与统计——另一个关键的数学基础。

> [!NOTE] 下一篇
> [04 - 概率与统计](./04-probability-statistics.md) —— 理解随机性、概率分布与统计推断，掌握机器学习中的不确定性建模基础。
