---
title: Python 编程基础
icon: code
order: 6
---

# Python 编程基础

## 引言

在 [机器学习入门](./02-machine-learning-intro.md) 中，我们已经使用了一些 Python 代码来演示机器学习概念。在 [微积分](./05-calculus-for-ml.md) 中，我们也用 Python 验证了数学推导。Python 是 AI 和机器学习领域最主流的编程语言——Scikit-learn、TensorFlow、PyTorch 等所有主流框架都提供了 Python API。

本篇将系统性地介绍 Python 编程的核心知识，从基础语法到高级特性，从数据处理到工程实践。无论你是编程初学者还是有其他语言经验的开发者，本篇都能帮助你快速掌握 Python 在 AI 开发中的核心用法。

## 为什么 Python 是 AI 的首选语言

| 优势 | 说明 |
|------|------|
| 简洁易读 | 语法接近自然语言，学习曲线平缓 |
| 生态丰富 | NumPy、Pandas、Scikit-learn、PyTorch、TensorFlow |
| 社区活跃 | 大量教程、问答、开源项目 |
| 胶水语言 | 可以调用 C/C++ 库，兼顾易用性和性能 |
| 交互式开发 | Jupyter Notebook 支持即时反馈 |

## Python 基础语法

### 变量与数据类型

Python 是动态类型语言，变量不需要声明类型。

```python
# === 基本数据类型 ===

# 整数（int）
age = 25
population = 7_800_000_000  # 下划线增加可读性

# 浮点数（float）
price = 19.99
pi = 3.14159
scientific = 6.022e23  # 科学计数法

# 布尔值（bool）
is_active = True
is_deleted = False

# 字符串（str）
name = "Python"
greeting = 'Hello, World!'
multi_line = """这是一个
多行字符串"""

# 空值（None）
result = None

# === 查看类型 ===
print(f"type(42) = {type(42)}")
print(f"type(3.14) = {type(3.14)}")
print(f"type(True) = {type(True)}")
print(f"type('hello') = {type('hello')}")
print(f"type(None) = {type(None)}")
```

### 字符串操作

```python
# === 字符串操作 ===
text = "Python for Machine Learning"

# 基本操作
print(f"长度: {len(text)}")
print(f"大写: {text.upper()}")
print(f"小写: {text.lower()}")
print(f"分割: {text.split()}")
print(f"替换: {text.replace('Python', 'AI')}")
print(f"查找: {text.find('Machine')}")
print(f"是否包含: {'Python' in text}")

# 字符串格式化
name = "Alice"
score = 95.678
print(f"姓名: {name}, 成绩: {score:.2f}")
print(f"姓名: {name:>10}, 成绩: {score:10.2f}")

# f-string 表达式
print(f"两年后成绩: {score + 5:.2f}")
print(f"是否及格: {'是' if score >= 60 else '否'}")
```

### 数据结构

#### 列表（List）

列表是最常用的数据结构，有序且可变。

```python
# === 列表 ===
fruits = ["apple", "banana", "cherry"]
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# 访问元素
print(f"第一个: {fruits[0]}")
print(f"最后一个: {fruits[-1]}")

# 切片
print(f"前两个: {fruits[:2]}")
print(f"从第二个开始: {fruits[1:]}")
print(f"反转: {numbers[::-1]}")

# 添加/删除
fruits.append("date")
fruits.insert(1, "apricot")
fruits.remove("banana")
last = fruits.pop()
print(f"操作后: {fruits}")

# 列表推导式
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
print(f"平方: {squares}")
print(f"偶数: {evens}")

# 常用操作
print(f"排序: {sorted(numbers)}")
print(f"最大值: {max(numbers)}")
print(f"最小值: {min(numbers)}")
print(f"求和: {sum(numbers)}")
print(f"长度: {len(numbers)}")
```

#### 元组（Tuple）

元组与列表类似，但不可变。

```python
# === 元组 ===
coordinates = (3.0, 4.0)
rgb = (255, 128, 0)

# 解包
x, y = coordinates
print(f"x={x}, y={y}")

r, g, b = rgb
print(f"R={r}, G={g}, B={b}")

# 元组不可变
try:
    coordinates[0] = 5.0
except TypeError as e:
    print(f"元组不可变: {e}")
```

#### 字典（Dict）

字典是键值对集合，查找效率高。

```python
# === 字典 ===
student = {
    "name": "Alice",
    "age": 20,
    "grades": [95, 87, 92],
    "major": "Computer Science",
}

# 访问
print(f"姓名: {student['name']}")
print(f"年龄: {student.get('age')}")
print(f"电话: {student.get('phone', '未提供')}")  # 默认值

# 添加/修改
student["email"] = "alice@example.com"
student["age"] = 21

# 遍历
for key, value in student.items():
    print(f"  {key}: {value}")

# 字典推导式
squares_dict = {x: x**2 for x in range(1, 6)}
print(f"平方字典: {squares_dict}")
```

#### 集合（Set）

集合是无重复元素的无序集合。

```python
# === 集合 ===
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

print(f"并集: {a | b}")
print(f"交集: {a & b}")
print(f"差集: {a - b}")
print(f"对称差: {a ^ b}")

# 去重
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique = list(set(numbers))
print(f"去重: {unique}")
```

### 控制流程

```python
# === 条件语句 ===
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(f"成绩 {score} 对应等级: {grade}")

# 三元表达式
status = "及格" if score >= 60 else "不及格"
print(f"状态: {status}")

# === for 循环 ===
# 遍历列表
for fruit in ["apple", "banana", "cherry"]:
    print(f"  {fruit}")

# enumerate：同时获取索引和值
for i, fruit in enumerate(["apple", "banana", "cherry"]):
    print(f"  {i}: {fruit}")

# range
for i in range(3):
    print(f"  第 {i+1} 次")

# === while 循环 ===
count = 0
while count < 5:
    count += 1
    if count == 3:
        continue  # 跳过
    if count == 6:
        break     # 退出
```

### 函数

```python
# === 函数定义 ===
def greet(name, greeting="Hello"):
    """向用户打招呼"""
    return f"{greeting}, {name}!"

print(greet("Alice"))
print(greet("Bob", "Hi"))

# === 多返回值 ===
def min_max(numbers):
    return min(numbers), max(numbers)

minimum, maximum = min_max([3, 1, 4, 1, 5, 9])
print(f"最小值: {minimum}, 最大值: {maximum}")

# === 可变参数 ===
def average(*args):
    """计算平均值"""
    return sum(args) / len(args)

print(f"平均: {average(1, 2, 3, 4, 5):.2f}")

# === Lambda 表达式 ===
square = lambda x: x ** 2
print(f"5² = {square(5)}")

# 与 map/filter 配合
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"平方: {squared}")
print(f"偶数: {evens}")

# === 类型注解（推荐） ===
def add(a: int, b: int) -> int:
    return a + b

def greet_user(name: str, times: int = 1) -> list[str]:
    return [f"Hello, {name}!" for _ in range(times)]
```

## 面向对象编程

### 类与对象

```python
class DataPreprocessor:
    """数据预处理器"""

    # 类变量
    version = "1.0"

    def __init__(self, name: str, method: str = "standard"):
        """初始化"""
        self.name = name          # 实例变量
        self.method = method
        self._fitted = False      # 私有变量约定
        self._mean = None
        self._std = None

    def fit(self, data: list[float]) -> None:
        """拟合数据"""
        import statistics
        self._mean = statistics.mean(data)
        self._std = statistics.stdev(data) if len(data) > 1 else 1.0
        self._fitted = True
        print(f"[{self.name}] 拟合完成: mean={self._mean:.2f}, std={self._std:.2f}")

    def transform(self, data: list[float]) -> list[float]:
        """转换数据"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        return [(x - self._mean) / self._std for x in data]

    def fit_transform(self, data: list[float]) -> list[float]:
        self.fit(data)
        return self.transform(data)

    def __repr__(self) -> str:
        return f"DataPreprocessor(name='{self.name}', method='{self.method}')"

# 使用
preprocessor = DataPreprocessor("age_scaler", method="z-score")
data = [25, 30, 35, 40, 45]
transformed = preprocessor.fit_transform(data)
print(f"原始数据: {data}")
print(f"标准化后: {[f'{x:.2f}' for x in transformed]}")
```

### 继承

```python
class BaseEstimator:
    """基础估计器"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    def fit(self, X, y):
        raise NotImplementedError("子类必须实现 fit()")

    def predict(self, X):
        raise NotImplementedError("子类必须实现 predict()")

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = sum(1 for p, t in zip(predictions, y) if p == t) / len(y)
        return accuracy


class SimpleClassifier(BaseEstimator):
    """简单分类器（基于阈值的分类）"""

    def __init__(self, name: str, threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold

    def fit(self, X, y):
        # 计算正样本的特征均值作为阈值
        self.threshold_ = sum(x for x, label in zip(X, y) if label == 1) / max(sum(1 for label in y if label == 1), 1)
        self.is_fitted = True
        print(f"[{self.name}] 学习到的阈值: {self.threshold_:.2f}")

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()")
        return [1 if x > self.threshold_ else 0 for x in X]

# 使用
clf = SimpleClassifier("threshold_classifier")
X_train = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]
y_train = [0, 0, 1, 1, 0, 1]

clf.fit(X_train, y_train)
X_test = [0.1, 0.5, 0.9]
predictions = clf.predict(X_test)
print(f"测试预测: {predictions}")
```

## 模块与包

### 导入模块

```python
# 导入整个模块
import math
print(f"π = {math.pi}")
print(f"e = {math.e}")

# 导入特定函数
from math import sqrt, factorial
print(f"√16 = {sqrt(16)}")
print(f"5! = {factorial(5)}")

# 导入并重命名
import numpy as np
import pandas as pd

print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
```

### 创建自己的模块

```python
# metrics.py（假设的文件）
"""自定义评估指标模块"""

def accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

def precision(y_true, y_pred, positive=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    return tp / max(tp + fp, 1)

def recall(y_true, y_pred, positive=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    return tp / max(tp + fn, 1)

def f1_score(y_true, y_pred, positive=1):
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return 2 * p * r / max(p + r, 1e-10)
```

## 文件操作

```python
import json
import csv

# === 写入/读取 JSON ===
data = {
    "name": "实验1",
    "parameters": {"lr": 0.01, "epochs": 100},
    "results": {"accuracy": 0.95, "loss": 0.05},
}

# 写入
with open("experiment.json", "w") as f:
    json.dump(data, f, indent=2)

# 读取
with open("experiment.json") as f:
    loaded = json.load(f)

print(f"实验名称: {loaded['name']}")
print(f"准确率: {loaded['results']['accuracy']}")

# === 写入/读取 CSV ===
import io

# 模拟 CSV 数据
csv_data = """name,age,score
Alice,25,95
Bob,30,87
Charlie,28,92"""

# 读取 CSV
reader = csv.DictReader(io.StringIO(csv_data))
rows = list(reader)
for row in rows:
    print(f"  {row['name']}: age={row['age']}, score={row['score']}")
```

## 异常处理

```python
def safe_divide(a, b):
    """安全除法"""
    try:
        result = a / b
    except ZeroDivisionError:
        print("错误: 除数不能为零")
        return None
    except TypeError:
        print("错误: 参数必须是数字")
        return None
    else:
        return result
    finally:
        print(f"  除法运算完成: {a} / {b}")

print(safe_divide(10, 3))
print(safe_divide(10, 0))
```

## AI 开发环境搭建

### 安装与配置

```bash
# 推荐使用 Miniconda 或 Anaconda 管理 Python 环境
# 安装 Miniconda 后：

# 创建 AI 开发环境
conda create -n ai-dev python=3.11 -y
conda activate ai-dev

# 安装核心数据科学库
pip install numpy pandas matplotlib seaborn scikit-learn

# 安装深度学习框架
pip install torch torchvision

# 安装 Jupyter
pip install jupyterlab

# 启动 Jupyter
jupyter lab
```

### Jupyter Notebook 常用快捷键

| 快捷键 | 功能 |
|--------|------|
| Shift+Enter | 运行当前单元格并跳到下一个 |
| Ctrl+Enter | 运行当前单元格 |
| Alt+Enter | 运行当前单元格并在下方插入新单元格 |
| A | 在上方插入新单元格 |
| B | 在下方插入新单元格 |
| D,D | 删除当前单元格 |
| M | 切换为 Markdown 模式 |
| Y | 切换为 Code 模式 |
| Tab | 自动补全 |
| Shift+Tab | 查看函数文档 |

### Python 在 AI 中的典型代码模式

```python
# === 数据加载 ===
import pandas as pd
from sklearn.datasets import load_iris

# 方式1：内置数据集
iris = load_iris()
X, y = iris.data, iris.target

# 方式2：CSV 文件
# df = pd.read_csv("data.csv")
# X = df.drop("target", axis=1).values
# y = df["target"].values

# === 数据划分 ===
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 模型训练 ===
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 模型评估 ===
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# === 保存/加载模型 ===
import joblib

joblib.dump(model, "model.pkl")
loaded_model = joblib.load("model.pkl")
```

## Python 高级特性（AI 开发常用）

### 装饰器

```python
import time
from functools import wraps

def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  {func.__name__} 耗时: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_model():
    """模拟模型训练"""
    time.sleep(0.5)
    return {"accuracy": 0.95}

result = train_model()
print(f"训练结果: {result}")
```

### 生成器

```python
def batch_generator(data, batch_size):
    """批量数据生成器"""
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# 使用
data = list(range(25))
for batch in batch_generator(data, batch_size=8):
    print(f"  批次: {batch}")
```

### 上下文管理器

```python
from contextlib import contextmanager

@contextmanager
def timer_context(name):
    """计时的上下文管理器"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"  [{name}] 耗时: {elapsed:.4f}s")

with timer_context("数据处理"):
    time.sleep(0.3)
    data = [x**2 for x in range(10000)]
```

## 总结

通过本篇的学习，你应该掌握了以下 Python 核心知识：

1. **基础语法**：变量、数据类型、字符串操作
2. **数据结构**：列表、元组、字典、集合及其操作
3. **控制流程**：条件语句、循环、列表推导式
4. **函数**：定义、参数、返回值、Lambda 表达式、类型注解
5. **面向对象**：类、对象、继承、魔术方法
6. **模块与包**：导入、创建、常用 AI 库
7. **文件操作**：JSON、CSV 读写
8. **异常处理**：try/except/finally
9. **高级特性**：装饰器、生成器、上下文管理器
10. **开发环境**：Conda、Jupyter、常用快捷键

Python 是 AI 开发的利器。掌握基础后，下一篇我们将学习 NumPy 和 Pandas——AI 数据处理的两个核心库。

> [!NOTE] 下一篇
> [07 - NumPy 与 Pandas](./07-numpy-pandas.md) —— 掌握 Python 数据科学的核心工具库，学习高效的数据处理与分析方法。
