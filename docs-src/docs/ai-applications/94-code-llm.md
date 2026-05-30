---
title: 代码大语言模型
icon: code
order: 94
---

# 94. 代码大语言模型

## 概述

代码大语言模型（Code LLM）是专门针对编程任务训练的大语言模型，它们不仅能够理解自然语言，还能深度理解和生成多种编程语言的代码。在视觉语言模型教会 AI"看"之后，代码大语言模型则让 AI 学会了"写代码"。Code LLM 的出现正在深刻改变软件开发的范式，从辅助编码的 Copilot 到自主完成开发任务的 Devin，代码 AI 的能力边界在持续扩展。

### Code LLM 与传统 LLM 的区别

| 维度 | 传统 LLM | 代码 LLM |
|------|---------|---------|
| 训练数据 | 自然语言文本 | 代码 + 注释 + 文档 + 技术讨论 |
| 理解能力 | 语义理解 | 代码语法、逻辑、依赖关系 |
| 输出要求 | 语言流畅 | 语法正确、可执行、无 Bug |
| 上下文长度 | 通常较短 | 需要支持整个代码库 |
| 评估标准 | BLEU/ROUGE | Pass@k、代码执行通过率 |
| 特殊能力 | 无 | AST 理解、执行反馈、工具调用 |

### 主流 Code LLM 模型对比

| 模型 | 开发者 | 参数量 | 支持语言 | 核心优势 | 开源情况 |
|------|--------|--------|---------|---------|---------|
| GPT-4 / o1 | OpenAI | 未公开 | 所有主流语言 | 综合推理能力强 | 闭源 |
| Claude 3.5 Sonnet | Anthropic | 未公开 | 所有主流语言 | 代码质量和安全性 | 闭源 |
| Code Llama | Meta | 7B-70B | 20+ | 开源，专为代码优化 | 开源 |
| StarCoder 2 | BigCode | 3B-15B | 100+ | 多语言覆盖广 | 开源 |
| DeepSeek-Coder | DeepSeek | 1B-33B | 80+ | 中英文代码理解 | 开源 |
| Qwen2.5-Coder | 阿里巴巴 | 0.5B-32B | 90+ | 中文生态好 | 开源 |
| Gemini | Google | 未公开 | 所有主流语言 | 多模态代码理解 | 闭源 |

## Code LLM 训练方法论

### 训练数据构建

Code LLM 的训练数据构建是模型质量的关键。高质量的数据源包括：

```python
"""
代码训练数据构建管道
"""
import json
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

class CodeDataBuilder:
    """代码训练数据构建器"""

    # 支持的编程语言及其文件扩展名
    LANGUAGE_EXTENSIONS = {
        "python": [".py"],
        "java": [".java"],
        "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
        "rust": [".rs"],
        "cpp": [".cpp", ".cc", ".cxx"],
        "c": [".c", ".h"],
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        # 初始化 Tree-sitter 解析器
        self.python_parser = Parser(Language(tspython.language()))

    def collect_files(self, languages: List[str] = None) -> List[Path]:
        """收集指定编程语言的源文件"""
        if languages is None:
            languages = list(self.LANGUAGE_EXTENSIONS.keys())

        extensions = set()
        for lang in languages:
            if lang in self.LANGUAGE_EXTENSIONS:
                extensions.update(self.LANGUAGE_EXTENSIONS[lang])

        files = []
        for ext in extensions:
            files.extend(self.data_dir.rglob(f"*{ext}"))

        return files

    def extract_code_blocks(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        从源文件中提取代码块和文档字符串

        Args:
            file_path: 源文件路径
        Returns:
            代码块列表
        """
        code_blocks = []

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 提取 Python 函数（含文档字符串）
        if file_path.suffix == '.py':
            code_blocks = self._extract_python_functions(content, str(file_path))

        # 提取类定义
        class_blocks = self._extract_class_definitions(content, str(file_path))
        code_blocks.extend(class_blocks)

        return code_blocks

    def _extract_python_functions(self, content: str, source: str) -> List[Dict[str, Any]]:
        """提取 Python 函数及其文档字符串"""
        blocks = []

        # 正则匹配函数定义
        func_pattern = re.compile(
            r'(def\s+\w+\s*\([^)]*\)\s*:(?:\s*\n(?:\s*""".*?"""|\s*\'\'\'.*?\'\'\'|[^\n]*\n)*?(?:\n(?:\s+.*)?)*)',
            re.DOTALL
        )

        for match in func_pattern.finditer(content):
            func_code = match.group(1)

            # 提取文档字符串
            docstring_match = re.search(r'("""|\'\'\')(.+?)\1', func_code, re.DOTALL)
            docstring = docstring_match.group(2).strip() if docstring_match else ""

            blocks.append({
                "type": "function",
                "code": func_code.strip(),
                "docstring": docstring,
                "source": source
            })

        return blocks

    def _extract_class_definitions(self, content: str, source: str) -> List[Dict[str, Any]]:
        """提取 Python 类定义"""
        blocks = []

        class_pattern = re.compile(
            r'(class\s+\w+(?:\s*\([^)]*\))?\s*:(?:\s*\n(?:\s*""".*?"""|\s*\'\'\'.*?\'\'\'|[^\n]*\n)*?(?:\n(?:\s+.*)?)*)',
            re.DOTALL
        )

        for match in class_pattern.finditer(content):
            class_code = match.group(1)

            docstring_match = re.search(r'("""|\'\'\')(.+?)\1', class_code, re.DOTALL)
            docstring = docstring_match.group(2).strip() if docstring_match else ""

            blocks.append({
                "type": "class",
                "code": class_code.strip(),
                "docstring": docstring,
                "source": source
            })

        return blocks

    def build_instruction_dataset(self, code_blocks: List[Dict]) -> List[Dict]:
        """
        构建指令微调数据集
        格式：instruction (自然语言描述) -> output (代码实现)
        """
        dataset = []

        for block in code_blocks:
            if block["docstring"]:
                # 使用文档字符串作为指令，代码作为输出
                instruction = block["docstring"]
                if block["type"] == "function":
                    instruction = f"请编写一个 Python 函数，要求：\n{instruction}"
                elif block["type"] == "class":
                    instruction = f"请编写一个 Python 类，要求：\n{instruction}"

                dataset.append({
                    "instruction": instruction,
                    "output": block["code"],
                    "type": block["type"],
                    "source": block["source"]
                })

        return dataset

    def save_dataset(self, dataset: List[Dict], output_path: str):
        """保存数据集到 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(dataset)} 条数据到 {output_path}")
```

### 代码专用的预训练目标

```python
"""
代码预训练的特殊目标设计
"""

# 1. 填充式掩码（Fill-in-the-Middle, FIM）
# 传统 LLM 使用因果掩码（仅能看到前面的 token）
# 代码 LLM 使用 FIM 掩码，可以同时看到前后文

FIM_FORMAT = """
<PREFIX> def calculate_sum(a, b):
<PREFIX_END><SUFFIX>    return result
<MIDDLE>    result = a + b
<MIDDLE_END>
"""

# 2. 代码结构感知训练
# 利用 AST（抽象语法树）信息增强模型对代码结构的理解

class ASTAwareTrainer:
    """感知 AST 结构的代码训练器"""

    def augment_with_ast(self, code: str) -> List[str]:
        """基于 AST 生成增强的训练样本"""
        # 方法 1：变量名混淆
        # 方法 2：控制流重排
        # 方法 3：等价代码变换
        pass
```

## Code LLM 应用开发实战

### 1. 代码补全系统

```python
"""
基于 Code LLM 的实时代码补全系统
"""
import re
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class CompletionResult:
    """补全结果"""
    text: str
    start_offset: int
    end_offset: int
    confidence: float

class CodeCompletionEngine:
    """代码补全引擎"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_completion(self,
                       code_prefix: str,
                       code_suffix: str,
                       file_path: str = "",
                       max_tokens: int = 200) -> CompletionResult:
        """
        获取代码补全建议

        Args:
            code_prefix: 光标前的代码
            code_suffix: 光标后的代码
            file_path: 文件路径（用于语言推断）
            max_tokens: 最大补全长度
        Returns:
            补全结果
        """
        # 推断编程语言
        language = self._detect_language(file_path, code_prefix)

        # 构建提示词
        prompt = self._build_completion_prompt(
            language, code_prefix, code_suffix
        )

        # 调用模型
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"请补全以下代码：\n\n{code_prefix}[CURSOR]"}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95,
            stop=["\n\n\n", "```"]
        )

        completion_text = response.choices[0].message.content.strip()

        # 清理代码块标记
        completion_text = re.sub(r'^```[a-z]*\n', '', completion_text)
        completion_text = re.sub(r'\n```$', '', completion_text)

        return CompletionResult(
            text=completion_text,
            start_offset=len(code_prefix),
            end_offset=len(code_prefix),
            confidence=0.9 if response.choices[0].finish_reason == "stop" else 0.7
        )

    def _detect_language(self, file_path: str, code: str) -> str:
        """检测编程语言"""
        if file_path.endswith('.py'):
            return "python"
        elif file_path.endswith('.java'):
            return "java"
        elif file_path.endswith(('.js', '.jsx')):
            return "javascript"
        elif file_path.endswith(('.ts', '.tsx')):
            return "typescript"
        return "unknown"

    def _build_completion_prompt(self, language: str, prefix: str, suffix: str) -> str:
        """构建补全提示词"""
        return f"""你是一个专业的 {language} 编程助手。请根据以下代码上下文，
补全光标 [CURSOR] 位置的代码。

要求：
1. 补全的代码必须符合 {language} 语法
2. 保持与前后代码的逻辑一致性
3. 只输出补全的代码，不要重复已有代码
4. 不要包含解释文字

前面的代码：
```{language}
{prefix}
"""

    def get_multi_completions(self,
                               code_prefix: str,
                               code_suffix: str,
                               n: int = 3) -> List[CompletionResult]:
        """获取多个补全建议供用户选择"""
        results = []
        for _ in range(n):
            result = self.get_completion(code_prefix, code_suffix)
            results.append(result)
        return results
```

### 2. 代码审查与 Bug 检测

```python
"""
基于 Code LLM 的自动化代码审查系统
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class IssueSeverity(Enum):
    """问题严重程度"""
    CRITICAL = "critical"    # 严重
    WARNING = "warning"      # 警告
    INFO = "info"            # 建议
    STYLE = "style"          # 风格

class IssueType(Enum):
    """问题类型"""
    SECURITY = "security"          # 安全问题
    PERFORMANCE = "performance"    # 性能问题
    BUG = "bug"                    # 潜在 Bug
    MAINTAINABILITY = "maintainability"  # 可维护性
    BEST_PRACTICE = "best_practice"      # 最佳实践

@dataclass
class CodeIssue:
    """代码问题"""
    line_number: int
    severity: IssueSeverity
    issue_type: IssueType
    message: str
    suggestion: str
    code_snippet: str

class CodeReviewEngine:
    """自动化代码审查引擎"""

    REVIEW_PROMPT = """你是一个资深代码审查专家。请审查以下代码，找出以下类型的问题：

1. 安全问题：如 SQL 注入、XSS、敏感信息泄露等
2. 潜在 Bug：如空指针、资源泄露、边界条件等
3. 性能问题：如不必要的循环、内存浪费等
4. 可维护性：如复杂度过高、命名不规范等
5. 最佳实践：如缺少错误处理、缺少注释等

对于每个发现的问题，请提供：
- 行号
- 严重程度（critical/warning/info/style）
- 问题描述
- 修复建议
- 相关代码片段

请以 JSON 格式输出，格式如下：
{{
    "issues": [
        {{
            "line": 行号,
            "severity": "严重程度",
            "type": "问题类型",
            "message": "问题描述",
            "suggestion": "修复建议",
            "code": "相关代码"
        }}
    ]
}}

代码：
{code}
"""

    def __init__(self, client):
        self.client = client

    def review(self, code: str, language: str = "python") -> List[CodeIssue]:
        """
        审查代码

        Args:
            code: 要审查的代码
            language: 编程语言
        Returns:
            问题列表
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.REVIEW_PROMPT.format(code=code)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        import json
        result = json.loads(response.choices[0].message.content)

        issues = []
        for issue_data in result.get("issues", []):
            issues.append(CodeIssue(
                line_number=issue_data.get("line", 0),
                severity=IssueSeverity(issue_data.get("severity", "info")),
                issue_type=IssueType(issue_data.get("type", "best_practice")),
                message=issue_data.get("message", ""),
                suggestion=issue_data.get("suggestion", ""),
                code_snippet=issue_data.get("code", "")
            ))

        return issues

    def generate_review_report(self, issues: List[CodeIssue]) -> str:
        """生成代码审查报告"""
        report_lines = ["## 代码审查报告\n"]

        # 按严重程度排序
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.INFO: 2,
            IssueSeverity.STYLE: 3
        }
        issues.sort(key=lambda x: severity_order[x.severity])

        # 统计
        summary = {}
        for issue in issues:
            summary[issue.severity] = summary.get(issue.severity, 0) + 1

        report_lines.append("### 问题统计")
        for severity, count in summary.items():
            report_lines.append(f"- {severity.value}: {count} 个")

        report_lines.append("\n### 详细问题\n")
        for i, issue in enumerate(issues, 1):
            report_lines.extend([
                f"#### 问题 {i}",
                f"- **严重程度**: {issue.severity.value}",
                f"- **类型**: {issue.issue_type.value}",
                f"- **描述**: {issue.message}",
                f"- **建议**: {issue.suggestion}",
                f"- **代码**:\n```python\n{issue.code_snippet}\n```\n"
            ])

        return "\n".join(report_lines)
```

### 3. 代码解释与文档生成

```python
"""
代码解释与文档生成系统
"""
class CodeDocumentationGenerator:
    """代码文档生成器"""

    def __init__(self, client):
        self.client = client

    def generate_docstring(self, code: str, language: str = "python",
                           style: str = "google") -> str:
        """
        为函数生成文档字符串

        Args:
            code: 函数代码
            language: 编程语言
            style: 文档风格 (google/sphinx/numpy)
        Returns:
            生成的文档字符串
        """
        style_templates = {
            "google": """
请为以下函数生成 Google 风格的文档字符串：

{code}

只输出文档字符串部分，格式如下：
'''
描述文本

Args:
    参数名: 参数描述

Returns:
    返回值描述

Raises:
    异常类型: 异常描述
'''
""",
            "sphinx": """
请为以下函数生成 Sphinx 风格的文档字符串：

{code}

只输出文档字符串部分，格式如下：
'''
描述文本

:param 参数名: 参数描述
:type 参数名: 参数类型
:return: 返回值描述
:rtype: 返回类型
:raises 异常类型: 异常描述
'''
"""
        }

        prompt = style_templates.get(style, style_templates["google"]).format(code=code)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    def generate_readme(self, project_structure: Dict[str, List[str]],
                        description: str = "") -> str:
        """
        为项目生成 README.md

        Args:
            project_structure: 项目文件结构
            description: 项目描述（可选）
        Returns:
            README 内容
        """
        prompt = f"""请为以下项目生成一份完整的 README.md 文档：

项目描述：{description}

项目结构：
```
{self._format_tree(project_structure)}
```

请包含以下部分：
1. 项目简介
2. 功能特性
3. 安装说明
4. 使用方法
5. 项目结构说明
6. 贡献指南
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def _format_tree(self, structure: Dict[str, List[str]], prefix: str = "") -> str:
        """格式化项目树结构"""
        lines = []
        items = list(structure.items())
        for i, (name, children) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")
            if children:
                extension = "    " if is_last else "│   "
                child_structure = {child: [] for child in children}
                lines.append(self._format_tree(child_structure, prefix + extension))
        return "\n".join(lines)
```

### 4. 代码转换与迁移

```python
"""
代码转换引擎：将代码从一种语言转换为另一种
"""
class CodeTranspiler:
    """代码转换器"""

    def __init__(self, client):
        self.client = client

    def transpile(self, source_code: str, source_lang: str,
                  target_lang: str) -> str:
        """
        将代码从源语言转换为目标语言

        Args:
            source_code: 源代码
            source_lang: 源语言
            target_lang: 目标语言
        Returns:
            转换后的代码
        """
        prompt = f"""你是一个专业的代码转换专家。请将以下 {source_lang} 代码转换为 {target_lang} 代码。

要求：
1. 保持功能完全等价
2. 使用 {target_lang} 的最佳实践和惯用法
3. 保持代码风格整洁
4. 添加必要的注释

源 {source_lang} 代码：
```{source_lang}
{source_code}
```

请输出转换后的 {target_lang} 代码：
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        # 清理代码块标记
        result = re.sub(r'^```[a-z]*\n', '', result)
        result = re.sub(r'\n```$', '', result)
        return result

# 使用示例
# transpiler = CodeTranspiler(client)
# java_code = transpiler.transpile(
#     source_code="def greet(name): return f'Hello, {name}!'",
#     source_lang="python",
#     target_lang="java"
# )
```

## 代码 LLM 的评估指标

| 指标 | 说明 | 适用场景 | 计算方式 |
|------|------|---------|---------|
| Pass@k | 生成 k 个方案中至少一个通过测试的概率 | 代码生成 | 蒙特卡洛采样 |
| Exact Match | 生成代码与参考答案完全匹配 | 简单代码补全 | 字符串比较 |
| CodeBLEU | 考虑语法树和语义的代码 BLEU | 代码翻译 | 综合 n-gram + AST |
| Execution Accuracy | 代码执行结果正确 | 算法题求解 | 测试用例通过率 |
| HumanEval | OpenAI 的代码生成基准 | 通用代码能力 | 164 道编程题 |
| MBPP | Google 的编程基准 | 编程能力评估 | 974 道编程题 |

## 总结

代码大语言模型正在深刻改变软件开发的每一个环节。从代码补全到自动审查，从文档生成到代码转换，Code LLM 已经成为开发者不可或缺的智能助手。理解 Code LLM 的训练方法、评估指标和应用模式，对于提升开发效率、保障代码质量具有重要意义。随着模型能力的持续增强，未来的 AI 编程助手将能够处理更加复杂的工程任务。

---

**下一篇**: [95. AI 搜索技术](./95-ai-search.md)
