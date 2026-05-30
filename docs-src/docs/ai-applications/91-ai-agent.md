---
title: AI Agent 智能体开发
icon: robot
order: 91
---

# 91. AI Agent 智能体开发

## 概述

AI Agent（人工智能智能体）是近年来人工智能领域最具革命性的技术范式之一。与传统 AI 模型不同，Agent 不仅仅是一个被动的语言生成器，而是一个能够感知环境、自主规划、调用工具、执行任务并持续学习的独立实体。2023 年以来，随着 ReAct、AutoGPT、LangChain、CrewAI 等框架的出现，AI Agent 从学术研究迅速走向工业级应用。

### 什么是 AI Agent

AI Agent 可以被理解为一个具备以下核心能力的智能系统：

| 能力维度 | 说明 | 传统 AI | AI Agent |
|---------|------|---------|----------|
| 感知能力 | 接收和理解环境信息 | 仅处理输入文本 | 多模态感知、实时数据流 |
| 推理规划 | 将复杂任务分解为子任务 | 无规划能力 | 自动分解、排序、优化 |
| 工具使用 | 调用外部 API 和工具 | 不能调用外部工具 | 函数调用、API 交互 |
| 记忆管理 | 存储和检索历史经验 | 仅对话上下文 | 长期记忆、向量数据库 |
| 自主学习 | 从反馈中持续改进 | 需要重新训练 | 在线学习、自我反思 |

### Agent 的核心架构

一个完整的 AI Agent 通常包含以下核心组件：

```
┌─────────────────────────────────────────┐
│              AI Agent 架构               │
├─────────────────────────────────────────┤
│  感知层：环境观测 → 信息提取 → 状态维护   │
│  决策层：任务理解 → 规划生成 → 策略选择   │
│  执行层：工具调用 → 动作执行 → 结果收集   │
│  记忆层：短期记忆 → 长期记忆 → 经验检索   │
│  反思层：结果评估 → 策略调整 → 自我优化   │
└─────────────────────────────────────────┘
```

## Agent 的核心组件详解

### 1. 规划引擎（Planning Engine）

规划引擎是 Agent 的"大脑"，负责任务分解和策略制定。常见的规划模式包括：

- **思维链（Chain of Thought, CoT）**：引导模型逐步推理
- **ReAct 框架**：结合推理（Reasoning）和行动（Acting）
- **Tree of Thoughts（ToT）**：多路径探索后选择最优解
- **Reflexion**：通过自我反思改进策略

#### ReAct 模式实现

ReAct 是当前最主流的 Agent 规划模式，其核心思想是将推理和行动交织进行：

```python
"""
ReAct Agent 模式实现
推理（Reason）和行动（Act）交替进行
"""
import openai
from typing import List, Dict, Any

class ReActAgent:
    """基于 ReAct 范式的智能体"""

    REACT_PROMPT = """你是一个智能助手。请按照以下步骤处理用户的请求：

思考（Thought）：分析当前状态和已知信息
行动（Action）：选择一个可用的工具来执行
观察（Observation）：记录工具的执行结果
...（重复以上步骤直到得出答案）
最终答案（Final Answer）：给出完整的回答

可用工具：
{tools}

历史对话：
{history}

用户问题：{question}

请开始你的思考："""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.tools = {}
        self.history: List[Dict[str, str]] = []
        self.max_iterations = 10

    def register_tool(self, name: str, description: str, func):
        """注册工具到 Agent"""
        self.tools[name] = {
            "description": description,
            "function": func
        }

    def _format_tools(self) -> str:
        """格式化可用工具列表"""
        tool_str = ""
        for name, info in self.tools.items():
            tool_str += f"- {name}: {info['description']}\n"
        return tool_str

    def _parse_response(self, response: str) -> Dict[str, str]:
        """解析模型响应，提取 Thought、Action 和 Observation"""
        result = {}
        if "Thought:" in response:
            result["thought"] = response.split("Thought:")[1].split("\n")[0].strip()
        if "Action:" in response:
            result["action"] = response.split("Action:")[1].split("\n")[0].strip()
        if "Action Input:" in response:
            result["action_input"] = response.split("Action Input:")[1].split("\n")[0].strip()
        if "Final Answer:" in response:
            result["final_answer"] = response.split("Final Answer:")[1].strip()
        return result

    def run(self, question: str) -> str:
        """执行 Agent 主循环"""
        for i in range(self.max_iterations):
            # 构建提示词
            prompt = self.REACT_PROMPT.format(
                tools=self._format_tools(),
                history="\n".join([f"{h['role']}: {h['content']}" for h in self.history[-5:]]),
                question=question
            )

            # 调用大语言模型
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = response.choices[0].message.content

            # 解析响应
            parsed = self._parse_response(content)

            # 检查是否有最终答案
            if "final_answer" in parsed:
                self.history.append({"role": "assistant", "content": parsed["final_answer"]})
                return parsed["final_answer"]

            # 执行工具调用
            if "action" in parsed and parsed["action"] in self.tools:
                tool_name = parsed["action"]
                tool_func = self.tools[tool_name]["function"]

                # 执行工具
                action_input = parsed.get("action_input", "")
                observation = tool_func(action_input)

                # 将观察结果添加到历史
                self.history.append({
                    "role": "observation",
                    "content": f"工具 {tool_name} 返回: {observation}"
                })
            else:
                # 如果没有找到有效工具，直接返回响应
                self.history.append({"role": "assistant", "content": content})
                return content

        return "达到最大迭代次数，未能完成任务。"
```

### 2. 工具系统（Tool System）

工具系统是 Agent 与外部世界交互的桥梁。现代 Agent 框架支持多种工具类型：

```python
"""
Agent 工具系统实现
包含搜索、计算、代码执行等常用工具
"""
from typing import Callable, Any
import requests
import json
import re
import math

class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: dict):
        """装饰器：注册工具"""
        def decorator(func: Callable):
            self._tools[name] = {
                "function": func,
                "description": description,
                "parameters": parameters,
                "name": name
            }
            return func
        return decorator

    def get_tool(self, name: str) -> Dict[str, Any]:
        """获取已注册的工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return list(self._tools.values())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """转换为 OpenAI Function Calling 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for tool in self._tools.values()
        ]

# 创建全局工具注册实例
registry = ToolRegistry()

@registry.register(
    name="web_search",
    description="搜索互联网获取最新信息",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"]
    }
)
def web_search(query: str) -> str:
    """执行网络搜索（示例实现）"""
    # 实际项目中应接入搜索 API（如 Google Search API、Bing API）
    return f"搜索 '{query}' 的结果：[模拟搜索结果 - 在实际部署中接入真实搜索 API]"

@registry.register(
    name="calculator",
    description="执行数学计算",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式，如 '2 + 3 * 4'"}
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    # 仅允许安全的数学运算
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含不安全字符"
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@registry.register(
    name="file_reader",
    description="读取文件内容",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "文件路径"}
        },
        "required": ["file_path"]
    }
)
def file_reader(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content[:5000]  # 限制返回长度
    except Exception as e:
        return f"读取文件失败：{str(e)}"

@registry.register(
    name="code_executor",
    description="执行 Python 代码片段",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 代码"}
        },
        "required": ["code"]
    }
)
def code_executor(code: str) -> str:
    """在沙箱中执行 Python 代码"""
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})
        output = sys.stdout.getvalue()
        return f"执行成功\n输出：\n{output}"
    except Exception as e:
        return f"执行失败：{str(e)}"
    finally:
        sys.stdout = old_stdout
```

### 3. 记忆系统（Memory System）

记忆系统是 Agent 区别于普通大语言模型的关键特征。它分为三个层次：

| 记忆类型 | 存储内容 | 技术实现 | 生命周期 |
|---------|---------|---------|---------|
| 短期记忆 | 当前对话上下文 | 对话历史列表 | 单次会话 |
| 长期记忆 | 历史经验和知识 | 向量数据库 | 持久化 |
| 程序性记忆 | 工具使用模式和策略 | 配置文件/权重 | 持久化 |

```python
"""
Agent 记忆系统实现
包含短期记忆、长期记忆和向量检索
"""
import json
import hashlib
from typing import List, Dict, Optional
from datetime import datetime

class ShortTermMemory:
    """短期记忆：维护当前对话上下文"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """添加消息到短期记忆"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # 超出限制时移除最早的消息
        self._trim_to_limit()

    def _trim_to_limit(self):
        """裁剪历史消息以适应 token 限制"""
        total_tokens = sum(len(msg["content"]) // 4 for msg in self.conversation_history)
        while total_tokens > self.max_tokens and len(self.conversation_history) > 2:
            self.conversation_history.pop(0)
            total_tokens = sum(len(msg["content"]) // 4 for msg in self.conversation_history)

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """获取对话历史"""
        if last_n:
            return self.conversation_history[-last_n:]
        return self.conversation_history

    def clear(self):
        """清空短期记忆"""
        self.conversation_history = []


class LongTermMemory:
    """长期记忆：使用向量存储进行语义检索"""

    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path
        self.memories: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """从文件加载记忆"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.memories = []

    def _save(self):
        """保存记忆到文件"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add_memory(self, content: str, metadata: Optional[Dict] = None):
        """添加长期记忆"""
        memory = {
            "id": hashlib.md5(content.encode()).hexdigest()[:8],
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "access_count": 0
        }
        self.memories.append(memory)
        self._save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于关键词相似度检索记忆"""
        # 简化实现：使用关键词匹配
        # 实际项目应使用向量数据库（如 Chroma、Pinecone、Milvus）
        query_terms = set(query.lower().split())
        scored_memories = []

        for memory in self.memories:
            content_terms = set(memory["content"].lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored_memories.append({
                    **memory,
                    "score": overlap
                })

        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        return scored_memories[:top_k]

    def forget(self, memory_id: str):
        """删除指定记忆"""
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self._save()


class AgentMemory:
    """Agent 综合记忆系统"""

    def __init__(self, short_term_max: int = 4000):
        self.short_term = ShortTermMemory(max_tokens=short_term_max)
        self.long_term = LongTermMemory()
        self.system_prompt: str = ""

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入并维护记忆"""
        # 检索相关长期记忆
        relevant_memories = self.long_term.search(user_input, top_k=3)

        # 构建上下文
        context = {
            "system": self.system_prompt,
            "short_term": self.short_term.get_history(last_n=10),
            "long_term": [m["content"] for m in relevant_memories]
        }

        # 保存用户输入到短期记忆
        self.short_term.add_message("user", user_input)

        return context

    def store_important_info(self, content: str, metadata: Optional[Dict] = None):
        """存储重要信息到长期记忆"""
        self.long_term.add_memory(content, metadata)
```

## Agent 开发框架对比

目前市场上有多个成熟的 Agent 开发框架，各有特色：

| 框架 | 语言 | 核心特色 | 适用场景 | GitHub Stars |
|------|------|---------|---------|-------------|
| LangChain | Python/JS | 生态最完整，组件丰富 | 通用 Agent 开发 | 90k+ |
| AutoGen (Microsoft) | Python | 多 Agent 协作 | 复杂任务协同 | 40k+ |
| CrewAI | Python | 角色定义清晰，易用 | 团队化 Agent 编排 | 20k+ |
| LlamaIndex | Python | 数据索引和 RAG 优化 | 知识增强 Agent | 35k+ |
| OpenAI Agent SDK | Python | 官方支持，函数调用优化 | OpenAI 生态 | - |

### LangChain Agent 实战

```python
"""
使用 LangChain 构建完整的 Agent 应用
"""
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_openai_tools_agent,
    AgentExecutor,
    Tool
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# 定义工具
@tool
def weather_lookup(location: str) -> str:
    """查询指定城市的天气信息"""
    # 实际项目中调用天气 API
    weather_data = {
        "北京": "晴天，25°C，湿度 30%",
        "上海": "多云，28°C，湿度 65%",
        "深圳": "雷阵雨，30°C，湿度 80%"
    }
    return weather_data.get(location, f"未找到 {location} 的天气信息")

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """货币汇率转换"""
    rates = {
        ("USD", "CNY"): 7.24,
        ("CNY", "USD"): 0.138,
        ("EUR", "CNY"): 7.85,
        ("CNY", "EUR"): 0.127
    }
    rate = rates.get((from_currency, to_currency), None)
    if rate:
        return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"
    return f"不支持 {from_currency} 到 {to_currency} 的转换"

# 构建 Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [weather_lookup, currency_converter]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，擅长帮助用户查询信息和进行计算。请使用提供的工具来回答用户的问题。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 执行
# result = agent_executor.invoke({"input": "北京今天天气怎么样？"})
# print(result["output"])
```

### CrewAI 多 Agent 协作

```python
"""
使用 CrewAI 构建多 Agent 协作系统
模拟一个内容创作团队
"""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 定义角色 Agent
researcher = Agent(
    role="高级研究员",
    goal="深入研究指定主题，收集关键信息和数据",
    backstory="你是一位经验丰富的研究员，擅长快速收集和分析信息。"
              "你能从大量信息中提取关键点，并提供结构化的研究报告。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer = Agent(
    role="资深内容创作者",
    goal="基于研究资料撰写高质量的技术文章",
    backstory="你是一位资深技术写作者，擅长将复杂的技术概念"
              "用通俗易懂的语言表达出来。你的文章结构清晰、逻辑严密。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

reviewer = Agent(
    role="内容审核编辑",
    goal="审核文章质量，确保准确性和可读性",
    backstory="你是一位经验丰富的内容编辑，擅长发现文章中的问题"
              "并提出具体的改进建议。你关注事实准确性、逻辑性和语言表达。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 定义任务
research_task = Task(
    description="研究主题：AI Agent 在软件开发中的应用。"
                "请收集以下信息：1) 当前主流应用场景 2) 典型案例 3) 技术挑战",
    expected_output="一份结构化的研究报告，包含关键发现和数据支撑",
    agent=researcher
)

writing_task = Task(
    description="基于研究报告撰写一篇 2000 字的技术文章",
    expected_output="一篇结构完整、语言流畅的技术文章",
    agent=writer
)

review_task = Task(
    description="审核技术文章，指出需要改进的地方并给出修改建议",
    expected_output="包含具体修改建议的审核报告",
    agent=reviewer
)

# 组建 Crew
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # 顺序执行
    verbose=True
)

# result = crew.kickoff()
# print(result)
```

## Agent 开发最佳实践

### 1. 提示词工程

- 使用结构化提示词模板
- 明确角色定义和任务边界
- 提供具体的输出格式要求
- 包含异常处理指引

### 2. 工具设计原则

- 工具描述要清晰具体
- 参数设计要简单明确
- 错误处理要友好详细
- 工具功能要原子化

### 3. 安全考量

- 限制 Agent 的操作权限
- 对敏感操作进行二次确认
- 在沙箱环境中执行代码
- 记录所有操作日志

### 4. 性能优化

- 合理使用缓存减少重复调用
- 控制上下文长度避免超出限制
- 并行执行独立任务
- 定期清理无效记忆

## 应用场景

AI Agent 已经在多个领域展现出巨大潜力：

| 应用领域 | 具体场景 | 代表案例 |
|---------|---------|---------|
| 软件开发 | 代码生成、Bug 修复、代码审查 | Devin、GitHub Copilot Workspace |
| 数据分析 | 自动报表、数据挖掘、可视化 | ChatGPT Data Analyst |
| 客户服务 | 智能客服、工单处理 | 各企业智能客服系统 |
| 内容创作 | 文案撰写、营销内容生成 | Jasper、Copy.ai |
| 科学研究 | 文献检索、实验设计 | AI 科研助手 |
| 个人助理 | 日程管理、邮件处理、信息检索 | AutoGPT、BabyAGI |

## 总结

AI Agent 代表了人工智能从"对话式"向"行动式"的重要转变。通过规划引擎、工具系统和记忆系统的有机结合，Agent 能够自主完成复杂任务。LangChain、CrewAI 等框架降低了开发门槛，但开发者仍需关注安全性、可靠性和性能优化。随着技术的持续演进，Agent 将成为连接 AI 能力与实际业务需求的核心桥梁。

---

**下一篇**: [92. 多模态 AI 开发](./92-multimodal.md)
