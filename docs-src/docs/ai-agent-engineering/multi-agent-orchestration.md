---
title: Multi-Agent 编排模式
icon: sitemap
order: 3
category:
  - AI Agent
tag:
  - Multi-Agent
  - LangGraph
  - 编排
---

# Multi-Agent 编排模式

> 任务编号：MA-01 / MA-02 / MA-03 / MA-04 / MA-05

## 一、三种编排模式全景

```
┌──────────────────────────────────────────────────────┐
│                   Multi-Agent 编排                    │
├───────────────┬──────────────────┬───────────────────┤
│  Sequential   │    Parallel      │     Router        │
│  (顺序执行)    │   (并行执行)      │    (路由分发)      │
├───────────────┼──────────────────┼───────────────────┤
│  A → B → C   │   A ─┬─ B        │   ┌─→ Agent A    │
│               │      └─ C        │  Input─→ Agent B  │
│               │                  │   └─→ Agent C    │
├───────────────┼──────────────────┼───────────────────┤
│ 适用：流水线   │ 适用：多源搜索    │ 适用：意图分类      │
│ 需求分析→编码  │ 同时查多个数据源  │ 客服/技术支持分流   │
│ →测试→审查    │ 然后汇总结果     │                   │
└───────────────┴──────────────────┴───────────────────┘
```

## 二、Sequential 编排（MA-01）

> 3 个 Agent 顺序执行：需求分析 → 编码 → 审查

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    requirement: str
    analysis: str
    code: str
    review: str
    messages: Annotated[list, operator.add]

# === 定义三个 Agent 节点 ===

def analyst_agent(state: AgentState) -> AgentState:
    """需求分析 Agent：把用户需求拆解为技术方案"""
    requirement = state["requirement"]
    # 模拟 LLM 分析
    analysis = f"""
    需求分析结果：
    - 输入：{requirement}
    - 技术栈：Python + LangGraph
    - 模块划分：API 层、Service 层、Model 层
    - 关键难点：并发安全、错误处理
    """
    return {"analysis": analysis}

def coder_agent(state: AgentState) -> AgentState:
    """编码 Agent：根据分析结果生成代码"""
    analysis = state["analysis"]
    code = f'''
# 根据分析生成的代码
# {analysis.split(chr(10))[1]}

from typing import Optional

class OrderService:
    def __init__(self):
        self.orders = {{}}
    
    def create_order(self, user_id: str, items: list) -> dict:
        """创建订单"""
        order_id = f"ORD-{{len(self.orders) + 1}}"
        order = {{
            "id": order_id,
            "user_id": user_id,
            "items": items,
            "status": "pending"
        }}
        self.orders[order_id] = order
        return order
    
    def get_order(self, order_id: str) -> Optional[dict]:
        """查询订单"""
        return self.orders.get(order_id)
'''
    return {"code": code}

def reviewer_agent(state: AgentState) -> AgentState:
    """代码审查 Agent：检查代码质量"""
    code = state["code"]
    review = f"""
    代码审查结果：
    - 命名规范：✅ 符合 PEP 8
    - 错误处理：⚠️ create_order 未校验 items 非空
    - 类型注解：✅ 使用了 Optional
    - 并发安全：⚠️ 字典操作在非并发场景可接受
    - 测试覆盖：❌ 缺少单元测试
    
    总评：基本可用，建议补充输入校验和单元测试。
    """
    return {"review": review}

# === 构建 LangGraph ===

graph = StateGraph(AgentState)

graph.add_node("analyst", analyst_agent)
graph.add_node("coder", coder_agent)
graph.add_node("reviewer", reviewer_agent)

# 顺序连接
graph.set_entry_point("analyst")
graph.add_edge("analyst", "coder")
graph.add_edge("coder", "reviewer")
graph.add_edge("reviewer", END)

app = graph.compile()

# === 执行 ===
result = app.invoke({
    "requirement": "实现一个订单管理系统，支持创建和查询订单"
})
print(result["review"])
```

## 三、Parallel 编排（MA-02）

> 2 个 Agent 并行搜索不同数据源，汇总结果。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor
import json

class SearchState(TypedDict):
    query: str
    results_db: str
    results_api: str
    merged: str

def search_database(state: SearchState) -> SearchState:
    """搜索内部数据库"""
    query = state["query"]
    # 模拟数据库查询
    results = [
        {"source": "DB", "title": f"{query} 相关文档1", "score": 0.92},
        {"source": "DB", "title": f"{query} 相关文档2", "score": 0.85},
    ]
    return {"results_db": json.dumps(results, ensure_ascii=False)}

def search_external_api(state: SearchState) -> SearchState:
    """搜索外部 API（如搜索引擎）"""
    query = state["query"]
    # 模拟外部 API 调用
    results = [
        {"source": "API", "title": f"外部：{query} 最新动态", "score": 0.88},
        {"source": "API", "title": f"外部：{query} 行业报告", "score": 0.79},
    ]
    return {"results_api": json.dumps(results, ensure_ascii=False)}

def merge_results(state: SearchState) -> SearchState:
    """汇总结果"""
    db_results = json.loads(state["results_db"])
    api_results = json.loads(state["results_api"])
    
    all_results = db_results + api_results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    merged = "搜索结果汇总：\n" + "\n".join(
        f"  [{r['source']}] {r['title']} (score: {r['score']})"
        for r in all_results
    )
    return {"merged": merged}

# === 构建 LangGraph（并行模式）===

graph = StateGraph(SearchState)

graph.add_node("search_db", search_database)
graph.add_node("search_api", search_external_api)
graph.add_node("merge", merge_results)

# 并行分支：从入口同时发到 search_db 和 search_api
graph.set_entry_point("search_db")  # 两者都从入口开始
graph.set_entry_point("search_api")

# 两个分支都完成后再汇总
graph.add_edge("search_db", "merge")
graph.add_edge("search_api", "merge")
graph.add_edge("merge", END)

app = graph.compile()
```

## 四、Router 编排（MA-03）

> 根据用户意图路由到不同 Agent。

```python
from langgraph.graph import StateGraph, END
from typing import Literal

class RouterState(TypedDict):
    user_input: str
    intent: str
    response: str

def classify_intent(state: RouterState) -> RouterState:
    """意图分类"""
    user_input = state["user_input"].lower()
    
    if any(word in user_input for word in ["bug", "报错", "异常", "错误"]):
        intent = "tech_support"
    elif any(word in user_input for word in ["怎么", "如何", "教程", "使用"]):
        intent = "faq"
    else:
        intent = "general"
    
    return {"intent": intent}

def tech_support_agent(state: RouterState) -> RouterState:
    return {"response": f"[技术支持] 收到问题：{state['user_input']}，正在排查..."}

def faq_agent(state: RouterState) -> RouterState:
    return {"response": f"[FAQ] 关于「{state['user_input']}」的解答：请参考文档第3章..."}

def general_agent(state: RouterState) -> RouterState:
    return {"response": f"[通用助手] 收到：{state['user_input']}，有什么可以帮助你的？"}

# 路由函数
def route_by_intent(state: RouterState) -> Literal["tech_support", "faq", "general"]:
    return state["intent"]

# === 构建 Graph ===

graph = StateGraph(RouterState)

graph.add_node("classifier", classify_intent)
graph.add_node("tech_support", tech_support_agent)
graph.add_node("faq", faq_agent)
graph.add_node("general", general_agent)

graph.set_entry_point("classifier")

# 条件路由
graph.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "tech_support": "tech_support",
        "faq": "faq",
        "general": "general"
    }
)

graph.add_edge("tech_support", END)
graph.add_edge("faq", END)
graph.add_edge("general", END)

app = graph.compile()
```

## 五、Agent 间文件传输协议（MA-04）

定义统一的 Agent 间消息格式：

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone
import json

class MessageType(Enum):
    TASK = "task"           # 下发任务
    RESULT = "result"       # 返回结果
    ERROR = "error"         # 错误通知
    HEARTBEAT = "heartbeat" # 心跳

@dataclass
class AgentMessage:
    """Agent 间统一消息格式"""
    msg_id: str
    msg_type: MessageType
    sender: str              # 发送方 Agent ID
    receiver: str            # 接收方 Agent ID
    task_id: str             # 关联任务 ID
    payload: Any             # 载荷（JSON/Excel/文本）
    content_type: str = "application/json"  # MIME 类型
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retry_count: int = 0
    max_retries: int = 3
    
    def to_json(self) -> str:
        return json.dumps({
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "task_id": self.task_id,
            "payload": self.payload,
            "content_type": self.content_type,
            "timestamp": self.timestamp
        }, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, data: str) -> "AgentMessage":
        d = json.loads(data)
        return cls(
            msg_id=d["msg_id"],
            msg_type=MessageType(d["msg_type"]),
            sender=d["sender"],
            receiver=d["receiver"],
            task_id=d["task_id"],
            payload=d["payload"],
            content_type=d.get("content_type", "application/json"),
        )
```

## 六、失败处理策略（MA-05）

```python
from enum import Enum
from typing import Callable
import time

class FailureStrategy(Enum):
    RETRY = "retry"         # 重试
    FALLBACK = "fallback"   # 降级
    COMPENSATE = "compensate"  # 补偿/回滚

class AgentOrchestrator:
    """Agent 编排器 — 内置失败处理"""
    
    def __init__(self):
        self.completed_steps = []  # 已完成步骤（用于回滚）
    
    def run_with_retry(
        self, 
        func: Callable, 
        state: dict, 
        max_retries: int = 3,
        backoff: float = 2.0
    ):
        """带退避重试的执行"""
        for attempt in range(max_retries):
            try:
                result = func(state)
                self.completed_steps.append(func.__name__)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = backoff ** attempt
                print(f"重试 {attempt + 1}/{max_retries}，等待 {wait}s...")
                time.sleep(wait)
    
    def run_with_fallback(
        self,
        primary: Callable,
        fallback: Callable,
        state: dict
    ):
        """主方案失败时降级到备选方案"""
        try:
            return primary(state)
        except Exception:
            print(f"主方案失败，降级到备选方案")
            return fallback(state)
    
    def compensate(self):
        """回滚已完成步骤"""
        while self.completed_steps:
            step = self.completed_steps.pop()
            print(f"回滚步骤: {step}")
            # 调用对应的补偿操作...
```

### 失败处理决策矩阵

| 场景 | 策略 | 示例 |
|------|------|------|
| 网络超时（偶发） | 重试 3 次 + 指数退避 | API 调用超时 |
| 部分 Agent 成功 | 降级：跳过失败步骤，继续执行 | 非关键搜索源不可用 |
| 全部流程失败 | 补偿：回滚已完成步骤 | 支付流程中某步报错 |
| 结果质量不合格 | 重新分配 Agent | 代码审查不通过，返回编码 Agent |
| 超过最大重试 | 告警 + 人工介入 | 死信队列 |
