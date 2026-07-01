---
title: Agent Memory 机制
icon: memory
order: 5
category:
  - AI Agent
tag:
  - Memory
  - PG Vector
  - Redis
---

# Agent Memory 机制

> 任务编号：MM-01 / MM-02 / MM-03

## 一、Memory 三层架构

```
┌──────────────────────────────────────────────────────┐
│                   Agent Memory                       │
├───────────────┬──────────────────┬───────────────────┤
│  短期记忆      │    长期记忆       │    工作记忆        │
│  Short-term   │   Long-term      │   Working Memory  │
├───────────────┼──────────────────┼───────────────────┤
│ 存储：Context  │ 存储：向量数据库   │ 存储：Redis/内存KV  │
│ Window        │ (PG Vector)      │                   │
├───────────────┼──────────────────┼───────────────────┤
│ 生命周期：     │ 生命周期：        │ 生命周期：         │
│ 当前会话       │ 跨会话持久        │ 当前任务           │
├───────────────┼──────────────────┼───────────────────┤
│ 用途：         │ 用途：            │ 用途：             │
│ 记住刚说的内容  │ 记住你是谁/历史   │ 记住任务中间状态    │
├───────────────┼──────────────────┼───────────────────┤
│ 典型容量：     │ 典型容量：         │ 典型容量：         │
│ 4K-128K tokens│ 百万级向量        │ 百条 KV           │
└───────────────┴──────────────────┴───────────────────┘
```

## 二、短期记忆（MM-01）

> 基于会话级 Context Window，在对话过程中保持上下文

```python
from openai import OpenAI

class ShortTermMemory:
    """短期记忆 — 基于对话历史的上下文管理"""
    
    def __init__(self, max_tokens: int = 8000):
        self.client = OpenAI()
        self.messages = []
        self.max_tokens = max_tokens
    
    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_if_needed()
    
    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """当上下文超过限制时，裁剪最早的对话"""
        estimated_tokens = sum(len(m["content"]) // 4 for m in self.messages)
        
        while estimated_tokens > self.max_tokens and len(self.messages) > 2:
            removed = self.messages.pop(1)  # 保留 system message
            estimated_tokens -= len(removed["content"]) // 4
            print(f"[MEMORY] 裁剪: {removed['content'][:50]}...")
    
    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.add_user_message(user_input)
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.messages
        )
        
        reply = response.choices[0].message.content
        self.add_assistant_message(reply)
        return reply

# === 使用 ===
memory = ShortTermMemory(max_tokens=4000)

memory.chat("我叫张三，今年 30 岁")
memory.chat("我喜欢吃川菜")
# Agent 记得上下文
print(memory.chat("我叫什么名字？我喜欢吃什么？"))
# → 你叫张三，你喜欢吃川菜
```

### 短期记忆的局限

| 问题 | 说明 |
|------|------|
| 上下文窗口有限 | 128K tokens 也可能不够长对话 |
| 注意力衰减 | LLM 对中间部分的记忆不如首尾 |
| 跨会话丢失 | 新开对话 = 全新记忆 |

## 三、长期记忆（MM-02）

> 对话内容写入向量数据库（PG Vector），下次对话时检索相关历史

### 3.1 PG Vector 表结构

```sql
-- 创建 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 长期记忆表
CREATE TABLE agent_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64),
    content TEXT NOT NULL,
    content_type VARCHAR(32) DEFAULT 'conversation',
    embedding VECTOR(1536),       -- OpenAI text-embedding-3-small
    metadata JSONB DEFAULT '{}',
    importance FLOAT DEFAULT 0.5, -- 记忆重要性（0-1）
    created_at TIMESTAMP DEFAULT NOW()
);

-- 向量索引
CREATE INDEX idx_memory_embedding 
    ON agent_memory 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### 3.2 完整实现

```python
import openai
import psycopg2
import json
from datetime import datetime

class LongTermMemory:
    """长期记忆 — 基于 PG Vector 的持久化记忆"""
    
    def __init__(self, db_url: str, user_id: str):
        self.conn = psycopg2.connect(db_url)
        self.user_id = user_id
        self.client = openai.OpenAI()
    
    def _embed(self, text: str) -> list[float]:
        """文本转向量"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def remember(self, content: str, importance: float = 0.5, 
                 metadata: dict = None):
        """存储记忆"""
        embedding = self._embed(content)
        
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO agent_memory 
                (user_id, content, embedding, importance, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            self.user_id,
            content,
            embedding,
            importance,
            json.dumps(metadata or {})
        ))
        self.conn.commit()
        print(f"[MEMORY] 已存储: {content[:50]}...")
    
    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """检索相关记忆"""
        query_embedding = self._embed(query)
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT content, metadata, importance,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM agent_memory
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, self.user_id, query_embedding, top_k))
        
        results = cur.fetchall()
        memories = []
        for content, meta, importance, similarity in results:
            memories.append({
                "content": content,
                "metadata": meta,
                "importance": importance,
                "similarity": float(similarity)
            })
        
        # 按重要性 × 相似度 重新排序
        memories.sort(key=lambda m: m["importance"] * m["similarity"], 
                      reverse=True)
        return memories
    
    def forget_old(self, days: int = 90):
        """遗忘旧记忆（低重要性的）"""
        cur = self.conn.cursor()
        cur.execute("""
            DELETE FROM agent_memory
            WHERE user_id = %s 
              AND importance < 0.3 
              AND created_at < NOW() - INTERVAL '%s days'
        """, (self.user_id, days))
        self.conn.commit()
        print(f"[MEMORY] 清理了 {cur.rowcount} 条旧记忆")

# === 使用 ===
memory = LongTermMemory(db_url="postgresql://...", user_id="user_123")

# 存储记忆
memory.remember("用户张三喜欢吃川菜，尤其是麻婆豆腐", importance=0.8,
                metadata={"category": "food_preference"})
memory.remember("张三的工作是做 AI Agent 开发", importance=0.9,
                metadata={"category": "career"})
memory.remember("上次讨论了 MCP 协议的网关设计", importance=0.7,
                metadata={"category": "tech_discussion"})

# 下次对话时检索
context = memory.recall("用户喜欢吃什么？对什么技术感兴趣？")
for mem in context:
    print(f"[相似度 {mem['similarity']:.2f}] {mem['content']}")
```

### 3.3 记忆注入策略

```python
def build_context(query: str, ltm: LongTermMemory) -> list:
    """将长期记忆注入到对话上下文"""
    memories = ltm.recall(query, top_k=5)
    
    memory_context = "以下是与用户相关的历史记忆：\n"
    for i, mem in enumerate(memories, 1):
        memory_context += f"{i}. {mem['content']}\n"
    
    return [
        {"role": "system", "content": memory_context},
        {"role": "user", "content": query}
    ]
```

## 四、工作记忆（MM-03）

> 任务过程中的中间状态存入 Redis/内存 KV，任务结束后清理

```python
import redis
import json
from typing import Optional
from datetime import timedelta

class WorkingMemory:
    """工作记忆 — 单任务生命周期内的状态存储"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.task_prefix = "task:"
    
    def set(self, task_id: str, key: str, value: any, ttl: int = 3600):
        """设置任务中间状态"""
        redis_key = f"{self.task_prefix}{task_id}:{key}"
        self.redis.setex(
            redis_key,
            timedelta(seconds=ttl),
            json.dumps(value, ensure_ascii=False)
        )
    
    def get(self, task_id: str, key: str) -> Optional[any]:
        """读取任务中间状态"""
        redis_key = f"{self.task_prefix}{task_id}:{key}"
        data = self.redis.get(redis_key)
        if data:
            return json.loads(data)
        return None
    
    def get_all(self, task_id: str) -> dict:
        """获取任务所有中间状态"""
        pattern = f"{self.task_prefix}{task_id}:*"
        keys = self.redis.keys(pattern)
        result = {}
        for key in keys:
            short_key = key.decode().split(":", 2)[-1]
            result[short_key] = json.loads(self.redis.get(key))
        return result
    
    def cleanup(self, task_id: str):
        """任务完成后清理"""
        pattern = f"{self.task_prefix}{task_id}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
            print(f"[WORKING MEMORY] 清理任务 {task_id} 的 {len(keys)} 个状态")

# === 使用 ===
wm = WorkingMemory()

task_id = "task_20240701_001"

# 任务开始
wm.set(task_id, "step", 1)
wm.set(task_id, "collected_data", {"sources": 3, "items": 150})
wm.set(task_id, "current_agent", "researcher", ttl=300)

# 步骤 2
wm.set(task_id, "step", 2)
wm.set(task_id, "analysis_result", {"sentiment": "positive", "score": 0.87})

# 查询进度
progress = wm.get_all(task_id)
print(f"当前步骤: {progress['step']}, 数据量: {progress['collected_data']['items']}")

# 任务完成，清理
wm.cleanup(task_id)
```

## 五、三种 Memory 的协作

```python
class AgentWithMemory:
    """集成三种记忆的 Agent"""
    
    def __init__(self, user_id: str):
        self.stm = ShortTermMemory()               # 短期
        self.ltm = LongTermMemory(db_url="...",     # 长期
                                  user_id=user_id)
        self.wm = WorkingMemory()                   # 工作
    
    def handle_message(self, user_input: str, task_id: str = None):
        # 1. 注入长期记忆到上下文
        relevant_memories = self.ltm.recall(user_input, top_k=3)
        for mem in relevant_memories:
            self.stm.add_system_message(
                f"[历史记忆] {mem['content']}"
            )
        
        # 2. 如果有进行中的任务，恢复工作记忆
        if task_id:
            state = self.wm.get_all(task_id)
            if state:
                self.stm.add_system_message(
                    f"[任务状态] 当前步骤 {state.get('step')}, "
                    f"已完成: {json.dumps(state)}"
                )
        
        # 3. 处理消息
        reply = self.stm.chat(user_input)
        
        # 4. 重要信息存入长期记忆
        if self._is_important(reply):
            self.ltm.remember(
                f"用户说了: {user_input}，Agent 回复: {reply[:100]}",
                importance=0.7
            )
        
        return reply
    
    def _is_important(self, text: str) -> bool:
        """判断是否值得存入长期记忆"""
        keywords = ["偏好", "喜欢", "工作", "项目", "技术栈", "MCP", "Agent"]
        return any(kw in text for kw in keywords)
```

## 📚 延伸阅读

- [Multi-Agent 编排模式](./multi-agent-orchestration.md)
- [RAG 全链路](../rag-vector-db/rag-end-to-end.md)
- [向量数据库](../rag-vector-db/vector-database.md)
