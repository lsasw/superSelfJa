---
title: 系统设计答题框架与题库
icon: project-diagram
order: 1
category:
  - 系统设计
tag:
  - 系统设计
  - 面试
---

# 系统设计答题框架与题库

> 任务编号：SD-01 ~ SD-13

## 一、七步答题框架（SD-01）

```
① 需求澄清（功能/非功能）
② 量级估算（QPS/存储/带宽）
③ 核心实体与数据模型
④ 接口设计（API 契约）
⑤ 架构图（组件+数据流）
⑥ 关键问题深挖（一致性/可用性/扩展性）
⑦ 瓶颈与优化
```

## 二、12 周系统设计题库

### Week 1: 短链接系统（SD-02）

```
需求：将长 URL 转为短 URL，访问短链接时 302 跳转
QPS 估算：写入 100/s，读取 10000/s（读写比 1:100）
存储：100 亿条 → 每条 200B → 2TB

关键决策：
- 哈希算法：MurmurHash（性能）vs 分布式 ID（无碰撞）
- 跳转方式：302（临时，方便统计）vs 301（永久，减少请求）
- 过期策略：惰性删除 + 定期清理
```

### Week 2: 限流器（SD-03）

```
四种算法对比：
┌──────────┬──────────┬──────────┬──────────┐
│ 固定窗口  │ 滑动窗口  │ 令牌桶    │ 漏桶      │
├──────────┼──────────┼──────────┼──────────┤
│ 简单     │ 精确     │ 允许突发  │ 强制平滑  │
│ 边界突刺  │ Redis有序 │ 自适应   │ 恒定速率  │
└──────────┴──────────┴──────────┴──────────┘

分布式限流：Redis 集中计数 / 本地 + 远程两级
```

### Week 3: 分布式 ID 生成器（SD-04）

```
方案对比：
- UUID：简单但无序、不利于索引
- 雪花算法：趋势递增、依赖时钟同步       ← 推荐
- 号段模式：批量分配、减少 DB 压力（美团 Leaf）
- Redis：简单但需持久化
```

### Week 4: 消息推送系统（SD-05）

```
架构：Client → WebSocket → Push Server → Message Queue → Business
关键：在线状态管理（Redis）、消息可靠性（ACK+重试）、离线消息存储
```

### Week 5: AI 智能体平台（SD-06）

```
核心模块：
- Agent Registry：Agent 注册与发现
- Tool Manager：MCP Tool 注册与管理
- Memory Service：短期/长期/工作记忆
- Orchestrator：Multi-Agent 编排引擎
- Observability：链路追踪、Token 消耗统计
```

### Week 6: 实时排行榜（SD-07）

```
Redis Sorted Set → 实时排名
分段排名：日榜/周榜/月榜 → 历史榜单存储 MySQL
```
```java
// 更新分数
redis.zadd("rank:daily", score, userId);
// Top 100
redis.zrevrange("rank:daily", 0, 99);
```

### Week 7: API 网关（SD-08）

```
功能矩阵：路由 → 限流 → 鉴权 → 协议转换 → 日志/监控
核心选型：Spring Cloud Gateway / Kong / APISIX
```

### Week 8: BI 数据分析智能体（SD-09）

```
NL2SQL Pipeline：
用户问题 → LLM 解析意图 → 生成 SQL → 执行 → 可视化自动生成
关键难点：复杂 JOIN 的 SQL 生成准确性、数据权限隔离
```

### Week 9: 微服务上下文传递（SD-10）

```
链路：网关(JWT解析) → ThreadLocal(set) → Feign拦截器(add header) → 下游(恢复)
```
```java
// 核心方案
public class UserContext {
    private static final ThreadLocal<UserInfo> CONTEXT = new ThreadLocal<>();
    
    public static void set(UserInfo user) { CONTEXT.set(user); }
    public static UserInfo get() { return CONTEXT.get(); }
    public static void clear() { CONTEXT.remove(); }
}

// Feign 拦截器自动透传
public class UserContextInterceptor implements RequestInterceptor {
    @Override
    public void apply(RequestTemplate template) {
        UserInfo user = UserContext.get();
        if (user != null) {
            template.header("X-User-Id", user.getId());
        }
    }
}
```

### Week 10: 文件存储系统（SD-10）

```
分块上传 → 合并 → 秒传（MD5 去重）→ CDN 分发 → 缩略图
存储：MinIO / 阿里云 OSS / AWS S3
```

### Week 11: 分布式任务调度（SD-12）

```
方案：XXL-JOB / Elastic-Job / PowerJob
核心：分片策略 / 故障转移 / 任务依赖 DAG
```

### Week 12: MCP 网关高可用设计（SD-13）

```
架构：
┌─────────────────────────────────────┐
│         Nginx (LB + TLS)            │
└─────────────┬───────────────────────┘
              │
  ┌───────────┼───────────┐
  ▼           ▼           ▼
┌──────┐  ┌──────┐  ┌──────┐
│ GW-1 │  │ GW-2 │  │ GW-3 │  ← 无状态
└──┬───┘  └──┬───┘  └──┬───┘
   │         │         │
   └─────────┼─────────┘
             ▼
      ┌──────────┐
      │  Redis   │  ← 限流计数、会话管理
      └──────────┘
             │
   ┌─────────┼─────────┐
   ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐
│MCP-1 │ │MCP-2 │ │MCP-3 │  ← MCP 服务集群
└──────┘ └──────┘ └──────┘

高可用保障：无状态 + Redis 共享 + 限流 + 熔断 + 灰度发布
```

## 三、每道题的产出规范

每道题必须包含：
1. **架构图**（draw.io / PlantUML → SVG）
2. **关键决策说明**（为什么选 A 不选 B）
3. **量级估算过程**（QPS / 存储 / 带宽）

团队内部交叉评审，每周五下午预留 1 小时。
