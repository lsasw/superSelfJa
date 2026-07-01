---
title: MCP 网关搭建实战
icon: server
order: 2
category:
  - MCP
tag:
  - 网关
  - 限流
  - 鉴权
---

# MCP 网关搭建实战

> 任务编号：GW-01 / GW-02 / GW-03 / GW-04

## 一、架构总览

```
                    ┌──────────────┐
                    │   Nginx/Envoy │  ← 前置代理（TLS 终结、负载均衡）
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Gateway 1│ │ Gateway 2│ │ Gateway 3│  ← 无状态网关实例
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          │
                    ┌─────▼─────┐
                    │   Redis   │  ← 共享状态（限流计数、会话、配置）
                    └─────┬─────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │MCP Svr 1 │ │MCP Svr 2 │ │MCP Svr 3 │  ← 后端 MCP 服务
        └──────────┘ └──────────┘ └──────────┘
```

## 二、Nginx 前置配置

```nginx
# mcp-gateway.conf
upstream mcp_gateway {
    least_conn;
    server 127.0.0.1:3001 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:3002 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:3003 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name mcp-gateway.example.com;

    ssl_certificate     /etc/nginx/certs/mcp-gateway.crt;
    ssl_certificate_key /etc/nginx/certs/mcp-gateway.key;

    # 请求体限制
    client_max_body_size 10m;

    location / {
        proxy_pass http://mcp_gateway;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 超时配置
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
    }
}
```

## 三、三种限流策略（GW-02）

### 3.1 令牌桶（Token Bucket）

```python
import time
import redis
from typing import Optional

class TokenBucket:
    """令牌桶限流器 — 允许突发流量"""
    
    def __init__(self, redis_client: redis.Redis, 
                 key_prefix: str = "rate_limit"):
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def is_allowed(self, client_id: str, 
                   rate: int = 100,      # 每秒生成令牌数
                   capacity: int = 200,  # 桶容量（允许突发）
                   cost: int = 1) -> bool:
        """检查是否允许请求"""
        key = f"{self.key_prefix}:token_bucket:{client_id}"
        now = time.time()
        
        # Lua 脚本保证原子性
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local cost = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- 计算新增令牌
        local elapsed = now - last_refill
        local new_tokens = math.min(capacity, tokens + elapsed * rate)
        
        if new_tokens >= cost then
            new_tokens = new_tokens - cost
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 60)
            return 1
        else
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
            return 0
        end
        """
        
        result = self.redis.eval(lua_script, 1, key, 
                                  rate, capacity, cost, now)
        return result == 1
```

### 3.2 滑动窗口（Sliding Window）

```python
class SlidingWindow:
    """滑动窗口限流器 — 精确控制时间窗口内的请求数"""
    
    def is_allowed(self, client_id: str,
                   limit: int = 100,      # 窗口内最大请求数
                   window: int = 60) -> bool:  # 窗口大小（秒）
        key = f"{self.key_prefix}:sliding:{client_id}"
        now = time.time()
        window_start = now - window
        
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window = tonumber(ARGV[4])
        
        -- 删除窗口外的记录
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- 统计当前窗口内的请求数
        local current = redis.call('ZCARD', key)
        
        if current < limit then
            redis.call('ZADD', key, now, now .. ':' .. math.random())
            redis.call('EXPIRE', key, window * 2)
            return 1
        else
            return 0
        end
        """
        
        result = self.redis.eval(lua_script, 1, key,
                                  now, window_start, limit, window)
        return result == 1
```

### 3.3 漏桶（Leaky Bucket）

```python
class LeakyBucket:
    """漏桶限流器 — 平滑流量，强制恒定速率"""
    
    def is_allowed(self, client_id: str,
                   rate: int = 10,         # 每秒漏出速率
                   capacity: int = 100) -> bool:  # 桶容量
        key = f"{self.key_prefix}:leaky:{client_id}"
        now = time.time()
        
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'water', 'last_leak')
        local water = tonumber(bucket[1]) or 0
        local last_leak = tonumber(bucket[2]) or now
        
        -- 漏水：按速率减少水量
        local elapsed = now - last_leak
        water = math.max(0, water - elapsed * rate)
        
        if water + 1 <= capacity then
            water = water + 1
            redis.call('HMSET', key, 'water', water, 'last_leak', now)
            redis.call('EXPIRE', key, 60)
            return 1
        else
            return 0
        end
        """
        
        result = self.redis.eval(lua_script, 1, key, rate, capacity, now)
        return result == 1
```

### 限流策略选型

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| API 网关全局限流 | 滑动窗口 | 精确控制时间窗口内请求量 |
| 需要允许短时突发 | 令牌桶 | 桶容量可吸收突发流量 |
| 下游处理能力恒定 | 漏桶 | 强制平滑，保护下游 |
| 多维度限流 | 组合使用 | IP + API Key + Endpoint 三级 |

## 四、双通道鉴权（GW-03）

```python
import jwt
import hashlib
import hmac
from functools import wraps
from flask import request, jsonify

# === 通道 1: JWT（Web 端用户）===
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"

def create_jwt(user_id: str, roles: list[str], 
               expire_minutes: int = 60) -> str:
    """签发 JWT"""
    import datetime
    payload = {
        "user_id": user_id,
        "roles": roles,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=expire_minutes),
        "iat": datetime.datetime.utcnow(),
        "type": "access"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt(token: str) -> dict | None:
    """验证 JWT"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# === 通道 2: API Key（服务端调用）===
API_KEYS = {
    "ak-prod-001": {"name": "CI/CD Pipeline", "roles": ["deploy"]},
    "ak-prod-002": {"name": "Monitor Service", "roles": ["read"]},
}

def verify_api_key(api_key: str) -> dict | None:
    """验证 API Key"""
    return API_KEYS.get(api_key)

# === 统一鉴权装饰器 ===
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # 1. 优先检查 JWT（Web 端）
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_jwt(token)
            if payload:
                request.user = payload
                return f(*args, **kwargs)
        
        # 2. 检查 API Key（服务端）
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            key_info = verify_api_key(api_key)
            if key_info:
                request.user = key_info
                return f(*args, **kwargs)
        
        return jsonify({"error": "Unauthorized"}), 401
    
    return decorated

# === 使用 ===
@app.route("/api/tools", methods=["GET"])
@require_auth
def list_tools():
    user = request.user
    # 根据用户角色过滤可用的 tools
    return jsonify({"tools": get_tools_for_user(user)})
```

## 五、可观测性（GW-04）

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# === Prometheus 指标定义 ===

# QPS 计数器
request_count = Counter(
    'mcp_gateway_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

# 延迟直方图
request_latency = Histogram(
    'mcp_gateway_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# 限流触发计数器
rate_limit_hits = Counter(
    'mcp_gateway_rate_limit_hits_total',
    'Rate limit hits',
    ['client_id', 'limit_type']
)

# 当前活跃连接
active_connections = Gauge(
    'mcp_gateway_active_connections',
    'Active connections'
)

# === 中间件 ===
def metrics_middleware(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        active_connections.inc()
        
        try:
            result = f(*args, **kwargs)
            status = "200"
            return result
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.time() - start
            request_count.labels(
                method=request.method,
                endpoint=request.path,
                status=status
            ).inc()
            request_latency.labels(
                method=request.method,
                endpoint=request.path
            ).observe(duration)
            active_connections.dec()
    
    return decorated

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}
```

### Grafana 面板配置建议

| 面板 | 指标 | 阈值 |
|------|------|------|
| QPS 趋势 | `rate(mcp_gateway_requests_total[1m])` | >1000 告警 |
| P99 延迟 | `histogram_quantile(0.99, ...)` | >1s 告警 |
| 错误率 | `rate(...{status="500"}[1m])` / total | >1% 告警 |
| 限流触发 | `rate(mcp_gateway_rate_limit_hits_total[5m])` | >10/min 关注 |
| 活跃连接数 | `mcp_gateway_active_connections` | >1000 告警 |
