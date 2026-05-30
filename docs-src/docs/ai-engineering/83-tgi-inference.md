---
title: TGI 推理服务框架
icon: file-alt
order: 8
---

# 83. TGI 推理服务框架

在上一篇文档中，我们学习了 vLLM 这一高效的 LLM 推理框架。HuggingFace 作为 NLP 领域最重要的开源组织，也推出了自己的推理服务框架——TGI（Text Generation Inference）。TGI 与 vLLM 在许多设计理念上有相似之处，但也有一些独特的功能和优势。本文将详细介绍 TGI 的架构特点、部署方法和工程实践，并与 vLLM 进行对比分析。

## TGI 概述

### 什么是 TGI

TGI（Text Generation Inference）是由 HuggingFace 官方开发的文本生成推理框架，专为生产环境中的大语言模型服务而设计。它是 HuggingFace 推理生态的核心组件，被广泛应用于 HuggingFace Inference API 和 HuggingChat 等产品。

### TGI 核心特性

| 特性 | 说明 | 优势 |
|------|------|------|
| Tensor Parallelism | 多 GPU 张量并行 | 支持超大模型部署 |
| Token Streaming | Token 级别流式输出 | 实时响应体验 |
| Continuous Batching | 连续批处理 | 高吞吐量 |
| Speculative Decoding | 推测解码 | 加速生成过程 |
| Watermarking | 文本水印 | 内容溯源 |
| 安全过滤 | 输出内容过滤 | 合规性保障 |
| Prometheus 指标 | 内置监控指标 | 运维友好 |
| 量化支持 | GPTQ、AWQ、bitsandbytes | 降低显存需求 |

### TGI vs vLLM 对比

| 维度 | TGI | vLLM |
|------|-----|------|
| 维护方 | HuggingFace | 加州大学伯克利分校 + 社区 |
| 框架依赖 | PyTorch + Candle (Rust) | PyTorch |
| KV Cache 管理 | 自定义分配器 | PagedAttention |
| 量化支持 | GPTQ, AWQ, bitsandbytes | GPTQ, AWQ, bitsandbytes |
| LoRA 支持 | 否 | 是 |
| 推测解码 | 是 | 是 |
| 多模态支持 | 部分支持 | 部分支持 |
| API 兼容性 | OpenAI 兼容 | OpenAI 兼容 |
| 社区生态 | HuggingFace 集成 | 独立生态 |

## TGI 安装与部署

### Docker 部署（推荐）

```bash
# 启动 TGI 服务
docker run --gpus all \
    -p 8080:80 \
    -v /path/to/data:/data \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 1 \
    --max-input-length 2048 \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 4096 \
    --max-batch-total-tokens 8192

# 测试服务
curl http://localhost:8080/generate \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "请介绍一下 Python 编程语言",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }'
```

### HuggingFace CLI 部署

```bash
# 安装 TGI CLI
pip install text-generation

# 启动本地服务
text-generation-launcher \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 1 \
    --port 8080

# 使用 Python 客户端
```

```python
from text_generation import Client

# 创建客户端
client = Client("http://127.0.0.1:8080")

# 文本生成
response = client.generate(
    prompt="解释一下什么是深度学习",
    max_new_tokens=200,
    temperature=0.7,
)
print(response.generated_text)

# 流式生成
for response in client.generate_stream(
    prompt="用 Python 写一个冒泡排序",
    max_new_tokens=100,
    temperature=0.7,
):
    print(response.token.text, end="", flush=True)
```

### 原生安装

```bash
# 安装 Rust 工具链
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 克隆 TGI 仓库
git clone https://github.com/huggingface/text-generation-inference.git
cd text-generation-inference

# 安装
make install

# 编译 release 版本
cargo build --release

# 启动服务
./target/release/text-generation-launcher \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 1
```

## TGI 配置详解

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model-id | - | HuggingFace Hub 模型 ID |
| --revision | main | 模型版本 |
| --sharded | false | 是否启用分片 |
| --num-shard | 1 | 分片数量 |
| --quantize | None | 量化方式 (gptq, awq, bitsandbytes) |
| --max-input-length | 1024 | 最大输入长度 |
| --max-total-tokens | 2048 | 最大总 token 数 |
| --max-batch-prefill-tokens | 4096 | 最大预填充 token 数 |
| --max-batch-total-tokens | 8192 | 最大批处理 token 数 |
| --max-waiting-tokens | 20 | 最大等待 token 数 |
| --cuda-memory-fraction | 1.0 | CUDA 显存使用比例 |

### 量化部署

```bash
# GPTQ 量化模型部署
docker run --gpus all \
    -p 8080:80 \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id TheBloke/Llama-2-7B-GPTQ \
    --quantize gptq \
    --max-input-length 2048 \
    --max-total-tokens 4096

# AWQ 量化模型部署
docker run --gpus all \
    -p 8080:80 \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id TheBloke/Llama-2-7B-AWQ \
    --quantize awq

# bitsandbytes 量化部署
docker run --gpus all \
    -p 8080:80 \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --quantize bitsandbytes-nf4 \
    --max-input-length 2048 \
    --max-total-tokens 4096
```

### 多 GPU 部署

```bash
# 使用 4 个 GPU 部署 70B 模型
docker run --gpus all \
    -p 8080:80 \
    --shm-size 2g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-70b-hf \
    --num-shard 4 \
    --quantize gptq \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

## TGI API 详解

### REST API

```python
import requests
import json

# 基础生成
def generate_text(prompt, **kwargs):
    """调用 TGI 生成文本"""
    url = "http://localhost:8080/generate"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", None),
            "do_sample": kwargs.get("do_sample", True),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "return_full_text": kwargs.get("return_full_text", False),
            "stop": kwargs.get("stop", []),
            "seed": kwargs.get("seed", None),
            "details": kwargs.get("details", False),
            "decoder_input_details": kwargs.get("decoder_input_details", False),
        }
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# 使用示例
result = generate_text(
    prompt="什么是注意力机制？",
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9
)
print(result["generated_text"])

# 获取详细信息
result = generate_text(
    prompt="解释一下反向传播算法",
    max_new_tokens=100,
    details=True
)
print(json.dumps(result, indent=2))
```

### 流式 API

```python
import requests
import json

def generate_stream(prompt, **kwargs):
    """流式生成文本"""
    url = "http://localhost:8080/generate_stream"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("do_sample", True),
        }
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    generated_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data:"):
                data = json.loads(line[5:])
                token = data["token"]["text"]
                generated_text += token
                print(token, end="", flush=True)
                
                if data["generated_text"] is not None:
                    print()
                    return data
    
    return {"generated_text": generated_text}

# 使用示例
generate_stream("请写一首关于人工智能的诗", max_new_tokens=100)
```

### OpenAI 兼容 API

```bash
# TGI 兼容 OpenAI API 格式的端点
# POST /v1/chat/completions

curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tgi",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "什么是微服务架构？"}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }'
```

## 推测解码（Speculative Decoding）

```bash
# 使用推测解码加速生成
docker run --gpus all \
    -p 8080:80 \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-70b-hf \
    --num-shard 4 \
    --speculative-decoding \
    --draft-model-id meta-llama/Llama-2-7b-hf \
    --num-shard-draft 1 \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

推测解码原理：
```
传统解码:  [小模型] 生成 token 1 → [小模型] 生成 token 2 → ...
推测解码:  [小模型] 批量预测 token 1-5 → [大模型] 并行验证 1-5 → 接受/拒绝
```

| 指标 | 传统解码 | 推测解码 | 加速比 |
|------|----------|----------|--------|
| LLaMA-2 70B | 8 tokens/s | 15 tokens/s | 1.9x |
| LLaMA-2 13B | 35 tokens/s | 58 tokens/s | 1.7x |
| LLaMA-2 7B | 60 tokens/s | 95 tokens/s | 1.6x |

## 自定义插件与中间件

### 请求验证中间件

```python
from fastapi import Request, HTTPException
import re

class RequestValidator:
    """TGI 请求验证中间件"""
    
    def __init__(self, max_prompt_length=2000, banned_patterns=None):
        self.max_prompt_length = max_prompt_length
        self.banned_patterns = banned_patterns or []
    
    def validate(self, request: dict) -> dict:
        """验证请求参数"""
        inputs = request.get("inputs", "")
        
        # 长度检查
        if len(inputs) > self.max_prompt_length:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt 长度超过限制 ({self.max_prompt_length})"
            )
        
        # 模式检查
        for pattern in self.banned_patterns:
            if re.search(pattern, inputs, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="请求内容包含禁止的模式"
                )
        
        return request
    
    def validate_response(self, response: dict) -> dict:
        """验证响应"""
        generated_text = response.get("generated_text", "")
        
        # 可以添加内容安全检查
        return response

# 使用示例
validator = RequestValidator(
    max_prompt_length=2000,
    banned_patterns=[
        r"malicious_pattern_1",
        r"malicious_pattern_2",
    ]
)
```

### 自定义采样策略

```python
class CustomSampler:
    """自定义采样策略"""
    
    @staticmethod
    def top_k_top_p_sampling(logits, top_k=50, top_p=0.9, temperature=1.0):
        """Top-K + Top-P 采样"""
        import torch
        
        # 应用温度
        logits = logits / temperature
        
        # Top-K
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        
        # Top-P
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除概率累积超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("inf")
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
```

## 性能监控与可观测性

### Prometheus 指标

TGI 内置了丰富的 Prometheus 指标，可以通过 `/metrics` 端点获取：

```
# TGI 主要指标:
tgi_request_success_total            # 成功请求总数
tgi_request_duration_seconds         # 请求延迟
tgi_request_input_length             # 输入长度分布
tgi_request_generated_tokens         # 生成 token 数分布
tgi_queue_size                       # 队列大小
tgi_batch_current_size               # 当前批处理大小
tgi_inference_duration_seconds       # 推理时间
```

### Grafana 仪表板配置

```yaml
# grafana-dashboard.json
{
  "dashboard": {
    "title": "TGI 监控仪表板",
    "panels": [
      {
        "title": "请求吞吐量",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(tgi_request_success_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "请求延迟",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(tgi_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(tgi_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(tgi_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "队列深度",
        "type": "timeseries",
        "targets": [
          {
            "expr": "tgi_queue_size",
            "legendFormat": "Queue Size"
          }
        ]
      }
    ]
  }
}
```

### 健康检查与自动重启

```python
import requests
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

class TGIMonitor:
    """TGI 服务监控器"""
    
    def __init__(self, endpoint="http://localhost:8080", check_interval=30):
        self.endpoint = endpoint
        self.check_interval = check_interval
    
    def health_check(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def monitor(self):
        """持续监控"""
        while True:
            if not self.health_check():
                logging.warning("TGI 服务不健康，尝试重启...")
                self.restart_service()
                time.sleep(60)  # 等待服务启动
            
            time.sleep(self.check_interval)
    
    def restart_service(self):
        """重启 TGI 服务"""
        # Docker 重启示例
        subprocess.run(["docker", "restart", "tgi-container"], check=True)
        logging.info("TGI 服务已重启")

# 使用示例
# monitor = TGIMonitor()
# monitor.monitor()
```

## Docker Compose 生产部署

```yaml
# docker-compose.yml
version: "3.8"

services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      --model-id meta-llama/Llama-2-7b-hf
      --num-shard 1
      --quantize gptq
      --max-input-length 2048
      --max-total-tokens 4096
      --max-batch-prefill-tokens 4096
      --max-batch-total-tokens 8192
    ports:
      - "8080:80"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - CUDA_MEMORY_FRACTION=0.95
    volumes:
      - ~/.cache/huggingface:/data
    shm_size: 2g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - tgi

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

## TGI 与 vLLM 选择指南

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| HuggingFace 生态重度用户 | TGI | 无缝集成，模型自动下载 |
| 需要 LoRA 多适配器服务 | vLLM | 内置 LoRA 支持 |
| 极致吞吐量要求 | vLLM | PagedAttention 显存管理更高效 |
| 企业安全合规要求 | TGI | 内置内容过滤和水印 |
| 推测解码场景 | TGI | 官方支持更成熟 |
| 多模型混合部署 | vLLM | 灵活的模型管理 |
| 需要 Rust 高性能组件 | TGI | Candle 推理引擎 |

## 总结

TGI 作为 HuggingFace 官方维护的推理框架，在 HuggingFace 生态集成、安全性、稳定性方面具有独特优势。本文介绍了：

- TGI 的核心特性与 vLLM 的对比分析
- TGI 的安装部署方法（Docker、CLI、原生）
- REST API、流式 API 和 OpenAI 兼容 API 的使用
- GPTQ、AWQ、bitsandbytes 量化部署
- 推测解码加速技术
- 性能监控与生产部署实践

在实际项目中，TGI 和 vLLM 都是优秀的 LLM 推理方案，选择取决于具体需求和技术栈偏好。在下一篇文档中，我们将站在更高的视角，系统探讨 LLM 服务化（LLM Serving）的整体架构设计、资源规划、负载均衡和服务治理策略。

[下一篇：84-LLM 服务化架构设计](84-llm-serving.md)
