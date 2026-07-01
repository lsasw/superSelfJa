---
title: vLLM 高并发推理框架
icon: server
order: 7
---

# 83. vLLM 高并发推理框架

在上一篇文档中，我们学习了知识蒸馏技术，通过教师-学生模型框架压缩模型体积。当我们将模型压缩到适合部署的规模后，下一步就是构建高效的推理服务。vLLM 是目前最流行的 LLM 推理框架之一，其独创的 PagedAttention 技术解决了 LLM 推理中的显存管理难题，实现了显著的吞吐量提升。本文将全面介绍 vLLM 的原理、部署和调优方法。

## vLLM 概述

### 为什么需要 vLLM

大型语言模型的推理面临独特的挑战：

| 挑战 | 说明 | 传统方案 | vLLM 方案 |
|------|------|----------|-----------|
| 显存碎片化 | KV Cache 动态变化导致显存碎片 | 预分配固定大小 | PagedAttention 分页管理 |
| 批处理效率低 | 不同请求序列长度差异大 | 静态批处理 | 连续批处理 |
| 并发能力弱 | 单 GPU 只能服务少量用户 | 多实例部署 | 高效显存利用 |
| 调度复杂 | 请求优先级、抢占等 | 手动实现 | 内置调度器 |

### PagedAttention 原理

PagedAttention 借鉴了操作系统虚拟内存的分页思想，将 KV Cache 分割为固定大小的块（block），通过块表（block table）管理非连续的显存分配：

```
传统 KV Cache 管理:
┌────────────────────────────────────┐
│  Request A  │    空闲    │ Req B   │  ← 显存碎片化
└────────────────────────────────────┘

PagedAttention 管理:
Block Table A: [3] → [7] → [1] → [12]  ← 非连续分配
Block Table B: [5] → [2] → [9]         ← 无碎片
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ...
└───┴───┴───┴───┴───┴───┴───┴───┘
```

### vLLM 核心特性

- **PagedAttention**：自动分页管理 KV Cache，消除显存碎片
- **连续批处理（Continuous Batching）**：动态插入新请求到批处理中
- **高性能 CUDA Kernel**：优化的注意力计算实现
- **多 LoRA 支持**：共享基础模型，同时服务多个 LoRA 适配器
- **并行解码**：支持张量并行和流水线并行
- **OpenAI 兼容 API**：直接替换 OpenAI API 调用

## vLLM 安装与部署

### 安装

```bash
# 使用 pip 安装
pip install vllm

# 指定 CUDA 版本（如果需要）
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# Docker 方式（推荐用于生产环境）
docker run --gpus all \
    -p 8000:8000 \
    --shm-size=10g \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-hf
```

### 基础使用

```python
from vllm import LLM, SamplingParams

# 创建 LLM 引擎
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # GPU 数量
    dtype="float16",         # 数据类型
    max_model_len=4096,      # 最大序列长度
    gpu_memory_utilization=0.9,  # GPU 显存利用率
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
    stop=["\n\n"],
)

# 推理
prompts = [
    "请解释一下什么是人工智能？",
    "用 Python 写一个快速排序算法。",
    "翻译以下句子到英文：今天天气真好。",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt}")
    print(f"生成: {generated_text}")
    print("---")
```

### OpenAI 兼容 API 服务

```bash
# 启动 vLLM API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# 使用 curl 调用
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "请介绍一下 Python 的优势",
        "max_tokens": 200,
        "temperature": 0.7
    }'

# 使用 Python 客户端
```

```python
from openai import OpenAI

# 创建客户端（使用 vLLM 作为后端）
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# 文本补全
completion = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="请解释一下机器学习中的过拟合问题",
    max_tokens=200,
    temperature=0.7,
)
print(completion.choices[0].text)

# 聊天补全
chat_completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "什么是 Transformer 模型？"}
    ],
    max_tokens=500,
    temperature=0.7,
)
print(chat_completion.choices[0].message.content)
```

## vLLM 核心配置详解

### 引擎配置

```python
from vllm import LLM, EngineArgs, EngineConfig

# 方式一：使用 LLM 构造函数
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # 模型配置
    tokenizer="meta-llama/Llama-2-7b-hf",
    tokenizer_mode="auto",
    trust_remote_code=False,
    download_dir=None,
    load_format="auto",
    dtype="auto",
    seed=0,
    
    # 显存配置
    gpu_memory_utilization=0.9,    # GPU 显存利用率
    swap_space=4,                   # CPU 交换空间 (GB)
    
    # 并行配置
    tensor_parallel_size=1,         # 张量并行 GPU 数
    pipeline_parallel_size=1,       # 流水线并行 GPU 数
    distributed_executor_backend=None,
    
    # 模型限制
    max_model_len=None,             # 最大序列长度
    max_num_batched_tokens=None,    # 最大批处理 token 数
    max_num_seqs=256,               # 最大并发序列数
    
    # 调度配置
    disable_log_stats=False,
    disable_log_requests=False,
    max_paddings=0,
    
    # 性能优化
    enable_prefix_caching=True,     # 启用前缀缓存
    enable_chunked_prefill=False,   # 启用分块预填充
)

# 方式二：使用 EngineArgs
engine_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    enable_prefix_caching=True,
)
engine_config = engine_args.create_engine_config()
```

### 关键参数调优

| 参数 | 默认值 | 调优建议 | 影响 |
|------|--------|----------|------|
| gpu_memory_utilization | 0.9 | 根据模型大小调整，大模型可降低 | 显存分配比例 |
| max_model_len | 自动 | 设置合理上限，过高浪费显存 | 最大序列长度 |
| max_num_seqs | 256 | 根据并发需求调整 | 最大并发请求数 |
| enable_prefix_caching | False | 多轮对话场景开启 | 前缀复用加速 |
| enable_chunked_prefill | False | 长序列场景开启 | 降低首 token 延迟 |

## 连续批处理与调度

### 连续批处理原理

```python
# vLLM 的连续批处理 vs 传统静态批处理

# 传统静态批处理
class StaticBatching:
    """传统静态批处理"""
    
    def process_batch(self, requests):
        """等待所有请求完成后才能处理下一批"""
        # 所有请求必须等待最慢的那个完成
        results = []
        for req in requests:
            result = self._generate(req)
            results.append(result)
        return results

# vLLM 连续批处理
class ContinuousBatching:
    """vLLM 连续批处理"""
    
    def process_requests(self, request_queue):
        """动态调度和完成请求"""
        running_requests = []
        
        while request_queue or running_requests:
            # 完成已结束的请求
            running_requests = [
                req for req in running_requests
                if not req.is_finished
            ]
            
            # 从队列中添加新请求
            while request_queue and self._has_capacity():
                req = request_queue.pop(0)
                running_requests.append(req)
            
            # 执行一步推理
            self._step(running_requests)
            
            # 返回已完成的请求结果
            for req in running_requests:
                if req.is_finished:
                    yield req.result
```

### 调度器配置

```python
from vllm.core.scheduler import Scheduler, SchedulerConfig

scheduler_config = SchedulerConfig(
    max_num_batched_tokens=4096,    # 每步最大 token 数
    max_num_seqs=256,               # 最大序列数
    max_model_len=4096,             # 模型最大长度
)

scheduler = Scheduler(scheduler_config)

# 调度策略
# 1. 优先级调度：根据请求等待时间和资源需求
# 2. 抢占调度：当资源不足时，暂停低优先级请求
# 3. 公平调度：确保所有请求都能获得处理机会
```

## 多 GPU 并行推理

### 张量并行（Tensor Parallelism）

```python
from vllm import LLM

# 使用多 GPU 进行张量并行
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",  # 70B 模型需要多 GPU
    tensor_parallel_size=4,              # 使用 4 个 GPU
    pipeline_parallel_size=1,            # 不使用流水线并行
)

# 张量并行将模型的每一层分割到多个 GPU 上
# 适用于单个 GPU 无法容纳整个模型权重的场景
```

### 流水线并行（Pipeline Parallelism）

```python
from vllm import LLM

# 流水线并行：将模型的不同层分配到不同 GPU
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,              # 每阶段 2 个 GPU
    pipeline_parallel_size=4,            # 4 个阶段
    # 总共使用 2 * 4 = 8 个 GPU
)

# 流水线并行将模型按层分割，每个 GPU 负责一部分层
# 适用于模型非常大的场景
```

### 并行配置选择

| 模型规模 | 可用 GPU | 推荐配置 | 说明 |
|----------|----------|----------|------|
| 7B | 1 | TP=1, PP=1 | 单 GPU 即可 |
| 7B | 2 | TP=2, PP=1 | 张量并行加速 |
| 13B | 1 | TP=1, PP=1 | 需要量化 |
| 13B | 2 | TP=2, PP=1 | 推荐配置 |
| 70B | 4 | TP=4, PP=1 | FP16 需要 4 卡 |
| 70B | 8 | TP=4, PP=2 | 推荐配置 |

## 多 LoRA 服务

```python
from vllm import LLM, SamplingParams

# 加载基础模型并支持多 LoRA
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,              # 启用 LoRA
    max_loras=4,                   # 最大 LoRA 数量
    max_lora_rank=16,              # 最大 LoRA rank
)

# 配置 LoRA 适配器
lora_request_1 = LoRARequest(
    lora_name="code_adapter",
    lora_int_id=1,
    lora_path="./lora_code_adapter",
)

lora_request_2 = LoRARequest(
    lora_name="medical_adapter",
    lora_int_id=2,
    lora_path="./lora_medical_adapter",
)

# 使用不同 LoRA 进行推理
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# 代码任务
outputs_code = llm.generate(
    prompts=["def quick_sort(arr):"],
    sampling_params=sampling_params,
    lora_request=lora_request_1,
)

# 医疗任务
outputs_medical = llm.generate(
    prompts=["请解释一下糖尿病的发病机制"],
    sampling_params=sampling_params,
    lora_request=lora_request_2,
)
```

## 前缀缓存（Prefix Caching）

```python
from vllm import LLM

# 启用前缀缓存
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,   # 开启前缀缓存
)

# 多轮对话场景：前缀会被缓存和复用
prompts = [
    "系统提示：你是一个编程助手。请帮我解答以下问题：问题1：什么是递归？",
    "系统提示：你是一个编程助手。请帮我解答以下问题：问题2：什么是迭代？",
    "系统提示：你是一个编程助手。请帮我解答以下问题：问题3：什么是动态规划？",
]

# 相同的前缀部分会被缓存，只计算差异部分
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

## 性能基准测试

### 吞吐量测试

```python
import time
import asyncio
from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs

class VLlmBenchmark:
    """vLLM 性能基准测试"""
    
    def __init__(self, model_name):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            max_model_len=2048,
        )
    
    def benchmark_throughput(
        self,
        prompts,
        max_tokens=100,
        temperature=0.7
    ):
        """测试吞吐量"""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # 预热
        _ = self.llm.generate(prompts[:5], sampling_params)
        
        # 测量
        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        end = time.time()
        
        total_time = end - start
        total_tokens = sum(
            len(output.outputs[0].token_ids)
            for output in outputs
        )
        
        print(f"=== 吞吐量测试 ===")
        print(f"请求数: {len(prompts)}")
        print(f"总生成 token 数: {total_tokens}")
        print(f"总时间: {total_time:.2f}s")
        print(f"吞吐量: {len(prompts)/total_time:.2f} requests/s")
        print(f"token 吞吐量: {total_tokens/total_time:.2f} tokens/s")
        print(f"平均延迟: {total_time/len(prompts)*1000:.2f} ms/request")
        
        return {
            "requests_per_second": len(prompts) / total_time,
            "tokens_per_second": total_tokens / total_time,
            "avg_latency_ms": total_time / len(prompts) * 1000,
        }
    
    def benchmark_latency(
        self,
        prompt,
        max_tokens=100,
        num_runs=10
    ):
        """测试首 token 延迟和端到端延迟"""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,  # 贪婪解码以获得一致结果
        )
        
        # 预热
        _ = self.llm.generate([prompt], sampling_params)
        
        # 测量
        latencies = []
        ttft = []  # Time to First Token
        
        for _ in range(num_runs):
            start = time.time()
            outputs = self.llm.generate([prompt], sampling_params)
            end = time.time()
            
            total_latency = (end - start) * 1000
            latencies.append(total_latency)
            
            # TTFT 需要通过 stream 方式获取
            # 此处简化处理
        
        import numpy as np
        print(f"=== 延迟测试 ===")
        print(f"平均延迟: {np.mean(latencies):.2f} ms")
        print(f"P50 延迟: {np.percentile(latencies, 50):.2f} ms")
        print(f"P99 延迟: {np.percentile(latencies, 99):.2f} ms")

# 使用示例
# benchmark = VLlmBenchmark("meta-llama/Llama-2-7b-hf")
# prompts = ["解释一下机器学习是什么"] * 100
# benchmark.benchmark_throughput(prompts)
```

### 与 HuggingFace 性能对比

| 框架 | 模型 | 批处理 | 吞吐量 (req/s) | Token/s | 显存利用率 |
|------|------|--------|----------------|---------|------------|
| HuggingFace | LLaMA-2 7B | 静态 | 2.1 | 420 | 65% |
| vLLM | LLaMA-2 7B | 连续 | 8.5 | 2100 | 92% |
| HuggingFace | LLaMA-2 13B | 静态 | 1.2 | 280 | 70% |
| vLLM | LLaMA-2 13B | 连续 | 5.2 | 1350 | 95% |

## 生产部署实践

### Docker Compose 部署

```yaml
# docker-compose.yml
version: "3.8"

services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      --model meta-llama/Llama-2-7b-hf
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size 1
      --max-model-len 4096
      --gpu-memory-utilization 0.9
      --enable-prefix-caching
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    shm_size: 10g
    restart: unless-stopped

  # 可选：添加反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vllm
```

### 生产配置建议

```python
# 生产环境启动脚本
#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512 \
    --enable-prefix-caching \
    --disable-log-requests \
    --enable-log-stats \
    --swap-space 8 \
    --trust-remote-code
```

### 监控与告警

```python
# vLLM 内置 Prometheus 指标
# 访问 http://localhost:8000/metrics 获取

"""
# vLLM 暴露的主要指标:
vllm:num_requests_running        # 当前运行请求数
vllm:num_requests_waiting        # 等待请求数
vllm:num_requests_swapped        # 被交换到 CPU 的请求数
vllm:gpu_cache_usage_perc        # GPU 缓存使用率
vllm:gpu_prefix_cache_hit_rate   # 前缀缓存命中率
vllm:iteration_tokens_total      # 总处理 token 数
vllm:e2e_request_latency_seconds # 端到端请求延迟
vllm:time_per_output_token       # 每输出 token 的时间
vllm:time_to_first_token_seconds # 首 token 时间
"""

# 使用 Prometheus + Grafana 监控
from prometheus_client import generate_latest

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

## 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM 错误 | 显存不足 | 降低 gpu_memory_utilization 或 max_model_len |
| 性能低于预期 | 未开启连续批处理 | 确保使用最新版本，检查配置 |
| 多 GPU 不生效 | NCCL 配置问题 | 设置 NCCL_P2P_LEVEL=PIX |
| 模型加载失败 | 显存碎片化 | 重启服务，设置 GPU_MEMORY_UTILIZATION |

## 总结

vLLM 通过 PagedAttention 和连续批处理技术，为 LLM 推理带来了革命性的性能提升。本文介绍了：

- vLLM 的核心优势与 PagedAttention 原理
- vLLM 的安装、部署与 OpenAI 兼容 API 使用
- 关键配置参数详解与调优建议
- 多 GPU 并行推理与多 LoRA 服务
- 前缀缓存与性能基准测试
- 生产部署实践与监控方案

vLLM 是目前 LLM 推理服务的首选方案，特别适合高并发场景。在下一篇文档中，我们将学习 Triton 推理服务器，这是 NVIDIA 官方维护的企业级推理方案，支持多框架、多模型的高性能推理服务。

[下一篇：84-Triton 推理服务](84-triton.md)
