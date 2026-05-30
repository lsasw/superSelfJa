---
title: AI 模型部署概述
icon: rocket
order: 1
---

# 76. AI 模型部署概述

在完成了 AI 基础理论的学习后，我们将正式进入 AI 工程化与部署领域。作为整个模块的开篇，本文将系统性地介绍 AI 模型从训练环境到生产环境部署的完整流程、核心挑战与主流解决方案，为后续深入学习 ONNX、TensorRT、vLLM 等具体技术奠定全局视野。

## 什么是 AI 模型部署

AI 模型部署是将训练完成的机器学习或深度学习模型从研发环境迁移到生产环境，使其能够接收真实数据输入并返回推理结果的全过程。这个过程远不止于"复制模型文件到服务器"，而是涉及模型格式转换、性能优化、服务封装、资源调度、监控告警等一系列工程化工作。

一个典型的 AI 模型部署流程包含以下阶段：

| 阶段 | 核心任务 | 关键产出 |
|------|----------|----------|
| 模型导出 | 将训练框架的模型导出为通用格式 | ONNX、SavedModel、TorchScript |
| 模型优化 | 量化、剪枝、图优化等 | 优化后的模型文件 |
| 服务封装 | 构建推理服务接口 | REST API、gRPC 服务 |
| 资源分配 | CPU/GPU 资源配置、批处理策略 | 部署配置文件 |
| 上线发布 | 灰度发布、A/B 测试、回滚机制 | 生产服务实例 |
| 监控运维 | 性能监控、数据漂移检测 | 监控面板、告警规则 |

💡 **提示**：模型部署不是单纯的"搬运"工作。一个在 Jupyter Notebook 中准确率 95% 的模型，未经合理的部署优化，在生产环境中可能出现延迟过高、吞吐量不足、精度下降等问题。部署质量直接决定了 AI 项目的商业价值能否真正实现。

## 模型部署的核心挑战

### 延迟与吞吐量的平衡

延迟（Latency）指从请求发出到收到响应的时间，吞吐量（Throughput）指单位时间内处理的请求数量。在实际生产中，这两者往往是矛盾的：

```python
# 高吞吐低延迟的批处理策略示例
import torch
from concurrent.futures import ThreadPoolExecutor

class BatchInferenceServer:
    """动态批处理推理服务器"""
    
    def __init__(self, model, max_batch_size=32, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue = []
        self.lock = threading.Lock()
    
    def add_request(self, input_data):
        """添加请求到批处理队列"""
        with self.lock:
            self.request_queue.append(input_data)
            
            # 当队列满或等待超时，触发批处理
            if len(self.request_queue) >= self.max_batch_size:
                batch = self.request_queue[:self.max_batch_size]
                self.request_queue = self.request_queue[self.max_batch_size:]
                return self._process_batch(batch)
    
    def _process_batch(self, batch):
        """执行批处理推理"""
        input_tensor = torch.stack(batch)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
```

💡 **提示**：对于在线推理场景（如推荐系统），通常要求延迟低于 100ms；对于离线推理场景（如批量图像分析），可以容忍秒级延迟以换取更高的吞吐量。

### 模型体积与部署成本

现代 AI 模型规模持续增长，从早期的 LeNet（6 万参数）到 GPT-4（预估万亿参数），模型体积的膨胀带来了严峻的部署成本问题：

| 模型 | 参数量 | FP32 模型体积 | FP16 模型体积 | INT8 模型体积 |
|------|--------|---------------|---------------|---------------|
| ResNet-50 | 2560 万 | ~102 MB | ~51 MB | ~26 MB |
| BERT-Base | 1.1 亿 | ~440 MB | ~220 MB | ~110 MB |
| LLaMA-2 7B | 70 亿 | ~28 GB | ~14 GB | ~7 GB |
| LLaMA-2 70B | 700 亿 | ~280 GB | ~140 GB | ~70 GB |

模型压缩技术（量化、剪枝、知识蒸馏）是降低部署成本的关键手段，我们将在后续文档中深入探讨。

### 异构硬件适配

生产环境中存在多种计算硬件，每种硬件的推理性能和工具链各不相同：

```python
# 多硬件适配的推理引擎抽象层
from abc import ABC, abstractmethod
import torch

class InferenceEngine(ABC):
    """推理引擎抽象基类"""
    
    @abstractmethod
    def load_model(self, model_path: str):
        pass
    
    @abstractmethod
    def infer(self, input_data) -> dict:
        pass

class PyTorchEngine(InferenceEngine):
    """PyTorch CPU/GPU 推理引擎"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
    
    def load_model(self, model_path: str):
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def infer(self, input_data):
        with torch.no_grad():
            input_tensor = torch.tensor(input_data).to(self.device)
            output = self.model(input_tensor)
            return output.cpu().numpy()

class TensorRTEngine(InferenceEngine):
    """TensorRT GPU 推理引擎"""
    
    def __init__(self):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def load_model(self, model_path: str):
        with open(model_path, "rb") as f:
            self.runtime = self.trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution上下文()
    
    def infer(self, input_data):
        # TensorRT 推理实现
        pass

class ONNXRuntimeEngine(InferenceEngine):
    """ONNX Runtime 跨平台推理引擎"""
    
    def __init__(self, providers: list = None):
        import onnxruntime as ort
        self.providers = providers or ["CPUExecutionProvider"]
    
    def load_model(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def infer(self, input_data):
        return self.session.run(None, {self.input_name: input_data})
```

## 主流部署方案对比

### 框架原生部署

PyTorch 和 TensorFlow 都提供了模型导出和部署工具：

```python
# PyTorch TorchScript 导出
import torch

model = MyModel()
model.eval()

# 方式一：TorchScript Scripting
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 方式二：TorchScript Tracing
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# 加载 TorchScript 模型进行推理
loaded_model = torch.jit.load("model_scripted.pt")
output = loaded_model(torch.randn(1, 3, 224, 224))
```

```python
# TensorFlow SavedModel 导出
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 保存为 SavedModel 格式
model.save("saved_model/")

# 使用 TensorFlow Serving 加载
loaded_model = tf.saved_model.load("saved_model/")
infer = loaded_model.signatures["serving_default"]
output = infer(tf.constant([[0.1] * 10]))
```

### 跨框架部署（ONNX）

ONNX（Open Neural Network Exchange）是目前最流行的跨框架模型交换格式：

```python
# PyTorch 模型导出为 ONNX 格式
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    export_params=True,
    opset_version=15,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
```

### 专用推理框架

| 框架 | 适用场景 | 支持的硬件 | 主要优势 |
|------|----------|------------|----------|
| TensorRT | GPU 高性能推理 | NVIDIA GPU | 极致性能优化 |
| OpenVINO | CPU 边缘推理 | Intel CPU/VPU | 硬件感知优化 |
| TVM | 全平台编译优化 | 多种硬件 | 自动调优 |
| vLLM | LLM 高并发服务 | NVIDIA GPU/AMD GPU | PagedAttention |
| TGI | LLM 推理服务 | NVIDIA GPU/AMD GPU | 官方维护 |

## 模型服务化架构

### REST API 服务模式

```python
# 使用 FastAPI 构建模型推理服务
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI(title="AI Model Inference Service")

class PredictionRequest(BaseModel):
    input_data: list[float]
    model_version: str = "v1"

class PredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    latency_ms: float

# 全局模型实例
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_model():
    global model
    # 加载模型逻辑
    pass

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time
    start_time = time.time()
    
    try:
        input_tensor = torch.tensor(request.input_data).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        predictions = output.squeeze().tolist()
        
        latency = (time.time() - start_time) * 1000
        return PredictionResponse(
            predictions=predictions,
            model_version=request.model_version,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### gRPC 服务模式

```protobuf
// inference.proto
syntax = "proto3";

package inference;

service InferenceService {
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc BatchPredict(BatchPredictRequest) returns (BatchPredictResponse);
}

message PredictRequest {
    string model_name = 1;
    repeated float input_data = 2;
}

message PredictResponse {
    repeated float output_data = 1;
    float latency_ms = 2;
}

message BatchPredictRequest {
    string model_name = 1;
    repeated PredictRequest requests = 2;
}

message BatchPredictResponse {
    repeated PredictResponse responses = 1;
}
```

### Triton Inference Server 方案

NVIDIA Triton Inference Server 是企业级模型部署的行业标准：

```python
# triton_model_repository/resnet50/1/model.py
import triton_python_backend_utils as pb_utils
import numpy as np
import torch

class TritonPythonModel:
    """Triton Python Backend 模型实现"""
    
    def initialize(self, args):
        self.model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda")
        self.model.to(self.device)
    
    def execute(self, requests):
        responses = []
        
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            input_data = in_0.as_numpy()
            
            input_tensor = torch.tensor(input_data).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            
            out_0 = pb_utils.Tensor("OUTPUT_0", output.cpu().numpy())
            responses.append(pb_utils.InferenceResponse([out_0]))
        
        return responses
```

对应的 `config.pbtxt` 配置文件：

```
name: "resnet50"
backend: "python"

max_batch_size: 32

input [
  {
    name: "INPUT_0"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "OUTPUT_0"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 100000
}
```

## 部署策略与发布模式

### 蓝绿部署

蓝绿部署通过维护两套相同的生产环境来实现零停机发布：

```
                 ┌─────────────┐
    流量 ────────▶│  负载均衡器  │
                 └──────┬──────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
    ┌─────────────┐           ┌─────────────┐
    │  蓝色环境    │           │  绿色环境    │
    │ (当前生产)   │           │ (新版本)     │
    │  Model v1   │           │  Model v2   │
    └─────────────┘           └─────────────┘
```

### 金丝雀发布

```python
# 金丝雀发布流量分配示例
import random

class CanaryDeployment:
    """金丝雀发布控制器"""
    
    def __init__(self, stable_version, canary_version, canary_percentage=5):
        self.stable_version = stable_version
        self.canary_version = canary_version
        self.canary_percentage = canary_percentage
    
    def route_request(self, request_data):
        """根据流量比例路由请求"""
        if random.randint(1, 100) <= self.canary_percentage:
            return self.canary_version, self._infer(self.canary_version, request_data)
        else:
            return self.stable_version, self._infer(self.stable_version, request_data)
    
    def _infer(self, version, request_data):
        # 实际推理逻辑
        pass
    
    def adjust_traffic(self, new_percentage):
        """动态调整金丝雀流量比例"""
        if self._check_metrics_ok():
            self.canary_percentage = new_percentage
    
    def _check_metrics_ok(self):
        """检查金丝雀版本的指标是否正常"""
        # 检查延迟、错误率、准确率等
        pass
```

### A/B 测试

```python
# A/B 测试框架
class ABTestFramework:
    """A/B 测试管理框架"""
    
    def __init__(self):
        self.variants = {}
        self.metrics = {}
    
    def add_variant(self, name, model, traffic_percentage):
        """添加测试变体"""
        self.variants[name] = {
            "model": model,
            "traffic_percentage": traffic_percentage,
            "request_count": 0,
            "metrics": {}
        }
    
    def get_variant(self, user_id):
        """根据用户 ID 分配变体"""
        hash_value = hash(user_id) % 100
        cumulative = 0
        for name, variant in self.variants.items():
            cumulative += variant["traffic_percentage"]
            if hash_value < cumulative:
                return name
    
    def record_metric(self, variant_name, metric_name, value):
        """记录实验指标"""
        if variant_name in self.variants:
            if metric_name not in self.variants[variant_name]["metrics"]:
                self.variants[variant_name]["metrics"][metric_name] = []
            self.variants[variant_name]["metrics"][metric_name].append(value)
    
    def analyze_results(self):
        """分析 A/B 测试结果"""
        results = {}
        for name, variant in self.variants.items():
            results[name] = {
                "request_count": variant["request_count"],
                "avg_metrics": {
                    metric: sum(values) / len(values)
                    for metric, values in variant["metrics"].items()
                }
            }
        return results
```

## 部署环境规划

### 典型部署架构

```
                    ┌───────────────────┐
                    │   API Gateway     │
                    │  (Kong/Nginx)     │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Service A   │  │  Service B   │  │  Service C   │
    │  (GPU)       │  │  (CPU)       │  │  (GPU)       │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
    ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
    │  GPU Pool   │  │  CPU Pool   │  │  GPU Pool   │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### 资源估算公式

```python
def estimate_gpu_memory(model_size_gb, batch_size, sequence_length, hidden_size):
    """估算 GPU 显存需求
    
    Args:
        model_size_gb: 模型参数量（GB）
        batch_size: 批处理大小
        sequence_length: 序列长度
        hidden_size: 隐藏层维度
    
    Returns:
        预估显存用量（GB）
    """
    # 模型权重显存
    weights_memory = model_size_gb
    
    # KV Cache 显存（对于 Transformer 模型）
    # 公式：2 * num_layers * batch_size * seq_len * hidden_size * dtype_size
    num_layers = 32  # 假设
    dtype_size = 2 / 1e9  # FP16
    kv_cache_memory = 2 * num_layers * batch_size * sequence_length * hidden_size * dtype_size
    
    # 激活显存
    activation_memory = batch_size * sequence_length * hidden_size * dtype_size * 10
    
    total_memory = weights_memory + kv_cache_memory + activation_memory
    
    return total_memory

# 示例：估算 LLaMA-2 7B 的显存需求
estimated_gb = estimate_gpu_memory(
    model_size_gb=14,        # FP16 模型
    batch_size=32,
    sequence_length=2048,
    hidden_size=4096
)
print(f"预估显存需求: {estimated_gb:.2f} GB")
```

## 模型部署的最佳实践

1. **模型版本管理**：为每个模型分配唯一版本号，支持快速回滚
2. **输入验证**：在服务层进行严格的输入格式和范围验证
3. **超时控制**：设置合理的推理超时时间，防止请求堆积
4. **降级策略**：当主模型不可用时，提供备选方案或默认响应
5. **日志记录**：记录推理请求的关键信息，便于问题排查和模型迭代
6. **定期评估**：定期检查模型在生产环境的准确率，及时发现模型退化
7. **安全防护**：防范对抗样本攻击，对敏感数据进行脱敏处理

```python
# 完整的推理服务中间件示例
from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import time
import hashlib

logger = logging.getLogger(__name__)

class InferenceMiddleware:
    """推理服务中间件"""
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        request_id = hashlib.md5(f"{time.time()}{request.client.host}".encode()).hexdigest()[:8]
        
        try:
            # 请求验证
            if request.method == "POST":
                body = await request.body()
                if len(body) > 10 * 1024 * 1024:  # 限制 10MB
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Request body too large"}
                    )
            
            response = await call_next(request)
            
            # 响应头添加
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id}
            )
```

## 总结

AI 模型部署是连接算法研究与商业价值的桥梁。本文系统介绍了模型部署的核心概念、常见挑战、主流方案和服务化架构。我们了解到：

- 模型部署是一个多阶段的工程化过程，涉及导出、优化、服务封装、上线发布和监控运维
- 延迟与吞吐量的平衡、模型体积控制、异构硬件适配是三大核心挑战
- ONNX 作为跨框架标准，Triton 作为企业级服务框架，是部署方案的重要组成
- 蓝绿部署、金丝雀发布、A/B 测试等策略保障了模型上线的安全性和可控性

在下一篇文档中，我们将深入探讨 ONNX 标准，学习如何将不同训练框架的模型统一导出为 ONNX 格式，实现跨平台的模型部署。

[下一篇：77-ONNX 跨框架模型交换标准](77-onnx.md)
