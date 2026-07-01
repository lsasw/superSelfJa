---
title: OpenAPI → MCP Tool 转换
icon: exchange-alt
order: 3
category:
  - MCP
tag:
  - OpenAPI
  - MCP
  - Schema
---

# OpenAPI → MCP Tool 转换规范

> 任务编号：OT-01 / OT-02

## 一、核心挑战

```
OpenAPI Spec (Swagger)          →        MCP Tool Schema
─────────────────────────────────        ──────────────────
路径: GET /api/users/{id}       →        Tool: get_user(id: string)
参数: path + query + body       →        JSON Schema inputSchema
响应: 200/400/500 schema        →        不需要（Tool 只关心入参）
描述: summary + description    →        Tool.description
认证: bearer/session           →        MCP Server 层面处理
```

### 三大难题

| 难题 | 说明 |
|------|------|
| **description 缺失** | 很多老接口的 Swagger 文档没有详细的 description |
| **参数类型映射** | OpenAPI type 和 JSON Schema type 不完全一致 |
| **嵌套 schema** | 复杂的 `$ref` 嵌套引用展平 |

## 二、参数类型映射表

```python
OPENAPI_TO_JSONSCHEMA = {
    "string":   {"type": "string"},
    "integer":  {"type": "integer"},
    "number":   {"type": "number"},
    "boolean":  {"type": "boolean"},
    "array":    {"type": "array"},
    "object":   {"type": "object"},
    # OpenAPI 特有格式
    "string:date":      {"type": "string", "format": "date"},
    "string:date-time": {"type": "string", "format": "date-time"},
    "string:uuid":      {"type": "string", "format": "uuid"},
    "string:email":     {"type": "string", "format": "email"},
    "integer:int32":    {"type": "integer", "format": "int32"},
    "integer:int64":    {"type": "integer", "format": "int64"},
    "number:float":     {"type": "number", "format": "float"},
    "number:double":    {"type": "number", "format": "double"},
}
```

## 三、核心转换器实现

```python
import json
import yaml
from typing import Any
from openai import OpenAI

class OpenAPIToMCP:
    """OpenAPI → MCP Tool 转换器"""
    
    def __init__(self, openapi_spec: dict):
        self.spec = openapi_spec
        self.llm = OpenAI()  # 用于 description 补全
    
    def convert_all(self) -> list[dict]:
        """转换所有 API 路径为 MCP Tools"""
        tools = []
        for path, methods in self.spec.get("paths", {}).items():
            for method, operation in methods.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue
                try:
                    tool = self._convert_operation(path, method, operation)
                    tools.append(tool)
                except Exception as e:
                    print(f"[WARN] 转换失败 {method.upper()} {path}: {e}")
        return tools
    
    def _convert_operation(self, path: str, method: str, 
                           operation: dict) -> dict:
        """转换单个 API 操作为 MCP Tool"""
        # 1. 生成 Tool name
        tool_name = self._generate_tool_name(method, path, operation)
        
        # 2. 构建 description（补全缺失的）
        description = self._build_description(operation, path, method)
        
        # 3. 转换参数为 JSON Schema
        input_schema = self._convert_parameters(operation)
        
        # 4. 添加 path 参数
        self._inject_path_params(path, input_schema)
        
        return {
            "name": tool_name,
            "description": description,
            "inputSchema": input_schema
        }
    
    def _generate_tool_name(self, method: str, path: str, 
                            operation: dict) -> str:
        """生成语义化的 Tool 名称"""
        # 优先使用 operationId
        if "operationId" in operation:
            return operation["operationId"]
        
        # 从路径和方法生成：GET /api/users/{id} → get_user_by_id
        parts = path.strip("/").split("/")
        # 移除动态参数
        clean_parts = [p.strip("{}") for p in parts if not p.startswith("{")]
        name = f"{method.lower()}_{'_'.join(clean_parts)}"[-64:]  # 限制长度
        return name
    
    def _build_description(self, operation: dict, 
                           path: str, method: str) -> str:
        """构建 Tool description"""
        summary = operation.get("summary", "")
        description = operation.get("description", "")
        
        # 如果 description 缺失，用 LLM 补全
        if not description or len(description) < 20:
            description = self._auto_generate_description(
                method, path, summary, operation
            )
        
        # 拼接：summary + description
        full = f"{summary}\n{description}".strip()
        
        # 追加使用场景提示
        parameters = operation.get("parameters", [])
        if parameters:
            param_names = [p["name"] for p in parameters]
            full += f"\n\n可用参数：{', '.join(param_names)}"
        
        return full
    
    def _auto_generate_description(self, method: str, path: str,
                                    summary: str, operation: dict) -> str:
        """LLM 自动补全 description"""
        prompt = f"""请为以下 API 生成一个简洁的功能描述（1-2 句话）：

方法：{method.upper()}
路径：{path}
摘要：{summary}
参数：{json.dumps(operation.get('parameters', []), ensure_ascii=False)}

请描述这个 API 的功能、输入什么、返回什么，以及什么场景下应该调用它。"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    def _convert_parameters(self, operation: dict) -> dict:
        """转换参数为 JSON Schema"""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in operation.get("parameters", []):
            name = param["name"]
            param_schema = self._convert_single_param(param)
            schema["properties"][name] = param_schema
            
            if param.get("required", False):
                schema["required"].append(name)
        
        # 处理 requestBody
        if "requestBody" in operation:
            body_schema = self._convert_request_body(operation["requestBody"])
            if body_schema:
                schema["properties"]["body"] = body_schema
                if operation["requestBody"].get("required", False):
                    schema["required"].append("body")
        
        return schema
    
    def _convert_single_param(self, param: dict) -> dict:
        """转换单个参数"""
        schema = param.get("schema", {})
        param_type = schema.get("type", param.get("type", "string"))
        fmt = schema.get("format", "")
        
        key = f"{param_type}:{fmt}" if fmt else param_type
        result = dict(OPENAPI_TO_JSONSCHEMA.get(key, {"type": param_type}))
        
        # 保留 description 和 enum
        if "description" in param:
            result["description"] = param["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]
        if "default" in schema:
            result["default"] = schema["default"]
        
        return result
    
    def _convert_request_body(self, body: dict) -> dict | None:
        """转换 requestBody"""
        content = body.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        
        if not schema:
            return None
        
        # 解析 $ref
        if "$ref" in schema:
            schema = self._resolve_ref(schema["$ref"])
        
        return self._convert_schema_recursive(schema)
    
    def _resolve_ref(self, ref: str) -> dict:
        """解析 $ref 引用"""
        # ref: "#/components/schemas/User"
        path_parts = ref.strip("#/").split("/")
        current = self.spec
        for part in path_parts:
            current = current.get(part, {})
        return current
    
    def _convert_schema_recursive(self, schema: dict) -> dict:
        """递归转换嵌套 schema"""
        result = {"type": schema.get("type", "object")}
        
        if "description" in schema:
            result["description"] = schema["description"]
        
        if schema.get("type") == "object" and "properties" in schema:
            result["properties"] = {}
            for name, prop in schema["properties"].items():
                result["properties"][name] = self._convert_schema_recursive(prop)
            if "required" in schema:
                result["required"] = schema["required"]
        
        elif schema.get("type") == "array" and "items" in schema:
            result["items"] = self._convert_schema_recursive(schema["items"])
        
        else:
            if "enum" in schema:
                result["enum"] = schema["enum"]
        
        return result
    
    def _inject_path_params(self, path: str, input_schema: dict):
        """注入路径参数"""
        import re
        path_params = re.findall(r'\{(\w+)\}', path)
        for param in path_params:
            if param not in input_schema["properties"]:
                input_schema["properties"][param] = {
                    "type": "string",
                    "description": f"路径参数: {param}"
                }
                input_schema["required"].append(param)
```

## 四、转换示例

### 输入：OpenAPI Spec

```yaml
paths:
  /api/users/{userId}:
    get:
      operationId: getUserById
      summary: 获取用户详情
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
        - name: includeOrders
          in: query
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
```

### 输出：MCP Tool

```json
{
  "name": "getUserById",
  "description": "获取用户详情\n根据用户ID查询用户的详细信息和关联数据。\n\n可用参数：userId, includeOrders",
  "inputSchema": {
    "type": "object",
    "properties": {
      "userId": {
        "type": "string",
        "description": "路径参数: userId"
      },
      "includeOrders": {
        "type": "boolean",
        "default": false
      }
    },
    "required": ["userId"]
  }
}
```

## 五、踩坑记录

| 坑 | 原因 | 解决 |
|----|------|------|
| `$ref` 循环引用导致死循环 | 模型间相互引用 | 维护已解析列表，检测到循环时跳过 |
| `oneOf`/`anyOf` 无法直接映射 | JSON Schema 支持但 MCP 端兼容性差 | 展平为单个 object + description 说明 |
| 超大 schema（100+ 属性） | 完整展开导致 Tool prompt 超长 | 只展开前 10 个关键属性，其余用 `additionalProperties` |
| 中文参数名乱码 | OpenAPI 编码问题 | 显式指定 `ensure_ascii=False` |
