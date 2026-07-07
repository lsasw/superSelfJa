---
title: Pi — AI 编程智能体
icon: robot
order: 3
---

# Pi — AI 编程智能体

**Pi** 是 earendil-works 开源的 AI 编程智能体框架（TypeScript/Node.js），提供 CLI + TUI + 多模型 API 的完整编码智能体解决方案。MIT 协议。

- **GitHub**: https://github.com/earendil-works/pi
- **官网**: https://pi.dev
- **语言**: TypeScript（100%）

---

## 一、设计理念

Pi 定位为 **"自扩展的编码智能体"**——它不仅能写代码，还能修改自己的工具和配置。核心原则：

1. **多模型统一 API**：一套接口调用 OpenAI / Anthropic / Google 等
2. **Agent 运行时**：工具调用 + 状态管理 + 可组合中间件
3. **无内置权限系统**：默认以用户权限运行，建议通过容器化隔离

---

## 二、包体系（4 个核心包）

| 包 | 说明 |
|---|------|
| **@earendil-works/pi-ai** | 统一多提供商 LLM API（OpenAI, Anthropic, Google 等） |
| **@earendil-works/pi-agent-core** | Agent 运行时：工具调用 + 状态管理 |
| **@earendil-works/pi-coding-agent** | 交互式编码智能体 CLI |
| **@earendil-works/pi-tui** | 终端 UI 库（差分渲染） |

另有独立仓库 [pi-chat](https://github.com/earendil-works/pi-chat) 提供 Slack/聊天自动化集成。

---

## 三、核心能力

### 3.1 统一多模型 API (`pi-ai`)

```typescript
import { createLLM } from '@earendil-works/pi-ai'

// 一行切换模型提供商
const llm = createLLM({ provider: 'anthropic', model: 'claude-sonnet-4-20250514' })
const response = await llm.chat([{ role: 'user', content: '写一个冒泡排序' }])
```

支持的提供商：OpenAI、Anthropic、Google、以及更多兼容 OpenAI API 的服务。

### 3.2 Agent 运行时 (`pi-agent-core`)

核心是工具调用循环 + 状态管理：

```
用户输入 → LLM 推理 → 工具调用决策 → 执行工具 → 结果回传 → 继续推理 → 最终输出
```

特性：
- **工具注册机制**：文件操作 / Shell 执行 / 网络请求 等内置工具
- **状态持久化**：跨轮次保持上下文
- **可扩展中间件**：日志、权限、速率限制等可插拔

### 3.3 编码 CLI (`pi-coding-agent`)

```bash
# 安装
npm install -g @earendil-works/pi-coding-agent

# 启动交互式编码会话
pi

# 从源码运行（开发模式）
./pi-test.sh
```

CLI 特性：
- 交互式 REPL 编码会话
- 文件系统读写
- Shell 命令执行
- 自扩展能力（可修改自己的配置和工具）

### 3.4 终端 UI (TUI)

基于差分渲染的高性能终端界面，支持：
- 流式输出
- 语法高亮
- 文件差异展示
- 会话历史

---

## 四、容器化与安全

Pi 默认以启动用户权限运行，**无内置权限系统**。官方推荐三种隔离方案：

| 方案 | 说明 |
|------|------|
| **Gondolin 扩展** | 将内置工具和 `!` 命令路由到 Linux micro-VM |
| **Plain Docker** | 整个 Pi 进程运行在容器内 |
| **OpenShell** | 策略控制沙箱 |

---

## 五、开发与构建

```bash
npm install --ignore-scripts    # 安装依赖（跳过生命周期脚本）
npm run build                   # 构建所有包
npm run check                   # Lint + 格式化 + 类型检查
./test.sh                       # 运行测试（无 API key 时跳过 LLM 相关测试）
./pi-test.sh                    # 从源码运行 Pi
```

---

## 六、供应链安全

Pi 对 npm 依赖有严格的控制措施：

- **精确版本锁定**：外部依赖固定精确版本，内部 workspace 保持范围版本
- **最小发布年龄**：`.npmrc` 设置 `min-release-age=2` 避免同日发布风险
- **lockfile 保护**：pre-commit hook 阻止意外锁文件提交
- **shrinkwrap**：发布的 CLI 包包含 `npm-shrinkwrap.json` 锁定传递依赖
- **发布前烟雾测试**：在隔离环境安装测试后再发布
- **`--ignore-scripts`**：CI 安装和本地发布安装均跳过生命周期脚本

---

## 七、与 HarnessX / DeepAgents 对比

| 维度 | Pi | HarnessX | DeepAgents |
|------|:--:|:--------:|:----------:|
| 语言 | TypeScript | Python | Python |
| 定位 | 自扩展编码 Agent | 可组合进化框架 | 开箱即用 Agent |
| 工具调用 | ✅ 运行时 | ✅ 处理器管线 | ✅ Middleware |
| 多模型 | ✅ 统一 API | ✅ 多后端 | ✅ LangChain 生态 |
| CLI | ✅ 交互式 | ✅ CLI + IM | ✅ dcode CLI |
| 上下文管理 | 状态持久化 | Light-Memory | 四层摘要 |
| 子代理 | ❌ | ❌ | ✅ task 工具 |
| 安全隔离 | 容器化推荐 | ❌ | 沙箱/QuickJS |

---

## 八、生态系统

- **Pi Chat**：Slack / Discord 集成
- **pi-share-hf**：将编码会话发布到 Hugging Face，助力开源 AI 训练
- **RFC 系统**：[rfc.earendil.com](https://rfc.earendil.com/keyword/pi/) 记录长期规划
