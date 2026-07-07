---
title: Caveman — 极简输出压缩技能
icon: compress
order: 3
---

# Caveman — 极简输出压缩技能

**Caveman** 是 Julius Brussee 开源的 AI Agent 技能，让 AI 智能体用"原始人风格"回答——**平均减少 65% 输出 Token**，同时保持代码、命令、错误信息逐字不变。支持 Claude Code、Codex、Gemini、Cursor 等 30+ 种智能体。MIT 协议。

- **GitHub**: https://github.com/JuliusBrussee/caveman
- **理念**: "why use many token when few do trick"

---

## 一、效果对比

| 正常 Agent（69 token） | Caveman（19 token） |
|:-----------------------|:--------------------|
| "The reason your React component is re-rendering is likely because you're creating a new object reference on each render cycle..." | "New ref each render. Wrap object in `useMemo`." |

```
┌───────────────────────────────────────┐
│  输出 Token 节省     █████████   65%  │
│  技术准确率          █████████  100%  │
│  代码/命令/错误      █████████  不变  │
└───────────────────────────────────────┘
```

> Caveman 不减脑容量。Caveman 只减**嘴**。缩的是 Agent **说什么**，不是它**知道什么**。

---

## 二、工作原理

1. 安装脚本将 Skill 文件放入智能体配置目录
2. Skill 指示模型：去填充词、保留实质、用短句——但**永不触碰**代码、命令、错误信息
3. Claude Code 下，hook 写入标记文件，从第一条消息起就进入 caveman 模式
4. `/caveman-stats` 读取会话日志，统计 Token 节省量

---

## 三、六级压缩强度

| 级别 | 同一句话的不同输出 |
|------|-------------------|
| *正常 Agent* | You should wrap the object in `useMemo`, since a new reference is created on every render. |
| `lite` | Wrap object in `useMemo`. New ref created every render. |
| `full` *(默认)* | New ref each render. Wrap object in `useMemo`. |
| `ultra` | New ref/render. `useMemo` it. |
| `wenyan` | 每渲染新引用，故包裹于 `useMemo` — 文言文输出，Token 密度最高 |

切换命令：`/caveman lite|full|ultra|wenyan`

> 注意：Caveman 保留你的语言。写中文就压缩中文，写英文就压缩英文。`wenyan` 模式是刻意例外——文言文是单位 Token 信息密度最高的选择。

---

## 四、命令总览

| 命令 | 功能 |
|------|------|
| `/caveman [level]` | 压缩所有回复，级别持续整个会话 |
| `/caveman-commit` | Conventional Commit 格式，≤50 字符标题 |
| `/caveman-review` | 单行 PR 评论：`L42: 🔴 bug: user null. Add guard.` |
| `/caveman-stats` | 实时会话 Token 用量、累计节省、USD 估算 |
| `/caveman-compress <file>` | 压缩记忆文件（如 CLAUDE.md），每次会话永久减少 ~46% 输入 Token |

---

## 五、安装

```bash
# macOS / Linux / WSL
curl -fsSL https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.sh | bash

# Windows PowerShell
irm https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.ps1 | iex
```

约 30 秒，需要 Node ≥18。自动跳过未安装的智能体，可安全重复运行。安装后 `/caveman` 或说 "talk like caveman" 即开启。

---

## 六、实测 Benchmark

基于 Claude API 真实 Token 统计，10 个不同任务的输出对比：

| 任务 | 正常 | Caveman | 节省 |
|------|-----:|-------:|-----:|
| 解释 React 重渲染 Bug | 1180 | 159 | **87%** |
| 修复认证中间件 | 704 | 121 | **83%** |
| 搭建 PostgreSQL 连接池 | 2347 | 380 | **84%** |
| 解释 git rebase vs merge | 702 | 292 | 58% |
| Docker 多阶段构建 | 1042 | 290 | **72%** |
| 实现 React 错误边界 | 3454 | 456 | **87%** |
| **平均** | **1214** | **294** | **65%** |

> 诚实声明：Caveman 只压缩**输出** Token，输入和推理 Token 不变。Skill 本身每轮消耗约 1-1.5k 输入 Token。实际全会话节省低于 65%，读速提升才是核心收益。

---

## 七、caveman-compress：永久压缩记忆文件

`/caveman-compress <file>` 将记忆文件（CLAUDE.md、project-notes.md 等）转换为 caveman 风格：

| 文件 | 原始 | 压缩后 | 节省 |
|---|---:|---:|---:|
| claude-md-preferences.md | 706 | 285 | **59.6%** |
| project-notes.md | 1145 | 535 | **53.3%** |
| claude-md-project.md | 1122 | 636 | **43.3%** |
| **平均** | **898** | **481** | **46%** |

每次会话该文件加载少 ~46%，永久受益。

---

## 八、生态矩阵

Caveman 是"压缩全家桶"的首发项目：

| 仓库 | 压缩什么 |
|------|---------|
| [**caveman**](https://github.com/JuliusBrussee/caveman) | Agent **说**的话 |
| [**caveman-code**](https://github.com/JuliusBrussee/caveman-code) | **整个** Agent 端到端 |
| [**cavemem**](https://github.com/JuliusBrussee/cavemem) | Agent **记住**的东西 |
| [**cavekit**](https://github.com/JuliusBrussee/cavekit) | **构建循环**——spec 驱动 |
| [**cavegemma**](https://github.com/JuliusBrussee/finetune-caveman) | 压缩**写入权重**（Gemma 微调） |

另有 5 个姐妹技能（grill-me、interface-kit、junior-to-senior、loop-factory）通过 `npx skills add JuliusBrussee/skills` 一键安装。

---

## 九、隐私

Caveman **零网络回传**——无遥测、无分析、无账号、无后端。安装后所有操作均为本地：Skill 是 prompt 文件，hook 是本地脚本，`/caveman-stats` 读取磁盘上已有的日志。

---

## 十、学术支撑

2026 年 3 月的论文 [*Brevity Constraints Reverse Performance Hierarchies in Language Models*](https://arxiv.org/abs/2604.00025) 测试了 31 个模型，发现约束大模型简短回答能**在某些 benchmark 上提升 ~26 分准确率**。少说话有时 = 更正确。
