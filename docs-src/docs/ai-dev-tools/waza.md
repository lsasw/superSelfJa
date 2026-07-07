---
title: Waza — AI 工程习惯技能集
icon: tool
order: 1
---

# Waza — AI 工程习惯技能集

**Waza**（技）是 tw93 开源的 AI Agent 技能集合，将工程师的日常习惯封装为可被 AI 智能体执行的 **Skill**。通过 `/` 命令触发，覆盖需求分析、UI 设计、代码审查、Bug 定位、写作润色、技术调研等全流程。MIT 协议。

- **GitHub**: https://github.com/tw93/Waza
- **作者**: tw93
- **名字由来**: 日语"技 (わざ)"——练到成为本能的技巧

---

## 一、设计哲学

> 好的工程师不止会写代码。他们会质疑需求、排查到根因、审查自己的改动、阅读原始资料。AI 有原始输出能力，但缺少结构化流程，输出就容易发散。每个 Waza Skill 设立清晰目标和约束，然后退后一步让模型工作——模型越强，这种克制越有价值。

Waza 是"三部曲"之一：
- **Kaku**（書く）：写代码
- **Waza**（技）：磨习惯 ← 此项目
- **Kami**（紙）：出文档

---

## 二、8 大技能速览

| Skill | 触发 | 职责 |
|:------|:-----|:-----|
| `/think` | 构建新功能前 | 质疑问题、压测设计、产出决策完备计划 |
| `/ui` | 前端界面开发 | 出品独特 UI，含截图驱动美学迭代 |
| `/check` | 任务完成/合并前 | 审查 diff、提取项目约束、验证证据 |
| `/hunt` | Bug/回归/意外行为 | 系统化调试，确认根因后才修复 |
| `/write` | 写作或编辑文稿 | 中英文自然重写，消除僵硬公式化措辞 |
| `/learn` | 深入陌生领域 | 六阶段调研：收集→消化→提纲→填充→精炼→自审 |
| `/read` | 读取 URL 或 PDF | 平台路由读取，输出简洁摘要或 Markdown |
| `/health` | 审计 Agent 健康 | 检查配置、指令、输出质量，预算感知摘要 |

---

## 三、一键安装

```bash
npx skills add tw93/Waza -a claude-code codex cursor -g -y
```

安装后在 `~/.agents/skills` 生成一份规范副本，Claude Code 通过 symlink 引用。Codex、Cursor、Kimi Code CLI、Amp、Cline 等所有读取该目录的智能体均自动拾取。

**原生插件安装**：

```bash
# Claude Code
/plugin marketplace add tw93/Waza
/plugin install waza@waza

# Codex
codex plugin marketplace add tw93/Waza
codex plugin add waza@waza
```

---

## 四、技能链编排

技能之间可串联，但每次切换需要手动触发——每个技能完成任务后停止，等待你的下一步决策。

**常见工作流**：

| 场景 | 链路 |
|------|------|
| 规划功能 | `/think` → 审批 → "implement X" → `/check` → merge |
| 修复发布 | `/hunt` → fix → `/check` → release/publish |
| 调研写作 | `/read`（抓源）→ `/learn`（合成）→ `/write`（润色） |
| 调试验证 | `/hunt`（找根因）→ fix → `/check`（审变更） |

---

## 五、项目上下文感知

`/check` 技能具备运行时项目感知能力——它会读取目标仓库的公开上下文（README、package.json、Makefile、CI 工作流）以及你的任务约束，**从不触碰私有路径、凭证或 token**。

---

## 六、附加功能

### 6.1 Statusline

极简状态栏：上下文窗口 + 5 小时配额 + 7 天配额。颜色编码用量，无进度条噪音。

```bash
curl -sL https://github.com/tw93/Waza/releases/latest/download/setup-statusline.sh | bash
```

Codex 原生支持 statusline 配置：

```toml
[tui]
status_line = ["model-with-reasoning", "current-dir", "context-used", "five-hour-limit", "weekly-limit"]
status_line_use_colors = true
```

### 6.2 可选规则

三个独立开关，按需启用：

```bash
# 英文教练：发现英文错误时附加纠正
curl -sL https://github.com/tw93/Waza/releases/latest/download/setup-rule.sh | bash -s -- english claude-code

# 反模式守卫：始终在线的跨技能护栏
curl -sL https://github.com/tw93/Waza/releases/latest/download/setup-rule.sh | bash -s -- anti-patterns claude-code

# 路由提示：告知非 Claude 主机优先使用 Waza Skills
curl -sL https://github.com/tw93/Waza/releases/latest/download/setup-rule.sh | bash -s -- waza-routing claude-code
```

---

## 七、设计约束

- 仅 8 个技能，每个单一职责
- 每个技能 = 文件夹（SKILL.md + references/ + helper scripts + 实战踩坑记录）
- 所有 gotcha 来源于真实失败案例（300+ 会话、7 个项目实战打磨）
- 不与 Superpowers/gstack 竞争——它们更重、更多配置；Waza 更轻、更聚焦
