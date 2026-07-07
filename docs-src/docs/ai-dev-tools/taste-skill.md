---
title: Taste Skill — 反 Slop 前端设计技能
icon: palette
order: 2
---

# Taste Skill — 反 Slop 前端设计技能

**Taste Skill** 是 Leonxlnx 开源的可移植 AI Agent 技能集合，专治 AI 生成的前端 UI "千篇一律"问题。提供代码实现技能和图片生成技能两大类，通过精确的设计约束让 AI 产出有品味的前端界面。MIT 协议。

- **GitHub**: https://github.com/Leonxlnx/taste-skill
- **官网**: https://tasteskill.dev
- **已获 Vercel 开源计划赞助**

---

## 一、核心问题：什么是 "Slop"？

AI 生成的前端代码往往有以下通病：

- 布局千篇一律（永远是居中卡片 + 两列网格）
- 字体/间距缺乏层次感
- 动画要么没有要么过度（千篇一律的 fade-in）
- 色彩单调（默认蓝 + 灰）
- 滥用 em-dash（—）、占位符注释

**Taste Skill 的目标**：通过结构化的设计约束，让 AI 在保持功能正确的前提下，产出有质感、有个性的前端代码。

---

## 二、技能矩阵

### 2.1 代码实现技能

| 技能 | 安装名 | 说明 |
|------|--------|------|
| **taste-skill** | `design-taste-frontend` | 🆕 v2 实验版——全面重写。推断设计语言，调节三个旋钮（VARIANCE/MOTION/DENSITY），禁止 em-dash，GSAP 骨架代码，重设计审计协议 |
| **taste-skill-v1** | `design-taste-frontend-v1` | 原版 v1，为依赖旧行为的项目保留 |
| **gpt-taste** | `gpt-taste` | GPT/Codex 严格版：更高布局变化、更强 GSAP 指令、激进反 Slop |
| **image-to-code** | `image-to-code` | 图片优先管线：生成参考图 → 分析 → 代码实现 |
| **redesign** | `redesign-existing-projects` | 现有项目审计：先审查 UI，再修复布局/间距/层级/样式 |
| **soft-skill** | `high-end-visual-design` | 高端视觉：柔和对比、大量留白、高级字体、弹簧动画 |
| **output** | `full-output-enforcement` | 输出强制：禁止占位注释，必须完整输出 |
| **minimalist** | `minimalist-ui` | 极简编辑风（Notion/Linear）：克制调色板、清晰结构 |
| **brutalist** | `industrial-brutalist-ui` | 工业粗野风：瑞士字体、锋利对比、实验性布局 |
| **stitch** | `stitch-design-taste` | Google Stitch 兼容规则，含可选 DESIGN.md 导出格式 |

### 2.2 图片生成技能（仅出图，不写代码）

| 技能 | 安装名 | 说明 |
|------|--------|------|
| **imagegen-web** | `imagegen-frontend-web` | 网页合成图：Hero、着陆页、多区块 |
| **imagegen-mobile** | `imagegen-frontend-mobile` | 移动端界面：iOS/Android/跨平台 |
| **brandkit** | `brandkit` | 品牌套件板：Logo 方向、调色板、字体、应用场景 |

---

## 三、安装与使用

```bash
# 一键安装所有技能
npx skills add https://github.com/Leonxlnx/taste-skill

# 安装单个技能（使用安装名）
npx skills add https://github.com/Leonxlnx/taste-skill --skill "design-taste-frontend"
```

支持 Claude Code、Codex、Cursor、ChatGPT 等任意支持 SKILL.md 的智能体。

---

## 四、v2 设计调参

taste-skill v2 引入三个 **1-10 刻度旋钮**，写在文件顶部：

```markdown
DESIGN_VARIANCE: 7   # 布局实验性（低→居中干净 / 高→不对称现代）
MOTION_INTENSITY: 5  # 动画深度（低→hover / 高→滚动磁性）
VISUAL_DENSITY: 4    # 信息密度（低→宽敞 / 高→密集仪表盘）
```

模型读取这些值后自动调节视觉方向。这是 v2 相比 v1 最大的创新。

---

## 五、图片优先工作流

推荐的"设计→代码"管线：

```
imagegen-frontend-web/mobile/brandkit
  → ChatGPT Images 生成参考图
    → image-to-code 技能
      → 分析参考图 → 代码实现
```

所有图片技能和代码技能通过同一个 `npx skills add` 安装，无需额外配置。

---

## 六、技能选择指南

| 场景 | 推荐技能 |
|------|----------|
| 一般前端开发 | `design-taste-frontend`（v2）|
| 严格代码风格 | `gpt-taste` |
| 图片→代码 | `image-to-code` |
| 改造现有项目 | `redesign-existing-projects` |
| 追求设计品质 | `high-end-visual-design` |
| AI 截断输出 | `full-output-enforcement` |
| 特定风格 | `minimalist-ui` / `industrial-brutalist-ui` |
| 纯出图 | `imagegen-frontend-web/mobile/brandkit` |

---

## 七、研究基础

Taste Skill 的约束规则基于专门的设计研究（见仓库 `research/` 目录），不是凭感觉写的 prompt。核心方法论：

1. **禁止通用模式**：显式排除特定 CSS 组合（如 `display:flex; justify-content:center` 的默认套路）
2. **强制差异化**：通过 VARIANCE 参数打破对称布局
3. **动画质量门禁**：GSAP 骨架代码 > CSS transition，不达标重做
4. **排版约束**：禁止 em-dash，指定字体分层规则
