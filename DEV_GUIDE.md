# superSelfJa 开发手册

> 版本：1.0 | 最后更新：2026-07-07 | 维护者：七智

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术栈](#2-技术栈)
- [3. 快速开始](#3-快速开始)
- [4. 项目结构详解](#4-项目结构详解)
- [5. 配置文件说明](#5-配置文件说明)
- [6. 开发工作流](#6-开发工作流)
- [7. 内容组织规范](#7-内容组织规范)
- [8. CI/CD 自动部署](#8-cicd-自动部署)
- [9. 主题定制与品牌](#9-主题定制与品牌)
- [10. 已知问题与故障排除](#10-已知问题与故障排除)
- [11. 贡献指南](#11-贡献指南)
- [12. 路线图与待办事项](#12-路线图与待办事项)

---

## 1. 项目概述

**superSelfJa**（超级个体技术文档）是一个综合性技术文档站点，覆盖 **Java 后端工程** 和 **AI 应用开发** 两大技术方向，旨在为技术人员提供从入门到前沿的体系化学习资源。

### 在线站点

| 项目 | 地址 |
|------|------|
| 在线文档 | [https://lsasw.github.io/superSelfJa/](https://lsasw.github.io/superSelfJa/) |
| 代码仓库 | [https://github.com/lsasw/superSelfJa](https://github.com/lsasw/superSelfJa) |

### 内容规模

- **208+** 篇技术文档
- **18** 个知识模块
- **3** 大技术方向（Java 后端 / AI 全栈 / 实战专题）
- **6** 个实战专题

### 内容板块

| 板块 | 模块 | 说明 |
|------|------|------|
| **Java 生态** | 设计模式 / Spring 核心 / Spring Boot / 数据库 / 并发与 JVM | 企业级后端开发完整技能树 |
| **AI 生态** | AI 基础 / 经典 ML / 深度学习 / PyTorch / LLM / AI 工程化 / AI 应用 | 从数学基础到大模型落地的全链路 |
| **实战专题** | Agent 工程化 / MCP 网关 / RAG 向量库 / 系统设计 / 团队建设 | 生产级实战能力 |

---

## 2. 技术栈

| 组件 | 版本 | 说明 |
|------|------|------|
| [VuePress](https://vuepress.vuejs.org/) | `2.0.0-rc.14` | 静态站点生成器 |
| [Theme Hope](https://theme-hope.vuejs.press/) | `2.0.0-rc.52` | 文档主题 |
| [@vuepress/bundler-vite](https://vitejs.dev/) | `2.0.0-rc.14` | Vite 构建工具 |
| [Sass](https://sass-lang.com/) | `^1.100.0` | CSS 预处理器 |
| [TypeScript](https://www.typescriptlang.org/) | — | 配置文件语言 |
| [search-pro](https://plugin-search-pro.vuejs.press/) | `^2.0.0-rc.59` | 全文搜索（已安装但暂未启用） |

### 环境要求

| 依赖 | 最低版本 | 推荐版本 |
|------|----------|----------|
| [Node.js](https://nodejs.org/) | 18.x | 20.x LTS |
| [npm](https://www.npmjs.com/) | 9.x | 10.x |
| [Git](https://git-scm.com/) | 2.x | latest |

---

## 3. 快速开始

### 3.1 克隆仓库

```bash
git clone https://github.com/lsasw/superSelfJa.git
cd superSelfJa
```

### 3.2 安装依赖

> **⚠️ 重要**：必须使用 `--legacy-peer-deps` 参数，否则 peer dependency 冲突会导致安装失败。

```bash
npm install --legacy-peer-deps
```

### 3.3 启动开发服务器

```bash
npm run docs:dev
```

浏览器访问 [http://localhost:8080](http://localhost:8080) 即可预览站点。

开发服务器支持 **热重载（HMR）**——修改 Markdown 或配置后页面自动刷新，无需手动重启。

### 3.4 生产构建

```bash
npm run docs:build
```

构建产物输出到 `docs-src/.vuepress/dist/`。

### 3.5 本地预览生产构建

```bash
npm run docs:serve
```

### 命令速查表

| 命令 | 说明 |
|------|------|
| `npm install --legacy-peer-deps` | 安装项目依赖 |
| `npm run docs:dev` | 启动开发服务器（HMR） |
| `npm run docs:build` | 生产构建 |
| `npm run docs:serve` | 本地预览生产构建 |

---

## 4. 项目结构详解

```
superSelfJa/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD 部署配置
├── docs-src/                   # 📁 文档源码目录（核心工作区）
│   ├── README.md               # 首页（Hero 区域 + 特性展示 + 快速导航）
│   ├── .vuepress/              # VuePress 配置目录
│   │   ├── config.ts           # VuePress 主配置（语言、标题、Markdown 增强）
│   │   ├── theme.ts            # Theme Hope 主题配置（导航栏、侧边栏、插件）
│   │   ├── .cache/             # 构建缓存（可安全删除）
│   │   └── .temp/              # 临时文件（可安全删除）
│   └── docs/                   # 📚 文档内容目录
│       ├── README.md           # 文档中心首页（模块速览 + 目录）
│       ├── java-patterns/      # Java 设计模式（3 篇）
│       ├── spring-core/        # Spring 核心（3 篇）
│       ├── spring-boot/        # Spring Boot（1 篇）
│       ├── database/           # 数据库（6 篇）
│       ├── java-concurrency-jvm/ # Java 并发与 JVM（3 篇）
│       ├── middleware/         # 中间件（待填充）
│       ├── microservices/      # 微服务（待填充）
│       ├── ai-foundations/     # AI 基础（15 篇）
│       ├── ai-ml-classic/      # 经典机器学习（10 篇）
│       ├── ai-dl-fundamentals/ # 深度学习基础（15 篇）
│       ├── ai-pytorch/         # PyTorch 实战（15 篇）
│       ├── ai-llm/            # 大语言模型（15 篇）
│       ├── ai-engineering/     # AI 工程化（14 篇）
│       ├── ai-applications/    # AI 应用开发（10 篇）
│       ├── ai-agent-engineering/ # AI Agent 工程化（5 篇）
│       ├── mcp-gateway/        # MCP 协议与网关（3 篇）
│       ├── rag-vector-db/      # RAG 与向量数据库（3 篇）
│       ├── system-design/      # 系统设计（1 篇）
│       └── team-process/       # 团队建设（1 篇）
├── package.json                # 项目配置（scripts + 依赖）
├── package-lock.json           # 依赖锁定文件
├── tsconfig.json               # TypeScript 配置
├── .gitignore                  # Git 忽略规则
├── AGENTS.md                   # AI Agent 辅助开发指引
├── CLAUDE.md                   # Claude Code 辅助开发指引
├── DEV_GUIDE.md                # 本文档
└── BUG_REPORT.md               # 历史调试与修复报告
```

### 核心文件职责

| 文件 | 职责 | 何时修改 |
|------|------|----------|
| `config.ts` | 站点元信息、语言、favicon、Open Graph | 修改站点标题/描述/SEO 时 |
| `theme.ts` | 导航栏、侧边栏、插件、页脚、作者信息 | 添加新模块、调整侧边栏、启用/禁用插件时 |
| `docs-src/README.md` | 站点首页 | 更新首页展示内容时 |
| `docs-src/docs/README.md` | 文档中心首页 | 调整模块速览表时 |
| `.github/workflows/deploy.yml` | CI/CD 部署 | 修改部署流程时 |

---

## 5. 配置文件说明

### 5.1 VuePress 配置（config.ts）

```typescript
// 文件位置：docs-src/.vuepress/config.ts
export default defineUserConfig({
  base: "/superSelfJa/",         // 部署基础路径（GitHub Pages 子路径）
  lang: "zh-CN",                 // 站点语言
  title: "超级个体技术文档",      // 站点标题
  description: "Java 后端工程 × AI 应用开发...",  // SEO 描述
  head: [/* favicon、OG 标签等 */],
  bundler: viteBundler(),        // Vite 构建器
  markdown: {
    importCode: true,            // 启用代码导入语法 @[code]
  },
});
```

**关键配置项**：

- `base` — 设为 `"/superSelfJa/"` 适配 GitHub Pages 子路径部署。本地开发时 VuePress 会自动处理。
- `lang: "zh-CN"` — 全局语言设置，影响主题内置文本。
- `head` — 配置了 favicon（多尺寸）、Apple Touch Icon、Web App Manifest、Theme Color（`#2563eb`）和 Open Graph 标签。

### 5.2 主题配置（theme.ts）

主题配置采用 **三套独立侧边栏 + 路径前缀映射** 的架构，实现"点击哪个导航，就显示哪套侧边栏"。

```typescript
// 文件位置：docs-src/.vuepress/theme.ts
export default hopeTheme({
  hostname: "https://lsasw.github.io",
  author: { name: "lsasw", url: "https://github.com/lsasw", avatar: "/avatar-lg.png" },
  repo: "https://github.com/lsasw/superSelfJa",
  docsDir: "docs-src",

  navbar: [
    // 顶部导航栏：首页 | 文档中心 | Java 生态 | AI 生态 | 实战专题 | 面试题库
  ],

  sidebar: {
    // 路径前缀 → 侧边栏映射
    "/docs/java-patterns/": javaSidebar,       // Java 生态各模块共享同一套侧边栏
    "/docs/ai-foundations/": aiSidebar,         // AI 生态各模块共享同一套侧边栏
    "/docs/ai-agent-engineering/": practicalSidebar, // 实战专题各模块共享同一套侧边栏
    "/docs/ai-applications/101-ai-interview-qa": [/* 面试题库独立侧边栏 */],
  },

  plugins: {
    copyCode: true,       // 代码块复制按钮
    photoSwipe: true,     // 图片点击放大
    components: {
      components: ["Badge", "BiliBili", "CodePen", "FontIcon", "PDF", "StackBlitz"],
    },
  },
});
```

### 5.3 侧边栏架构

```
┌─────────────────────────────────────────────────────┐
│                   顶部导航栏                          │
│  [首页] [文档中心] [Java生态▼] [AI生态▼] [实战▼] [面试] │
├─────────────────────────────────────────────────────┤
│  侧边栏自动切换                                       │
│  ┌──────────┬───────────┬──────────────┐            │
│  │ Java生态  │  AI生态    │  实战专题     │            │
│  │ 侧边栏    │  侧边栏    │  侧边栏       │            │
│  │          │           │              │            │
│  │ 设计模式  │  AI基础    │  Agent工程   │            │
│  │ Spring   │  经典ML    │  MCP网关     │            │
│  │ 数据库    │  深度学习   │  RAG向量库   │            │
│  │ 并发JVM  │  PyTorch   │  系统设计     │            │
│  │          │  LLM       │  团队建设     │            │
│  │          │  AI工程化  │              │            │
│  │          │  AI应用    │              │            │
│  └──────────┴───────────┴──────────────┘            │
└─────────────────────────────────────────────────────┘
```

---

## 6. 开发工作流

### 6.1 添加新文档

**步骤**：

1. **创建 Markdown 文件** — 在 `docs-src/docs/` 下对应模块目录创建 `.md` 文件

2. **编写 frontmatter** — 文件顶部添加 YAML 头：

   ```yaml
   ---
   title: 文档标题
   icon: book
   order: 1
   ---

   # 文档内容
   ```

3. **更新侧边栏** — 在 `theme.ts` 中找到对应模块的侧边栏配置，添加新条目：

   ```typescript
   { text: "新文档标题", icon: "相应的图标", link: "/docs/模块名/文件名" }
   ```

4. **更新导航栏**（如需要）— 如果是全新模块，在 `theme.ts` 的 `navbar` 中添加下拉项

5. **更新文档首页** — 在 `docs-src/docs/README.md` 的模块速览表中更新统计信息

6. **本地验证** — 运行 `npm run docs:dev`，确认新页面可访问、侧边栏导航正确

### 6.2 添加全新模块

相较于添加单篇文档，全新模块需要更多步骤：

1. 在 `docs-src/docs/` 下创建新目录
2. 创建文档文件（按编号前缀命名以保持顺序）
3. 在 `theme.ts` 中新增侧边栏配置分组
4. 在 `theme.ts` 的 `navbar` 中添加导航入口
5. 在 `theme.ts` 的 `sidebar` 中添加路径前缀映射
6. 在 `docs-src/docs/README.md` 中添加模块说明行
7. 在 `docs-src/README.md` 中更新首页内容

### 6.3 编辑现有文档

直接修改 `docs-src/docs/` 下对应 `.md` 文件即可。开发服务器会自动热重载。

### 6.4 调试技巧

| 问题 | 解决方案 |
|------|----------|
| 侧边栏链接 404 | 检查 `theme.ts` 中 link 路径与实际文件名是否一致 |
| 新页面不出现在导航 | 确认已在 `sidebar` 和 `navbar` 中添加对应条目 |
| 构建/开发服务器卡死 | 删除 `.vuepress/.cache` 和 `.vuepress/.temp` 后重试 |
| 样式不生效 | 检查 Sass 版本兼容性，确保使用 `^1.100.0` |
| 端口冲突 | VuePress 会自动尝试下一个端口，或手动指定 `--port 9090` |

### 6.5 AI 辅助开发

项目根目录提供了两份 AI 编程助手配置文件：

- **AGENTS.md** — 供 CodeBuddy Agent 使用
- **CLAUDE.md** — 供 Claude Code 使用

两份文件内容基本一致，包含项目概述、技术栈、常用命令、结构说明和配置要点。使用时，AI 编程助手会自动读取这些文件来理解项目上下文。

---

## 7. 内容组织规范

### 7.1 文件命名

| 规范 | 示例 |
|------|------|
| 英文模块使用英文文件名 | `creation.md`、`ioc.md`、`auto-config.md` |
| AI 基础模块使用英文文件名 | `01-ai-overview.md`、`02-machine-learning-basics.md` |
| 经典 ML 模块使用中文文件名 | `16-线性回归.md`、`17-逻辑回归.md` |
| 使用数字前缀控制排序 | `01-` 到 `101-` |
| 目录首页使用 `README.md` | `docs/java-patterns/README.md` |

### 7.2 Frontmatter 规范

```yaml
---
title: 文档标题          # 必填，显示在页面标题和侧边栏
icon: icon-name          # 可选，FontAwesome 图标名
order: 1                 # 可选，控制页面排序
---
```

### 7.3 Markdown 写作规范

- 使用 `#` 到 `####` 的标题层级，避免过深层级
- 代码块必须指定语言：```` ```java ````、```` ```python ````、```` ```bash ````
- 代码导入功能可用：`@[code](../../path/to/file.ts)` （需启用 `markdown.importCode`）
- 图片放在对应模块目录或 `.vuepress/public/` 下
- 任务列表使用 `- [ ]` 和 `- [x]` 语法（需启用 `markdown.tasklist`）
- 表格用于对比和速览型内容
- 数学公式使用 KaTeX 语法（Theme Hope 内置支持）

### 7.4 可用的自定义组件

Theme Hope 内置组件可在 Markdown 中直接使用：

| 组件 | 用法 | 用途 |
|------|------|------|
| `Badge` | `<Badge text="新" type="tip" />` | 标记状态/标签 |
| `BiliBili` | `<BiliBili bvid="xxx" />` | 嵌入 B 站视频 |
| `CodePen` | `<CodePen user="xxx" hash="xxx" />` | 嵌入 CodePen |
| `FontIcon` | `<FontIcon icon="github" />` | 内联图标 |
| `PDF` | `<PDF url="/demo.pdf" />` | 嵌入 PDF |
| `StackBlitz` | `<StackBlitz id="xxx" />` | 嵌入 StackBlitz |

### 7.5 孤立目录说明

以下目录存在于 `docs-src/docs/` 下但不在导航中（历史遗留，可通过直接 URL 访问）：

- `ml-classic/` — 经典 ML 英文版（与 `ai-ml-classic/` 内容重叠）
- `dl-fundamentals/` — 深度学习英文版（与 `ai-dl-fundamentals/` 内容重叠）
- `llm-tech/` — LLM 技术英文版（与 `ai-llm/` 内容重叠）
- `pytorch-practice/` — PyTorch 实践英文版（与 `ai-pytorch/` 内容重叠）
- `ai-deep-learning/` — AI 深度学习进阶（与已有模块内容重叠）
- `ai-development/` — AI 开发实践（与 AI 工程化/Agent 内容重叠）
- `microservices/` — 微服务（仅 README 占位）
- `middleware/` — 中间件（仅 README 占位）

> **计划**：这些目录最终需要整合或删除，在此之前请勿向这些目录添加新内容。

---

## 8. CI/CD 自动部署

### 8.1 工作流概览

项目使用 **GitHub Actions** 实现持续部署，配置文件位于 `.github/workflows/deploy.yml`。

```
push to main / 手动触发
        │
        ▼
   ┌─────────────┐
   │   Checkout   │  Node 20
   └──────┬──────┘
          ▼
   ┌──────────────┐
   │ npm install  │  --legacy-peer-deps
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ docs:build   │  VuePress 生产构建
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Upload Pages │  上传到 GitHub Pages Artifact
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  Deploy      │  部署到 GitHub Pages
   └──────────────┘
```

### 8.2 触发条件

- **自动触发**：push 到 `main` 分支
- **手动触发**：在 GitHub Actions 页面点击 "Run workflow"

### 8.3 部署地址

部署到 GitHub Pages 默认域名：`https://lsasw.github.io/superSelfJa/`

> **注意**：`config.ts` 中 `base` 配置为 `"/superSelfJa/"`，与 GitHub Pages 子路径部署匹配。如需切换到自定义域名，需要同步修改 `base` 配置。

---

## 9. 主题定制与品牌

### 9.1 品牌色

| 色值 | 用途 |
|------|------|
| `#2563eb`（蓝色） | 主题色（Theme Color / MS Tile Color） |
| 自动 | 暗色模式配色由 Theme Hope 自动生成 |

### 9.2 图标资源

| 文件 | 位置 | 用途 |
|------|------|------|
| `favicon.ico` | `.vuepress/public/` | 浏览器标签页图标 |
| `favicon-192.png` | `.vuepress/public/` | PWA 安装图标 |
| `favicon.svg` | `.vuepress/public/` | SVG 矢量图标 |
| `apple-touch-icon.png` | `.vuepress/public/` | iOS 主屏幕图标 |
| `avatar-lg.png` | `.vuepress/public/` | Logo 和作者头像 |
| `site.webmanifest` | `.vuepress/public/` | PWA 配置清单 |
| `hero-banner.svg` | `.vuepress/public/assets/icon/` | 首页背景图 |

### 9.3 自定义样式

如需覆盖主题默认样式，可在 `docs-src/.vuepress/styles/` 下创建以下文件：

- `index.scss` — 全局样式覆盖
- `palette.scss` — 颜色变量覆盖（如 `$theme-color`）
- `config.scss` — 主题 SCSS 变量覆盖

---

## 10. 已知问题与故障排除

### 10.1 版本锁定

> **🔒 重要**：VuePress `2.0.0-rc.14` + Theme Hope `2.0.0-rc.52` 是经过验证的稳定组合。**不要随意升级这两个包的版本**，新版本的 API 可能有破坏性变更。

### 10.2 search-pro 插件问题

`vuepress-plugin-search-pro` 已安装但**暂时禁用**：

- **问题**：启用后开发服务器和生产构建均卡死在 "Initializing and preparing data"
- **状态**：已在 `theme.ts` 中注释掉
- **替代方案**：建议后续评估 `@vuepress/plugin-docsearch` 或仅依赖侧边栏导航

### 10.3 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `npm install` 失败 | peer dependency 冲突 | 使用 `npm install --legacy-peer-deps` |
| 构建后页面空白 | `base` 路径不匹配 | 确认 `config.ts` 中 `base` 与部署路径一致 |
| 侧边栏链接 404 | 文件名与 link 不匹配 | 检查中文/英文文件名差异（`ai-ml-classic/` 的链接已从英文修正为中文） |
| 构建/Dev 卡死 | search-pro 插件问题或缓存损坏 | 删除 `.cache` 和 `.temp`，确认 search-pro 已禁用 |
| Hot Reload 不生效 | Windows 文件系统监听限制 | 手动刷新浏览器 |

### 10.4 历史修复记录

详细的修复历史见 [BUG_REPORT.md](./BUG_REPORT.md)，包含：
- ai-ml-classic 侧边栏断链修复
- docs/README.md 批量断链修复
- ai-engineering 文件编号冲突修复
- 废弃组件移除
- 交叉引用修复

---

## 11. 贡献指南

欢迎贡献！请遵循以下流程：

### 11.1 贡献类型

| 类型 | 说明 |
|------|------|
| 📝 **新增文档** | 在已有模块中添加新的技术文档 |
| 🆕 **新增模块** | 创建全新的技术板块（需先讨论大纲） |
| 🐛 **修复错误** | 修正文档中的错误信息、断链、过期内容 |
| 🎨 **优化体验** | 改善站点导航、样式、功能插件 |
| 📊 **更新示例** | 更新代码示例、版本信息、最佳实践 |

### 11.2 贡献流程

1. **Fork 仓库** — 从 [lsasw/superSelfJa](https://github.com/lsasw/superSelfJa) fork
2. **创建分支** — 使用语义化分支名：`feat/xxx`、`fix/xxx`、`docs/xxx`
3. **本地开发** — `npm install --legacy-peer-deps` → `npm run docs:dev`
4. **验证** — 确认所有修改的页面可访问，侧边栏导航正确
5. **提交** — 使用清晰的 commit message
6. **Pull Request** — 提交到 main 分支

### 11.3 文档写作标准

- 使用简体中文（代码注释和变量名可用英文）
- 每个主题至少包含：**概念解释 → 核心原理 → 代码示例 → 最佳实践**
- 代码示例确保可运行
- 引用外部资料时添加链接
- 配合图表（Mermaid / 图片）提升可读性

### 11.4 提交信息规范

```
<type>(<scope>): <subject>

示例：
docs(spring-core): 新增 IoC 容器源码分析章节
fix(theme): 修复 ai-ml-classic 侧边栏断链
feat(rag): 新增 RAG 端到端实现文档
```

---

## 12. 路线图与待办事项

### 短期（近期计划）

- [ ] 整合孤立目录（`ml-classic/`、`dl-fundamentals/`、`llm-tech/` 等）
- [ ] 填充 `microservices/` 和 `middleware/` 模块内容
- [ ] 解决 search-pro 插件卡死问题，或切换到替代搜索方案
- [ ] 补充 ai-llm、ai-dl-fundamentals、ai-pytorch 模块的占位文档内容

### 中期（规划中）

- [ ] 新增 Spring Cloud 微服务系列文档
- [ ] 新增 MLOps 实践系列
- [ ] 新增 Fine-tuning 实战教程
- [ ] 添加多语言支持（英文版）
- [ ] 接入文档反馈与评论区（Giscus / Waline）

### 长期（展望）

- [ ] 视频教程配套（B 站嵌入）
- [ ] 在线交互式代码演示（CodePen / StackBlitz）
- [ ] 自动化文档健康检查（断链检测、时效性标记）
- [ ] 社区贡献者激励计划

---

> 📬 **反馈与建议**：欢迎通过 [GitHub Issues](https://github.com/lsasw/superSelfJa/issues) 提交问题或建议。
>
> 📄 **许可证**：MIT License | Copyright © 2024-2026 七智
