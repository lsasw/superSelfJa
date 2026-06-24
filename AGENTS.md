# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**superSelfJa** — Java 后端技术文档站点，基于 VuePress 2 + Theme Hope 构建，部署到 GitHub Pages。

- 在线地址：https://lsasw.github.io/superSelfJa/
- 仓库：https://github.com/lsasw/superSelfJa

## Tech Stack

| 组件 | 版本 | 说明 |
|------|------|------|
| VuePress | 2.0.0-rc.14 | 静态站点生成器 |
| Theme Hope | 2.0.0-rc.52 | 文档主题 |
| Bundler | @vuepress/bundler-vite rc.14 | Vite 构建 |
| Sass | ^1.100.0 | CSS 预处理器 |

**重要**：安装依赖时必须使用 `--legacy-peer-deps`，否则 peer dependency 冲突会导致安装失败。

## 常用命令

```bash
npm install --legacy-peer-deps    # 安装依赖（必须加 --legacy-peer-deps）
npm run docs:dev                  # 本地开发服务器 http://localhost:8080
npm run docs:build                # 构建生产版本，输出到 docs-src/.vuepress/dist/
npm run docs:serve                # 预览生产构建
```

## 项目结构

```
superSelfJa/
├── package.json                  # 项目配置，包含 scripts 和依赖
├── tsconfig.json                 # TypeScript 配置
├── .gitignore                    # Git 忽略规则
├── .github/workflows/deploy.yml  # GitHub Actions CI/CD
└── docs-src/                     # 文档源码目录
    ├── README.md                 # 首页（Hero 区域 + 特性展示）
    ├── .vuepress/
    │   ├── config.ts             # VuePress 主配置（语言、标题、Markdown 增强）
    │   └── theme.ts              # Theme Hope 主题配置（导航栏、侧边栏、插件）
    └── docs/                     # 文档内容目录
        ├── README.md             # 文档目录页
        ├── java-patterns/        # Java 设计模式
        │   ├── creation.md       # 创建型模式
        │   ├── structural.md     # 结构型模式
        │   └── behavioral.md     # 行为型模式
        ├── spring-core/          # Spring 核心
        │   ├── ioc.md            # IoC 容器
        │   ├── aop.md            # AOP 切面
        │   └── transaction.md    # 事务管理
        ├── spring-boot/          # Spring Boot
        │   └── auto-config.md    # 自动配置原理
        └── database/             # 数据库
            ├── mysql.md          # MySQL 进阶
            ├── mongodb.md        # MongoDB 实战
            └── redis.md          # Redis 缓存
```

## 配置说明

### VuePress 配置（config.ts）

- `base: "/"` — 部署基础路径
- `lang: "zh-CN"` — 站点语言
- `bundler: viteBundler()` — 使用 Vite 构建
- `markdown.importCode: true` — 启用代码导入

### Theme Hope 配置（theme.ts）

- **navbar** — 顶部导航栏，按模块分组
- **sidebar** — 侧边栏，与导航栏结构对应
- **plugins** — 主题插件（photoSwipe 已启用）
- **markdown.tasklist: true** — 启用任务列表

## 添加新文档

1. 在 `docs-src/docs/` 下创建对应的目录和 `.md` 文件
2. 在 `theme.ts` 的 `navbar` 和 `sidebar` 中添加导航项
3. 在 `docs/README.md` 中添加目录链接

文档 frontmatter 示例：

```yaml
---
title: 文档标题
icon: book
order: 1
---

# 文档内容
```

## CI/CD 流程

GitHub Actions 工作流（`.github/workflows/deploy.yml`）：
- **触发**：push 到 main 分支或手动触发
- **Node 版本**：20
- **安装命令**：`npm install --legacy-peer-deps`
- **构建命令**：`npm run docs:build`
- **部署**：推送到 GitHub Pages

## 已知注意事项

1. **版本锁定**：VuePress rc.14 + Theme Hope rc.52 是经过验证的稳定组合，不要随意升级
2. **安装依赖**：必须使用 `--legacy-peer-deps` 参数
3. **清理缓存**：构建异常时可删除 `docs-src/.vuepress/.cache` 和 `docs-src/.vuepress/.temp`
4. **GitHub Pages**：使用默认域名 https://lsasw.github.io/superSelfJa/，无需配置自定义域名
