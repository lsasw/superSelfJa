---
title: 配置说明
icon: cog
order: 2
---

# 配置说明

## 站点配置

站点配置位于 `docs-src/.vuepress/config.ts`。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `base` | 部署基础路径 | `/` |
| `lang` | 站点语言 | `zh-CN` |
| `title` | 站点标题 | - |
| `description` | 站点描述 | - |

## 主题配置

主题配置位于 `docs-src/.vuepress/theme.ts`。

主要配置项：

- **navbar** — 导航栏链接
- **sidebar** — 侧边栏结构
- **plugins** — 主题插件
- **iconAssets** — 图标库

## 添加新页面

在 `docs-src/docs/` 目录下创建 `.md` 文件，并在 frontmatter 中配置：

```yaml
---
title: 页面标题
icon: book
order: 1
---
```
