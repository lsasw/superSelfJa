---
title: Pretext — 高性能文本测量与布局
icon: text-height
order: 1
---

# Pretext — 高性能文本测量与布局

**Pretext** 是 chenglou 开源的纯 JavaScript/TypeScript 库，用于多行文本测量与布局。核心特点是**绕开 DOM 测量、零 layout reflow**，支持几乎所有语言（CJK、阿拉伯语等）。可渲染到 DOM、Canvas、SVG，MIT 协议。

- **GitHub**: https://github.com/chenglou/pretext
- **npm**: `@chenglou/pretext`
- **Demo**: https://chenglou.me/pretext/

---

## 一、为什么绕开 DOM？

浏览器中测量文本的传统方式（`getBoundingClientRect`、`offsetHeight`）会触发 **layout reflow**——浏览器最昂贵的操作之一。Pretext 实现了一套独立的文本测量引擎：

```
传统方式： DOM 渲染 → 测量 → reflow → 测量 → reflow → ...
Pretext：  Canvas measureText → 纯算术布局 → 一次渲染
```

**关键公式**：

```
prepare(text, font)       ← 一次性预计算（重）
layout(prepared, w, h)    ← 纯算术热路径（轻）
```

不要对相同文本重复 `prepare()`——窗口 resize 时只需重新 `layout()`。

---

## 二、核心 API

### 2.1 纯测量路径（无需 DOM）

```typescript
import { prepare, layout } from '@chenglou/pretext'

const prepared = prepare('AGI 春天到了. بدأت الرحلة 🚀‎', '16px Inter')
const { height, lineCount } = layout(prepared, 320, 20)
// height: 段落实际像素高度
// lineCount: 换行后的行数
```

`prepare()` 选项：

```typescript
{
  whiteSpace?: 'normal' | 'pre-wrap',  // pre-wrap 保留 \t 和 \n
  wordBreak?: 'normal' | 'keep-all',   // keep-all 用于 CJK/Hangul
  letterSpacing?: number                // 匹配 CSS letter-spacing
}
```

### 2.2 行级布局（手动渲染）

```typescript
import { prepareWithSegments, layoutWithLines } from '@chenglou/pretext'

const prepared = prepareWithSegments('AGI 春天到了', '18px "Helvetica Neue"')
const { lines } = layoutWithLines(prepared, 320, 26)

// 渲染到 Canvas
for (let i = 0; i < lines.length; i++) {
  ctx.fillText(lines[i].text, 0, i * 26)
}
```

### 2.3 变宽布局（文本环绕图片）

```typescript
import { layoutNextLineRange, materializeLineRange, type LayoutCursor } from '@chenglou/pretext'

let cursor: LayoutCursor = { segmentIndex: 0, graphemeIndex: 0 }
let y = 0

while (true) {
  const width = y < imageBottom ? columnWidth - imageWidth : columnWidth
  const range = layoutNextLineRange(prepared, cursor, width)
  if (range === null) break

  const line = materializeLineRange(prepared, range)
  ctx.fillText(line.text, 0, y)
  cursor = range.end
  y += lineHeight
}
```

### 2.4 高级 API 速览

| 函数 | 说明 |
|------|------|
| `measureLineStats(prepared, maxWidth)` | 仅返回 `{ lineCount, maxLineWidth }`，无字符串分配 |
| `walkLineRanges(prepared, maxWidth, onLine)` | 遍历每行范围和宽度，适合二分搜索最佳宽度 |
| `measureNaturalWidth(prepared)` | 最宽强制行宽度 |
| `layoutNextLineRange(prepared, start, maxWidth)` | 迭代器式 API，每行可使用不同宽度 |

---

## 三、Rich Inline Flow

`@chenglou/pretext/rich-inline` 模块支持富文本内联流：code spans、mentions、chips、边界空白折叠。

```typescript
import { prepareRichInline, materializeRichInlineLineRange } from '@chenglou/pretext/rich-inline'

const prepared = prepareRichInline([
  { text: 'Ship ', font: '500 17px Inter' },
  { text: '@maya', font: '700 12px Inter', break: 'never', extraWidth: 22 },
  { text: "'s rich-note", font: '500 17px Inter' },
])

// walkRichInlineLineRanges / materializeRichInlineLineRange 处理每行
```

**设计约束（刻意窄范围）**：
- 仅处理原始内联文本（含边界空格）
- 调用者拥有 `extraWidth`（padding + border）
- `break: 'never'` 用于原子项（如 chip）
- **不是**嵌套标记树，**不是**通用 CSS 内联格式化引擎

---

## 四、应用场景

| 场景 | 解决方案 |
|------|----------|
| **虚拟列表** | 无需猜测和缓存即可实现正确虚拟化 |
| **Masonry 布局** | JS 驱动的自定义布局 |
| **AI 验证** | 无浏览器环境验证按钮标签是否溢出 |
| **防止布局偏移** | 新文本加载时重新锚定滚动位置 |
| **Canvas/SVG 渲染** | 完整文本布局引擎，无需 DOM |
| **服务端渲染**（计划中） | 无浏览器环境精确测量 |

---

## 五、技术栈

| 层面 | 技术 |
|------|------|
| 语言 | TypeScript |
| 测量引擎 | Canvas 2D `measureText`（浏览器字体引擎） |
| 文本分段 | `Intl.Segmenter`（grapheme 级别） |
| Bidi 算法 | 源自 pdf.js（`segLevels` 自定义 bidi 渲染） |
| 渲染目标 | DOM、Canvas、SVG、WebGL（未来：服务端） |
| 运行时要求 | `Intl.Segmenter` + Canvas 2D |

---

## 六、支持的 CSS 文本特性

| 特性 | 状态 |
|------|:----:|
| `white-space: normal / pre-wrap` | ✅ |
| `word-break: normal / keep-all` | ✅ |
| `overflow-wrap: break-word` | ✅（极窄宽度下 grapheme 断词） |
| `line-break: auto` | ✅ |
| `letter-spacing` | ✅（数字像素值传入） |
| `tab-size: 8` | ✅（默认浏览器风格） |
| Soft hyphen (`\u00AD`) | ✅（可选断点，选中显示 `-`） |

---

## 七、已知限制

- **`system-ui` 字体在 macOS 上 `layout()` 不准确**——使用具名字体
- Canvas `font` 简写之外的 CSS 特性不单独建模（如 `font-optical-sizing`）
- 段宽度用于换行而非精确 glyph-position 数据
- `prepare()` 仅做水平方向工作，`lineHeight` 是布局时输入

---

## 八、历史渊源

> Sebastian Markbage 十年前通过 [text-layout](https://github.com/chenglou/text-layout) 种下了这颗种子。他的设计——canvas `measureText` 做 shaping、bidi 来自 pdf.js、流式换行——奠定了 Pretext 的架构基础。

---

## 九、快速开发

```bash
bun install
bun start            # 打开 /demos/index
bun run start:windows
```
