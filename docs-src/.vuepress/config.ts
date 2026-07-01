import { defineUserConfig } from "vuepress";
import theme from "./theme.js";
import { viteBundler } from "@vuepress/bundler-vite";

export default defineUserConfig({
  base: "/superSelfJa/",

  lang: "zh-CN",
  title: "超级个体技术文档",
  description: "Java 后端工程 × AI 应用开发 — 从设计模式到大模型，系统构建全栈技术体系",

  head: [
    // Favicon
    ["link", { rel: "icon", href: "/superSelfJa/favicon.ico" }],
    ["link", { rel: "icon", type: "image/svg+xml", href: "/superSelfJa/favicon.svg" }],
    // Apple Touch Icon
    ["link", { rel: "apple-touch-icon", sizes: "180x180", href: "/superSelfJa/apple-touch-icon.png" }],
    // Web App Manifest
    ["link", { rel: "manifest", href: "/superSelfJa/site.webmanifest" }],
    // Theme Color
    ["meta", { name: "theme-color", content: "#2563eb" }],
    ["meta", { name: "msapplication-TileColor", content: "#2563eb" }],
    // Open Graph
    ["meta", { property: "og:title", content: "超级个体技术文档" }],
    ["meta", { property: "og:description", content: "Java 后端工程 × AI 应用开发 — 系统构建全栈技术体系" }],
    ["meta", { property: "og:image", content: "/superSelfJa/logo.svg" }],
  ],

  bundler: viteBundler(),

  theme,

  // 启用 Markdown 增强
  markdown: {
    importCode: true,
  },
});
