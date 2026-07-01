import { defineUserConfig } from "vuepress";
import theme from "./theme.js";
import { viteBundler } from "@vuepress/bundler-vite";

export default defineUserConfig({
  base: "/superSelfJa/",

  lang: "zh-CN",
  title: "超级个体技术文档",
  description: "Java 后端工程 × AI 应用开发 — 从设计模式到大模型，系统构建全栈技术体系",

  bundler: viteBundler(),

  theme,

  // 启用 Markdown 增强
  markdown: {
    importCode: true,
  },
});
