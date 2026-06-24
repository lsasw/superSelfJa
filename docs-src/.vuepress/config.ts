import { defineUserConfig } from "vuepress";
import theme from "./theme.js";
import { viteBundler } from "@vuepress/bundler-vite";

export default defineUserConfig({
  base: "/superSelfJa/",

  lang: "zh-CN",
  title: "文档中心",
  description: "基于 VuePress + Theme Hope 构建的文档站点",

  bundler: viteBundler(),

  theme,

  // 启用 Markdown 增强
  markdown: {
    importCode: true,
  },
});
