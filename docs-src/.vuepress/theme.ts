import { hopeTheme } from "vuepress-theme-hope";

export default hopeTheme({
  hostname: "https://your-domain.com",

  author: {
    name: "作者名称",
    url: "https://your-domain.com",
  },

  logo: "https://theme-hope-assets.vuejs.press/logo.svg",

  repo: "https://github.com/your-repo",

  docsDir: "docs-src",

  // 导航栏
  navbar: [
    "/",
    "/docs/",
    {
      text: "指南",
      icon: "book",
      prefix: "/docs/",
      children: [
        { text: "快速开始", icon: "rocket", link: "quick-start" },
        { text: "配置说明", icon: "cog", link: "config" },
      ],
    },
  ],

  // 侧边栏
  sidebar: {
    "/docs/": [
      "",
      {
        text: "入门指南",
        icon: "lightbulb",
        prefix: "",
        children: [
          { text: "快速开始", icon: "rocket", link: "quick-start" },
          { text: "配置说明", icon: "cog", link: "config" },
        ],
      },
    ],
  },

  footer: "基于 VuePress + Theme Hope 构建",

  displayFooter: true,

  // 页面元数据
  metaLocales: {
    editLink: "在 GitHub 上编辑此页",
  },

  // Markdown 增强
  markdown: {
    tasklist: true,
  },

  // 主题插件
  plugins: {
    comment: false,
    photoSwipe: true,
    icon: {
      assets: "fontawesome-with-brands",
    },
  },
});
