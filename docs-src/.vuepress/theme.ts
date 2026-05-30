import { hopeTheme } from "vuepress-theme-hope";

export default hopeTheme({
  hostname: "https://lsasw.github.io",

  author: {
    name: "lsasw",
    url: "https://github.com/lsasw",
  },

  logo: "https://theme-hope-assets.vuejs.press/logo.svg",

  repo: "https://github.com/lsasw/superSelfJa",

  docsDir: "docs-src",

  // 导航栏
  navbar: [
    "/",
    "/docs/",
    {
      text: "设计模式",
      icon: "puzzle-piece",
      prefix: "/docs/java-patterns/",
      children: [
        { text: "创建型模式", icon: "code", link: "creation" },
        { text: "结构型模式", icon: "project-diagram", link: "structural" },
        { text: "行为型模式", icon: "exchange-alt", link: "behavioral" },
      ],
    },
    {
      text: "Spring",
      icon: "leaf",
      prefix: "/docs/spring-core/",
      children: [
        { text: "IoC 容器", icon: "box", link: "ioc" },
        { text: "AOP 切面", icon: "layer-group", link: "aop" },
        { text: "事务管理", icon: "sync", link: "transaction" },
      ],
    },
    {
      text: "Spring Boot",
      icon: "boot",
      prefix: "/docs/spring-boot/",
      children: [
        { text: "自动配置", icon: "cogs", link: "auto-config" },
      ],
    },
    {
      text: "数据库",
      icon: "database",
      prefix: "/docs/database/",
      children: [
        { text: "MySQL", icon: "server", link: "mysql" },
        { text: "MongoDB", icon: "leaf", link: "mongodb" },
        { text: "Redis", icon: "bolt", link: "redis" },
      ],
    },
  ],

  // 侧边栏
  sidebar: {
    "/docs/": [
      "",
      {
        text: "Java 设计模式",
        icon: "puzzle-piece",
        prefix: "java-patterns/",
        children: [
          { text: "创建型模式", icon: "code", link: "creation" },
          { text: "结构型模式", icon: "project-diagram", link: "structural" },
          { text: "行为型模式", icon: "exchange-alt", link: "behavioral" },
        ],
      },
      {
        text: "Spring 核心",
        icon: "leaf",
        prefix: "spring-core/",
        children: [
          { text: "IoC 容器", icon: "box", link: "ioc" },
          { text: "AOP 切面", icon: "layer-group", link: "aop" },
          { text: "事务管理", icon: "sync", link: "transaction" },
        ],
      },
      {
        text: "Spring Boot",
        icon: "boot",
        prefix: "spring-boot/",
        children: [
          { text: "自动配置原理", icon: "cogs", link: "auto-config" },
        ],
      },
      {
        text: "数据库",
        icon: "database",
        prefix: "database/",
        children: [
          { text: "MySQL 进阶", icon: "server", link: "mysql" },
          { text: "MongoDB 实战", icon: "leaf", link: "mongodb" },
          { text: "Redis 缓存", icon: "bolt", link: "redis" },
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
