import { hopeTheme } from "vuepress-theme-hope";

// ============================================================
//  共享侧边栏配置 — 按导航分类拆分为三套独立侧边栏
//  通过 sidebar 对象的路径前缀映射，实现"点击哪个导航，就显示哪套侧边栏"
// ============================================================

/** Java 生态侧边栏（设计模式 / Spring / 数据库 / 并发JVM） */
const javaSidebar = [
  {
    text: "Java 设计模式",
    icon: "puzzle-piece",
    prefix: "java-patterns/",
    collapsible: true,
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
    collapsible: true,
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
    collapsible: true,
    children: [
      { text: "自动配置原理", icon: "cogs", link: "auto-config" },
    ],
  },
  {
    text: "数据库",
    icon: "database",
    prefix: "database/",
    collapsible: true,
    children: [
      { text: "MySQL 进阶", icon: "server", link: "mysql" },
      { text: "MongoDB 实战", icon: "leaf", link: "mongodb" },
      { text: "Redis 缓存", icon: "bolt", link: "redis" },
      { text: "MySQL 索引与 MVCC", icon: "server", link: "mysql-advanced" },
      { text: "Redis 三大问题", icon: "bolt", link: "redis-advanced" },
      { text: "Kafka + 自动配置", icon: "stream", link: "kafka-spring-autoconfig" },
    ],
  },
  {
    text: "Java 并发与 JVM",
    icon: "sync",
    prefix: "java-concurrency-jvm/",
    collapsible: true,
    children: [
      { text: "并发编程实战", icon: "sync", link: "thread-concurrency" },
      { text: "分布式锁设计", icon: "lock", link: "distributed-lock" },
      { text: "JVM 调优实战", icon: "wrench", link: "jvm-tuning" },
    ],
  },
];

/** AI 生态侧边栏（AI基础 → 机器学习 → 深度学习 → PyTorch → LLM → 工程化 → 应用） */
const aiSidebar = [
  {
    text: "AI 基础",
    icon: "globe",
    prefix: "ai-foundations/",
    collapsible: true,
    children: [
      { text: "AI 全景概览", icon: "globe", link: "01-ai-overview" },
      { text: "机器学习基础", icon: "brain", link: "02-machine-learning-basics" },
      { text: "线性代数", icon: "superscript", link: "03-linear-algebra" },
      { text: "概率与统计", icon: "chart-bar", link: "04-probability-statistics" },
      { text: "微积分", icon: "square-root-alt", link: "05-calculus-for-ml" },
      { text: "Python 基础", icon: "code", link: "06-python-basics" },
      { text: "NumPy 与 Pandas", icon: "table", link: "07-numpy-pandas" },
      { text: "数据预处理", icon: "filter", link: "08-data-preprocessing" },
      { text: "评估指标", icon: "bullseye", link: "09-evaluation-metrics" },
      { text: "过拟合与欠拟合", icon: "balance-scale", link: "10-overfitting-underfitting" },
      { text: "偏差与方差", icon: "project-diagram", link: "11-bias-variance" },
      { text: "交叉验证", icon: "sync", link: "12-cross-validation" },
      { text: "梯度下降", icon: "arrow-down", link: "13-gradient-descent" },
      { text: "优化算法", icon: "cogs", link: "14-optimization-algorithms" },
      { text: "AI 伦理", icon: "handshake", link: "15-ai-ethics" },
    ],
  },
  {
    text: "经典机器学习",
    icon: "brain",
    prefix: "ai-ml-classic/",
    collapsible: true,
    children: [
      { text: "线性回归", icon: "chart-line", link: "16-linear-regression" },
      { text: "逻辑回归", icon: "chart-line", link: "17-logistic-regression" },
      { text: "支持向量机", icon: "project-diagram", link: "18-svm" },
      { text: "决策树", icon: "tree", link: "19-decision-tree" },
      { text: "随机森林", icon: "tree", link: "20-random-forest" },
      { text: "XGBoost 与 GBDT", icon: "chart-bar", link: "21-xgboost" },
      { text: "K 近邻", icon: "dot-circle", link: "22-knn" },
      { text: "K-Means 聚类", icon: "th", link: "23-kmeans" },
      { text: "PCA 降维", icon: "compress", link: "24-pca" },
      { text: "朴素贝叶斯", icon: "envelope", link: "25-naive-bayes" },
      { text: "隐马尔可夫模型", icon: "random", link: "26-hmm" },
      { text: "集成学习", icon: "layer-group", link: "27-ensemble" },
      { text: "特征工程", icon: "filter", link: "28-feature-engineering" },
      { text: "模型选择与调优", icon: "sliders-h", link: "29-model-selection" },
      { text: "ML 完整流程", icon: "tasks", link: "30-ml-pipeline" },
    ],
  },
  {
    text: "深度学习基础",
    icon: "network-wired",
    prefix: "ai-dl-fundamentals/",
    collapsible: true,
    children: [
      { text: "神经网络基础", icon: "network-wired", link: "31-neural-network-basics" },
      { text: "反向传播", icon: "retweet", link: "32-backpropagation" },
      { text: "激活函数", icon: "wave-square", link: "33-activation-functions" },
      { text: "损失函数", icon: "chart-area", link: "34-loss-functions" },
      { text: "权重初始化", icon: "sliders-h", link: "35-weight-initialization" },
      { text: "批量归一化", icon: "align-center", link: "36-batch-normalization" },
      { text: "CNN 基础", icon: "th", link: "37-cnn-basics" },
      { text: "CNN 架构", icon: "sitemap", link: "38-cnn-architectures" },
      { text: "RNN 基础", icon: "redo", link: "39-rnn-basics" },
      { text: "LSTM 与 GRU", icon: "memory", link: "40-lstm-gru" },
      { text: "注意力机制", icon: "eye", link: "41-attention-mechanism" },
      { text: "Transformer", icon: "project-diagram", link: "42-transformer" },
      { text: "GAN", icon: "random", link: "43-gan" },
      { text: "自编码器", icon: "compress-alt", link: "44-autoencoder" },
      { text: "扩散模型", icon: "water", link: "45-diffusion-models" },
    ],
  },
  {
    text: "PyTorch 实战",
    icon: "fire",
    prefix: "ai-pytorch/",
    collapsible: true,
    children: [
      { text: "PyTorch 基础", icon: "fire", link: "46-pytorch-basics" },
      { text: "张量操作", icon: "cubes", link: "47-tensor-operation" },
      { text: "自动求导", icon: "calculator", link: "48-autograd" },
      { text: "构建神经网络", icon: "network-wired", link: "49-build-nn" },
      { text: "Dataset 与 DataLoader", icon: "database", link: "50-dataset-dataloader" },
      { text: "自定义层与模块", icon: "puzzle-piece", link: "51-custom-layers" },
      { text: "模型保存与加载", icon: "save", link: "52-model-io" },
      { text: "训练循环设计", icon: "sync", link: "53-training-loop" },
      { text: "GPU 加速", icon: "bolt", link: "54-gpu-computing" },
      { text: "分布式训练", icon: "network-wired", link: "55-distributed-training" },
      { text: "模型可视化", icon: "chart-line", link: "56-visualization" },
      { text: "迁移学习", icon: "exchange-alt", link: "57-transfer-learning" },
      { text: "自定义损失函数", icon: "chart-area", link: "58-custom-loss" },
      { text: "调试与性能分析", icon: "bug", link: "59-debugging" },
      { text: "模型部署入门", icon: "rocket", link: "60-model-deployment-basics" },
    ],
  },
  {
    text: "大语言模型",
    icon: "robot",
    prefix: "ai-llm/",
    collapsible: true,
    children: [
      { text: "LLM 综述", icon: "brain", link: "61-llm-overview" },
      { text: "GPT 演进", icon: "code-branch", link: "62-gpt-evolution" },
      { text: "LLaMA 家族", icon: "paw", link: "63-llama-family" },
      { text: "Claude 分析", icon: "robot", link: "64-claude-analysis" },
      { text: "Tokenizer", icon: "cut", link: "65-tokenizer" },
      { text: "位置编码", icon: "ruler", link: "66-positional-encoding" },
      { text: "注意力变体", icon: "eye", link: "67-attention-variants" },
      { text: "FFN 与 MoE", icon: "project-diagram", link: "68-ffn-moe" },
      { text: "预训练策略", icon: "cogs", link: "69-pretraining" },
      { text: "SFT 指令微调", icon: "terminal", link: "70-sft" },
      { text: "RLHF 对齐", icon: "balance-scale", link: "71-rlhf" },
      { text: "DPO 优化", icon: "sliders-h", link: "72-dpo" },
      { text: "Prompt 工程", icon: "terminal", link: "73-prompt-engineering" },
      { text: "RAG 检索增强", icon: "search", link: "74-rag" },
      { text: "模型评估", icon: "bullseye", link: "75-evaluation-benchmarks" },
    ],
  },
  {
    text: "AI 工程化",
    icon: "rocket",
    prefix: "ai-engineering/",
    collapsible: true,
    children: [
      { text: "模型部署概述", icon: "rocket", link: "76-model-deployment" },
      { text: "ONNX 标准", icon: "exchange-alt", link: "77-onnx" },
      { text: "TensorRT", icon: "bolt", link: "78-tensorrt" },
      { text: "模型量化", icon: "compress", link: "79-model-quantization" },
      { text: "模型剪枝", icon: "cut", link: "80-model-pruning" },
      { text: "知识蒸馏", icon: "graduation-cap", link: "82-knowledge-distillation" },
      { text: "vLLM 服务", icon: "server", link: "83-vllm" },
      { text: "Triton 推理", icon: "server", link: "84-triton" },
      { text: "Docker 部署", icon: "docker", link: "85-docker-deployment" },
      { text: "K8s 部署", icon: "cubes", link: "86-k8s-deployment" },
      { text: "监控告警", icon: "bell", link: "87-monitoring" },
      { text: "版本管理", icon: "tags", link: "88-model-versioning" },
      { text: "CI/CD", icon: "sync", link: "89-cicd-ml" },
      { text: "边缘部署", icon: "mobile-alt", link: "90-edge-deployment" },
    ],
  },
  {
    text: "AI 应用开发",
    icon: "user-astronaut",
    prefix: "ai-applications/",
    collapsible: true,
    children: [
      { text: "AI Agent", icon: "robot", link: "91-ai-agent" },
      { text: "多模态 AI", icon: "image", link: "92-multimodal" },
      { text: "视觉语言模型", icon: "eye", link: "93-vision-language" },
      { text: "代码大模型", icon: "code", link: "94-code-llm" },
      { text: "AI 搜索", icon: "search", link: "95-ai-search" },
      { text: "推荐系统", icon: "thumbs-up", link: "96-recommendation-ai" },
      { text: "AI for Science", icon: "flask", link: "97-ai-for-science" },
      { text: "具身智能", icon: "robot", link: "98-embodied-ai" },
      { text: "AI 面试题库", icon: "clipboard-list", link: "101-ai-interview-qa" },
    ],
  },
];

/** 实战专题侧边栏（Agent工程化 / MCP / RAG / 系统设计 / 团队建设） */
const practicalSidebar = [
  {
    text: "AI Agent 工程化",
    icon: "robot",
    prefix: "ai-agent-engineering/",
    collapsible: true,
    children: [
      { text: "Function Call 全链路", icon: "code", link: "function-call" },
      { text: "Multi-Agent 编排", icon: "sitemap", link: "multi-agent-orchestration" },
      { text: "Skill vs MCP vs A2A", icon: "balance-scale", link: "skill-mcp-a2a" },
      { text: "Agent Memory 机制", icon: "memory", link: "agent-memory" },
    ],
  },
  {
    text: "MCP 协议与网关",
    icon: "server",
    prefix: "mcp-gateway/",
    collapsible: true,
    children: [
      { text: "MCP 协议深度", icon: "book-open", link: "protocol-deep-dive" },
      { text: "网关搭建实战", icon: "server", link: "gateway-build" },
      { text: "OpenAPI 转换", icon: "exchange-alt", link: "openapi-to-mcp" },
    ],
  },
  {
    text: "RAG 与向量数据库",
    icon: "search",
    prefix: "rag-vector-db/",
    collapsible: true,
    children: [
      { text: "RAG 端到端搭建", icon: "search", link: "rag-end-to-end" },
      { text: "Embedding 选型", icon: "chart-bar", link: "embedding-guide" },
      { text: "向量数据库选型", icon: "database", link: "vector-database" },
    ],
  },
  {
    text: "系统设计",
    icon: "project-diagram",
    prefix: "system-design/",
    collapsible: true,
    children: [
      { text: "答题框架与题库", icon: "project-diagram", link: "framework-and-cases" },
    ],
  },
  {
    text: "团队建设",
    icon: "users-cog",
    prefix: "team-process/",
    collapsible: true,
    children: [
      { text: "流程建设标准", icon: "users-cog", link: "standards" },
    ],
  },
];

// ============================================================
//  导出主题配置
// ============================================================
export default hopeTheme({
  hostname: "https://lsasw.github.io",

  author: {
    name: "lsasw",
    url: "https://github.com/lsasw",
  },

  logo: "/logo.svg",
  logoDark: "/logo.svg",

  favicon: "/favicon.ico",

  repo: "https://github.com/lsasw/superSelfJa",

  docsDir: "docs-src",

  // ============================================================
  //  顶部导航栏
  // ============================================================
  navbar: [
    "/",
    "/docs/",
    {
      text: "Java 生态",
      icon: "mug-hot",
      prefix: "/docs/",
      children: [
        { text: "设计模式", icon: "puzzle-piece", link: "java-patterns/" },
        { text: "Spring 核心", icon: "leaf", link: "spring-core/" },
        { text: "Spring Boot", icon: "boot", link: "spring-boot/" },
        { text: "数据库", icon: "database", link: "database/" },
        { text: "并发与 JVM", icon: "sync", link: "java-concurrency-jvm/" },
      ],
    },
    {
      text: "AI 生态",
      icon: "brain",
      prefix: "/docs/",
      children: [
        { text: "AI 基础", icon: "globe", link: "ai-foundations/" },
        { text: "经典机器学习", icon: "chart-line", link: "ai-ml-classic/" },
        { text: "深度学习", icon: "network-wired", link: "ai-dl-fundamentals/" },
        { text: "PyTorch 实战", icon: "fire", link: "ai-pytorch/" },
        { text: "大语言模型", icon: "robot", link: "ai-llm/" },
        { text: "AI 工程化", icon: "rocket", link: "ai-engineering/" },
        { text: "AI 应用", icon: "user-astronaut", link: "ai-applications/" },
      ],
    },
    {
      text: "实战专题",
      icon: "laptop-code",
      prefix: "/docs/",
      children: [
        { text: "Agent 工程化", icon: "robot", link: "ai-agent-engineering/" },
        { text: "MCP 网关", icon: "server", link: "mcp-gateway/" },
        { text: "RAG 向量数据库", icon: "search", link: "rag-vector-db/" },
        { text: "系统设计", icon: "project-diagram", link: "system-design/" },
        { text: "团队建设", icon: "users-cog", link: "team-process/" },
      ],
    },
    {
      text: "面试题库",
      icon: "clipboard-list",
      link: "/docs/ai-applications/101-ai-interview-qa",
    },
  ],

  // ============================================================
  //  ⭐ 侧边栏 — 按路径前缀映射 → 点击顶部导航，左侧只显示对应内容
  //  Theme Hope 会遍历 keys 查找第一个匹配当前路径的，显示对应侧边栏
  // ============================================================
  sidebar: {
    // -----------------------------------------------
    //  Java 生态 — 点击顶部 "Java 生态" 时显示
    // -----------------------------------------------
    "/docs/java-patterns/": javaSidebar,
    "/docs/spring-core/": javaSidebar,
    "/docs/spring-boot/": javaSidebar,
    "/docs/database/": javaSidebar,
    "/docs/java-concurrency-jvm/": javaSidebar,

    // -----------------------------------------------
    //  AI 生态 — 点击顶部 "AI 生态" 时显示
    // -----------------------------------------------
    "/docs/ai-foundations/": aiSidebar,
    "/docs/ai-ml-classic/": aiSidebar,
    "/docs/ai-dl-fundamentals/": aiSidebar,
    "/docs/ai-pytorch/": aiSidebar,
    "/docs/ai-llm/": aiSidebar,
    "/docs/ai-engineering/": aiSidebar,
    "/docs/ai-applications/": aiSidebar,

    // -----------------------------------------------
    //  实战专题 — 点击顶部 "实战专题" 时显示
    // -----------------------------------------------
    "/docs/ai-agent-engineering/": practicalSidebar,
    "/docs/mcp-gateway/": practicalSidebar,
    "/docs/rag-vector-db/": practicalSidebar,
    "/docs/system-design/": practicalSidebar,
    "/docs/team-process/": practicalSidebar,

    // -----------------------------------------------
    //  文档中心首页 — 显示 Java 侧边栏作为默认
    // -----------------------------------------------
    "/docs/": javaSidebar,
  },

  // ============================================================
  //  页脚
  // ============================================================
  footer: `基于 <a href="https://vuepress.vuejs.org/">VuePress</a> + <a href="https://theme-hope.vuejs.press/">Theme Hope</a> 构建 | 
<a href="https://github.com/lsasw/superSelfJa">GitHub</a>`,

  displayFooter: true,

  // ============================================================
  //  页面元数据
  // ============================================================
  metaLocales: {
    editLink: "在 GitHub 上编辑此页",
  },

  // ============================================================
  //  主题插件 & Markdown 增强
  // ============================================================
  markdown: {
    tasklist: true,
  },

  plugins: {
    copyCode: true,
    photoSwipe: true,
    components: {
      components: ["Badge", "BiliBili", "CodePen", "FontIcon", "PDF", "StackBlitz", "VideoPlayer", "YouTube"],
    },
    searchPro: {
      indexContent: true,
      autoSuggestions: true,
    },
  },
});
