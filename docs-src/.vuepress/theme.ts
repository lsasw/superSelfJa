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
    collapsible: true,
    children: [
      { text: "创建型模式", icon: "code", link: "/docs/java-patterns/creation" },
      { text: "结构型模式", icon: "project-diagram", link: "/docs/java-patterns/structural" },
      { text: "行为型模式", icon: "exchange-alt", link: "/docs/java-patterns/behavioral" },
    ],
  },
  {
    text: "Spring 核心",
    icon: "leaf",
    collapsible: true,
    children: [
      { text: "IoC 容器", icon: "box", link: "/docs/spring-core/ioc" },
      { text: "AOP 切面", icon: "layer-group", link: "/docs/spring-core/aop" },
      { text: "事务管理", icon: "sync", link: "/docs/spring-core/transaction" },
    ],
  },
  {
    text: "Spring Boot",
    icon: "boot",
    collapsible: true,
    children: [
      { text: "自动配置原理", icon: "cogs", link: "/docs/spring-boot/auto-config" },
    ],
  },
  {
    text: "数据库",
    icon: "database",
    collapsible: true,
    children: [
      { text: "MySQL 进阶", icon: "server", link: "/docs/database/mysql" },
      { text: "MongoDB 实战", icon: "leaf", link: "/docs/database/mongodb" },
      { text: "Redis 缓存", icon: "bolt", link: "/docs/database/redis" },
      { text: "MySQL 索引与 MVCC", icon: "server", link: "/docs/database/mysql-advanced" },
      { text: "Redis 三大问题", icon: "bolt", link: "/docs/database/redis-advanced" },
      { text: "Kafka + 自动配置", icon: "stream", link: "/docs/database/kafka-spring-autoconfig" },
    ],
  },
  {
    text: "Java 并发与 JVM",
    icon: "sync",
    collapsible: true,
    children: [
      { text: "并发编程实战", icon: "sync", link: "/docs/java-concurrency-jvm/thread-concurrency" },
      { text: "分布式锁设计", icon: "lock", link: "/docs/java-concurrency-jvm/distributed-lock" },
      { text: "JVM 调优实战", icon: "wrench", link: "/docs/java-concurrency-jvm/jvm-tuning" },
    ],
  },
];

/** AI 生态侧边栏（AI基础 → 机器学习 → 深度学习 → PyTorch → LLM → 工程化 → 应用） */
const aiSidebar = [
  {
    text: "AI 基础",
    icon: "globe",
    collapsible: true,
    children: [
      { text: "AI 全景概览", icon: "globe", link: "/docs/ai-foundations/01-ai-overview" },
      { text: "机器学习基础", icon: "brain", link: "/docs/ai-foundations/02-machine-learning-basics" },
      { text: "线性代数", icon: "superscript", link: "/docs/ai-foundations/03-linear-algebra" },
      { text: "概率与统计", icon: "chart-bar", link: "/docs/ai-foundations/04-probability-statistics" },
      { text: "微积分", icon: "square-root-alt", link: "/docs/ai-foundations/05-calculus-for-ml" },
      { text: "Python 基础", icon: "code", link: "/docs/ai-foundations/06-python-basics" },
      { text: "NumPy 与 Pandas", icon: "table", link: "/docs/ai-foundations/07-numpy-pandas" },
      { text: "数据预处理", icon: "filter", link: "/docs/ai-foundations/08-data-preprocessing" },
      { text: "评估指标", icon: "bullseye", link: "/docs/ai-foundations/09-evaluation-metrics" },
      { text: "过拟合与欠拟合", icon: "balance-scale", link: "/docs/ai-foundations/10-overfitting-underfitting" },
      { text: "偏差与方差", icon: "project-diagram", link: "/docs/ai-foundations/11-bias-variance" },
      { text: "交叉验证", icon: "sync", link: "/docs/ai-foundations/12-cross-validation" },
      { text: "梯度下降", icon: "arrow-down", link: "/docs/ai-foundations/13-gradient-descent" },
      { text: "优化算法", icon: "cogs", link: "/docs/ai-foundations/14-optimization-algorithms" },
      { text: "AI 伦理", icon: "handshake", link: "/docs/ai-foundations/15-ai-ethics" },
    ],
  },
  {
    text: "经典机器学习",
    icon: "brain",
    collapsible: true,
    children: [
      { text: "线性回归", icon: "chart-line", link: "/docs/ai-ml-classic/16-线性回归" },
      { text: "逻辑回归", icon: "chart-line", link: "/docs/ai-ml-classic/17-逻辑回归" },
      { text: "决策树", icon: "tree", link: "/docs/ai-ml-classic/18-决策树" },
      { text: "随机森林", icon: "tree", link: "/docs/ai-ml-classic/19-随机森林" },
      { text: "支持向量机", icon: "project-diagram", link: "/docs/ai-ml-classic/20-支持向量机" },
      { text: "K 近邻", icon: "dot-circle", link: "/docs/ai-ml-classic/21-K近邻" },
      { text: "朴素贝叶斯", icon: "envelope", link: "/docs/ai-ml-classic/22-朴素贝叶斯" },
      { text: "K-Means 聚类", icon: "th", link: "/docs/ai-ml-classic/23-K-Means聚类" },
      { text: "PCA 降维", icon: "compress", link: "/docs/ai-ml-classic/24-PCA降维" },
      { text: "集成学习", icon: "layer-group", link: "/docs/ai-ml-classic/25-集成学习" },
    ],
  },
  {
    text: "深度学习基础",
    icon: "network-wired",
    collapsible: true,
    children: [
      { text: "神经网络基础", icon: "network-wired", link: "/docs/ai-dl-fundamentals/31-neural-network-basics" },
      { text: "反向传播", icon: "retweet", link: "/docs/ai-dl-fundamentals/32-backpropagation" },
      { text: "激活函数", icon: "wave-square", link: "/docs/ai-dl-fundamentals/33-activation-functions" },
      { text: "损失函数", icon: "chart-area", link: "/docs/ai-dl-fundamentals/34-loss-functions" },
      { text: "权重初始化", icon: "sliders-h", link: "/docs/ai-dl-fundamentals/35-weight-initialization" },
      { text: "批量归一化", icon: "align-center", link: "/docs/ai-dl-fundamentals/36-batch-normalization" },
      { text: "CNN 基础", icon: "th", link: "/docs/ai-dl-fundamentals/37-cnn-basics" },
      { text: "CNN 架构", icon: "sitemap", link: "/docs/ai-dl-fundamentals/38-cnn-architectures" },
      { text: "RNN 基础", icon: "redo", link: "/docs/ai-dl-fundamentals/39-rnn-basics" },
      { text: "LSTM 与 GRU", icon: "memory", link: "/docs/ai-dl-fundamentals/40-lstm-gru" },
      { text: "注意力机制", icon: "eye", link: "/docs/ai-dl-fundamentals/41-attention-mechanism" },
      { text: "Transformer", icon: "project-diagram", link: "/docs/ai-dl-fundamentals/42-transformer" },
      { text: "GAN", icon: "random", link: "/docs/ai-dl-fundamentals/43-gan" },
      { text: "自编码器", icon: "compress-alt", link: "/docs/ai-dl-fundamentals/44-autoencoder" },
      { text: "扩散模型", icon: "water", link: "/docs/ai-dl-fundamentals/45-diffusion-models" },
    ],
  },
  {
    text: "PyTorch 实战",
    icon: "fire",
    collapsible: true,
    children: [
      { text: "PyTorch 基础", icon: "fire", link: "/docs/ai-pytorch/46-pytorch-basics" },
      { text: "张量操作", icon: "cubes", link: "/docs/ai-pytorch/47-tensor-operation" },
      { text: "自动求导", icon: "calculator", link: "/docs/ai-pytorch/48-autograd" },
      { text: "构建神经网络", icon: "network-wired", link: "/docs/ai-pytorch/49-build-nn" },
      { text: "Dataset 与 DataLoader", icon: "database", link: "/docs/ai-pytorch/50-dataset-dataloader" },
      { text: "自定义层与模块", icon: "puzzle-piece", link: "/docs/ai-pytorch/51-custom-layers" },
      { text: "模型保存与加载", icon: "save", link: "/docs/ai-pytorch/52-model-io" },
      { text: "训练循环设计", icon: "sync", link: "/docs/ai-pytorch/53-training-loop" },
      { text: "GPU 加速", icon: "bolt", link: "/docs/ai-pytorch/54-gpu-computing" },
      { text: "分布式训练", icon: "network-wired", link: "/docs/ai-pytorch/55-distributed-training" },
      { text: "模型可视化", icon: "chart-line", link: "/docs/ai-pytorch/56-visualization" },
      { text: "迁移学习", icon: "exchange-alt", link: "/docs/ai-pytorch/57-transfer-learning" },
      { text: "自定义损失函数", icon: "chart-area", link: "/docs/ai-pytorch/58-custom-loss" },
      { text: "调试与性能分析", icon: "bug", link: "/docs/ai-pytorch/59-debugging" },
      { text: "模型部署入门", icon: "rocket", link: "/docs/ai-pytorch/60-model-deployment-basics" },
    ],
  },
  {
    text: "大语言模型",
    icon: "robot",
    collapsible: true,
    children: [
      { text: "LLM 综述", icon: "brain", link: "/docs/ai-llm/61-llm-overview" },
      { text: "GPT 演进", icon: "code-branch", link: "/docs/ai-llm/62-gpt-evolution" },
      { text: "LLaMA 家族", icon: "paw", link: "/docs/ai-llm/63-llama-family" },
      { text: "Claude 分析", icon: "robot", link: "/docs/ai-llm/64-claude-analysis" },
      { text: "Tokenizer", icon: "cut", link: "/docs/ai-llm/65-tokenizer" },
      { text: "位置编码", icon: "ruler", link: "/docs/ai-llm/66-positional-encoding" },
      { text: "注意力变体", icon: "eye", link: "/docs/ai-llm/67-attention-variants" },
      { text: "FFN 与 MoE", icon: "project-diagram", link: "/docs/ai-llm/68-ffn-moe" },
      { text: "预训练策略", icon: "cogs", link: "/docs/ai-llm/69-pretraining" },
      { text: "SFT 指令微调", icon: "terminal", link: "/docs/ai-llm/70-sft" },
      { text: "RLHF 对齐", icon: "balance-scale", link: "/docs/ai-llm/71-rlhf" },
      { text: "DPO 优化", icon: "sliders-h", link: "/docs/ai-llm/72-dpo" },
      { text: "Prompt 工程", icon: "terminal", link: "/docs/ai-llm/73-prompt-engineering" },
      { text: "RAG 检索增强", icon: "search", link: "/docs/ai-llm/74-rag" },
      { text: "模型评估", icon: "bullseye", link: "/docs/ai-llm/75-evaluation-benchmarks" },
    ],
  },
  {
    text: "AI 工程化",
    icon: "rocket",
    collapsible: true,
    children: [
      { text: "模型部署概述", icon: "rocket", link: "/docs/ai-engineering/76-model-deployment" },
      { text: "ONNX 标准", icon: "exchange-alt", link: "/docs/ai-engineering/77-onnx" },
      { text: "TensorRT", icon: "bolt", link: "/docs/ai-engineering/78-tensorrt" },
      { text: "模型量化", icon: "compress", link: "/docs/ai-engineering/79-model-quantization" },
      { text: "模型剪枝", icon: "cut", link: "/docs/ai-engineering/80-model-pruning" },
      { text: "知识蒸馏", icon: "graduation-cap", link: "/docs/ai-engineering/82-knowledge-distillation" },
      { text: "vLLM 服务", icon: "server", link: "/docs/ai-engineering/83-vllm" },
      { text: "Triton 推理", icon: "server", link: "/docs/ai-engineering/84-triton" },
      { text: "Docker 部署", icon: "docker", link: "/docs/ai-engineering/85-docker-deployment" },
      { text: "K8s 部署", icon: "cubes", link: "/docs/ai-engineering/86-k8s-deployment" },
      { text: "监控告警", icon: "bell", link: "/docs/ai-engineering/87-monitoring" },
      { text: "版本管理", icon: "tags", link: "/docs/ai-engineering/88-model-versioning" },
      { text: "CI/CD", icon: "sync", link: "/docs/ai-engineering/89-cicd-ml" },
      { text: "边缘部署", icon: "mobile-alt", link: "/docs/ai-engineering/90-edge-deployment" },
    ],
  },
  {
    text: "AI 应用开发",
    icon: "user-astronaut",
    collapsible: true,
    children: [
      { text: "AI Agent", icon: "robot", link: "/docs/ai-applications/91-ai-agent" },
      { text: "多模态 AI", icon: "image", link: "/docs/ai-applications/92-multimodal" },
      { text: "视觉语言模型", icon: "eye", link: "/docs/ai-applications/93-vision-language" },
      { text: "代码大模型", icon: "code", link: "/docs/ai-applications/94-code-llm" },
      { text: "AI 搜索", icon: "search", link: "/docs/ai-applications/95-ai-search" },
      { text: "推荐系统", icon: "thumbs-up", link: "/docs/ai-applications/96-recommendation-ai" },
      { text: "AI for Science", icon: "flask", link: "/docs/ai-applications/97-ai-for-science" },
      { text: "具身智能", icon: "robot", link: "/docs/ai-applications/98-embodied-ai" },
      { text: "AI 面试题库", icon: "clipboard-list", link: "/docs/ai-applications/101-ai-interview-qa" },
    ],
  },
];

/** 实战专题侧边栏（Agent工程化 / MCP / RAG / 系统设计 / 团队建设） */
const practicalSidebar = [
  {
    text: "AI Agent 工程化",
    icon: "robot",
    collapsible: true,
    children: [
      { text: "Function Call 全链路", icon: "code", link: "/docs/ai-agent-engineering/function-call" },
      { text: "Multi-Agent 编排", icon: "sitemap", link: "/docs/ai-agent-engineering/multi-agent-orchestration" },
      { text: "Skill vs MCP vs A2A", icon: "balance-scale", link: "/docs/ai-agent-engineering/skill-mcp-a2a" },
      { text: "Agent Memory 机制", icon: "memory", link: "/docs/ai-agent-engineering/agent-memory" },
    ],
  },
  {
    text: "MCP 协议与网关",
    icon: "server",
    collapsible: true,
    children: [
      { text: "MCP 协议深度", icon: "book-open", link: "/docs/mcp-gateway/protocol-deep-dive" },
      { text: "网关搭建实战", icon: "server", link: "/docs/mcp-gateway/gateway-build" },
      { text: "OpenAPI 转换", icon: "exchange-alt", link: "/docs/mcp-gateway/openapi-to-mcp" },
    ],
  },
  {
    text: "RAG 与向量数据库",
    icon: "search",
    collapsible: true,
    children: [
      { text: "RAG 端到端搭建", icon: "search", link: "/docs/rag-vector-db/rag-end-to-end" },
      { text: "Embedding 选型", icon: "chart-bar", link: "/docs/rag-vector-db/embedding-guide" },
      { text: "向量数据库选型", icon: "database", link: "/docs/rag-vector-db/vector-database" },
    ],
  },
  {
    text: "系统设计",
    icon: "project-diagram",
    collapsible: true,
    children: [
      { text: "答题框架与题库", icon: "project-diagram", link: "/docs/system-design/framework-and-cases" },
    ],
  },
  {
    text: "团队建设",
    icon: "users-cog",
    collapsible: true,
    children: [
      { text: "流程建设标准", icon: "users-cog", link: "/docs/team-process/standards" },
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
    avatar: "/avatar-lg.png",
  },

  logo: "/avatar-lg.png",
  logoDark: "/avatar-lg.png",

  favicon: "/favicon.ico",

  repo: "https://github.com/lsasw/superSelfJa",

  docsDir: "docs-src",

  // 禁用文章元信息（作者、阅读时间等），目录页不需要显示
  pageInfo: false,

  // ============================================================
  //  顶部导航栏
  // ============================================================
  navbar: [
    "/",
    "/docs/",
    {
      text: "Java 生态",
      icon: "mug-hot",
      children: [
        { text: "设计模式", icon: "puzzle-piece", link: "/docs/java-patterns/" },
        { text: "Spring 核心", icon: "leaf", link: "/docs/spring-core/" },
        { text: "Spring Boot", icon: "boot", link: "/docs/spring-boot/" },
        { text: "数据库", icon: "database", link: "/docs/database/" },
        { text: "并发与 JVM", icon: "sync", link: "/docs/java-concurrency-jvm/" },
      ],
    },
    {
      text: "AI 生态",
      icon: "brain",
      children: [
        { text: "AI 基础", icon: "globe", link: "/docs/ai-foundations/" },
        { text: "经典机器学习", icon: "chart-line", link: "/docs/ai-ml-classic/" },
        { text: "深度学习", icon: "network-wired", link: "/docs/ai-dl-fundamentals/" },
        { text: "PyTorch 实战", icon: "fire", link: "/docs/ai-pytorch/" },
        { text: "大语言模型", icon: "robot", link: "/docs/ai-llm/" },
        { text: "AI 工程化", icon: "rocket", link: "/docs/ai-engineering/" },
        { text: "AI 应用", icon: "user-astronaut", link: "/docs/ai-applications/" },
      ],
    },
    {
      text: "实战专题",
      icon: "laptop-code",
      children: [
        { text: "Agent 工程化", icon: "robot", link: "/docs/ai-agent-engineering/" },
        { text: "MCP 网关", icon: "server", link: "/docs/mcp-gateway/" },
        { text: "RAG 向量数据库", icon: "search", link: "/docs/rag-vector-db/" },
        { text: "系统设计", icon: "project-diagram", link: "/docs/system-design/" },
        { text: "团队建设", icon: "users-cog", link: "/docs/team-process/" },
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
      components: ["Badge", "BiliBili", "CodePen", "FontIcon", "PDF", "StackBlitz"],
    },
    // searchPro: {
    //   indexContent: true,
    //   autoSuggestions: true,
    // },
  },
});
