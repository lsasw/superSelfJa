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
      text: "AI 基础",
      icon: "globe",
      prefix: "/docs/ai-foundations/",
      children: [
        { text: "AI 全景概览", icon: "globe", link: "01-ai-overview" },
        { text: "机器学习基础", icon: "brain", link: "02-machine-learning-basics" },
        { text: "线性代数基础", icon: "superscript", link: "03-linear-algebra" },
        { text: "概率与统计", icon: "chart-bar", link: "04-probability-statistics" },
        { text: "微积分基础", icon: "square-root-alt", link: "05-calculus-for-ml" },
      ],
    },
    {
      text: "机器学习",
      icon: "brain",
      prefix: "/docs/ai-ml-classic/",
      children: [
        { text: "线性回归", icon: "chart-line", link: "16-线性回归" },
        { text: "逻辑回归", icon: "chart-line", link: "17-逻辑回归" },
        { text: "决策树", icon: "tree", link: "18-决策树" },
        { text: "随机森林", icon: "tree", link: "19-随机森林" },
        { text: "支持向量机", icon: "project-diagram", link: "20-支持向量机" },
      ],
    },
    {
      text: "深度学习",
      icon: "network-wired",
      prefix: "/docs/ai-dl-fundamentals/",
      children: [
        { text: "神经网络基础", icon: "network-wired", link: "31-neural-network-basics" },
        { text: "反向传播", icon: "retweet", link: "32-backpropagation" },
        { text: "CNN 基础", icon: "th", link: "37-cnn-basics" },
        { text: "Transformer", icon: "project-diagram", link: "42-transformer" },
        { text: "GAN", icon: "random", link: "43-gan" },
      ],
    },
    {
      text: "PyTorch 实战",
      icon: "fire",
      prefix: "/docs/ai-pytorch/",
      children: [
        { text: "PyTorch 基础", icon: "fire", link: "46-pytorch-basics" },
        { text: "张量操作", icon: "cubes", link: "47-tensor-operation" },
        { text: "自动求导", icon: "calculator", link: "48-autograd" },
        { text: "构建神经网络", icon: "network-wired", link: "49-build-nn" },
      ],
    },
    {
      text: "大语言模型",
      icon: "robot",
      prefix: "/docs/ai-llm/",
      children: [
        { text: "LLM 综述", icon: "brain", link: "61-llm-overview" },
        { text: "GPT 演进", icon: "code-branch", link: "62-gpt-evolution" },
        { text: "Prompt 工程", icon: "terminal", link: "73-prompt-engineering" },
        { text: "RAG 检索增强", icon: "search", link: "74-rag" },
      ],
    },
    {
      text: "AI 工程化",
      icon: "rocket",
      prefix: "/docs/ai-engineering/",
      children: [
        { text: "模型部署概述", icon: "rocket", link: "76-model-deployment" },
        { text: "ONNX 标准", icon: "exchange-alt", link: "77-onnx" },
        { text: "TensorRT", icon: "bolt", link: "78-tensorrt" },
        { text: "模型量化", icon: "compress", link: "79-model-quantization" },
        { text: "vLLM 服务", icon: "server", link: "83-vllm" },
      ],
    },
    {
      text: "AI 应用",
      icon: "robot",
      prefix: "/docs/ai-applications/",
      children: [
        { text: "AI Agent", icon: "robot", link: "91-ai-agent" },
        { text: "多模态 AI", icon: "image", link: "92-multimodal" },
        { text: "视觉语言模型", icon: "eye", link: "93-vision-language" },
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
        text: "AI 基础",
        icon: "globe",
        prefix: "ai-foundations/",
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
        icon: "robot",
        prefix: "ai-applications/",
        children: [
          { text: "AI Agent", icon: "robot", link: "91-ai-agent" },
          { text: "多模态 AI", icon: "image", link: "92-multimodal" },
          { text: "视觉语言模型", icon: "eye", link: "93-vision-language" },
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
  },
});
