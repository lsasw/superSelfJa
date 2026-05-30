---
title: Tokenizer 技术
icon: scissors
order: 5
---

# Tokenizer 技术（Tokenizer）

Tokenizer（分词器）是大语言模型中不可或缺的基础组件。它将人类可读的自然语言文本转换为模型可以处理的数字序列（Token），并在生成时将数字序列转换回文本。Tokenization 的质量直接影响模型的训练效率、推理质量和多语言能力。本文将系统介绍 LLM 中主流的 Tokenizer 技术，包括 BPE、WordPiece、SentencePiece 和 TikToken 等。

## 为什么需要 Tokenizer

神经网络只能处理数字，不能直接处理文本。因此需要将文本转换为数值表示：

```
原始文本 → Tokenizer → Token IDs → Embedding → 神经网络
"I love AI" → [4523, 892, 1234] → [向量, 向量, 向量] → Transformer
```

Tokenizer 需要解决的核心问题：

| 问题 | 描述 | 影响 |
|------|------|------|
| 词表大小 | 多少个不同的 Token | 模型参数、效率 |
| 未登录词 | 遇到训练中未见过的词 | 模型泛化能力 |
| 多语言 | 如何覆盖多种语言 | 国际化能力 |
| 压缩率 | 平均多少字符对应一个 Token | 上下文利用效率 |
| 边界处理 | 词边界的表示 | 生成质量 |

## Tokenization 方法对比

| 方法 | 提出者 | 词表构建 | 优点 | 缺点 | 代表模型 |
|------|--------|----------|------|------|----------|
| 字符级 | — | 固定字符集 | 无未登录词 | 序列长、上下文短 | — |
| 词级 | — | 固定词典 | 语义明确 | 词表爆炸、未登录词 | 早期 NLP |
| WordPiece | Google | 概率最大化合并 | 平衡效率与覆盖 | 训练慢 | BERT、T5 |
| BPE | OpenAI | 频率统计合并 | 简单高效、流行广 | 依赖预处理 | GPT 系列 |
| Unigram | SentencePiece | 概率子词 | 多语言友好 | 训练复杂 | mBART |
| SentencePiece | Google | 框架（支持多种算法） | 端到端、多语言 | 需要重新训练 | LLaMA、T5 |
| TikToken | OpenAI | BPE 变种 | 极快、可控 | 仅支持特定模型 | GPT-4、Claude |

## Byte Pair Encoding（BPE）

BPE 是最广泛使用的子词分词算法，由 GPT 系列推广。

### 算法原理

BPE 的核心思想是：**从字符级别开始，迭代合并出现频率最高的字符对**。

```
初始状态（字符级）：
h e l l o   w o r l d
  ↓ 合并频率最高的字符对
he ll o   wo r ld
  ↓ 继续合并
hell o   wor ld
  ↓ 继续...
hello   world
```

### BPE 算法步骤

```python
import re
from collections import defaultdict, Counter

class BPETokenizer:
    """Byte Pair Encoding 分词器实现"""
    
    def __init__(self):
        self.vocab = {}          # Token -> ID 映射
        self.merges = {}         # 合并规则 (A, B) -> AB
        self.encoder = {}        # ID -> Token 映射
    
    def get_stats(self, vocab):
        """统计字符对频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并指定的字符对"""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def learn_bpe(self, corpus, num_merges=1000):
        """学习 BPE 合并规则"""
        # 1. 初始化词表（所有字符）
        vocab = {}
        for text in corpus:
            for word in text.split():
                # 添加词尾标记
                word = ' '.join(list(word)) + ' </w>'
                vocab[word] = vocab.get(word, 0) + 1
        
        # 2. 迭代合并
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            
            # 选择频率最高的字符对
            best_pair = max(pairs, key=pairs.get)
            
            # 记录合并规则
            self.merges[best_pair] = ''.join(best_pair)
            
            # 执行合并
            vocab = self.merge_vocab(best_pair, vocab)
        
        # 3. 构建词表
        tokens = set()
        for word in vocab:
            for token in word.split():
                tokens.add(token)
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(tokens))}
        self.encoder = {idx: token for token, idx in self.vocab.items()}
        
        return self.merges
    
    def encode(self, text):
        """编码文本"""
        # 预处理
        word = ' '.join(list(text)) + ' </w>'
        symbols = word.split()
        
        # 应用合并规则
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            
            # 找到最优合并
            best_merge = None
            best_idx = float('inf')
            for pair in pairs:
                if pair in self.merges:
                    merge_idx = list(self.merges.keys()).index(pair)
                    if merge_idx < best_idx:
                        best_idx = merge_idx
                        best_merge = pair
            
            if best_merge is None:
                break
            
            # 执行合并
            merged = self.merges[best_merge]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and 
                    symbols[i] == best_merge[0] and 
                    symbols[i+1] == best_merge[1]):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        
        # 转换为 IDs
        return [self.vocab.get(s, self.vocab.get('<unk>', 0)) for s in symbols]

# 使用示例
tokenizer = BPETokenizer()
corpus = ["hello world", "hello there", "world peace", "hello peace"]
tokenizer.learn_bpe(corpus, num_merges=50)
print(tokenizer.encode("hello world"))
```

### BPE 的词表大小与效果

| 词表大小 | 平均词长 | 未登录词率 | 适用场景 |
|----------|----------|-----------|----------|
| 5000 | 1-2 字符 | ~0% | 字符级替代 |
| 30000 | 4-6 字符 | <1% | GPT-2、小型模型 |
| 50000 | 5-8 字符 | <0.5% | GPT-3、中型模型 |
| 100000 | 6-10 字符 | <0.2% | 大型多语言模型 |
| 128000 | 7-12 字符 | <0.1% | LLaMA 3、TikToken |

## WordPiece

WordPiece 由 Google 提出，用于 BERT 和 T5 等模型。与 BPE 不同，WordPiece 基于概率选择合并。

### 算法原理

WordPiece 的合并准则是选择使训练数据似然最大化的字符对：

```
Score(A, B) = (freq(AB) - 1) / (freq(A) × freq(B))

选择 Score 最高的字符对进行合并。
```

### BPE vs WordPiece

| 对比项 | BPE | WordPiece |
|--------|-----|-----------|
| 合并准则 | 频率最高 | 似然增益最大 |
| 初始词表 | 字符 | 字符 |
| 停止条件 | 固定合并次数 | 固定词表大小 |
| 子词标记 | 无前缀 | ## 后缀标记 |
| 未知词处理 | <unk> | 逐字符分解 |

### WordPiece 示例

```python
class WordPieceTokenizer:
    """简化版 WordPiece 分词器"""
    
    def __init__(self, vocab, unk_token="[UNK]"):
        self.vocab = vocab  # 词表（包含 ## 前缀标记）
        self.unk_token = unk_token
    
    def tokenize(self, text):
        """将文本分词为 WordPiece tokens"""
        tokens = []
        for word in text.lower().split():
            # 第一个子词不需要 ## 前缀
            sub_tokens = self._wordpiece_tokenize_word(word)
            tokens.extend(sub_tokens)
        return tokens
    
    def _wordpiece_tokenize_word(self, word):
        """对单个词进行 WordPiece 分词"""
        # 贪心最长匹配
        sub_tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr  # 非首子词添加 ##
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                return [self.unk_token]  # 未登录词
            
            sub_tokens.append(cur_substr)
            start = end
        
        return sub_tokens

# WordPiece 词表示例：
# "playing" → ["play", "##ing"]
# "unhappiness" → ["un", "##happi", "##ness"]
# "hello" → ["hello"]  (如果在词表中)
# "xyzabc" → ["[UNK]"]  (完全未登录)
```

## SentencePiece

SentencePiece 由 Google 开发，是一个统一的分词框架，支持 BPE 和 Unigram 两种算法。

### 核心创新

SentencePiece 的最大创新是**直接从原始文本（无需预分词）训练分词器**：

```
传统流程（BPE/WordPiece）：
原始文本 → 预分词（空格/标点） → 训练分词器

SentencePiece 流程：
原始文本 → 直接训练分词器
```

这使得 SentencePiece 天然支持所有语言，包括不分词的语言（如中文、日文）。

### 字符处理方式

```
SentencePiece 的特殊标记：
- ▁（U+2581）：词边界标记（替代空格）
- <s>：句子开始
- </s>：句子结束
- <unk>：未知词
- <pad>：填充

示例：
"Hello world!" → ["▁Hello", "▁world", "!"]
"你好世界" → ["▁你", "▁好", "▁世", "▁界"]  (字符级)
```

### SentencePiece 训练与使用

```python
import sentencepiece as spm

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(
    input='corpus.txt',          # 训练语料
    model_prefix='mymodel',      # 输出模型前缀
    vocab_size=32000,            # 词表大小
    model_type='unigram',        # 或 'bpe'
    character_coverage=0.9995,   # 字符覆盖率
    unk_id=0,                    # <unk> ID
    bos_id=1,                    # <s> ID
    eos_id=2,                    # </s> ID
    pad_id=-1,                   # <pad> ID
    split_by_whitespace=True,    # 是否按空格分割
    user_defined_symbols=[]      # 用户自定义符号
)

# 加载模型
sp = spm.SentencePieceProcessor()
sp.load('mymodel.model')

# 编码
text = "Hello, world! 你好世界"
tokens = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

print(f"Tokens: {tokens}")
print(f"IDs: {ids}")

# 解码
decoded = sp.decode_ids(ids)
print(f"Decoded: {decoded}")

# 采样编码（用于数据增强）
sp.set_enable_sampling(True)
sp.set_nbest_size(-1)
sampled_ids = sp.encode_as_ids(text)

# 逆运算
reversed_ids = sp.encode_as_ids(text, reverse=True)
```

### Unigram 算法

SentencePiece 的 Unigram 模式与 BPE 不同，它使用概率语言模型：

```python
# Unigram 算法伪代码
def unigram_train(corpus, vocab_size):
    """Unigram 分词器训练"""
    
    # 1. 初始化大词表（所有字符和常见子串）
    vocab = initialize_large_vocab(corpus)
    
    # 2. 迭代修剪
    while len(vocab) > vocab_size:
        # 计算每个 token 对损失函数的贡献
        contributions = compute_losses(vocab, corpus)
        
        # 保留贡献最大的 tokens
        vocab = keep_top_tokens(contributions, reduction_ratio=0.2)
        
        # 添加一些新的子词组合
        vocab = add_new_subwords(vocab, corpus)
    
    return vocab

# Unigram 编码：选择最优分词序列
def unigram_encode(text, vocab):
    """使用 Unigram 模型编码"""
    # Viterbi 算法寻找最优分词路径
    # P(sequence) = Π P(token_i)
    # 选择概率最高的分词序列
    return viterbi_decode(text, vocab)
```

## TikToken

TikToken 是 OpenAI 开发的 BPE 分词器，以速度和精确控制著称。

### 特点

| 特点 | 说明 |
|------|------|
| 极快 | 比 HuggingFace Tokenizer 快数倍 |
| 精确 | 与 GPT 系列模型使用的分词器完全一致 |
| 可控 | 允许精确指定词表和合并规则 |
| 专用 | 为特定模型定制（gpt-4、cl100k_base 等） |

### 使用方式

```python
import tiktoken

# 加载预训练编码器
encoding = tiktoken.get_encoding("cl100k_base")
# 或按模型名加载
encoding = tiktoken.encoding_for_model("gpt-4")

# 编码
text = "Hello, world! 你好世界"
tokens = encoding.encode(text)
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")

# 解码
decoded = encoding.decode(tokens)
print(f"Decoded: {decoded}")

# 获取 token 字符串
token_strings = [encoding.decode_single_token_bytes(t).decode('utf-8', errors='replace') 
                 for t in tokens]
print(f"Tokens: {token_strings}")

# 检查 token 是否在词表中
def is_valid_token(token_str):
    return token_str in encoding._special_tokens.values()

# 强制使用特定 token（特殊 token）
special_tokens = {
    "<|endoftext|>": encoding.eot_token,
    "<|fim_prefix|>": encoding._special_tokens.get("<|fim_prefix|>"),
}

# 计算对话的 token 数量
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

num_tokens = 0
for message in messages:
    num_tokens += 4  # 每条消息的固定开销
    for key, value in message.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += -1  # name 字段少一个 token
num_tokens += 2  # 对话的固定开销
print(f"Total tokens: {num_tokens}")
```

### 不同编码器的词表

| 编码器 | 词表大小 | 使用模型 | 特点 |
|--------|----------|----------|------|
| gpt2 | 50257 | GPT-2、GPT-3 | 基础 BPE |
| p50k_base | 50281 | GPT-3.5、Codex | 代码增强 |
| cl100k_base | 100277 | GPT-4、Embedding | 多语言 |
| o200k_base | 200019 | GPT-4o | 最新、最高效 |

## Tokenization 对模型性能的影响

### 压缩率与上下文效率

```
压缩率 = 原始字符数 / Token 数

不同语言/文本类型的压缩率：
- 英文常规文本：~3.5 字符/Token
- 中文：~1.5 字符/Token（每个字通常是 1 个 Token）
- 代码：~2.5 字符/Token
- URL：~5 字符/Token
- 数学公式：~1.8 字符/Token
```

### 词表大小的权衡

```python
# 词表大小对模型的影响
def analyze_vocab_impact(vocab_size):
    """分析词表大小的影响"""
    
    embedding_params = vocab_size * hidden_size  # 嵌入层参数
    
    # 词表大小的影响
    impacts = {
        "small_vocab": {
            "size": "< 30K",
            "pros": ["嵌入层参数少", "训练快"],
            "cons": ["频繁分解为子词", "语义信息碎片化"]
        },
        "medium_vocab": {
            "size": "30K - 50K",
            "pros": ["平衡性能和效率", "常见词完整保留"],
            "cons": ["低频词仍需分解"]
        },
        "large_vocab": {
            "size": "100K - 200K",
            "pros": ["更多完整词", "更好的多语言支持", "更高压缩率"],
            "cons": ["嵌入层参数多", "训练成本增加"]
        }
    }
    
    return impacts

# LLaMA 3 的词表扩大分析
# 32K → 128K 词表
# 嵌入层参数增加：32K * 4096 → 128K * 4096 = 4 倍
# 但压缩率提升 15%，同等上下文容纳更多内容
```

### 多语言 Tokenization

```python
# 多语言 tokenization 分析
def compare_multilingual_tokenization():
    """比较不同 Tokenizer 的多语言能力"""
    
    texts = {
        "english": "The quick brown fox jumps over the lazy dog.",
        "chinese": "人工智能正在改变世界。",
        "japanese": "人工知能が世界を変えています。",
        "korean": "인공지능이 세상을 바꾸고 있습니다.",
        "arabic": "الذكاء الاصطناعي يغير العالم.",
        "russian": "Искусственный интеллект меняет мир.",
        "code": "def hello(): print('Hello, world!')",
    }
    
    # 不同编码器的 token 数量
    results = {}
    for lang, text in texts.items():
        gpt2_tokens = len(tiktoken.get_encoding("gpt2").encode(text))
        cl100k_tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
        results[lang] = {
            "gpt2": gpt2_tokens,
            "cl100k": cl100k_tokens,
            "improvement": f"{(1 - cl100k_tokens/gpt2_tokens)*100:.1f}%"
        }
    
    return results

# 典型结果（近似）：
# 英文：gpt2=10, cl100k=10 → 0% 变化
# 中文：gpt2=21, cl100k=9 → 57% 减少
# 日文：gpt2=25, cl100k=13 → 48% 减少
```

## 特殊 Token 与标记

现代 Tokenizer 使用特殊 Token 来控制模型行为：

```python
# 常见特殊 Token
SPECIAL_TOKENS = {
    # 控制标记
    "<|endoftext|>": "文本结束",
    "<|begin_of_text|>": "文本开始（LLaMA 3）",
    "<s>": "句子开始",
    "</s>": "句子结束",
    "<pad>": "填充",
    "<unk>": "未知词",
    "<mask>": "掩码（BERT）",
    
    # 对话标记（LLaMA 3）
    "<|start_header_id|>": "角色开始",
    "<|end_header_id|>": "角色结束",
    "<|eot_id|>": "对话轮次结束",
    
    # 代码标记
    "<|fim_prefix|>": "代码填充前缀",
    "<|fim_middle|>": "代码填充中间",
    "<|fim_suffix|>": "代码填充后缀",
    
    # 工具使用
    "<|tool_call|>": "工具调用",
    "<|tool_response|>": "工具响应",
}

# LLaMA 3 对话格式编码
def encode_llama3_dialog(messages):
    """编码 LLaMA 3 对话格式"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = [encoding.encode("<|begin_of_text|>")]
    
    for msg in messages:
        role_tokens = encoding.encode(
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
        )
        content_tokens = encoding.encode(msg['content'])
        eot_tokens = encoding.encode("<|eot_id|>")
        
        tokens.append(role_tokens)
        tokens.append(content_tokens)
        tokens.append(eot_tokens)
    
    # 添加助手角色头
    tokens.append(encoding.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ))
    
    # 展平
    return [t for sublist in tokens for t in sublist]
```

## Tokenizer 训练实践

```python
# 使用 HuggingFace Tokenizers 库训练自定义 Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_custom_tokenizer(files, vocab_size=50000):
    """训练自定义 BPE Tokenizer"""
    
    # 初始化 Tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # 预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # 训练器配置
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|endoftext|>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<pad>",
            "<unk>",
        ]
    )
    
    # 训练
    tokenizer.train(files, trainer)
    
    # 配置解码器
    tokenizer.decoder = decoders.ByteLevel()
    
    # 保存
    tokenizer.save("custom-tokenizer.json")
    
    return tokenizer

# 使用训练好的 Tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("custom-tokenizer.json")

# 编码
output = tokenizer.encode("Hello, world!")
print(output.ids)        # Token IDs
print(output.tokens)     # Token 字符串
print(output.attention_mask)  # 注意力掩码
```

## 总结

Tokenizer 是 LLM 的基础设施，决定了文本到数字的映射方式：

1. **BPE** 是最广泛使用的方法，GPT 系列和 LLaMA 3（通过 TikToken）均采用
2. **WordPiece** 由 Google 提出，用于 BERT 和 T5，通过概率准则选择合并
3. **SentencePiece** 是统一的分词框架，支持直接从原始文本训练
4. **TikToken** 提供了高速、精确的分词，是 OpenAI 模型的标准选择

选择合适的 Tokenizer 需要考虑：
- 目标语言（单语言 vs 多语言）
- 词表大小（参数预算 vs 压缩率）
- 速度要求（在线推理 vs 离线训练）
- 与模型的兼容性（必须使用模型训练时相同的分词器）

Tokenizer 将文本转换为 Token ID 序列后，下一步就是将这些位置信息注入序列中，这就是位置编码要解决的问题。

💡 **提示**：在实际开发中，不要混用不同 Tokenizer。每个模型都有专属的分词器，使用错误的 Tokenizer 会导致完全错误的输出。可以通过 `tokenizer = AutoTokenizer.from_pretrained(model_name)` 自动获取正确的分词器。

## 下一篇

继续阅读 [位置编码](./66-position-encoding.md)，了解 LLM 如何理解 Token 的位置关系。
