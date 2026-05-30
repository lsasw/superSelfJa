---
title: TorchText 自然语言处理实战
icon: type
order: 56
---

# TorchText 自然语言处理实战

TorchText 是 PyTorch 官方的自然语言处理（NLP）库，提供了文本数据处理、词表构建、数据管道等核心功能。虽然近年来 Hugging Face Transformers 库在 NLP 领域占据主导地位，但理解 TorchText 的基础概念对于构建自定义 NLP 系统仍然非常有价值。本文将系统学习 TorchText 的使用方法。

## TorchText 核心组件

| 组件 | 功能 | 说明 |
|------|------|------|
| `torchtext.data` | 数据管道 | 数据迭代器和字段定义 |
| `torchtext.vocab` | 词表 | 词到 ID 的映射 |
| `torchtext.datasets` | 数据集 | AG News、IMDB 等预置数据集 |
| `torchtext.functional` | 函数 | 转换函数 |
| `torchtext.transforms` | 变换 | 文本到 Tensor 的转换 |

## 文本预处理基础

### 分词（Tokenization）

```python
import torch
from torchtext.data.utils import get_tokenizer

# 内置分词器
tokenizer = get_tokenizer('basic_english')

text = "Hello world! This is PyTorch. NLP is fascinating."
tokens = tokenizer(text)
print(f"Tokens: {tokens}")
# ['hello', 'world', '!', 'this', 'is', 'pytorch', '.', 'nlp', 'is', 'fascinating', '.']

# 其他分词器
# tokenizer_spacy = get_tokenizer('spacy', 'en_core_web_sm')  # 需要安装 spacy
# tokenizer_moses = get_tokenizer('moses')                     # 需要安装 sacremoses

# 中文分词（使用 jieba）
# import jieba
# tokenizer_zh = lambda x: list(jieba.cut(x))
# tokens_zh = tokenizer_zh("我爱自然语言处理")
# print(f"中文分词: {tokens_zh}")
```

### 构建词表

```python
from torchtext.vocab import build_vocab_from_iterator, Vocab

def yield_tokens(data_iter):
    """从数据迭代器中生成 token"""
    for text in data_iter:
        yield tokenizer(text)

# 示例数据
texts = [
    "I love machine learning",
    "PyTorch is great for deep learning",
    "Natural language processing is fun",
    "Deep learning models need lots of data",
    "I love PyTorch",
]

# 构建词表
vocab = build_vocab_from_iterator(
    yield_tokens(texts),
    min_freq=1,           # 最小词频
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],  # 特殊 token
    special_first=True
)

# 设置默认索引（用于未知词）
vocab.set_default_index(vocab['<unk>'])

print(f"Vocab size: {len(vocab)}")
print(f"Word to ID: 'learning' -> {vocab['learning']}")
print(f"ID to Word: 3 -> {vocab.lookup_token(3)}")
print(f"Lookup words: {vocab.lookup_tokens(['i', 'love', 'pytorch'])}")
```

### 词表操作

```python
# 查看词表信息
print(f"Vocab items: {list(vocab.get_stoi().items())}")

# 编码文本
text = "I love deep learning"
token_ids = vocab(tokenizer(text))
print(f"Encoded: {token_ids}")

# 批量编码
token_ids_batch = vocab(tokenizer("PyTorch is awesome"))
print(f"Batch encoded: {token_ids_batch}")

# 解码
decoded = vocab.lookup_tokens(token_ids)
print(f"Decoded: {decoded}")

# 添加新词
# vocab.insert_token('transformer', 5)
```

## Dataset 与 DataLoader

### 使用内置数据集

```python
from torchtext.datasets import AGNews, IMDB, SogouNews, YahooAnswers

# AG News 数据集（新闻分类）
# 下载并缓存到 ~/.torchtext/cache
# train_iter = AGNews(split='train')
# test_iter = AGNews(split='test')

# 遍历数据
# for label, text in train_iter:
#     print(f"Label: {label}")
#     print(f"Text: {text[:100]}...")
#     break
```

### 自定义文本 Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, texts, labels, tokenizer, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 分词
        tokens = self.tokenizer(text)

        # 编码
        token_ids = self.vocab(tokens)

        # 截断/填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.max_length - len(token_ids))

        return torch.tensor(token_ids), torch.tensor(label)

# 创建数据集
texts = [
    "I love machine learning and deep learning",
    "This movie is terrible and boring",
    "PyTorch is an excellent deep learning framework",
    "The food was delicious and the service was great",
    "I hate waiting in long lines",
]
labels = [1, 0, 1, 1, 0]  # 1=正面, 0=负面

tokenizer = get_tokenizer('basic_english')

# 构建词表
def yield_tokens(texts):
    for text in texts:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(texts),
    specials=['<unk>', '<pad>'],
    special_first=True
)
vocab.set_default_index(vocab['<unk>'])

dataset = TextClassificationDataset(texts, labels, tokenizer, vocab, max_length=20)

# DataLoader
def collate_fn(batch):
    """自定义 collate 函数"""
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    texts = torch.stack(texts)
    labels = torch.tensor(labels)
    return texts, labels

# loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
# for batch_texts, batch_labels in loader:
#     print(f"Texts: {batch_texts.shape}")
#     print(f"Labels: {batch_labels}")
```

### Padding 处理

```python
from torch.nn.utils.rnn import pad_sequence

def padded_collate_fn(batch):
    """处理变长序列的 collate 函数"""
    texts = []
    labels = []

    for text, label in batch:
        texts.append(text)
        labels.append(label)

    # padding（batch_first=True）
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    labels = torch.tensor(labels)

    # 创建 attention mask
    mask = (texts_padded != vocab['<pad>']).long()

    return texts_padded, labels, mask

# 使用变长数据
dataset_variable = TextClassificationDataset(texts, labels, tokenizer, vocab, max_length=50)
# loader = DataLoader(dataset_variable, batch_size=2, collate_fn=padded_collate_fn)
```

## 文本分类模型

### TextCNN

```python
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """TextCNN 文本分类模型"""

    def __init__(self, vocab_size, embed_dim, num_classes,
                 filter_sizes=(3, 4, 5), num_filters=100, dropout=0.5,
                 pad_idx=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 多尺寸卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)          # [batch, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)       # [batch, 1, seq_len, embed_dim]

        # 多尺寸卷积 + 最大池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded)).squeeze(3)  # [batch, num_filters, seq_len - fs + 1]
            pooled = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)  # [batch, num_filters]
            conv_outputs.append(pooled)

        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(filter_sizes)]
        concatenated = self.dropout(concatenated)

        return self.fc(concatenated)

# 创建模型
vocab_size = len(vocab)
embed_dim = 128
num_classes = 2

model = TextCNN(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_classes=num_classes,
    filter_sizes=(3, 4, 5),
    num_filters=100,
    dropout=0.5,
    pad_idx=vocab['<pad>']
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### RNN 文本分类

```python
class RNNClassifier(nn.Module):
    """RNN/LSTM 文本分类"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.5, bidirectional=True, pad_idx=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        rnn_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # RNN
        if lengths is not None:
            # 打包变长序列
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.rnn(packed)
        else:
            output, (hidden, cell) = self.rnn(embedded)

        # 使用最后一个隐藏状态
        if self.rnn.bidirectional:
            # 拼接前向和后向的最后一个隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        return self.fc(hidden)

# 创建模型
rnn_model = RNNClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=256,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.5,
    bidirectional=True,
    pad_idx=vocab['<pad>']
)

print(f"RNN parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
```

## 预训练词向量

```python
from torchtext.vocab import GloVe, FastText

# 加载 GloVe 词向量
# glove = GloVe(name='6B', dim=100)
# print(f"GloVe vocab size: {len(glove)}")

# 查找词的向量
# vector = glove['king']
# print(f"Vector for 'king': {vector[:5]}...")

# 查找相似词
# most_similar = glove.get_most_similar('king', topn=5)
# print(f"Most similar to 'king': {most_similar}")

# 词类比推理
# result = glove.get analogy('man', 'woman', 'king')
# print(f"man is to woman as king is to: {result}")

# 加载 FastText
# fasttext = FastText(language='en')

# 在模型中使用预训练词向量
def load_pretrained_embeddings(vocab, glove_model, embed_dim=100):
    """加载预训练词向量到 Embedding 层"""
    embedding_matrix = torch.zeros(len(vocab), embed_dim)

    for word, idx in vocab.get_stoi().items():
        if word in glove_model:
            embedding_matrix[idx] = glove_model[word]
        # 未知词保持零向量或使用随机初始化

    return embedding_matrix

# embedding_matrix = load_pretrained_embeddings(vocab, glove)
# model.embedding.weight.data.copy_(embedding_matrix)
# model.embedding.weight.requires_grad = False  # 冻结词向量
```

## 序列到序列模型（Seq2Seq）

```python
class Seq2Seq(nn.Module):
    """Seq2Seq 模型（机器翻译）"""

    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def create_src_mask(self, src):
        """创建源序列 mask"""
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt):
        """创建目标序列 mask（包含 padding 和因果 mask）"""
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return decoder_output

class Encoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 forward_dim, dropout, pad_idx, max_len=5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=forward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float))

    def forward(self, src, src_mask):
        seq_len = src.shape[1]
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0)

        x = self.embedding(src) * self.scale + self.pos_embedding(positions)
        x = self.dropout(x)

        output = self.transformer_encoder(x, src_key_padding_mask=(src == self.encoder.transformer_encoder.layers[0].norm1.eps))
        return output

class Decoder(nn.Module):
    """Transformer Decoder"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 forward_dim, dropout, pad_idx, max_len=5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=forward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float))

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        seq_len = tgt.shape[1]
        positions = torch.arange(0, seq_len, device=tgt.device).unsqueeze(0)

        x = self.embedding(tgt) * self.scale + self.pos_embedding(positions)
        x = self.dropout(x)

        output = self.transformer_decoder(x, encoder_output,
                                          tgt_mask=tgt_mask,
                                          memory_key_padding_mask=src_mask)
        return self.fc_out(output)
```

## 完整训练流程

```python
def train_text_classifier(model, train_loader, val_loader, num_epochs, device):
    """训练文本分类器"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(avg_loss)

    return model

# 使用示例
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trained_model = train_text_classifier(model, train_loader, val_loader, num_epochs=10, device=device)
```

## 总结

本文系统学习了 TorchText 的核心功能：

- **分词与词表**：文本预处理的基础操作
- **Dataset 与 DataLoader**：构建 NLP 数据管道
- **变长序列处理**：padding、masking 和 pack_padded_sequence
- **TextCNN**：使用卷积进行文本分类
- **RNN/LSTM**：循环神经网络文本分类
- **预训练词向量**：GloVe、FastText 的加载与使用
- **Seq2Seq**：基于 Transformer 的序列到序列模型
- **训练流程**：完整的文本分类训练代码

在下一篇文章中，我们将学习如何在 PyTorch 中自定义网络层，构建更加灵活和创新的模型架构。

[上一篇：TorchVision 计算机视觉实战](./55-torchvision.md) | [下一篇：自定义网络层 →](./57-custom-layer.md)
