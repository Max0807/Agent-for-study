# Agent Doc

# **什么是RAG**

**LangChain**的官方文档：https://docs.langchain.com/oss/python/deepagents/retrieval

## **1. RAG 解决的两个问题**

RAG 的本质是：**在回答时临时检索外部知识，再把相关内容交给 LLM 生成答案**。它主要解决两个限制：

| 问题 | 含义 | RAG 怎么解决 |
| --- | --- | --- |
| 有限上下文 | LLM 不能一次读完整个知识库、全部文档、所有历史数据 | 先**检索最相关的片段**，只把少量高价值上下文**放进 prompt** |
| 静态知识 | 模型训练数据停留在某个时间点，无法天然知道新文档、新规则、新业务数据 | 查询时连接**外部知识源**，如文档库、SQL、CRM、API、网页、向量库 |

所以 RAG 不是“让模型记住更多”，而是“让模型在需要时查资料”。

## **2. Loader、Splitter、Embedding、Vector Store、Retriever**

典型流程可以理解为：

```
原始数据 -> Loader -> Document -> Splitter -> Chunks
       -> Embedding -> Vectors -> Vector Store
       -> Retriever -> 相关 Documents -> LLM 生成答案
```

| 组件 | 作用 | 你可以这样理解 |
| --- | --- | --- |
| Loader | 从外部来源加载数据，并转成 LangChain 标准 `Document` 对象 | “把 PDF、网页、Notion、Slack、Google Drive 等资料**搬进系统**” |
| Splitter | 把大文档切成小块 | “**切片**”，让每个片段足够小、语义集中、能放进上下文 |
| Embedding | 把文本转成数字向量 | “**语义坐标**”，意思相近的文本在向量空间里更接近 |
| Vector Store | **存储 embedding**，并支持相似度搜索 | “语义搜索数据库”，如 Chroma、Qdrant、Pinecone、PGVector |
| Retriever | 根据用户问题**返回**相关 `Document` | “统一检索接口”，上层 RAG/Agent 不必关心底层是向量库、API 还是数据库 |

几个容易踩坑但很重要的点：

- `Document` 不只是文本，通常包含 `page_content`、`metadata`、可选 `id`。`metadata` 很关键，用来保存来源、页码、时间、权限、业务标签等。
- **Splitter 不只是为了“变短”，更是为了避免相关信息被大段无关文本稀释。**
- 文档里推荐通用文本场景用 `RecursiveCharacterTextSplitter`，常见参数是 `chunk_size`、`chunk_overlap`、`add_start_index`。
- Embedding 检索适合语义相似，不依赖完全相同关键词；但关键词、数字、专有名词场景可能还要结合 BM25/SQL/filter。
- **Vector Store 的相似度分数**含义可能因实现而异，有的分数越高越相似，有的距离越低越相似。
- Retriever 比 Vector Store 更适合**接入链路**，因为 retriever 是 Runnable，支持 `invoke`、`batch`、async 等标准调用方式。