"""FastAPI 自动校验请求与响应时使用的 Pydantic 模型。
Pydantic 模型既是数据校验规则，也是 API 文档和接口契约
定义 API 的请求体和响应体长什么样
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentRequest(BaseModel):
    """POST /agent 的请求体。它定义 /agent 接收的 JSON"""

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="发送给销售助手的问题。",
        examples=["华东 2026-09 的销售额是多少？"],
    )  # 在进入 ask_agent() 前，FastAPI 自动做校验


class AgentResponse(BaseModel):
    """POST /agent 成功时返回的模型回答。它约束 /agent 成功时的返回格式："""

    answer: str
    tool_rounds: int = Field(ge=0, description="本次请求实际完成的工具调用轮数。")  # ge 是 Pydantic 的 Field 函数中的一个参数，代表 “大于等于”


class HealthResponse(BaseModel):
    """GET /health 的服务状态响应。"""

    status: str
    database_ready: bool


class SalesRow(BaseModel):
    """一条销售记录。"""

    region: str
    month: str
    amount: float


class SalesQueryResponse(BaseModel):
    """GET /sales 的查询结果。"""

    count: int
    rows: list[SalesRow]


class DocumentChunk(BaseModel):
    """清洗和分块后的单个文档片段。"""

    text: str = Field(description="当前 chunk 的正文。")
    source: str = Field(description="原始文件名，用于后续引用来源。")
    file_type: str = Field(description="pdf、markdown、html 或 csv。")  # 文件格式
    page_number: int | None = Field(
        default=None,
        description="PDF 的页码；没有分页概念的格式为 null。",
    )
    title: str = Field(description="文档标题；无法提取时使用文件名。")
    chunk_index: int = Field(ge=0, description="chunk 在本次解析结果中的顺序编号。")  # 分块顺序


class DocumentParseResponse(BaseModel):
    """POST /documents/parse 和命令行解析工具的统一输出格式。"""

    filename: str
    file_type: str
    title: str
    chunk_count: int = Field(ge=0)
    chunks: list[DocumentChunk]


class IndexFileResult(BaseModel):
    """单个知识库文件建立索引后的结果，用于表示单个文件是否索引成功。"""

    source: str = Field(description="知识库目录中的相对文件路径。")
    success: bool = Field(description="当前文件是否成功写入向量库。")
    chunk_count: int = Field(default=0, ge=0, description="成功写入的文本块数量。")
    error: str | None = Field(default=None, description="失败原因；成功时为 null。")


class IndexDirectoryResult(BaseModel):
    """批量建立知识库索引后的汇总信息。"""

    total_files: int = Field(ge=0)
    indexed_files: int = Field(ge=0)
    failed_files: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    files: list[IndexFileResult]


class KnowledgeSearchRequest(BaseModel):
    """POST /knowledge/search 接收的语义搜索条件。接收可选 metadata 过滤条件"""

    query: str = Field(
        min_length=1,
        max_length=2_000,
        description="需要在知识库中检索的自然语言问题。",
        examples=["员工出差前需要谁审批？"],
    )  # 
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="最多返回多少个结果；不填写时读取 SEARCH_TOP_K。",
    )
    source: str | None = Field(default=None, description="可选：只检索指定来源文件。")
    file_type: str | None = Field(
        default=None,
        description="可选：只检索 pdf、markdown、html 或 csv。",
    )
    title: str | None = Field(default=None, description="可选：只检索指定标题。")


class KnowledgeSearchHit(BaseModel):
    """一个命中的知识库文本块及其可追溯元数据。"""

    text: str
    source: str
    file_type: str
    page_number: int | None = None
    title: str
    chunk_index: int = Field(ge=0)
    distance: float = Field(description="Chroma 返回的距离；越小通常表示越相似。")


class KnowledgeSearchResponse(BaseModel):
    """知识库语义搜索的统一返回结构。"""

    query: str
    count: int = Field(ge=0)
    results: list[KnowledgeSearchHit]


class KnowledgeStatsResponse(BaseModel):
    """向量知识库的集合名称和当前文本块数量。"""

    collection: str
    chunk_count: int = Field(ge=0)


class RagRequest(BaseModel):
    """POST /rag/ask 接口的请求体模型: 规定了前端调用这个接口时, 接收的问题、召回数量和可选检索过滤条件。"""

    # 给 Swagger 一个可以直接执行的示例，避免可选字段被误填成 "string"。
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "员工出差前需要谁审批？",
                "top_k": 3,
            }
        }
    )

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="需要根据知识库回答的问题。",
    )  # 必填字段
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="召回的知识块数量；不填写时使用 RAG_TOP_K。",
    )  # | None：这个是可选字段，不填时有settings.default_top_k
    source: str | None = Field(default=None, description="可选：只检索指定来源文件。")
    file_type: str | None = Field(
        default=None,
        description="可选：只检索 pdf、markdown、html 或 csv。",
    )  # 
    title: str | None = Field(default=None, description="可选：只检索指定标题。")
    # 这三个可选的过滤条件，它们都会传给 search_knowledge_base，最终转换成 Chroma 的 where 过滤条件


class ChatTurn(BaseModel):
    """一条由客户端传入的历史对话，只允许用户和助手两种角色。"""

    role: Literal["user", "assistant"] = Field(description="这条消息的发送方。")
    content: str = Field(
        min_length=1,
        max_length=2_000,
        description="历史消息正文。",
    )


class RagStreamRequest(RagRequest):
    """
    POST /rag/stream 的请求体：当前问题加客户端保存的历史。
    它既包含当前问题和 top_k，又增加了历史对话
    重点理解：服务端没有保存历史，历史由客户端每次重新传入。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "那住宿方面呢？",
                "history": [
                    {"role": "user", "content": "员工出差前需要谁审批？"},
                    {"role": "assistant", "content": "需要直属主管审批。[S1]"},
                ],
                "top_k": 4,
            }
        }
    )

    # 允许客户端传入多于 8 条，业务层会按配置裁剪并保留最近的消息。
    # 50 是接口的绝对安全上限，防止异常大的请求占用过多内存。
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)  # 一个 ChatTurn 对象的列表


class RagCitation(BaseModel):  # 表示一个引用来源
    """一条提供给模型的知识库来源，用于追溯答案依据。这条答案依据哪个地方"""

    source_id: str = Field(description="上下文中的来源编号，例如 S1。")
    source: str  # 来源文件名
    title: str  # 来源标题
    page_number: int | None = None  # 来自哪一页
    chunk_index: int = Field(ge=0)  # 来自哪个chunk
    distance: float  # 检索距离是多少


class CitationValidation(BaseModel):
    """模型答案的引用校验结果，用来识别来源列表里不存在的例如 [S99] 等编号。"""

    cited_source_ids: list[str] = Field(description="答案中实际出现的去重引用编号。")
    invalid_source_ids: list[str] = Field(description="不在当前来源列表中的编号。")
    is_valid: bool = Field(description="是否没有发现非法引用。")


class RagResponse(BaseModel):
    """Naive RAG 返回的答案、检索数量和引用来源。"""

    question: str
    answer: str
    retrieved_count: int = Field(ge=0)
    citations: list[RagCitation]
    model: str | None = Field(
        default=None,
        description="实际生成答案的模型；未调用模型时为 null。",
    )
    citation_validation: CitationValidation | None = Field(
        default=None,
        description="答案中 [Sx] 编号的合法性检查结果。",
    )
