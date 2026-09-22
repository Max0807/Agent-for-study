"""FastAPI 自动校验请求与响应时使用的 Pydantic 模型。
Pydantic 模型既是数据校验规则，也是 API 文档和接口契约
定义 API 的请求体和响应体长什么样
"""

from pydantic import BaseModel, Field


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
