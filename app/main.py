"""FastAPI 应用入口：暴露健康检查、销售查询和 Agent 问答接口。HTTP 请求真正从哪里进入"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.agent_service import AgentServiceError, run_agent
from app.config import DATABASE_FILE, get_document_settings
from app.document_service import (
    DocumentParseError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
    parse_document_bytes,
)
from app.logging_config import configure_logging
from app.schemas import (
    AgentRequest,
    AgentResponse,
    DocumentParseResponse,
    HealthResponse,
    SalesQueryResponse,
)
from app.tools import initialize_database, query_sales


logger = logging.getLogger(__name__)  # 获取当前模块的日志器,后面每个接口和中间件都用它记录日志,方便排查问题

# @asynccontextmanager 是 Python 标准库 contextlib 里的一个装饰器,把一个异步生成器函数，变成一个异步上下文管理器
# 这样就可以用 async with 来使用它，而不需要手动写一个类去实现 __aenter__ 和 __aexit__ 方法
@asynccontextmanager  
async def lifespan(_: FastAPI):  # 应用生命周期管理
    """在服务启动时配置日志并确保 SQLite 数据库已初始化。"""
    configure_logging()  # 配置日志
    initialize_database()  # 确保数据库已存在
    logger.info("应用启动完成，数据库：%s", DATABASE_FILE)  # 记录一条启动完成的日志
    yield  # 运行应用生命周期管理代码:yield 之前：应用启动时执行;yield 之后：应用关闭时执行，记录一条关闭日志。
    logger.info("应用已关闭")


app = FastAPI(
    title="公司销售 Agent API",
    description="通过 DeepSeek Tool Calling 调用计算器和 SQLite 销售查询工具。",
    version="1.0.0",
    lifespan=lifespan,  # 表示服务启动和关闭时使用上述生命周期函数。
)  # 创建 FastAPI 应用实例;这些元信息会自动出现在 FastAPI 生成的 Swagger 文档页面上

@app.middleware("http")  # HTTP 请求日志中间件,中间件是每个 HTTP 请求都会经过的一层
async def log_request(request: Request, call_next):
    """记录每个 HTTP 请求的方法、路径、状态码和耗时。"""
    start_time = time.perf_counter()  # 记录请求开始时间
    response = await call_next(request)  # 把请求交给真实路由函数处理
    elapsed_ms = (time.perf_counter() - start_time) * 1_000  # 计算请求耗时
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )  # 记录请求方法、路径、状态码和耗时。
    return response  # 返回处理后的响应


@app.get("/health", response_model=HealthResponse, tags=["system"])  # 创建健康检查接口;路径：GET /health
def health_check() -> HealthResponse:
    """
    返回服务状态和数据库文件是否存在。database_ready 通过检查数据库文件路径是否存在来判断;
    运维和监控系统用来判断服务是否正常
    """
    return HealthResponse(status="ok", database_ready=Path(DATABASE_FILE).exists())


@app.get("/sales", response_model=SalesQueryResponse, tags=["sales"])  # 创建销售查询接口;路径：GET /sales
def get_sales(
    region: str | None = Query(default=None, description="区域，例如：华东"),
    month: str | None = Query(default=None, description="月份，例如：2026-09"),
) -> SalesQueryResponse:  # region 和 month 都是可选的查询参数，通过 URL 传入，比如 /sales?region=华东。
    """直接调用 query_sales 函数查数据库，把结果包装成 SalesQueryResponse 返回，便于验证数据库和工具行为。"""
    result = query_sales(region=region, month=month)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result["error"])  # 抛出 HTTP 500 异常，把错误信息作为详情返回
    return SalesQueryResponse(count=result["count"], rows=result["rows"])


@app.post("/agent", response_model=AgentResponse, tags=["agent"])  # 创建销售查询接口;路径：POST /agent
def ask_agent(request: AgentRequest) -> AgentResponse:
    """前端把用户问题发到这里，Agent 自主决定是否调用工具，最终返回答案。"""
    try:
        result = run_agent(request.question.strip())  # 接收用户问题，交给 run_agent 处理
    except AgentServiceError as error:
        logger.warning("Agent 请求失败：%s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error
    return AgentResponse(answer=result.answer, tool_rounds=result.tool_rounds)  # 返回最终答案和工具调用轮次


@app.post(
    "/documents/parse",
    response_model=DocumentParseResponse,
    tags=["documents"],
)
async def parse_uploaded_document(
    file: UploadFile = File(description="支持 PDF、Markdown、HTML 和 CSV。"),
) -> DocumentParseResponse:
    """接收上传文件，完成读取、清洗、分块和元数据生成。"""
    settings = get_document_settings()
    filename = file.filename or ""

    # 只多读取 1 字节，用于判断文件是否超过限制，避免无限制占用内存。
    try:
        content = await file.read(settings.max_upload_size + 1)
    finally:
        await file.close()

    try:
        # PDF 解析和文本分块属于同步工作，放在线程池避免阻塞异步事件循环。
        return await run_in_threadpool(  # run_in_threadpool ：把一个同步函数提交到线程池里执行，并返回一个可等待的异步结果。
            parse_document_bytes,  # 把 parse_document_bytes 这个同步函数丢到线程池里去跑
            content,  # parse_document_bytes 的三个参数
            filename,
            settings,
        )
    except DocumentTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except UnsupportedDocumentTypeError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except DocumentParseError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
