"""应用配置：读取 .env，并创建 DeepSeek 兼容客户端。"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# app 目录的上一级就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_FILE = DATA_DIR / "company.db"
SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".md", ".markdown", ".html", ".htm", ".csv"}
)

# 启动时读取项目根目录的 .env,并把内容放入 Python 的环境变量中；其中的变量不会被提交到 Git。
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)  # frozen=True 表示创建后不能随意修改，避免程序运行中配置被改坏
class Settings:
    """保存服务运行所需的外部配置。"""

    api_key: str
    model: str
    base_url: str


@dataclass(frozen=True)
class DocumentSettings:
    """保存文档上传和分块相关配置。"""

    chunk_size: int  # 每段最多多少字符；
    chunk_overlap: int  # 相邻 chunk 重复多少字符
    max_upload_size: int  # 上传文件大小限制,防止用户上传过大文件
    supported_extensions: frozenset[str]  # 支持的文档格式;格式白名单


@dataclass(frozen=True)
class EmbeddingSettings:
    """保存 Day7 文本向量化和 Chroma 向量库相关配置。"""

    model_name: str  # Sentence Transformers 模型名称或本地模型目录
    vector_db_path: Path  # Chroma 持久化文件的保存目录
    collection_name: str  # Chroma collection 名称，作用类似数据库中的表名
    batch_size: int  # 每批送入 Embedding 模型的文本块数量
    default_top_k: int  # 没有显式指定时，默认返回的相似文本块数量


@dataclass(frozen=True)
class RagSettings:
    """保存 Day8/Day9 RAG 的检索、上下文、输出和历史限制。"""

    default_top_k: int  # 一次 RAG 问答默认召回多少个知识库 chunk;默认从知识库找几段资料
    max_context_chars: int  # 最多向大模型发送多少个上下文字符
    max_output_tokens: int  # 大模型回答最多生成多少个 Token
    max_distance: float = 0.55  # 检索距离上限；超过该值的文本块不进入模型上下文
    history_max_messages: int = 8  # 多轮对话最多保留的历史消息数
    history_max_chars: int = 6_000  # 多轮对话历史的总字符数上限


def get_settings() -> Settings:
    """从环境变量读取并校验 DeepSeek 配置。服务启动时不一定会调用它；只有真正请求 /agent、需要调用模型时才会调用"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY,请检查项目根目录的 .env 文件。")

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    return Settings(api_key=api_key, model=model, base_url=base_url)


def create_client() -> tuple[OpenAI, Settings]:
    """根据环境配置创建 OpenAI 兼容的 DeepSeek 客户端。"""
    settings = get_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    return client, settings


def get_document_settings() -> DocumentSettings:
    """读取文档解析配置，并检查 chunk 参数是否合理。"""
    chunk_size = int(os.getenv("DOCUMENT_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "120"))
    max_upload_size = int(os.getenv("DOCUMENT_MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))  # 10MB

    if chunk_size <= 0:
        raise RuntimeError("DOCUMENT_CHUNK_SIZE 必须大于 0。")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise RuntimeError("DOCUMENT_CHUNK_OVERLAP 必须大于等于 0 且小于 chunk size。")
    if max_upload_size <= 0:
        raise RuntimeError("DOCUMENT_MAX_UPLOAD_SIZE 必须大于 0。")

    return DocumentSettings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_upload_size=max_upload_size,
        supported_extensions=SUPPORTED_DOCUMENT_EXTENSIONS,
    )


def get_embedding_settings() -> EmbeddingSettings:
    """读取并校验 Embedding 模型、批处理和向量库配置。"""
    model_name = os.getenv(
        "EMBEDDING_MODEL",  # 模型名称，用哪个 Embedding 模型
        "BAAI/bge-small-zh-v1.5",
    ).strip()
    vector_db_value = os.getenv("VECTOR_DB_PATH", "data/chroma").strip()  # 向量数据库存在哪
    collection_name = os.getenv(
        "VECTOR_COLLECTION",  # 向量库里的集合名
        "company_knowledge",
    ).strip()
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))  # 批处理大小，一次处理多少条文本
    default_top_k = int(os.getenv("SEARCH_TOP_K", "5"))  # 默认召回数，检索时默认返回几条结果

    if not model_name:
        raise RuntimeError("EMBEDDING_MODEL 不能为空。")
    if not vector_db_value:
        raise RuntimeError("VECTOR_DB_PATH 不能为空。")
    if not collection_name:
        raise RuntimeError("VECTOR_COLLECTION 不能为空。")
    if batch_size <= 0:
        raise RuntimeError("EMBEDDING_BATCH_SIZE 必须大于 0。")
    if not 1 <= default_top_k <= 20:
        raise RuntimeError("SEARCH_TOP_K 必须在 1 到 20 之间。")

    vector_db_path = Path(vector_db_value)
    if not vector_db_path.is_absolute():
        vector_db_path = PROJECT_ROOT / vector_db_path

    return EmbeddingSettings(
        model_name=model_name,
        vector_db_path=vector_db_path.resolve(),
        collection_name=collection_name,
        batch_size=batch_size,
        default_top_k=default_top_k,
    )


def get_rag_settings() -> RagSettings:
    """读取并校验 Day8/Day9 RAG 的运行参数。"""
    default_top_k = int(os.getenv("RAG_TOP_K", "5"))
    max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
    max_output_tokens = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "800"))
    max_distance = float(os.getenv("RAG_MAX_DISTANCE", "0.55"))  # 检索距离超过0.55的资料不要
    history_max_messages = int(os.getenv("RAG_HISTORY_MAX_MESSAGES", "8"))  # 最多保留8条历史消息
    history_max_chars = int(os.getenv("RAG_HISTORY_MAX_CHARS", "6000"))  # 历史消息最多6000字符

    if not 1 <= default_top_k <= 20:
        raise RuntimeError("RAG_TOP_K 必须在 1 到 20 之间。")
    if max_context_chars <= 0:
        raise RuntimeError("RAG_MAX_CONTEXT_CHARS 必须大于 0。")
    if max_output_tokens <= 0:
        raise RuntimeError("RAG_MAX_OUTPUT_TOKENS 必须大于 0。")
    if max_distance < 0:
        raise RuntimeError("RAG_MAX_DISTANCE 必须大于等于 0。")
    if history_max_messages <= 0:
        raise RuntimeError("RAG_HISTORY_MAX_MESSAGES 必须大于 0。")
    if history_max_chars <= 0:
        raise RuntimeError("RAG_HISTORY_MAX_CHARS 必须大于 0。")

    return RagSettings(
        default_top_k=default_top_k,
        max_context_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        max_distance=max_distance,
        history_max_messages=history_max_messages,
        history_max_chars=history_max_chars,
    )
