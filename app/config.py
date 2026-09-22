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
