"""文档解析编排层：验证文件、调用 loader、分块并组装响应。"""

from pathlib import Path

from app.chunking import split_sections
from app.config import DocumentSettings, get_document_settings
from app.document_loader import DocumentLoadError, load_document
from app.schemas import DocumentParseResponse


class DocumentParseError(ValueError):
    """所有可展示给 API 调用方的文档解析错误的基类。"""


class UnsupportedDocumentTypeError(DocumentParseError):
    """文件扩展名不在允许列表中。"""


class DocumentTooLargeError(DocumentParseError):
    """上传文件超过配置的大小限制。"""


def parse_document_bytes(
    content: bytes,
    filename: str,
    settings: DocumentSettings | None = None,
) -> DocumentParseResponse:
    """把内存中的文件内容解析为带元数据的 chunks。"""
    settings = settings or get_document_settings()
    safe_filename = Path(filename).name
    extension = Path(safe_filename).suffix.lower()

    if not safe_filename:  # 检查文件名
        raise DocumentParseError("文件名不能为空。")
    if extension not in settings.supported_extensions:  # 检查文件扩展名
        supported = ", ".join(sorted(settings.supported_extensions))
        raise UnsupportedDocumentTypeError(
            f"不支持 {extension or '无扩展名文件'}；支持的格式：{supported}。"
        )
    if not content:  # 检查文件是否为空
        raise DocumentParseError("上传的文件为空。")
    if len(content) > settings.max_upload_size:  # 检查文件大小
        limit_mb = settings.max_upload_size / (1024 * 1024)
        raise DocumentTooLargeError(f"文件超过 {limit_mb:g} MB 的大小限制。")

    try:
        sections = load_document(content, safe_filename)
    except DocumentLoadError as error:
        raise DocumentParseError(str(error)) from error

    if not sections:
        raise DocumentParseError("文档中没有可提取的文本；扫描版 PDF 可能需要 OCR。")

    chunks = split_sections(
        sections,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise DocumentParseError("文档清洗后没有可用内容。")

    return DocumentParseResponse(
        filename=safe_filename,
        file_type=sections[0].file_type,
        title=sections[0].title,
        chunk_count=len(chunks),
        chunks=chunks,
    )


def parse_document_path(
    path: Path,
    settings: DocumentSettings | None = None,
) -> DocumentParseResponse:
    """读取本地文件，再复用 parse_document_bytes 完成解析。"""
    if not path.is_file():
        raise DocumentParseError(f"文件不存在：{path}")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DocumentParseError(f"读取文件失败：{error}") from error
    return parse_document_bytes(content, path.name, settings=settings)
