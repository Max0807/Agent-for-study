"""读取 PDF、Markdown、HTML 和 CSV，并转成统一的文档段落。"""

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.text_cleaner import clean_text, html_to_clean_text, markdown_to_clean_text


class DocumentLoadError(ValueError):
    """表示文件内容无法被对应解析器读取。"""


@dataclass(frozen=True)  # 保证只可读和安全
class DocumentSection:
    """分块前的标准文档段落；PDF 通常每页对应一个段落。"""

    text: str
    source: str
    file_type: str
    page_number: int | None
    title: str


def fallback_title(filename: str) -> str:
    """无法从文档内部提取标题时，使用不带扩展名的文件名。"""
    return Path(filename).stem or "未命名文档"


def decode_text_content(content: bytes) -> str:
    """优先按 UTF-8 读取文本；失败时兼容常见的中文 GB18030 编码。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("文本文件不是有效的 UTF-8 或 GB18030 编码。")


def load_pdf(content: bytes, filename: str) -> list[DocumentSection]:
    """逐页提取 PDF 文本，并为每个段落保留从 1 开始的真实页码。"""
    try:
        reader = PdfReader(BytesIO(content))  # 读取 PDF 文件
    except (PdfReadError, EOFError, ValueError) as error:
        raise DocumentLoadError(f"PDF 文件无法读取：{error}") from error

    metadata_title = reader.metadata.title if reader.metadata else None
    title = clean_text(metadata_title) if metadata_title else fallback_title(filename)  # 获取文档标题
    sections: list[DocumentSection] = []  # 逐页提取文本并封装

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")  # 提取文本
        if text:
            sections.append(
                DocumentSection(
                    text=text,
                    source=filename,
                    file_type="pdf",
                    page_number=page_number,
                    title=title,
                )
            )
    return sections


def load_markdown(content: bytes, filename: str) -> list[DocumentSection]:
    """读取 Markdown，以第一个一级标题作为文档标题。"""
    markdown = decode_text_content(content)
    extracted_title, text = markdown_to_clean_text(markdown)
    if not text:
        return []
    return [
        DocumentSection(
            text=text,
            source=filename,
            file_type="markdown",
            page_number=None,
            title=extracted_title or fallback_title(filename),
        )
    ]


def load_html(content: bytes, filename: str) -> list[DocumentSection]:
    """读取 HTML，删除不可见元素并提取 title 或 h1。"""
    html = decode_text_content(content)
    extracted_title, text = html_to_clean_text(html)
    if not text:
        return []
    return [
        DocumentSection(
            text=text,
            source=filename,
            file_type="html",
            page_number=None,
            title=extracted_title or fallback_title(filename),
        )
    ]


def load_csv(content: bytes, filename: str) -> list[DocumentSection]:
    """把 CSV 的每一行转换为“字段名: 值”的可检索文本。"""
    csv_text = decode_text_content(content)  # 把二进制内容解码成文本字符串
    reader = csv.DictReader(StringIO(csv_text))  # 把字符串包装成“内存中的文件对象”，然后读取 CSV，把每一行变成一个字典
    if not reader.fieldnames:
        raise DocumentLoadError("CSV 文件缺少表头。")

    rows: list[str] = []
    for row_number, row in enumerate(reader, start=1):
        fields: list[str] = []
        for key, value in row.items():
            if key is None:
                continue
            value_text = clean_text(str(value or ""))  # 把值转成字符串并清洗
            if value_text:
                fields.append(f"{clean_text(str(key))}: {value_text}")
        if fields:
            rows.append(f"第 {row_number} 行\n" + "\n".join(fields))

    text = clean_text("\n\n".join(rows))
    if not text:
        return []
    return [
        DocumentSection(
            text=text,
            source=filename,
            file_type="csv",
            page_number=None,
            title=fallback_title(filename),
        )
    ]


def load_document(content: bytes, filename: str) -> list[DocumentSection]:
    """根据文件扩展名选择解析器，并返回统一的 DocumentSection 列表。"""
    extension = Path(filename).suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".md": load_markdown,
        ".markdown": load_markdown,
        ".html": load_html,
        ".htm": load_html,
        ".csv": load_csv,
    }
    loader = loaders.get(extension)
    if loader is None:
        raise DocumentLoadError(f"没有可用于 {extension or '无扩展名文件'} 的解析器。")
    return loader(content, filename)
