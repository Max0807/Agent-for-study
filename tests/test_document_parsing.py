"""Day6 文档读取、清洗、分块和元数据测试。"""

from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.config import DocumentSettings, SUPPORTED_DOCUMENT_EXTENSIONS
from app.document_service import (
    DocumentParseError,
    UnsupportedDocumentTypeError,
    parse_document_bytes,
)
from app.text_cleaner import clean_text


TEST_SETTINGS = DocumentSettings(
    chunk_size=80,
    chunk_overlap=10,
    max_upload_size=1024 * 1024,
    supported_extensions=SUPPORTED_DOCUMENT_EXTENSIONS,
)


def test_clean_text_normalizes_whitespace() -> None:
    """清洗函数应移除空字符、多余空格和过多空行。"""
    raw_text = "第一段\x00   内容\r\n\r\n\r\n第二段\t内容"
    assert clean_text(raw_text) == "第一段 内容\n\n第二段 内容"


def test_parse_markdown_keeps_title_and_source() -> None:
    """Markdown 的 h1 应成为标题，每个 chunk 都应保留来源元数据。"""
    markdown = "# 员工手册\n\n" + "员工请假需要提前申请。" * 15
    result = parse_document_bytes(
        markdown.encode("utf-8"),
        "employee.md",
        settings=TEST_SETTINGS,
    )

    assert result.file_type == "markdown"
    assert result.title == "员工手册"
    assert result.chunk_count > 1
    assert all(chunk.source == "employee.md" for chunk in result.chunks)
    assert all(chunk.page_number is None for chunk in result.chunks)
    assert [chunk.chunk_index for chunk in result.chunks] == list(range(result.chunk_count))


def test_parse_html_removes_script_and_reads_title() -> None:
    """HTML 解析结果不应包含脚本内容，并应优先读取 title。"""
    html = b"""
    <html><head><title>Travel Policy</title><script>secret()</script></head>
    <body><h1>Policy</h1><p>Employees need approval.</p></body></html>
    """
    result = parse_document_bytes(html, "policy.html", settings=TEST_SETTINGS)

    assert result.title == "Travel Policy"
    assert "Employees need approval" in result.chunks[0].text
    assert "secret" not in result.chunks[0].text


def test_parse_csv_converts_rows_to_readable_text() -> None:
    """CSV 应把表头和值转换为可检索文本。"""
    csv_content = "region,month,amount\n华东,2026-09,128000\n华南,2026-09,96000\n"
    result = parse_document_bytes(
        csv_content.encode("utf-8"),
        "sales.csv",
        settings=TEST_SETTINGS,
    )

    combined_text = "\n".join(chunk.text for chunk in result.chunks)
    assert result.file_type == "csv"
    assert result.title == "sales"
    assert "region: 华东" in combined_text
    assert "amount: 128000" in combined_text


def test_blank_pdf_reports_ocr_hint() -> None:
    """没有文字层的 PDF 应返回清晰错误，而不是生成空 chunks。"""
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)

    with pytest.raises(DocumentParseError, match="OCR"):
        parse_document_bytes(buffer.getvalue(), "scan.pdf", settings=TEST_SETTINGS)


def test_unsupported_extension_is_rejected() -> None:
    """不在白名单中的扩展名应在解析前被拒绝。"""
    with pytest.raises(UnsupportedDocumentTypeError):
        parse_document_bytes(b"hello", "notes.txt", settings=TEST_SETTINGS)
