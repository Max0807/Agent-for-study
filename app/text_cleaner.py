"""不同文档格式共用的文本清洗函数。"""

import re

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


def clean_text(text: str) -> str:
    """统一换行和空白，删除空字符，同时保留有意义的段落边界。"""
    text = text.replace("\x00", "").replace("\u00a0", " ")  # 替换空字符和不换行空格
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # 统一windows和linux的换行符

    # 按换行符把文本拆成一行一行的列表，然后把行内连续的多个空格或制表符压缩成一个空格；这里只压缩行内空格，不会把不同行合并。
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(cleaned_lines)  # 把清理好的各行用换行符重新拼成一整段文本
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # 把连续 3 个及以上的换行符压缩成 2 个。也就是把多个空行合并成一个空行
    return cleaned.strip()


def html_to_clean_text(html: str) -> tuple[str | None, str]:
    """从 HTML 中提取标题和可见正文，并移除脚本、样式等非正文元素。"""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()

    title: str | None = None
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)
    elif soup.h1:
        title = clean_text(soup.h1.get_text(" ", strip=True))

    body = soup.body if soup.body else soup
    text = clean_text(body.get_text("\n", strip=True))
    return title or None, text


def markdown_to_clean_text(markdown: str) -> tuple[str | None, str]:
    """把 Markdown 渲染成 HTML 后，复用 HTML 清洗逻辑提取标题和正文。"""
    rendered_html = MarkdownIt("commonmark").render(markdown)
    return html_to_clean_text(rendered_html)
