"""把标准文档段落切分为可供后续 Embedding 使用的 chunks。"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.document_loader import DocumentSection
from app.schemas import DocumentChunk


def split_sections(
    sections: list[DocumentSection],
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """逐段落分块，并把来源、页码和标题复制到每个 chunk。"""
    # RecursiveCharacterTextSplitter 是 LangChain 提供的文本分块工具。
    # 它的特点是按优先级依次尝试用不同的分隔符切分文本，直到每个块的长度不超过 chunk_size。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],  # 分隔符列表，按顺序尝试，保证了尽量在语义边界处断开
    )

    chunks: list[DocumentChunk] = []
    for section in sections:  # 遍历每个段落并分块
        for chunk_text in splitter.split_text(section.text):
            chunks.append(
                DocumentChunk(
                    text=chunk_text,  # 切分后的文本内容
                    source=section.source,
                    file_type=section.file_type,
                    page_number=section.page_number,
                    title=section.title,
                    chunk_index=len(chunks),
                )  # 为每个 chunk 创建一个 DocumentChunk 对象
            )
    return chunks  # 返回一个 DocumentChunk 列表。这个列表可以直接用于生成 Embedding 并存入向量数据库
