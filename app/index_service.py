"""Day7 索引编排层：复用 Day6 chunks，完成向量化、入库和语义检索。"""

import logging
from pathlib import Path
from typing import Protocol

from app.config import get_document_settings, get_embedding_settings
from app.document_service import parse_document_path
from app.embedding_service import get_embedding_service
from app.schemas import (
    DocumentChunk,
    IndexDirectoryResult,
    IndexFileResult,
    KnowledgeSearchHit,
    KnowledgeSearchResponse,
    KnowledgeStatsResponse,
)
from app.vector_store import VectorSearchResult, get_vector_store


logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):  # 一个协议类，用来定义“Embedding 服务应该长什么样”
    """
    索引服务需要的最小 Embedding 接口，便于测试时替换为假实现。
    继承Protocol之后，这个类就变成了一个“接口声明”，不能直接实例化，只能被其他类实现
    它不提供具体实现，只规定接口：任何类只要实现了 embed_documents 和 embed_query 这两个方法，就可以被当作 Embedding 服务使用。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    '''接收一批文本，返回一批向量。用于把知识库的 chunk 批量转成向量。'''
    def embed_query(self, query: str) -> list[float]: ...
    '''接收一条查询文本，返回一个向量。用于把用户问题转成查询向量。'''

class VectorStore(Protocol):  # 是一个协议类（Protocol），用来定义“向量库应该长什么样”
    """
    索引服务需要的最小向量库接口。
    它不提供具体实现，只规定接口：任何类只要实现了下面这些属性和方法，就可以被当作向量库使用
    """

    @property  # 只读属性
    def collection_name(self) -> str: ...
    '''返回当前使用的集合名称。上层代码可以通过它知道当前操作的是哪个集合，比如用于日志记录或展示'''

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int: ...
    '''接收一批 chunk 和它们对应的向量，写入向量库。返回写入的数量。语义是“更新或插入”，保证幂等。'''
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        source: str | None = None,
        file_type: str | None = None,
        title: str | None = None,
    ) -> list[VectorSearchResult]: ...
    '''接收查询向量和可选的过滤条件，返回最相似的 Top-k 条结果。返回的是 VectorSearchResult 列表'''
    def count(self) -> int: ...
    '''返回当前集合里保存的记录总数'''

    def reset(self) -> None: ...
    '''删除并重建集合，用于彻底清空向量库。没有返回值。'''

class KnowledgeIndexError(RuntimeError):
    """表示知识库目录或索引流程无法继续。"""


def index_file(
    path: Path,  # 要索引的文件路径
    *,  # 表示后面的参数必须用关键字传递，不能按位置传。
    source_name: str | None = None,  # 可选的来源名称
    embedding_provider: EmbeddingProvider | None = None,  # 可选的 Embedding 服务
    vector_store: VectorStore | None = None,   # 可选的向量库
) -> IndexFileResult:
    """
    解析并索引一个文件；
    接收一个文件路径，解析文件内容，切成 chunk，计算向量，写入向量库，最后返回一个结果对象
    如果出错，不会抛异常中断，而是把错误信息包在结果里返回，方便批量索引时继续处理下一个文件。
    """
    source = source_name or path.name
    try:
        parsed = parse_document_path(path)  # 文档解析函数，把文件解析成 DocumentSection 列表，再切成 DocumentChunk 列表。
        # Day6 默认只记录文件名；批量索引时替换成相对路径，避免子目录同名文件冲突。
        chunks = [
            chunk.model_copy(update={"source": source}) for chunk in parsed.chunks
        ]  # parsed.chunks 就是切好的 chunk 列表
        embedder = embedding_provider or get_embedding_service()  # 获取 Embedding 服务和向量库
        store = vector_store or get_vector_store()
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])  # 计算向量并写入
        written_count = store.upsert_chunks(chunks, embeddings)  # 把所有 chunk 的文本拿出来，批量计算向量
        return IndexFileResult(
            source=source,
            success=True,
            chunk_count=written_count,
        )
    except Exception as error:
        logger.exception("索引文件失败：%s", path)
        return IndexFileResult(
            source=source,
            success=False,
            error=f"{type(error).__name__}: {error}",
        )


def index_directory(
    directory: Path,
    *,
    rebuild: bool = False,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> IndexDirectoryResult:
    """递归索引目录内的支持格式文档，并返回逐文件结果和汇总数量。"""
    directory = directory.resolve()  # 转成绝对路径
    if not directory.is_dir():
        raise KnowledgeIndexError(f"知识库目录不存在：{directory}")

    extensions = get_document_settings().supported_extensions  # 获取支持的文件扩展名
    document_paths = sorted(
        (
            path
            for path in directory.rglob("*")  # 递归遍历目录下所有内容
            if path.is_file() and path.suffix.lower() in extensions  # 过滤出是文件、并且扩展名在支持列表里的路径
        ),
        key=lambda path: path.as_posix().lower(),  # 排序，按路径字符串的小写形式升序。这样保证每次索引的顺序一致，方便复现和调试
    )  # 递归查找所有支持的文件
    if not document_paths:  # 如果没有找到支持的文件，报错
        supported = ", ".join(sorted(extensions))
        raise KnowledgeIndexError(
            f"目录中没有支持的文档；当前支持：{supported}。"
        )

    embedder = embedding_provider or get_embedding_service()   # 获取 Embedding 服务和向量库
    store = vector_store or get_vector_store()
    if rebuild:  # 如果开启重建模式，先清空向量库
        store.reset()

    file_results = [
        index_file(
            path,
            source_name=path.relative_to(directory).as_posix(),  # 文件相对于根目录的路径
            embedding_provider=embedder,
            vector_store=store,
        )
        for path in document_paths
    ]  # 逐个索引文件，对每个文件调用之前学的 index_file
    indexed_files = sum(result.success for result in file_results)  # 求和就是成功索引的文件数
    total_chunks = sum(result.chunk_count for result in file_results)  # 所有文件写入的 chunk 总数
    return IndexDirectoryResult(
        total_files=len(file_results),  # 总文件数
        indexed_files=indexed_files,  # 成功索引的文件数
        failed_files=len(file_results) - indexed_files,  # 失败的文件数
        total_chunks=total_chunks,  # 总 chunk 数
        files=file_results,  # 每个文件的详细结果列表
    )  # 返回一个结果对象


def search_knowledge_base(
    query: str,
    *,
    top_k: int | None = None,
    source: str | None = None,
    file_type: str | None = None,
    title: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> KnowledgeSearchResponse:  # 知识库检索的业务入口函数
    """
    把自然语言问题向量化，执行 Top-k 检索并返回统一的业务响应。
    它把用户输入的自然语言问题转成向量，去向量库执行 Top-k 检索，
    最后把检索结果包装成统一的业务响应对象返回
    它位于 RAG 在线检索阶段，是连接用户问题和向量库的桥梁
    """
    query = query.strip()  # 清洗和校验查询内容
    if not query:
        raise KnowledgeIndexError("查询内容不能为空。")

    actual_top_k = top_k or get_embedding_settings().default_top_k  # 确定 top_k 并校验
    if not 1 <= actual_top_k <= 20:
        raise KnowledgeIndexError("top_k 必须在 1 到 20 之间。")

    embedder = embedding_provider or get_embedding_service()  # 获取 Embedding 服务和向量库
    store = vector_store or get_vector_store()
    query_embedding = embedder.embed_query(query)  # 把问题转成查询向量
    raw_results = store.search(
        query_embedding,  # 查询向量
        actual_top_k,  # top_k 
        source=source,  # 三个可选的过滤条件
        file_type=file_type,
        title=title,
    )  # 执行向量检索;调用向量库的 search 方法；返回的是 VectorSearchResult 列表
    hits = [
        KnowledgeSearchHit(
            text=result.text,
            source=result.source,
            file_type=result.file_type,
            page_number=result.page_number,
            title=result.title,
            chunk_index=result.chunk_index,
            distance=result.distance,
        )
        for result in raw_results
    ]  # 把原始结果转换成业务对象；遍历 raw_results，把每个 VectorSearchResult 转成 KnowledgeSearchHit
    return KnowledgeSearchResponse(query=query, count=len(hits), results=hits)  # 把参数包装成 KnowledgeSearchResponse 返回


def get_knowledge_stats(
    vector_store: VectorStore | None = None,
) -> KnowledgeStatsResponse:
    """返回当前向量 collection 的名称和已保存 chunk 数量。"""
    store = vector_store or get_vector_store()
    return KnowledgeStatsResponse(
        collection=store.collection_name,
        chunk_count=store.count(),
    )
