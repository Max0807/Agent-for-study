"""Day7 向量存储层：持久化 chunks，并按向量相似度执行 Top-k 检索。"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import EmbeddingSettings, get_embedding_settings
from app.schemas import DocumentChunk


class VectorStoreError(RuntimeError):
    """表示 Chroma 初始化、写入或查询失败。"""


@dataclass(frozen=True)  # 自动生成构造函数且不可变
class VectorSearchResult:
    """存储层返回的一条原始向量检索结果。"""

    text: str
    source: str
    file_type: str
    page_number: int | None
    title: str
    chunk_index: int
    distance: float


def build_chunk_id(chunk: DocumentChunk) -> str:
    """根据来源、页码和顺序生成稳定 ID，chroma 的 upsert 操作会根据 ID 判断是新增还是更新，从而避免重复插入，实现幂等写入"""
    identity = (
        f"{chunk.source}\0{chunk.file_type}\0"
        f"{chunk.page_number if chunk.page_number is not None else ''}\0"
        f"{chunk.chunk_index}"
    )  # 根据以上字段生成哈希 ID
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()  # 生成一个稳定的 SHA256 ID


def build_metadata_filter(
    source: str | None = None,
    file_type: str | None = None,
    title: str | None = None,
) -> dict[str, Any] | None:
    """把业务层的过滤参数（来源、文件类型、标题）转换成 Chroma 支持的 where 条件格式，不接受任意查询表达式。"""
    filters = [
        {name: value.strip()}
        for name, value in (
            ("source", source),
            ("file_type", file_type),
            ("title", title),
        )
        if value is not None and value.strip()
    ]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


class ChromaVectorStore:
    """
    封装 Chroma PersistentClient，隔离第三方向量库的底层返回格式。
    这个类是整个 RAG 流程中向量存储层的实现,核心职责:
    1. 管理连接和集合
    2. 写入和更新向量
    3. 按相似度检索
    4. 辅助维护
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        """创建持久化客户端，并获取或创建指定 collection。"""
        self.settings = settings or get_embedding_settings()
        self.settings.vector_db_path.mkdir(parents=True, exist_ok=True)

        try:
            # 延迟导入使未安装 Day7 依赖时，原有接口仍可正常导入和运行。
            import chromadb
        except ImportError as error:
            raise VectorStoreError(
                "缺少 chromadb，请先执行 pip install -r requirements.txt。"
            ) from error

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.settings.vector_db_path)
            )  # 创建磁盘持久化客户端
            self._collection = self._create_collection()
        except Exception as error:
            raise VectorStoreError(f"Chroma 初始化失败：{error}") from error

    @property
    def collection_name(self) -> str:
        """返回当前使用的 collection 名称。"""
        return self.settings.collection_name

    def _create_collection(self) -> Any:
        """获取或创建使用余弦距离的 collection。"""
        return self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )  # 打开或创建 collection;出现data/chroma/,这就是本地向量数据库文件

    @staticmethod
    def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        """把 chunk 元数据转成 Chroma 支持的标量类型，页码为 None 时直接省略这个字段，因为 Chroma 不支持存储 None。"""
        metadata: dict[str, str | int] = {
            "source": chunk.source,
            "file_type": chunk.file_type,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        return metadata

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """写入或更新文本块，并清理同一来源已不再存在的旧块。"""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks 数量必须与 embeddings 数量一致。")
        if any(not embedding for embedding in embeddings):
            raise VectorStoreError("Embedding 向量不能为空。")

        ids = [build_chunk_id(chunk) for chunk in chunks]  # 为每个 chunk 生成稳定 ID
        source_to_ids: dict[str, set[str]] = {}  # 创建一个字典，用于存储每个来源的 chunk ID
        for chunk, chunk_id in zip(chunks, ids, strict=True):
            source_to_ids.setdefault(chunk.source, set()).add(chunk_id)  # 按来源分组

        try:
            self._collection.upsert(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=[self._chunk_metadata(chunk) for chunk in chunks],
            )  # 调用 Chroma 的 upsert 方法，一次性写入这四样东西；upsert：ID存在，更新；ID不存在，新增；如果不存在，就新增

            # 如果文件更新后 chunk 数量变少，移除该来源遗留的旧 chunk。
            for source, current_ids in source_to_ids.items():
                stored = self._collection.get(where={"source": source})  # 查出该来源在向量库里当前存储的所有 ID
                stale_ids = [
                    stored_id
                    for stored_id in stored.get("ids", [])
                    if stored_id not in current_ids
                ]  # 把所以ID,stored_id 和本次写入的 current_ids 对比，找出在库里但不在本次写入中的 ID
                if stale_ids:
                    self._collection.delete(ids=stale_ids)  # 如果有陈旧 ID，调用 collection.delete(ids=stale_ids) 删掉它们
        except Exception as error:
            raise VectorStoreError(f"写入向量库失败：{error}") from error
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        source: str | None = None,
        file_type: str | None = None,
        title: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        接收一个查询向量和若干过滤条件，去 Chroma 向量库里找出最相似的 Top-k 条记录，
        把结果转换成统一的 VectorSearchResult 对象列表返回。
        """
        if not query_embedding:
            raise VectorStoreError("查询向量不能为空。")
        if not 1 <= top_k <= 20:  # 太小没意义，太大召回太多噪声，浪费后续送给大模型的 Token
            raise VectorStoreError("top_k 必须在 1 到 20 之间。")
        if self.count() == 0:
            return []

        query_arguments: dict[str, Any] = {
            "query_embeddings": [query_embedding],  # 查询向量，Chroma 要求传列表
            "n_results": min(top_k, self.count()),  # 要返回几条。取 top_k 和当前总数量的较小值
            "include": ["documents", "metadatas", "distances"],  # 指定返回结果里要包含哪些字段：文档文本、元数据和距离
        }
        metadata_filter = build_metadata_filter(source, file_type, title)  # 把业务参数翻译成 Chroma 的 where 语法
        if metadata_filter is not None:
            query_arguments["where"] = metadata_filter  # 加到查询参数里

        try:
            result = self._collection.query(**query_arguments)  # 调用 Chroma 的 query 方法，把参数解包传进去
        except Exception as error:
            raise VectorStoreError(f"向量检索失败：{error}") from error

        # Chroma 返回的结果是嵌套列表，取 [0] 拿到第一组结果；三个列表一一对应，长度相同
        documents = (result.get("documents") or [[]])[0]  # 命中的文本
        metadatas = (result.get("metadatas") or [[]])[0]  # 命中的元数据
        distances = (result.get("distances") or [[]])[0]  # 命中的距离
        hits: list[VectorSearchResult] = []
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
            strict=True,  #  保证三个列表长度一致
        ):  # 同时遍历三个列表
            metadata = metadata or {}
            page_number = metadata.get("page_number")
            hits.append(
                VectorSearchResult(
                    text=str(document or ""),
                    source=str(metadata.get("source", "")),
                    file_type=str(metadata.get("file_type", "")),
                    page_number=int(page_number) if page_number is not None else None,
                    title=str(metadata.get("title", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=float(distance),
                )
            )  # 每条结果封装成一个 VectorSearchResult 对象，加入 hits 列表
        return hits

    def count(self) -> int:
        """返回当前 collection 中保存的文本块数量（也就是多少个 chunk）"""
        try:
            return int(self._collection.count())
        except Exception as error:
            raise VectorStoreError(f"读取向量库数量失败：{error}") from error

    def reset(self) -> None:
        """把当前集合整个删掉，再重新创建一个空的。相当于 清空向量库 """
        try:
            existing_names = {
                collection
                if isinstance(collection, str)  # 判断 collection 是否是字符串
                else collection.name  # 如果不是字符串，则获取 collection 的 name 属性
                for collection in self._client.list_collections()
            }
            if self.settings.collection_name in existing_names:  # 检查当前配置的集合名是否已经存在
                self._client.delete_collection(self.settings.collection_name)  # 如果已经存在，则删除
            self._collection = self._create_collection()  # 重新创建一个同名的新集合，赋值给 self._collection
        except Exception as error:
            raise VectorStoreError(f"重建向量库失败：{error}") from error


@lru_cache(maxsize=1)  # 保证整个程序运行期间只创建一个实例，所有调用方拿到的都是同一个对象
def get_vector_store() -> ChromaVectorStore:
    """返回进程内共享的 Chroma 存储对象。"""
    return ChromaVectorStore()
