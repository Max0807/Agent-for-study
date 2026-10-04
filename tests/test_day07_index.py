"""Day7 索引编排测试：不下载真实模型，也不写入真实 Chroma 目录。"""

from pathlib import Path

from app.index_service import index_directory, search_knowledge_base
from app.schemas import DocumentChunk
from app.vector_store import (
    VectorSearchResult,
    build_chunk_id,
    build_metadata_filter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingProvider:
    """测试用向量模型：根据文本长度生成固定的二维向量。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为每段文本生成一个可预测向量。"""
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """为查询生成一个可预测向量。"""
        return [float(len(query)), 1.0]


class FakeVectorStore:
    """测试用内存向量库：模拟 upsert、查询、计数和重建。"""

    collection_name = "test_knowledge"

    def __init__(self) -> None:
        """创建按稳定 chunk ID 保存数据的内存字典。"""
        self.records: dict[str, tuple[DocumentChunk, list[float]]] = {}
        self.last_filters: dict[str, str | None] = {}

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """相同 chunk ID 会覆盖旧值，用来验证重复索引不重复。"""
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[build_chunk_id(chunk)] = (chunk, embedding)
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        source: str | None = None,
        file_type: str | None = None,
        title: str | None = None,
    ) -> list[VectorSearchResult]:
        """返回固定结果，并记录索引服务传入的过滤参数。"""
        self.last_filters = {
            "source": source,
            "file_type": file_type,
            "title": title,
        }
        return [
            VectorSearchResult(
                text="员工出差前需要直属主管审批。",
                source="travel.md",
                file_type="markdown",
                page_number=None,
                title="差旅制度",
                chunk_index=0,
                distance=0.08,
            )
        ][:top_k]

    def count(self) -> int:
        """返回内存记录数量。"""
        return len(self.records)

    def reset(self) -> None:
        """清空全部测试记录。"""
        self.records.clear()


def test_build_chunk_id_is_stable() -> None:
    """同一来源、页码和序号的 chunk 应始终生成相同 ID。"""
    chunk = DocumentChunk(
        text="第一版正文",
        source="policy.md",
        file_type="markdown",
        title="制度",
        chunk_index=0,
    )
    updated_chunk = chunk.model_copy(update={"text": "更新后的正文"})

    assert build_chunk_id(chunk) == build_chunk_id(updated_chunk)


def test_build_metadata_filter_only_accepts_business_fields() -> None:
    """多个业务条件应转换为 Chroma 的 $and 过滤结构。"""
    result = build_metadata_filter(
        source="policy.md",
        file_type="markdown",
        title=None,
    )

    assert result == {
        "$and": [
            {"source": "policy.md"},
            {"file_type": "markdown"},
        ]
    }


def test_index_directory_reuses_day6_and_is_idempotent(tmp_path: Path) -> None:
    """批量索引应复用 Day6 解析结果，重复运行后记录数量不应增加。"""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "employee.md").write_text(
        "# 员工制度\n\n员工请假需要提前申请。",
        encoding="utf-8",
    )
    (knowledge_dir / "travel.md").write_text(
        "# 差旅制度\n\n员工出差前需要直属主管审批。",
        encoding="utf-8",
    )
    embedder = FakeEmbeddingProvider()
    store = FakeVectorStore()

    first = index_directory(
        knowledge_dir,
        embedding_provider=embedder,
        vector_store=store,
    )
    count_after_first_run = store.count()
    second = index_directory(
        knowledge_dir,
        embedding_provider=embedder,
        vector_store=store,
    )

    assert first.total_files == 2
    assert first.indexed_files == 2
    assert first.failed_files == 0
    assert count_after_first_run == 2
    assert store.count() == count_after_first_run
    assert second.total_chunks == first.total_chunks


def test_search_knowledge_base_passes_filters_and_formats_result() -> None:
    """搜索服务应完成查询向量化、传递过滤条件并返回统一响应。"""
    store = FakeVectorStore()

    response = search_knowledge_base(
        "出差需要谁批准？",
        top_k=3,
        file_type="markdown",
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert response.count == 1
    assert response.results[0].source == "travel.md"
    assert response.results[0].distance == 0.08
    assert store.last_filters["file_type"] == "markdown"


def test_sample_knowledge_base_indexes_at_least_twenty_documents() -> None:
    """仓库自带的 Day7 示例库应满足 20+ 文档验收目标且都能被 Day6 解析。"""
    result = index_directory(
        PROJECT_ROOT / "data" / "knowledge_base",
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    assert result.total_files >= 20
    assert result.indexed_files == result.total_files
    assert result.failed_files == 0
    assert result.total_chunks >= 20
