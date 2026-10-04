"""Day8 Naive RAG 测试：Mock 检索和 DeepSeek，不请求网络、不产生费用。"""

from types import SimpleNamespace

from app.config import RagSettings
from app.rag_service import (
    NO_RESULT_ANSWER,
    ask_rag,
    build_context,
    filter_search_results,
    validate_citations,
)
from app.schemas import KnowledgeSearchHit, KnowledgeSearchResponse, RagCitation


TEST_RAG_SETTINGS = RagSettings(
    default_top_k=3,
    max_context_chars=2_000,
    max_output_tokens=300,
)


def make_hit(
    text: str = "员工出差前需要直属主管审批。",
    source: str = "travel.md",
    title: str = "差旅制度",
    chunk_index: int = 0,
    distance: float = 0.08,
) -> KnowledgeSearchHit:
    """创建一条可复用的测试检索结果。"""
    return KnowledgeSearchHit(
        text=text,
        source=source,
        file_type="markdown",
        page_number=None,
        title=title,
        chunk_index=chunk_index,
        distance=distance,
    )


class FakeCompletions:
    """记录发送给模型的参数，并返回固定回答。"""

    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        """模拟 client.chat.completions.create。"""
        self.arguments = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="员工出差前需要直属主管审批。[S1]"
                    )
                )
            ]
        )


class FakeClient:
    """提供与 OpenAI 兼容客户端相同的 chat.completions 访问结构。"""

    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_build_context_adds_source_ids_and_citations() -> None:
    """检索结果应被编号为 S1、S2，并生成相同顺序的引用信息。"""
    context = build_context(
        [
            make_hit(),
            make_hit(
                text="差旅住宿标准按照城市等级确定。",
                source="hotel.md",
                title="住宿标准",
                chunk_index=1,
            ),
        ],
        max_chars=2_000,
    )

    assert "[S1]" in context.text
    assert "[S2]" in context.text
    assert "来源：travel.md" in context.text
    assert context.citations[0].source_id == "S1"
    assert context.citations[1].source == "hotel.md"


def test_build_context_respects_character_limit() -> None:
    """过长文本应被截断，拼接后的上下文不能超过配置上限。"""
    context = build_context([make_hit(text="制度内容" * 100)], max_chars=100)

    assert len(context.text) <= 100
    assert context.text.endswith("…")
    assert len(context.citations) == 1


def test_ask_rag_uses_retrieval_context_and_returns_citations() -> None:
    """完整服务应检索资料、调用兼容客户端并返回答案和来源。"""
    fake_client = FakeClient()
    received_search_arguments: dict[str, object] = {}

    def fake_search(query: str, **kwargs: object) -> KnowledgeSearchResponse:
        received_search_arguments.update({"query": query, **kwargs})
        return KnowledgeSearchResponse(query=query, count=1, results=[make_hit()])

    result = ask_rag(
        "员工出差需要谁审批？",
        top_k=3,
        rag_settings=TEST_RAG_SETTINGS,
        search_function=fake_search,
        client_factory=lambda: (
            fake_client,
            SimpleNamespace(model="deepseek-test"),
        ),
    )

    messages = fake_client.completions.arguments["messages"]
    assert received_search_arguments["top_k"] == 3
    assert "<context>" in messages[1]["content"]
    assert "[S1]" in messages[1]["content"]
    assert result.answer.endswith("[S1]")
    assert result.citations[0].source == "travel.md"
    assert result.model == "deepseek-test"
    assert result.citation_validation is not None
    assert result.citation_validation.is_valid is True
    assert fake_client.completions.arguments["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_ask_rag_does_not_call_model_when_retrieval_is_empty() -> None:
    """没有检索结果时应直接说明资料不足，不调用可能产生费用的模型。"""
    def empty_search(query: str, **_: object) -> KnowledgeSearchResponse:
        return KnowledgeSearchResponse(query=query, count=0, results=[])

    def forbidden_client_factory() -> tuple[object, object]:
        raise AssertionError("没有检索结果时不应创建大模型客户端。")

    result = ask_rag(
        "知识库中不存在的问题",
        rag_settings=TEST_RAG_SETTINGS,
        search_function=empty_search,
        client_factory=forbidden_client_factory,
    )

    assert result.answer == NO_RESULT_ANSWER
    assert result.retrieved_count == 0
    assert result.citations == []
    assert result.model is None


def test_filter_search_results_applies_distance_gate() -> None:
    """距离大于 0.55 的低相关文本块不应进入 RAG 上下文。"""
    results = filter_search_results(
        [
            make_hit(text="高相关内容", distance=0.55),
            make_hit(text="低相关内容", distance=0.551),
        ],
        max_distance=0.55,
    )

    assert [hit.text for hit in results] == ["高相关内容"]


def test_filter_search_results_deduplicates_normalized_text() -> None:
    """仅空格和换行不同的文本块应视为同一条资料。"""
    results = filter_search_results(
        [
            make_hit(text="员工出差前 需要审批。", chunk_index=0),
            make_hit(text="员工出差前\n需要审批。", chunk_index=1),
        ],
        max_distance=0.55,
    )

    assert len(results) == 1
    assert results[0].chunk_index == 0


def test_validate_citations_detects_unknown_source_id() -> None:
    """模型写出上下文中不存在的 S99 时必须被识别。"""
    citation = RagCitation(
        source_id="S1",
        source="travel.md",
        title="差旅制度",
        chunk_index=0,
        distance=0.08,
    )

    validation = validate_citations("需要主管审批。[S1][S99][S99]", [citation])

    assert validation.cited_source_ids == ["S1", "S99"]
    assert validation.invalid_source_ids == ["S99"]
    assert validation.is_valid is False
