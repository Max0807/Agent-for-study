"""Day9 流式多轮 RAG 测试：全部使用 Fake，不访问 Chroma 或 DeepSeek。"""

import json
from types import SimpleNamespace

from app.config import RagSettings
from app.rag_service import NO_RESULT_ANSWER, build_context
from app.rag_stream_service import (
    build_chat_messages,
    build_retrieval_query,
    encode_sse,
    stream_rag,
    trim_history,
)
from app.schemas import (
    ChatTurn,
    KnowledgeSearchHit,
    KnowledgeSearchResponse,
    RagStreamRequest,
)


TEST_SETTINGS = RagSettings(
    default_top_k=3,
    max_context_chars=2_000,
    max_output_tokens=300,
    max_distance=0.55,
    history_max_messages=8,
    history_max_chars=6_000,
)


def make_hit(
    text: str = "员工出差前需要直属主管审批。",
    *,
    distance: float = 0.08,
) -> KnowledgeSearchHit:
    """构造一条可通过距离门控的检索结果。"""
    return KnowledgeSearchHit(
        text=text,
        source="travel.md",
        file_type="markdown",
        page_number=None,
        title="差旅制度",
        chunk_index=0,
        distance=distance,
    )


def fake_search(query: str, **_: object) -> KnowledgeSearchResponse:
    """返回固定知识库命中，避免测试加载 Embedding 模型。"""
    return KnowledgeSearchResponse(query=query, count=1, results=[make_hit()])


class FakeModelStream:
    """模拟 DeepSeek 返回的可迭代 Chunk 流，并记录是否被关闭。"""

    def __init__(self, parts: list[str], error_after: int | None = None) -> None:
        self.parts = parts
        self.error_after = error_after
        self.closed = False

    def __iter__(self):
        for index, part in enumerate(self.parts):
            if self.error_after == index:
                raise RuntimeError("模拟上游中断")
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
            )
        if self.error_after == len(self.parts):
            raise RuntimeError("模拟上游中断")

    def close(self) -> None:
        """模拟 SDK 流对象的 close() 方法。"""
        self.closed = True


class FakeCompletions:
    """记录模型调用参数，并返回指定的 Fake Stream。"""

    def __init__(self, model_stream: FakeModelStream) -> None:
        self.model_stream = model_stream
        self.arguments: dict[str, object] = {}

    def create(self, **kwargs: object) -> FakeModelStream:
        """模拟 client.chat.completions.create(stream=True)。"""
        self.arguments = kwargs
        return self.model_stream


class FakeClient:
    """提供与 OpenAI 兼容 SDK 相同的 chat.completions 访问层级。"""

    def __init__(self, model_stream: FakeModelStream) -> None:
        self.completions = FakeCompletions(model_stream)
        self.chat = SimpleNamespace(completions=self.completions)


def decode_sse(item: str) -> tuple[str, dict]:
    """解析单条测试 SSE 文本，便于断言事件顺序和数据。"""
    lines = item.rstrip("\n").splitlines()
    event_name = lines[0].removeprefix("event: ")
    data = json.loads(lines[1].removeprefix("data: "))
    return event_name, data


def test_encode_sse_uses_named_event_json_and_blank_line() -> None:
    """SSE 必须包含 event、data 且以两个换行结束。"""
    encoded = encode_sse("delta", {"content": "你好"})

    assert encoded == 'event: delta\ndata: {"content":"你好"}\n\n'


def test_build_retrieval_query_uses_latest_user_question() -> None:
    """追问检索应使用最近的 user 消息，不把 assistant 回答当作问题。"""
    history = [
        ChatTurn(role="user", content="谁审批出差？"),
        ChatTurn(role="assistant", content="直属主管。[S1]"),
    ]

    query = build_retrieval_query("那住宿呢？", history)

    assert query == "上一条用户问题：谁审批出差？\n当前问题：那住宿呢？"


def test_trim_history_keeps_most_recent_eight_messages() -> None:
    """历史超过消息数上限时应丢弃最旧消息。"""
    history = [
        ChatTurn(role="user" if index % 2 == 0 else "assistant", content=f"m{index}")
        for index in range(10)
    ]

    trimmed = trim_history(history, max_messages=8, max_chars=6_000)

    assert [turn.content for turn in trimmed] == [f"m{index}" for index in range(2, 10)]


def test_trim_history_respects_character_limit() -> None:
    """字符预算不足时只保留能容纳的最近连续历史。"""
    history = [
        ChatTurn(role="user", content="1111"),
        ChatTurn(role="assistant", content="2222"),
        ChatTurn(role="user", content="3333"),
    ]

    trimmed = trim_history(history, max_messages=8, max_chars=8)

    assert [turn.content for turn in trimmed] == ["2222", "3333"]
    assert sum(len(turn.content) for turn in trimmed) <= 8


def test_build_chat_messages_keeps_required_order() -> None:
    """模型消息必须是 system、历史、当前 user 的顺序。"""
    history = [
        ChatTurn(role="user", content="第一问"),
        ChatTurn(role="assistant", content="第一答[S1]"),
    ]
    context = build_context([make_hit()], max_chars=2_000)

    messages = build_chat_messages("追问", context, history)

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "第一问"
    assert "<question>\n追问\n</question>" in messages[-1]["content"]
    assert "[S1]" in messages[-1]["content"]


def test_stream_rag_sends_sources_before_deltas_and_done() -> None:
    """正常流必须先给来源，再分块给答案，最后给完整结果。"""
    model_stream = FakeModelStream(["需要", "主管审批", "。[S1]"])
    fake_client = FakeClient(model_stream)
    request = RagStreamRequest(question="出差需要谁审批？", history=[], top_k=3)

    raw_events = list(
        stream_rag(
            request,
            rag_settings=TEST_SETTINGS,
            search_function=fake_search,
            client_factory=lambda: (
                fake_client,
                SimpleNamespace(model="deepseek-test"),
            ),
        )
    )
    events = [decode_sse(item) for item in raw_events]

    assert [name for name, _ in events] == [
        "sources",
        "delta",
        "delta",
        "delta",
        "done",
    ]
    assert events[0][1]["citations"][0]["source_id"] == "S1"
    assert events[-1][1]["answer"] == "需要主管审批。[S1]"
    assert fake_client.completions.arguments["stream"] is True
    assert fake_client.completions.arguments["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert model_stream.closed is True


def test_stream_rag_passes_history_in_correct_order() -> None:
    """第二轮应把第一轮的 user/assistant 按原顺序重新发给模型。"""
    model_stream = FakeModelStream(["住宿按城市等级。[S1]"])
    fake_client = FakeClient(model_stream)
    received_query: dict[str, str] = {}

    def recording_search(query: str, **_: object) -> KnowledgeSearchResponse:
        received_query["value"] = query
        return KnowledgeSearchResponse(query=query, count=1, results=[make_hit()])

    request = RagStreamRequest(
        question="那住宿方面呢？",
        history=[
            ChatTurn(role="user", content="员工出差前需要谁审批？"),
            ChatTurn(role="assistant", content="需要直属主管审批。[S1]"),
        ],
    )

    list(
        stream_rag(
            request,
            rag_settings=TEST_SETTINGS,
            search_function=recording_search,
            client_factory=lambda: (
                fake_client,
                SimpleNamespace(model="deepseek-test"),
            ),
        )
    )

    messages = fake_client.completions.arguments["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert received_query["value"].startswith(
        "上一条用户问题：员工出差前需要谁审批？"
    )


def test_stream_rag_no_results_skips_model_but_finishes_protocol() -> None:
    """无合法资料时不应创建模型客户端，但 SSE 协议仍应完整。"""
    def empty_search(query: str, **_: object) -> KnowledgeSearchResponse:
        return KnowledgeSearchResponse(query=query, count=0, results=[])

    def forbidden_factory() -> tuple[object, object]:
        raise AssertionError("无检索结果时不应调用 DeepSeek。")

    request = RagStreamRequest(question="知识库不存在的问题")
    events = [
        decode_sse(item)
        for item in stream_rag(
            request,
            rag_settings=TEST_SETTINGS,
            search_function=empty_search,
            client_factory=forbidden_factory,
        )
    ]

    assert [name for name, _ in events] == ["sources", "delta", "done"]
    assert events[0][1] == {"retrieved_count": 0, "citations": []}
    assert events[1][1]["content"] == NO_RESULT_ANSWER
    assert events[2][1]["model"] is None


def test_stream_rag_done_detects_invalid_citation() -> None:
    """done 事件应将模型幻觉的 [S99] 标记为非法引用。"""
    model_stream = FakeModelStream(["错误的引用。[S99]"])
    fake_client = FakeClient(model_stream)

    events = [
        decode_sse(item)
        for item in stream_rag(
            RagStreamRequest(question="请回答"),
            rag_settings=TEST_SETTINGS,
            search_function=fake_search,
            client_factory=lambda: (
                fake_client,
                SimpleNamespace(model="deepseek-test"),
            ),
        )
    ]

    validation = events[-1][1]["citation_validation"]
    assert validation["is_valid"] is False
    assert validation["invalid_source_ids"] == ["S99"]


def test_stream_rag_emits_error_and_closes_failed_upstream() -> None:
    """上游流中途抛错时应发 error 事件，并在 finally 中关闭流。"""
    model_stream = FakeModelStream(["已生成部分"], error_after=1)
    fake_client = FakeClient(model_stream)

    events = [
        decode_sse(item)
        for item in stream_rag(
            RagStreamRequest(question="测试中断"),
            rag_settings=TEST_SETTINGS,
            search_function=fake_search,
            client_factory=lambda: (
                fake_client,
                SimpleNamespace(model="deepseek-test"),
            ),
        )
    ]

    assert [name for name, _ in events] == ["sources", "delta", "error"]
    assert "模拟上游中断" in events[-1][1]["message"]
    assert model_stream.closed is True
