"""FastAPI 接口测试：不会发起真实的模型 API 请求。"""

from fastapi.testclient import TestClient

from app.agent_service import AgentResult
from app.main import app
from app.schemas import (
    KnowledgeSearchHit,
    KnowledgeSearchResponse,
    KnowledgeStatsResponse,
    RagCitation,
    RagResponse,
)


def test_health_check() -> None:
    """健康检查应返回服务与数据库均已就绪。"""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database_ready": True}


def test_get_sales_with_filters() -> None:
    """销售接口应只返回匹配区域和月份的数据。"""
    with TestClient(app) as client:
        response = client.get("/sales", params={"region": "华东", "month": "2026-09"})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["rows"][0]["amount"] == 128_000.0


def test_agent_returns_mocked_answer(monkeypatch) -> None:
    """使用 mock 替代真实模型调用，测试 POST /agent 的响应结构。"""
    def fake_run_agent(question: str) -> AgentResult:
        return AgentResult(answer=f"已收到：{question}", tool_rounds=1)

    # 替换 main.py 中已导入的函数，确保测试不会请求 DeepSeek。
    monkeypatch.setattr("app.main.run_agent", fake_run_agent)

    with TestClient(app) as client:
        response = client.post("/agent", json={"question": "华东销售额是多少？"})

    assert response.status_code == 200
    assert response.json() == {"answer": "已收到：华东销售额是多少？", "tool_rounds": 1}


def test_agent_rejects_blank_question() -> None:
    """空问题应由 Pydantic 在进入业务逻辑前拒绝。"""
    with TestClient(app) as client:
        response = client.post("/agent", json={"question": ""})

    assert response.status_code == 422


def test_parse_markdown_upload() -> None:
    """上传 Markdown 后，接口应返回带来源和标题的 chunks。"""
    files = {
        "file": (
            "guide.md",
            "# 使用指南\n\n这是上传接口的测试内容。".encode("utf-8"),
            "text/markdown",
        )
    }
    with TestClient(app) as client:
        response = client.post("/documents/parse", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "guide.md"
    assert body["title"] == "使用指南"
    assert body["chunk_count"] == 1
    assert body["chunks"][0]["source"] == "guide.md"


def test_knowledge_search_returns_mocked_hits(monkeypatch) -> None:
    """知识库搜索接口应把请求参数交给服务层并返回检索结果。"""
    def fake_search(query: str, **_: object) -> KnowledgeSearchResponse:
        return KnowledgeSearchResponse(
            query=query,
            count=1,
            results=[
                KnowledgeSearchHit(
                    text="员工出差前需要直属主管审批。",
                    source="travel.md",
                    file_type="markdown",
                    title="差旅制度",
                    chunk_index=0,
                    distance=0.08,
                )
            ],
        )

    monkeypatch.setattr("app.main.search_knowledge_base", fake_search)

    with TestClient(app) as client:
        response = client.post(
            "/knowledge/search",
            json={"query": "出差需要谁批准？", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["results"][0]["source"] == "travel.md"


def test_knowledge_stats_returns_mocked_count(monkeypatch) -> None:
    """知识库状态接口应返回 collection 名称和 chunk 数量。"""
    monkeypatch.setattr(
        "app.main.get_knowledge_stats",
        lambda: KnowledgeStatsResponse(collection="test_knowledge", chunk_count=42),
    )

    with TestClient(app) as client:
        response = client.get("/knowledge/stats")

    assert response.status_code == 200
    assert response.json() == {
        "collection": "test_knowledge",
        "chunk_count": 42,
    }


def test_rag_ask_returns_mocked_answer_and_citations(monkeypatch) -> None:
    """RAG 接口应返回生成答案、召回数量、模型名和可追溯来源。"""
    def fake_ask_rag(question: str, **_: object) -> RagResponse:
        return RagResponse(
            question=question,
            answer="员工出差前需要直属主管审批。[S1]",
            retrieved_count=1,
            citations=[
                RagCitation(
                    source_id="S1",
                    source="travel.md",
                    title="差旅制度",
                    chunk_index=0,
                    distance=0.08,
                )
            ],
            model="deepseek-test",
        )

    monkeypatch.setattr("app.main.ask_rag", fake_ask_rag)

    with TestClient(app) as client:
        response = client.post(
            "/rag/ask",
            json={"question": "员工出差需要谁审批？", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["answer"].endswith("[S1]")
    assert response.json()["citations"][0]["source"] == "travel.md"
    assert response.json()["model"] == "deepseek-test"


def test_rag_ask_rejects_blank_question() -> None:
    """空问题应在进入 RAG 服务前被 Pydantic 拒绝。"""
    with TestClient(app) as client:
        response = client.post("/rag/ask", json={"question": "", "top_k": 3})

    assert response.status_code == 422


def test_rag_stream_returns_event_stream(monkeypatch) -> None:
    """流式路由应返回 text/event-stream，并保留生成器的事件顺序。"""
    def fake_stream_rag(_: object):
        yield 'event: sources\ndata: {"retrieved_count":0,"citations":[]}\n\n'
        yield 'event: delta\ndata: {"content":"资料不足"}\n\n'
        yield 'event: done\ndata: {"answer":"资料不足"}\n\n'

    monkeypatch.setattr("app.main.stream_rag", fake_stream_rag)

    with TestClient(app) as client:
        response = client.post(
            "/rag/stream",
            json={"question": "知识库里有什么？", "history": []},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.index("event: sources") < response.text.index("event: delta")
    assert response.text.index("event: delta") < response.text.index("event: done")
