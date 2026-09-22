"""FastAPI 接口测试：不会发起真实的模型 API 请求。"""

from fastapi.testclient import TestClient

from app.agent_service import AgentResult
from app.main import app


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
