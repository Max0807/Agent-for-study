"""Day9 命令行客户端：调用 POST /rag/stream，实时显示多轮 RAG 回答。"""

import argparse
import json
from collections.abc import Iterator

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000/rag/stream"


def iter_sse(response: httpx.Response) -> Iterator[tuple[str, dict]]:
    """把 HTTP 响应中的 SSE 文本行解析成 ``(事件名, JSON 数据)``。

    参数：response 是 ``httpx.stream()`` 打开的流式响应。
    返回：按服务端发送顺序逐个产生 SSE 事件。
    """
    event_name = "message"
    data_lines: list[str] = []

    for line in response.iter_lines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
        elif line == "" and data_lines:
            payload = json.loads("\n".join(data_lines))
            yield event_name, payload
            event_name = "message"
            data_lines = []


def ask_streaming_question(
    client: httpx.Client,
    api_url: str,
    question: str,
    history: list[dict[str, str]],
    top_k: int,
) -> str | None:
    """发送一轮请求，显示来源并把 delta 文本立即打印到终端。

    参数：
    - client：可复用连接的 httpx 客户端。
    - api_url：FastAPI 流式接口地址。
    - question：本轮用户问题。
    - history：前面轮次的 user/assistant 消息。
    - top_k：本轮最多召回的文本块数。

    返回：成功时返回完整答案，错误时返回 None。
    """
    payload = {
        "question": question,
        "history": history,
        "top_k": top_k,
    }
    answer = ""
    answer_started = False

    try:
        with client.stream("POST", api_url, json=payload) as response:
            response.raise_for_status()
            for event_name, data in iter_sse(response):
                if event_name == "sources":
                    citations = data.get("citations", [])
                    print(f"\n检索到 {data.get('retrieved_count', 0)} 个合法来源：")
                    for citation in citations:
                        print(
                            f"  [{citation['source_id']}] {citation['source']} "
                            f"- {citation['title']}"
                        )
                elif event_name == "delta":
                    if not answer_started:
                        print("\n助手：", end="", flush=True)
                        answer_started = True
                    content = data.get("content", "")
                    answer += content
                    print(content, end="", flush=True)
                elif event_name == "done":
                    print()
                    validation = data.get("citation_validation", {})
                    if not validation.get("is_valid", True):
                        invalid = ", ".join(validation.get("invalid_source_ids", []))
                        print(f"警告：答案包含非法引用：{invalid}")
                    return data.get("answer", answer)
                elif event_name == "error":
                    print(f"\n流式请求失败：{data.get('message', '未知错误')}")
                    return None
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        print(f"\n连接或解析失败：{error}")
        return None

    print("\n流已结束，但未收到 done 事件。")
    return None


def main() -> None:
    """运行可连续提问的命令行对话，支持 /clear 和 /exit。"""
    parser = argparse.ArgumentParser(description="Day9 流式多轮 RAG 客户端")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="/rag/stream 接口地址")
    parser.add_argument("--top-k", type=int, default=4, help="每轮召回的文本块数")
    args = parser.parse_args()

    history: list[dict[str, str]] = []
    timeout = httpx.Timeout(120.0, connect=10.0)
    print(f"Day9 流式 RAG CLI 已启动：{args.url}")
    print("输入 /clear 清空客户端历史，输入 /exit 退出。")

    with httpx.Client(timeout=timeout) as client:
        while True:
            question = input("\n你：").strip()
            if not question:
                continue
            if question == "/exit":
                print("对话结束。")
                break
            if question == "/clear":
                history.clear()
                print("客户端历史已清空。")
                continue

            answer = ask_streaming_question(
                client,
                args.url,
                question,
                history,
                args.top_k,
            )
            if answer is not None:
                # API 本身无状态，因此 CLI 要在本地记住对话并于下轮重新传入。
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
