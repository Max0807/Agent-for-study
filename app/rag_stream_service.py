"""Day9 流式 RAG：处理历史对话，读取 DeepSeek Chunk，并编码为 SSE 事件。"""

import json
import logging
from collections.abc import Iterator, Sequence
from typing import Any

from app.config import RagSettings, create_client, get_rag_settings
from app.rag_service import (
    NO_RESULT_ANSWER,
    BuiltContext,
    ClientFactory,
    RagServiceError,
    SearchFunction,
    build_rag_user_content,
    load_rag_instructions,
    prepare_rag,
    validate_citations,
)
from app.schemas import ChatTurn, RagStreamRequest


logger = logging.getLogger(__name__)


def trim_history(
    history: Sequence[ChatTurn],
    *,
    max_messages: int,
    max_chars: int,
) -> list[ChatTurn]:
    """同时按消息数和字符数裁剪历史，优先保留最近对话。

    参数：
    - history：客户端按时间顺序传入的历史消息。
    - max_messages：最多保留几条历史消息。
    - max_chars：所有保留消息的正文总字符数上限。

    返回：仍按从旧到新排列的历史列表。
    """
    if max_messages <= 0 or max_chars <= 0:
        raise RagServiceError("历史消息数和字符数上限必须大于 0。")

    kept_reversed: list[ChatTurn] = []
    remaining_chars = max_chars  # 剩余可用的字符数，初始等于 max_chars
    for turn in reversed(history):  # 倒序遍历
        if len(kept_reversed) >= max_messages or remaining_chars <= 0:
            break  # 如果已经保留了 max_messages 条，或者字符额度用完，就停止

        content = turn.content.strip()
        if not content:
            continue

        if len(content) > remaining_chars:
            # 如果最近一条自身就超长，保留它的末尾，因为末尾更接近当前语义。
            if not kept_reversed:
                kept_reversed.append(
                    turn.model_copy(update={"content": content[-remaining_chars:]})
                )  # 取末尾的 remaining_chars 个字符，然后复制一个新的 ChatTurn，只更新 content 字段
            break

        kept_reversed.append(turn.model_copy(update={"content": content}))  # 消息能完整放下，复制一份加入保留列表
        remaining_chars -= len(content)  # 从剩余字符额度里扣掉这条消息的长度

    kept_reversed.reverse()  # 因为前面是倒序遍历，kept_reversed 里是从新到旧的顺序
    return kept_reversed


def build_retrieval_query(
    question: str,
    history: Sequence[ChatTurn],
) -> str:
    """用最近一条用户问题补全当前指代不清的追问。

    例如当前只问“那住宿方面呢”时，检索文本会同时包含上一条用户问题。
    这里只做确定性拼接，不另外调用大模型改写 Query。
    """
    clean_question = question.strip()
    for turn in reversed(history):
        if turn.role == "user" and turn.content.strip():
            return (
                f"上一条用户问题：{turn.content.strip()}\n"
                f"当前问题：{clean_question}"
            )
    return clean_question


def build_chat_messages(
    question: str,
    context: BuiltContext,
    history: Sequence[ChatTurn],
) -> list[dict[str, str]]:
    """按“系统指令 → 历史 → 当前问题和知识”组装模型消息。

    参数：
    - question：用户这一轮的问题。
    - context：已经附加 S1、S2 编号的知识库上下文。
    - history：已裁剪的多轮历史。

    返回：可直接传给 ``chat.completions.create`` 的 messages。
    """
    messages = [
        {"role": "system", "content": load_rag_instructions()},
    ]
    messages.extend(
        {"role": turn.role, "content": turn.content}
        for turn in history
    )
    messages.append(
        {
            "role": "user",
            "content": build_rag_user_content(question, context.text),
        }
    )
    return messages  # 要特别注意：当前问题必须放在最后。


def encode_sse(event: str, data: dict[str, Any]) -> str:
    """把事件名和字典编码成一条符合 SSE 规范的 UTF-8 文本。

    ``ensure_ascii=False`` 让中文保持可读；末尾两个换行表示一条事件结束。
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _close_model_stream(model_stream: object | None) -> None:
    """如果上游流提供 close() 就关闭它，断开时也不留占用的连接。"""
    close = getattr(model_stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception as error:  # 关闭失败不应覆盖原始生成结果或错误
            logger.warning("关闭上游模型流失败：%s", error)


def stream_rag(
    request: RagStreamRequest,
    *,
    rag_settings: RagSettings | None = None,
    search_function: SearchFunction | None = None,
    client_factory: ClientFactory | None = None,
) -> Iterator[str]:  # 是 RAG 流式问答的顶层生成器函数
    """
    执行多轮流式 RAG，依次产生 sources、delta、done 或 error 事件。
    它接收流式请求，依次完成历史裁剪、检索准备、来源推送、流式调用大模型、逐块推送答案、引用校验，最后产生各种事件。
    返回一个 Iterator[str]，每 yield 一次就是一条 SSE 编码后的字符串，供上层 StreamingResponse 写入 HTTP 流。
    
    参数：
    - request：当前问题、历史、Top-k 和可选过滤条件。
    - rag_settings：可注入的 RAG 配置，方便测试边界。
    - search_function：可注入的检索函数，测试中不访问真实 Chroma。
    - client_factory：可注入的模型客户端工厂，测试中不请求 DeepSeek。

    注意：流开始后 HTTP 响应头已经发送，中途异常只能转成 ``error``
    事件，不能再把状态码改成 502。
    """
    model_stream: object | None = None  # 用于保存大模型返回的流对象，供 finally 关闭
    try:
        settings = rag_settings or get_rag_settings()
        history = trim_history(
            request.history,
            max_messages=settings.history_max_messages,
            max_chars=settings.history_max_chars,
        )  # 按消息数和字符数裁剪历史，优先保留最近对话。返回从旧到新的列表
        retrieval_query = build_retrieval_query(request.question, history)  # 结合当前问题和历史，生成更适合检索的查询词
        prepared = prepare_rag(
            request.question,
            retrieval_query=retrieval_query,
            top_k=request.top_k,
            source=request.source,
            file_type=request.file_type,
            title=request.title,
            rag_settings=settings,
            search_function=search_function,
        )  # 完成检索、距离过滤、去重、上下文构建, 返回一个 PreparedRag

        # 客户端先收到来源，之后才会看到答案增量。
        yield encode_sse(
            "sources",
            {
                "retrieved_count": prepared.retrieved_count,
                "citations": [
                    citation.model_dump(mode="json")
                    for citation in prepared.context.citations
                ],
            },
        )  # 先推送来源事件：在推送答案之前，先把来源推给客户端。客户端收到后可以先展示来源，再逐字显示答案。

        if not prepared.context.text:  # 上下文为空时不调大模型，直接推送“资料不足”的 delta 和 done 事件，然后结束
            validation = validate_citations(NO_RESULT_ANSWER, [])
            yield encode_sse("delta", {"content": NO_RESULT_ANSWER})
            yield encode_sse(
                "done",
                {
                    "answer": NO_RESULT_ANSWER,
                    "model": None,
                    "citation_validation": validation.model_dump(mode="json"),
                },
            )
            return

        factory = client_factory or create_client  # client_factory 是调用方传入的可替换工厂，create_client 是默认的真实实现
        client, llm_settings = factory()  # 创建模型客户端
        messages = build_chat_messages(
            prepared.question,
            prepared.context,
            history,
        )  # 组装模型消息
        model_stream = client.chat.completions.create(
            model=llm_settings.model,
            messages=messages,
            stream=True,  # 开启流式输出
            max_tokens=settings.max_output_tokens,
            # DeepSeek OpenAI 兼容接口要求通过 extra_body 关闭思考模式。
            # 同时下方只读 delta.content，不读、不存 reasoning_content。
            extra_body={"thinking": {"type": "disabled"}},  # 是 DeepSeek 特有参数，关闭思考模式
        )  # 流式调用大模型

        # 遍历流对象，每个 chunk 取增量内容，追加到 answer_parts，同时立即用 delta 事件推送。客户端收到后追加到聊天框，实现打字机效果。
        answer_parts: list[str] = []  # 用来收集所有增量片段
        for chunk in model_stream:  # model_stream 是返回的可迭代对象。每循环一次，就拿到一小块数据（一个 chunk）
            choices = getattr(chunk, "choices", None)  # 用 getattr 从 chunk 里取 choices 字段，因为流式返回的 chunk 不一定每个都带 choices
            if not choices:  # choices是一个列表， 不是“输出内容本身”，而是“包含输出内容的结构”
                continue
            content = getattr(choices[0].delta, "content", None)  # 从第一个 choice 的 delta 里取 content 字段。流式返回里，增量内容放在 delta 里，而不是 message 
            if not content:
                continue
            answer_parts.append(content)
            yield encode_sse("delta", {"content": content})  # 立即把这段增量用 SSE 编码成 delta 事件，yield 出去，上层 StreamingResponse 会把它写入 HTTP 流

        answer = "".join(answer_parts).strip()  # 拼接完整答案
        if not answer:
            raise RagServiceError("模型没有返回文本答案。")

        validation = validate_citations(answer, prepared.context.citations)  # 调用 validate_citations 校验引用是否合法
        yield encode_sse(
            "done",
            {
                "answer": answer,
                "model": llm_settings.model,
                "citation_validation": validation.model_dump(mode="json"),
            },
        )  # 推送 done 事件，包含完整答案、模型名、校验结果
    except GeneratorExit:
        # StreamingResponse 或客户端取消迭代时，直接进入 finally 释放上游连接。
        raise  # 客户端取消时抛出，直接 raise，让 finally 释放资源，不生成 error 事件
    except Exception as error:
        logger.exception("RAG 流式生成失败")
        yield encode_sse("error", {"message": f"RAG 流式生成失败：{error}"})
    finally:
        _close_model_stream(model_stream)  # 无论成功失败，都关闭模型流（资源清理）
