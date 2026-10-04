"""Day8 Naive RAG：检索知识库、拼接上下文，并调用 DeepSeek 生成带来源的回答。"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, RagSettings, create_client, get_rag_settings
from app.index_service import search_knowledge_base
from app.schemas import (
    CitationValidation,
    KnowledgeSearchHit,
    KnowledgeSearchResponse,
    RagCitation,
    RagResponse,
)


RAG_PROMPT_FILE = PROJECT_ROOT / "prompts" / "rag_answer.txt"
NO_RESULT_ANSWER = "根据当前知识库资料无法确定。"


class RagServiceError(RuntimeError):
    """表示 RAG Prompt、模型配置或生成调用失败。"""


@dataclass(frozen=True)  # 自动生成构造函数、打印方法、相等比较，省去手写样板代码
class BuiltContext:
    """已经编号并限制长度的模型上下文，以及与编号对应的来源列表。"""

    text: str
    citations: list[RagCitation]  # 和 text 里编号一一对应的来源列表


@dataclass(frozen=True)
class PreparedRag:
    """一次 RAG 生成前的准备结果，供普通和流式问答共用。"""

    question: str  # 用户当前真正要问的问题
    retrieval_query: str  # 送给向量检索的问题，多轮时可包含上一条问题
    context: BuiltContext  # 过滤、编号、限长后的知识库上下文
    retrieved_count: int  # 真正进入上下文的来源数量


SearchFunction = Callable[..., KnowledgeSearchResponse]
ClientFactory = Callable[[], tuple[Any, Any]]


@lru_cache(maxsize=1)  # 缓存结果，后续调用直接使用内存缓存，避免重复调用
def load_rag_instructions(prompt_file: Path = RAG_PROMPT_FILE) -> str:
    """读取并缓存 RAG 系统 Prompt，避免每次请求重复读取磁盘。后续是将读取到的 Prompt 放进模型的 system 消息"""
    try:
        instructions = prompt_file.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RagServiceError(f"读取 RAG Prompt 失败：{error}") from error
    if not instructions:
        raise RagServiceError("RAG Prompt 不能为空。")
    return instructions  # 返回Prompt字符串


def _format_source_header(source_id: str, hit: KnowledgeSearchHit) -> str:  #输入：来源编号S1，一条检索结果
    """把一个检索结果的来源信息格式化为模型可读的文本头。"""
    page_text = str(hit.page_number) if hit.page_number is not None else "无页码"
    return (
        f"[{source_id}]\n"
        f"来源：{hit.source}\n"
        f"标题：{hit.title}\n"
        f"页码：{page_text}\n"
        f"内容："
    )


def build_context(
    results: list[KnowledgeSearchHit],
    max_chars: int,
) -> BuiltContext:  # 这是“检索”和“生成”之间的桥梁。
    """给 Top-k 结果分配 S1、S2 编号，并在字符上限内拼成模型上下文。"""
    if max_chars <= 0:
        raise RagServiceError("上下文长度上限必须大于 0。")

    blocks: list[str] = []  # 存放每个片段的文本块（含来源头和正文）
    citations: list[RagCitation] = []  # 存放每个编号对应的来源信息
    current_length = 0  # 记录当前已拼装的总字符数，用来控制不超限

    for hit in results:
        source_id = f"S{len(citations) + 1}"  # 为结果编号;编号跟着 citations 的长度走。第一个是 S1，第二个是 S2，以此类推
        header = _format_source_header(source_id, hit)  # 生成来源头，比如 "[S1] 销售报告.pdf 第3页\n"
        separator = "\n\n" if blocks else ""  # 如果不是第一个块，就加两个换行分隔；第一个块不加。
        remaining = max_chars - current_length - len(separator)  # 计算剩余可用字符数，要减去分隔符和来源头的长度
        if remaining <= len(header):  # 如果剩余空间连来源头都放不下，直接 break，不再处理后续结果
            break

        content = hit.text.strip()  # 取出文本并去掉首尾空白
        block = header + content  # 组合来源头和文本;先假设能完整放下
        if len(block) > remaining:
            # 至少保留来源头，再把当前文本截到剩余长度；后续来源不再加入。
            content_limit = remaining - len(header)  # 正文能保留的最大字符数，等于剩余空间减去来源头长度
            clipped_content = content[:content_limit]  # 截取

            if len(clipped_content) < len(content) and content_limit > 1:  # 如果确实被截断了，且还能留至少一个字符，
                clipped_content = clipped_content[:-1].rstrip() + "…"  # 把最后一个字符替换成省略号 …，并在前面去掉尾部空格，让截断看起来自然
            block = header + clipped_content

        blocks.append(separator + block)  # 把 separator + block 加入 blocks
        current_length += len(separator) + len(block)  # 更新 current_length
        citations.append(
            RagCitation(
                source_id=source_id,  # 上面生成的 S1、S2
                source=hit.source,
                title=hit.title,
                page_number=hit.page_number,
                chunk_index=hit.chunk_index,
                distance=hit.distance,
            )  # 把当前片段的来源信息封装成 RagCitation，加入 citations
        )  # 记录引用

        if current_length >= max_chars:  # 如果已经达到字符上限，跳出循环，不再处理后续结果
            break

    return BuiltContext(text="".join(blocks), citations=citations)  # 把所有文本块拼成一整段字符串，和引用列表一起封装成 BuiltContext 返回。


def filter_search_results(
    results: list[KnowledgeSearchHit],
    max_distance: float,
) -> list[KnowledgeSearchHit]:
    """用距离阈值过滤低相关结果，再按规范化后的正文去重。它位于检索之后、构建上下文之前。

    参数：
    - results：向量库按相似度返回的文本块。
    - max_distance：允许的最大距离，距离越小通常越相关。

    返回：保留原有顺序的高质量、不重复文本块。
    """
    if max_distance < 0:
        raise RagServiceError("检索距离上限必须大于等于 0。")

    filtered: list[KnowledgeSearchHit] = []  # 存放通过过滤的文本块，保持原有顺序。
    seen_texts: set[str] = set()  # 一个集合，用来记录已经见过的规范化文本，用于去重。
    for hit in results:
        if hit.distance > max_distance:
            continue  # 如果距离超过阈值，说明这条结果和用户问题的相关性太低，直接跳过，不加入结果

        # 先折叠连续空白，然后再去重，避免仅换行或空格不同的重复 chunk。
        normalized_text = " ".join(hit.text.split())  # 比如 "员工出差\n\n需要审批" 会变成 "员工出差 需要审批"
        if not normalized_text or normalized_text in seen_texts:
            continue  #如果规范化后是空字符串，或者已经在 seen_texts 里出现过，说明是重复内容，跳过
        seen_texts.add(normalized_text)
        filtered.append(hit)  # 存入 filtered 的是原始 hit，保留了原始的换行和格式

    return filtered


def prepare_rag(
    question: str,
    *,  # 后面的参数必须用关键字传递
    retrieval_query: str | None = None,  # 可选的独立检索问题。这是为了多轮对话设计的，业务层可以先用大模型改写成完整查询再传进来
    top_k: int | None = None,
    source: str | None = None,
    file_type: str | None = None,
    title: str | None = None,
    rag_settings: RagSettings | None = None,
    search_function: SearchFunction | None = None,
) -> PreparedRag:
    """
    完成模型生成前的检索、距离过滤、去重和上下文构建。
    它负责把用户问题变成一份“准备好的 RAG 材料”：清洗问题、检索、过滤、构建上下文，最后打包成 PreparedRag 返回。
    它是 ask_rag 的前半段，把检索和上下文构建独立出来，方便流式和非流式两种接口复用
    
    参数：
    - question：当前用户问题，最终回答仍围绕它生成。
    - retrieval_query：可选的独立检索问题；不传时直接使用 question。
    - top_k/source/file_type/title：检索数量和元数据过滤条件。
    - rag_settings：RAG 配置；不传时从环境变量读取。
    - search_function：可替换的检索函数，测试时用 Fake 避免访问真实向量库。

    返回：PreparedRag，其中包含可直接发给模型的上下文。
    """
    clean_question = question.strip()  # 清洗问题
    if not clean_question:
        raise RagServiceError("问题不能为空。")

    settings = rag_settings or get_rag_settings()
    actual_top_k = top_k or settings.default_top_k
    if not 1 <= actual_top_k <= 20:
        raise RagServiceError("top_k 必须在 1 到 20 之间。")

    clean_retrieval_query = (retrieval_query or clean_question).strip()  # 确定检索查询词
    if not clean_retrieval_query:
        clean_retrieval_query = clean_question

    searcher = search_function or search_knowledge_base
    search_response = searcher(
        clean_retrieval_query,
        top_k=actual_top_k,
        source=source,
        file_type=file_type,
        title=title,
    )  # 执行检索
    filtered_results = filter_search_results(
        search_response.results,
        max_distance=settings.max_distance,
    )  # 用距离阈值筛掉低相关结果，并做空白折叠去重
    context = build_context(filtered_results, max_chars=settings.max_context_chars)  # 在字符上限内拼装成模型上下文
    return PreparedRag(
        question=clean_question,  # 清洗后的用户原始问题，用于最终生成回答
        retrieval_query=clean_retrieval_query,  # 实际拿去向量库检索的查询词，目的是让检索更准，retrieval_query是对原始问题的检索意图显式化
        context=context,  # 构建好的上下文对象，包含 text 和 citations
        retrieved_count=len(context.citations),  # 实际进入上下文的来源数量
    )


def build_rag_user_content(question: str, context: str) -> str:
    """把当前问题和检索上下文包装成模型易于区分的用户消息。标签可以帮助模型区分“用户问题”和“参考资料”"""
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<context>\n{context}\n</context>"
    )


def validate_citations(
    answer: str,
    citations: list[RagCitation],
) -> CitationValidation:
    """检查答案中的 [Sx] 是否都存在于本次提供的来源列表。

    参数：
    - answer：模型生成的完整答案。
    - citations：本次上下文实际使用的 S1、S2 等来源。

    返回：出现过的引用、非法引用以及总体是否合法。
    """
    cited_source_ids = list(dict.fromkeys(re.findall(r"\[S(\d+)\]", answer)))  # 用正则表达式在答案里找出所有形如[S1]的引用
    cited_source_ids = [f"S{number}" for number in cited_source_ids]  # 只捕获数字部分
    allowed_source_ids = {citation.source_id for citation in citations}  # 用集合推导式，把本次上下文实际使用的所有来源编号收集起来
    invalid_source_ids = [
        source_id
        for source_id in cited_source_ids
        if source_id not in allowed_source_ids
    ]                                     
    return CitationValidation(
        cited_source_ids=cited_source_ids,
        invalid_source_ids=invalid_source_ids,
        is_valid=not invalid_source_ids,
    )


def ask_rag(
    question: str,
    *,
    top_k: int | None = None,
    source: str | None = None,
    file_type: str | None = None,
    title: str | None = None,
    rag_settings: RagSettings | None = None,
    search_function: SearchFunction | None = None,
    client_factory: ClientFactory | None = None,
) -> RagResponse:  # RAG 问答的顶层入口函数
    """
    完成一次检索增强问答；没有资料时不调用大模型，直接返回资料不足。
    它把之前学的检索、上下文构建、大模型调用串成完整流程：
    先检索知识库，把结果拼成上下文，再送给大模型生成回答，最后把答案和引用来源一起返回。
    如果检索不到资料，直接返回“资料不足”，不调用大模型
    处于 RAG 在线流程的最终阶段：
        用户提问
        → search_knowledge_base 检索 Top-k
        → build_context 编号 + 限长 + 拼装
        → 【ask_rag：调大模型生成回答】→ RagResponse
        → 返回答案和引用来源
    它把检索、上下文构建、模型生成三步串成完整的问答流程，是 RAG 对外的最终接口。
    """
    settings = rag_settings or get_rag_settings()
    prepared = prepare_rag(
        question,
        top_k=top_k,
        source=source,
        file_type=file_type,
        title=title,
        rag_settings=settings,
        search_function=search_function,
    )
    if not prepared.context.text:
        return RagResponse(
            question=prepared.question,
            answer=NO_RESULT_ANSWER,
            retrieved_count=0,
            citations=[],
            model=None,
            citation_validation=validate_citations(NO_RESULT_ANSWER, []),
        )  # 过滤后没有合法资料时不调用大模型，避免无依据生成和额外费用

    factory = client_factory or create_client  # 支持依赖注入
    try:
        client, llm_settings = factory()  # 把配置读进来，把客户端建好，交给上层去调用
        response = client.chat.completions.create(
            model=llm_settings.model,
            messages=[
                {
                    "role": "system",  # 加载 RAG 指令，告诉模型怎么用上下文回答问题
                    "content": load_rag_instructions(),  # "RAG回答规则"
                },
                {
                    "role": "user",  # 把问题和上下文用 XML 标签包起来，方便模型区分
                    "content": build_rag_user_content(
                        prepared.question,
                        prepared.context.text,
                    ),  # 实际发给模型的用户消息："问题 + Context"
                },
            ],  # 调用大模型的 Chat Completions 接口，构造两条消息
            max_tokens=settings.max_output_tokens,
            # DeepSeek 的 OpenAI 兼容 SDK 通过 extra_body 传递思考模式开关。
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = (response.choices[0].message.content or "").strip()  # 模型返回的答案
    except Exception as error:
        raise RagServiceError(f"RAG 模型调用失败：{error}") from error

    if not answer:
        raise RagServiceError("模型没有返回文本答案。")

    return RagResponse(
        question=prepared.question,
        answer=answer,
        retrieved_count=prepared.retrieved_count,
        citations=prepared.context.citations,
        model=llm_settings.model,
        citation_validation=validate_citations(
            answer,
            prepared.context.citations,
        ),
    )
