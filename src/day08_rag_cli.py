"""Day8 命令行入口：执行检索、上下文拼接和 DeepSeek 生成的完整 RAG 流程。"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_service import ask_rag  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """定义问题、Top-k 和三个可选 metadata 过滤条件。"""
    parser = argparse.ArgumentParser(
        description="从 Chroma 检索资料，并调用 DeepSeek 生成带来源的答案。"
    )
    parser.add_argument("question", nargs="?", help="问题；不提供时会交互输入。")
    parser.add_argument("--top-k", type=int, default=None, help="召回数量，范围 1-20。")
    parser.add_argument("--source", help="只使用指定相对文件路径。")
    parser.add_argument("--file-type", help="只使用 pdf、markdown、html 或 csv。")
    parser.add_argument("--title", help="只使用指定文档标题。")
    return parser


def main() -> None:
    """执行一次完整 RAG 问答，并打印答案及模型实际收到的来源。"""
    arguments = build_argument_parser().parse_args()
    question = (arguments.question or input("请输入问题：")).strip()
    if not question:
        raise SystemExit("问题不能为空。")

    try:
        result = ask_rag(
            question,
            top_k=arguments.top_k,
            source=arguments.source,
            file_type=arguments.file_type,
            title=arguments.title,
        )
    except Exception as error:
        raise SystemExit(f"RAG 问答失败：{type(error).__name__}: {error}") from error

    print(f"\n回答：\n{result.answer}")
    print(f"\n检索到 {result.retrieved_count} 个文本块。")
    if result.citations:
        print("\n引用来源：")
        for citation in result.citations:
            page = f"第 {citation.page_number} 页" if citation.page_number else "无页码"
            print(
                f"[{citation.source_id}] {citation.title} | "
                f"{citation.source} | {page} | chunk {citation.chunk_index}"
            )
    if result.model:
        print(f"\n生成模型：{result.model}")


if __name__ == "__main__":
    main()
