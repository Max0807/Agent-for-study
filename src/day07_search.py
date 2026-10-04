"""Day7 命令行入口：验证 Top-k 语义检索和 metadata filter。"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.index_service import search_knowledge_base  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """定义查询文本、Top-k 和三个可选元数据过滤条件。"""
    parser = argparse.ArgumentParser(
        description="对已经建立的 Chroma 知识库执行语义检索。"
    )
    parser.add_argument("query", nargs="?", help="自然语言查询；不提供时会交互输入。")
    parser.add_argument("--top-k", type=int, default=None, help="返回结果数，范围 1-20。")
    parser.add_argument("--source", help="只搜索指定相对文件路径。")
    parser.add_argument("--file-type", help="只搜索 pdf、markdown、html 或 csv。")
    parser.add_argument("--title", help="只搜索指定文档标题。")
    return parser


def main() -> None:
    """执行一次知识库检索，并以便于学习和核对的格式打印结果。"""
    arguments = build_argument_parser().parse_args()
    query = (arguments.query or input("请输入问题：")).strip()
    if not query:
        raise SystemExit("查询内容不能为空。")

    try:
        response = search_knowledge_base(
            query,
            top_k=arguments.top_k,
            source=arguments.source,
            file_type=arguments.file_type,
            title=arguments.title,
        )
    except Exception as error:
        raise SystemExit(f"检索失败：{type(error).__name__}: {error}") from error

    print(f"\n找到 {response.count} 个相关文本块：")
    for number, hit in enumerate(response.results, start=1):
        location = f"第 {hit.page_number} 页" if hit.page_number else "无页码"
        print(f"\n{number}. {hit.title}")
        print(f"   来源：{hit.source} | {location} | chunk {hit.chunk_index}")
        print(f"   距离：{hit.distance:.4f}")
        print(f"   内容：{hit.text}")


if __name__ == "__main__":
    main()
