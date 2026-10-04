"""Day7 命令行入口：把知识库目录中的文档批量写入 Chroma。"""

import argparse
import sys
from pathlib import Path


# 直接执行 src 下的脚本时，把项目根目录加入模块搜索路径，以便导入 app。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding_service import EmbeddingServiceError  # noqa: E402
from app.index_service import KnowledgeIndexError, index_directory  # noqa: E402
from app.vector_store import VectorStoreError  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """定义知识库目录和是否重建 collection 两个命令行参数。"""
    parser = argparse.ArgumentParser(
        description="解析知识库文档、计算 Embedding，并写入本地 Chroma。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "knowledge_base",
        help="知识库文档目录，默认是 data/knowledge_base。",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="建立索引前删除旧 collection；普通重复运行不需要该参数。",
    )
    return parser


def main() -> None:
    """读取命令行参数，批量建立索引，并打印逐文件结果和汇总。"""
    arguments = build_argument_parser().parse_args()
    print(f"知识库目录：{arguments.input.resolve()}")
    print("正在加载 Embedding 模型并建立索引，首次运行可能需要下载模型……")

    try:
        result = index_directory(arguments.input, rebuild=arguments.rebuild)
    except (KnowledgeIndexError, EmbeddingServiceError, VectorStoreError) as error:
        raise SystemExit(f"建立索引失败：{error}") from error

    for file_result in result.files:
        if file_result.success:
            print(f"[成功] {file_result.source}: {file_result.chunk_count} chunks")
        else:
            print(f"[失败] {file_result.source}: {file_result.error}")

    print("\n索引完成：")
    print(f"  文件总数：{result.total_files}")
    print(f"  成功文件：{result.indexed_files}")
    print(f"  失败文件：{result.failed_files}")
    print(f"  写入 chunks：{result.total_chunks}")
    if result.failed_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
