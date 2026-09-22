"""Day6 命令行入口：解析一个本地文档并输出 chunks JSON。"""

import argparse
import json
import sys
from pathlib import Path


# 直接运行 src/day06_parse_documents.py 时，显式加入项目根目录以便导入 app。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.document_service import DocumentParseError, parse_document_path  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """定义命令行参数：输入文件和可选的 JSON 输出路径。"""
    parser = argparse.ArgumentParser(
        description="解析 PDF、Markdown、HTML 或 CSV，并生成带元数据的 chunks。"
    )
    parser.add_argument("file", type=Path, help="需要解析的文档路径。")
    parser.add_argument(
        "--output",
        type=Path,
        help="可选的 JSON 输出文件；不提供时直接打印到终端。",
    )
    return parser


def main() -> None:
    """解析命令行参数，执行文档解析并输出 UTF-8 JSON。"""
    arguments = build_argument_parser().parse_args()
    try:
        result = parse_document_path(arguments.file)
    except DocumentParseError as error:
        raise SystemExit(f"文档解析失败：{error}") from error

    json_text = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json_text, encoding="utf-8")
        print(f"解析完成：{result.chunk_count} 个 chunks，结果已写入 {arguments.output}")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
