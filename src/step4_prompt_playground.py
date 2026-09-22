'''
使用不同 Prompt 模板完成不同类型任务
让用户选择模板，读取 prompts 文件夹中对应的提示词，再组织用户输入并调用模型。
支持查询改写、基于资料问答、文档摘要、元数据提取、意图路由等任务。
'''
from pathlib import Path

from config import create_client, format_usage, print_api_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"

TEMPLATES = {
    "1": ("query_rewrite", "query_rewrite.txt"),
    "2": ("grounded_qa", "grounded_qa.txt"),
    "3": ("document_summary", "document_summary.txt"),
    "4": ("metadata_extract", "metadata_extract.txt"),
    "5": ("intent_router", "intent_router.txt"),
    "6": ("maths", "maths.txt"),
}


def read_block(label: str) -> str:
    print(f"请输入{label}，单独输入 END 表示结束：")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_input(template_name: str) -> str:
    if template_name in {"grounded_qa", "maths"}:
        context = read_block("参考资料")
        question = input("请输入问题：").strip()
        return f"<context>\n{context}\n</context>\n\n<question>{question}</question>"
    if template_name in {"document_summary", "metadata_extract"}:
        document = read_block("文档")
        return f"<document>\n{document}\n</document>"
    return input("请输入用户问题：").strip()


def main() -> None:
    print("可用 Prompt 模板：")
    for key, (name, _) in TEMPLATES.items():
        print(f"{key}. {name}")

    choice = input("请选择 1-6：").strip()
    if choice not in TEMPLATES:
        raise SystemExit("选择无效。")

    template_name, filename = TEMPLATES[choice]
    instructions = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    user_input = build_input(template_name)
    if not user_input:
        raise SystemExit("输入不能为空。")

    try:
        client, settings = create_client()
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
            max_tokens=700,
        )
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error

    print(f"\n模板：{template_name}")
    print("\n模型输出：")
    print(response.choices[0].message.content)
    print("\n" + format_usage(response))


if __name__ == "__main__":
    main()

