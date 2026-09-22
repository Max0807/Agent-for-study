'''
将模型回答转为结构化数据(非自然语言回答，而是得到程序可以直接处理的数据)
使用 Pydantic 定义 DocumentMetadata 数据格式，
让模型从企业文档中提取标题、类型、生效日期、关键词、摘要，以及是否需要人工复核。
'''
from typing import Literal  # # 用于限制字段只能取指定的几个字符串值。

from pydantic import BaseModel  # 用于定义、校验和管理结构化数据模型。

from config import create_client, format_usage, print_api_error


class DocumentMetadata(BaseModel):
    """
    定义企业文档需要提取的结构化元数据格式。
    该类继承 BaseModel，Pydantic 会自动校验模型输出的字段类型。
    """
    title: str | None  # # 文档标题；str | None 表示可以是字符串，也可以是空值。
    document_type: Literal["policy", "faq", "job_description", "other"]  # 文档类型；Literal 限制其值只能是列出的四种类型之一。
    effective_date: str | None  # # 文档生效日期；如果文档未提供日期，则允许为 None。
    keywords: list[str]  # 从文档中提取的关键词列表。
    summary: str  # 对文档内容的简要总结。
    needs_human_review: bool  # 是否需要人工复核；True 表示信息不完整或存在不确定性。


SAMPLE_DOCUMENT = """
标题：员工差旅管理办法
本办法适用于全体正式员工，自 2026 年 10 月 1 日起执行。
员工出差前需要获得直属主管审批。住宿标准由各城市等级决定。
文档没有说明海外出差的住宿标准。
""".strip()  # 定义一份用于测试的示例企业文档。


def main() -> None:
    """
    接收一份企业文档，并调用模型提取结构化元数据。
    返回：None：函数只负责交互、调用模型和输出结果。
    """
    print("直接回车使用示例文档，或者输入一行自己的文档。")
    document = input("文档：").strip() or SAMPLE_DOCUMENT  # 如果用户输入为空，则使用 SAMPLE_DOCUMENT 作为默认文档。

    try:
        client, settings = create_client()
        response = client.responses.parse(
            model=settings.model,
            instructions=(
                "提取企业文档元数据。只能使用文档中的信息；"
                "缺失内容使用 null，并据此设置 needs_human_review。"
            ),
            input=document,
            text_format=DocumentMetadata,
        )  # parse() 会请求模型按指定的 Pydantic 数据模型返回结构化结果。
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error

    metadata = response.output_parsed
    if metadata is None:
        raise SystemExit("模型没有返回可解析的结构化结果。")

    print("\nPydantic 校验后的结果：")
    print(metadata.model_dump_json(indent=2))
    print("\n" + format_usage(response))


if __name__ == "__main__":
    main()

