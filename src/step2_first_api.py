'''
第一次调用大模型 API
创建一个 API 客户端，并调用 DeepSeek API，获取模型回答并打印结果。
'''
from config import create_client, format_usage, print_api_error


def main() -> None:
    """
    调用 DeepSeek API，获取模型回答并打印结果。
    """
    try:
        client, settings = create_client()  # 创建 API 客户端和设置；
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {
                    "role": "system",  # system 角色用于设置模型的身份、行为和回答风格。
                    "content": "你是一名耐心的 AI 大模型入门老师。回答准确、简洁。",
                },
                {
                    "role": "user",  # user 角色表示用户实际提出的问题。
                    "content": "请用三句话解释 Token、上下文窗口和 Embedding 的区别。",
                },
            ],
            max_tokens=400,
        )  # 调用 DeepSeek 兼容的 Chat Completions 接口生成回答。
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error

    print(f"Model: {settings.model}")
    print("\n模型回答：")
    # print(response.output_text)
    print(response.choices[0].message.content)  # “从返回结果（response）里，拿第一个（[0]）生成选项（choices）中的消息体（message），再取里面的纯文本内容（content）。”
    print("\n" + format_usage(response))


if __name__ == "__main__":
    main()

