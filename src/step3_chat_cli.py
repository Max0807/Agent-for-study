'''
构建命令行多轮聊天机器人
保存 history 聊天记录，使模型能理解上下文；支持 /clear 清空上下文、/exit 退出；通过系统提示词约束回答风格。
'''
from config import create_client, format_usage, print_api_error


INSTRUCTIONS = """
你是一名 AI 大模型学习助手。
- 使用中文回答。
- 先给结论，再解释原因。
- 不确定时明确说不确定，不编造资料。
- 每次回答不超过 300 字。
""".strip()  # 定义系统提示词，用于规定助手的身份和回答规则。

def main() -> None:
    """
    启动命令行聊天程序，维护多轮对话历史并调用大模型生成回答。
    """
    try:
        client, settings = create_client()
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error  # 使用状态码 1 退出程序，并保留原始异常信息。

    history: list[dict[str, str]] = []  # history 用于保存本轮命令行会话中的聊天记录;每条消息包含 role（角色）和 content（内容）。
    print(f"Chat CLI 已启动，模型：{settings.model}")  # 输出启动提示，并显示当前使用的模型名称。
    print("输入 /clear 清空上下文，输入 /exit 退出。")   # 输出可用命令说明。

    while True:
        user_text = input("\n你：").strip()  # input() 等待用户在终端输入内容；
        if not user_text:
            continue  # 如果用户没有输入任何内容，则跳过本轮循环并重新等待输入。
        if user_text == "/exit":
            print("聊天结束。")
            break  # 如果用户输入 /exit，则结束聊天循环。
        if user_text == "/clear":
            history.clear()  # clear() 删除列表中的所有历史消息。
            print("上下文已清空。")
            continue  # 如果用户输入 /clear，则清空历史上下文。

        history.append({"role": "user", "content": user_text})  # append() 在历史消息列表末尾添加当前用户消息。
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": INSTRUCTIONS},  # content 包含系统提示词，指导模型如何回答。
                    *history,
                    # “解包运算符”，把 history 这个“列表”拆开，把里面的“每一个元素”单独拿出来，塞进外面的新列表 messages 里。
                ],
                max_tokens=600,
            )
        except Exception as error:
            history.pop()
            print_api_error(error)
            continue

        answer = (response.choices[0].message.content or "").strip()
        history.append({"role": "assistant", "content": answer})  # append() 在历史消息列表末尾添加当前助手的回答。
        print(f"\n助手：{answer}")
        print(format_usage(response))
        print(f"当前本地历史消息数：{len(history)}")


if __name__ == "__main__":
    main()

