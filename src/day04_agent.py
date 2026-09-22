"""Day4：带计算器和销售查询工具的最小 Agent。"""

import json
from typing import Any

from pydantic import ValidationError

from config import create_client, format_usage, print_api_error
from day04_tools import ARGUMENT_MODELS, TOOL_DEFINITIONS, TOOL_HANDLERS


MAX_TOOL_ROUNDS = 5
SYSTEM_PROMPT = """
你是公司业务助手。遇到明确的四则运算时调用 calculate；
遇到销售额、区域或月份的查询时调用 query_sales。
不要编造工具可以查询到的信息。拿到工具结果后，用中文简洁回答用户。
""".strip()


def execute_tool_call(tool_call: Any) -> str:
    """校验参数并执行一次工具调用，将结果序列化为 JSON 字符串。"""
    tool_name = tool_call.function.name
    argument_model = ARGUMENT_MODELS.get(tool_name)  # 工具参数校验模型
    handler = TOOL_HANDLERS.get(tool_name)  # 工具处理函数

    if argument_model is None or handler is None:
        return json.dumps(
            {"ok": False, "error": f"不允许调用工具：{tool_name}"},
            ensure_ascii=False,
        )  # 白名单机制

    try:
        raw_arguments = json.loads(tool_call.function.arguments)  # 将工具参数 JSON 字符串转为 Python 字典
        arguments = argument_model.model_validate(raw_arguments)  # 用之前定义好的 Pydantic 模型来校验这个字典
        result = handler(**arguments.model_dump())  # 把校验后的 Pydantic 模型转回一个纯字典,然后执行工具函数
    except json.JSONDecodeError:
        result = {"ok": False, "error": "工具参数不是合法 JSON。"}
    except ValidationError as error:
        result = {"ok": False, "error": f"工具参数校验失败：{error.errors()}"}
    except Exception as error:
        result = {"ok": False, "error": f"工具执行失败：{type(error).__name__}: {error}"}

    return json.dumps(result, ensure_ascii=False)


def run_agent(question: str) -> None:  # 整个 Agent 的主循环调度器
    """把用户问题发给我模型，模型如果要求调用工具，就去执行工具，
    把结果回传，再问模型，如此循环，直到模型不再请求工具、输出最终文本答案为止。
    同时设置了最大轮次上限，防止无限循环。
    """
    client, settings = create_client()
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _round in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=600,
        )
        message = response.choices[0].message

        # 完整保存助手消息，尤其是其中的 tool_calls，供下一次请求使用。
        messages.append(message.model_dump(exclude_none=True))
        tool_calls = message.tool_calls

        # 模型没有请求工具，当前文本就是最终回答。
        if not tool_calls:
            print("\n助手：" + (message.content or "模型未返回文本回答。"))
            print(format_usage(response))
            return

        # 一个助手消息可以包含多个工具调用，必须逐个处理。
        for tool_call in tool_calls:
            tool_result = execute_tool_call(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    print(f"工具调用已达到最多 {MAX_TOOL_ROUNDS} 轮，已停止以防止循环。")


def main() -> None:
    """读取用户问题并启动 Day4 Agent。"""
    question = input("请输入问题：").strip()
    if not question:
        raise SystemExit("问题不能为空。")

    try:
        run_agent(question)
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
