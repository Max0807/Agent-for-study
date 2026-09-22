"""协调模型与本地工具调用的 Agent 服务。整个 Agent 的核心"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.config import create_client
from app.tools import ARGUMENT_MODELS, TOOL_DEFINITIONS, TOOL_HANDLERS


logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5
SYSTEM_PROMPT = """
你是公司销售业务助手。
涉及明确的四则运算时调用 calculate；涉及销售额、区域、月份时调用 query_sales。
不要编造销售数据或计算结果。获得工具结果后，使用中文简洁回答用户。
""".strip()


@dataclass(frozen=True)
class AgentResult:
    """Agent 完成一次请求后返回给 API 层(FastAPI 路由层)的结果。"""

    answer: str
    tool_rounds: int


class AgentServiceError(RuntimeError):
    """表示模型调用或 Agent 执行无法继续的异常。"""


def execute_tool_call(tool_call: Any) -> str:
    """解析、校验并执行一次工具调用，返回 JSON 格式的工具结果。"""
    tool_name = tool_call.function.name  # 取出工具名
    argument_model = ARGUMENT_MODELS.get(tool_name)  # → 从白名单找参数模型
    handler = TOOL_HANDLERS.get(tool_name)  # → 从白名单找处理函数

    # 未在白名单中的名称不会被执行。
    if argument_model is None or handler is None:
        return json.dumps(
            {"ok": False, "error": f"不允许调用工具：{tool_name}"},
            ensure_ascii=False,
        )

    try:
        # 模型返回的是 JSON 字符串，必须先解析再交给 Pydantic 校验。
        raw_arguments = json.loads(tool_call.function.arguments)  # → 把模型生成的 JSON 字符串变成 Python 字典。
        arguments = argument_model.model_validate(raw_arguments)  # → 用 Pydantic 校验参数类型、字段和枚举值
        result = handler(**arguments.model_dump())  # → 执行 Python 工具函数
    except (json.JSONDecodeError, TypeError):
        result = {"ok": False, "error": "工具参数不是合法 JSON。"}
    except ValidationError as error:
        result = {"ok": False, "error": f"工具参数校验失败：{error.errors()}"}
    except Exception as error:
        logger.exception("执行工具 %s 时失败", tool_name)
        result = {"ok": False, "error": f"工具执行失败：{type(error).__name__}: {error}"}

    # tool 消息必须是字符串；ensure_ascii=False 保留中文可读性。
    return json.dumps(result, ensure_ascii=False)  # → json.dumps() 将工具结果转为字符串


def run_agent(question: str) -> AgentResult:
    """运行最多五轮工具调用；模型不再调用工具时返回最终文本回答。这是 Agent 主循环。"""
    client, settings = create_client()
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},  # 告诉模型你是销售业务助手，什么时候应该调用工具。
        {"role": "user", "content": question},  # 用户实际的问题。
    ]
    tool_rounds = 0

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=600,
            )
        except Exception as error:
            raise AgentServiceError(f"模型调用失败：{error}") from error

        assistant_message = response.choices[0].message
        # 将完整助手消息（包含 tool_calls）追加回历史，供下一轮模型理解。
        messages.append(assistant_message.model_dump(exclude_none=True))  # 追加 assistant message
        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            answer = assistant_message.content or "模型未返回文本回答。"
            return AgentResult(answer=answer, tool_rounds=tool_rounds)

        tool_rounds += 1
        for tool_call in tool_calls:
            tool_result = execute_tool_call(tool_call)
            # tool_call_id 将工具结果精确关联到模型发出的那一次调用。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    # 达到上限仍未结束时，不再请求模型，以防无限循环和额外费用。
    return AgentResult(
        answer=f"工具调用已达到最多 {MAX_TOOL_ROUNDS} 轮，已停止本次请求。",
        tool_rounds=tool_rounds,
    )
