import os  # 导入 os，用于读取系统环境变量。
from dataclasses import dataclass  # 导入 dataclass，用于快速创建数据配置类。
from pathlib import Path  # 导入 Path，用于处理文件和目录路径。

from dotenv import load_dotenv  # 导入 load_dotenv，用于读取 .env 文件中的环境变量。
from openai import OpenAI  # 导入 OpenAI 客户端；DeepSeek 可通过其兼容接口调用。


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")  # 读取项目根目录下的 .env 文件，加载环境变量到系统环境中。


@dataclass(frozen=True)  # dataclass 自动生成初始化方法；frozen=True 表示配置创建后不可修改。
class Settings:
    """保存调用大模型 API 所需的配置信息。"""

    api_key: str
    model: str
    base_url: str | None


# def load_settings() -> Settings:
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         raise RuntimeError(
#             "没有找到 OPENAI_API_KEY。请把 .env.example 另存为 .env，"
#             "然后填写你自己的 API Key。"
#         )

#     model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
#     base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
#     return Settings(api_key=api_key, model=model, base_url=base_url)
def load_settings() -> Settings:
    """
    从 .env 环境变量中读取 DeepSeek 配置。
    返回：Settings：包含 API Key、模型名和接口地址的配置对象。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()  # # strip() 去除密钥首尾可能存在的空格和换行。
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
    base_url = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ).strip()  # 接口地址

    return Settings(api_key=api_key, model=model, base_url=base_url)  # 创建并返回 Settings 配置对象。


def create_client() -> tuple[OpenAI, Settings]:
    """
    根据环境配置创建 API 客户端。
    返回：tuple[OpenAI, Settings]：
        第一个元素client是 OpenAI 兼容客户端，
        第二个元素settings是读取到的配置对象。
    """
    settings = load_settings()  # 调用 load_settings()，读取 .env 中的配置。
    if settings.base_url:  # 判断是否提供了自定义 API 地址。
        client = OpenAI(
            api_key=settings.api_key,  # api_key 用于身份验证；
            base_url=settings.base_url  # base_url 指向 DeepSeek 的兼容 API 地址。
            )
    else:
        client = OpenAI(api_key=settings.api_key)  # 若未提供地址，则仅使用 API Key 创建客户端。
    return client, settings  # 返回客户端和配置，供调用代码使用。


def format_usage(response: object) -> str:
    """
    从 API 响应中提取并格式化 Token 用量。
    参数：response (object)：DeepSeek API 返回的响应对象。
    返回：str：格式化后的输入、输出及总 Token 数量。
    """
    usage = getattr(response, "usage", None)  # getattr() 安全读取 response 的 usage 属性；
    if usage is None:
        return "Token usage: unavailable"

    input_tokens = getattr(usage, "prompt_tokens", "?")  # prompt_tokens 是输入给模型的 Token 数量；
    output_tokens = getattr(usage, "completion_tokens", "?")  # completion_tokens 是模型生成的 Token 数量；
    total_tokens = getattr(usage, "total_tokens", "?")  # total_tokens 是输入和输出的 Token 数量之和。
    return (
        f"Token usage: input={input_tokens}, output={output_tokens}, "
        f"total={total_tokens}"
    )


def print_api_error(error: Exception) -> None:
    """
    打印 API 调用失败时的错误信息和排查提示。
    参数：error (Exception)：捕获到的异常对象。
    """
    print(f"\nAPI 调用失败：{type(error).__name__}: {error}")
    print("依次检查：.env、API Key、模型名、账户额度和网络连接。")

