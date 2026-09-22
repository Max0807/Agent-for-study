"""Agent 可调用的本地工具，以及安全的 SQLite 访问逻辑。这个文件不是让模型“直接执行 Python”，而是提前定义好受控制的能力。"""

import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.config import DATABASE_FILE


# 固定主键配合 INSERT OR IGNORE，使初始化脚本可以安全地反复执行。
SAMPLE_SALES = [
    (1, "华东", "2026-09", 128_000.0),
    (2, "华东", "2026-10", 136_500.0),
    (3, "华南", "2026-09", 96_000.0),
    (4, "华南", "2026-10", 102_300.0),
    (5, "华北", "2026-09", 88_400.0),
    (6, "华北", "2026-10", 91_700.0),
]


class CalculateArgs(BaseModel):  # 这个函数执行实际运算：def calculate(operation, a, b)
    """calculate 工具允许的参数。"""

    # forbid 确保模型不能夹带未定义字段。
    model_config = ConfigDict(extra="forbid")  # 集中管理模型配置

    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class QuerySalesArgs(BaseModel):
    """query_sales 工具允许的业务筛选参数。"""

    model_config = ConfigDict(extra="forbid")

    region: str | None = None
    month: str | None = None

# 这个函数会在两个地方被调用： 服务启动时：lifespan()。 查询销售数据前：query_sales()。
def initialize_database() -> None:
    """创建 sales 表并写入幂等的演示销售数据。"""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)  # 创建 data/ 目录
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sales(
                id INTEGER PRIMARY KEY,
                region TEXT NOT NULL,
                month TEXT NOT NULL,
                amount REAL NOT NULL
            )
            """
        )  # 创建一张名为 sales的表格
        connection.executemany(
            """
            INSERT OR IGNORE INTO sales (id, region, month, amount)
            VALUES (?, ?, ?, ?)
            """,
            SAMPLE_SALES,
        )


def calculate(operation: str, a: float, b: float) -> dict[str, Any]:
    """执行四则运算；使用明确分支而非危险的 eval()。"""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"ok": False, "error": "除数不能为零。"}
        result = a / b
    else:
        return {"ok": False, "error": f"不支持的运算：{operation}"}

    return {"ok": True, "operation": operation, "a": a, "b": b, "result": result}


def query_sales(region: str | None, month: str | None) -> dict[str, Any]:
    """按区域和月份查询销售记录，不接受模型提供的原始 SQL。"""
    initialize_database()

    # SQL 主体固定在 Python 内；模型只能影响参数值，不能改变 SQL 结构。
    sql = "SELECT region, month, amount FROM sales"  # 固定 SQL 开始
    conditions: list[str] = []  # 根据用户是否提供 region、month，追加固定条件
    parameters: list[str] = []

    if region is not None:
        conditions.append("region = ?")
        parameters.append(region)
    if month is not None:
        conditions.append("month = ?")
        parameters.append(month)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY month, region"

    try:
        with sqlite3.connect(DATABASE_FILE) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, parameters).fetchall()
    except sqlite3.Error as error:
        return {"ok": False, "error": f"销售数据库查询失败：{error}"}

    sales_rows = [dict(row) for row in rows]  # 查询结果会转成 Python 字典
    return {"ok": True, "count": len(sales_rows), "rows": sales_rows}


# 工具定义会发送给模型，让模型知道可用函数及其 JSON 参数模式。
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行两个数字的加、减、乘、除计算。",
            "parameters": CalculateArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sales",
            "description": "按可选区域和月份查询公司销售数据。",
            "parameters": QuerySalesArgs.model_json_schema(),
        },
    },
]

# 这两个白名单是 Agent 执行工具前的最后一道保护;把“工具名称”和“该工具的参数校验规则”对应起来
TOOL_HANDLERS = {"calculate": calculate, "query_sales": query_sales}
ARGUMENT_MODELS = {"calculate": CalculateArgs, "query_sales": QuerySalesArgs}
