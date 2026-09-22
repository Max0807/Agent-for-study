"""提供给 Day4 Agent 调用的安全本地工具。"""

import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "data" / "company.db"


class CalculateArgs(BaseModel):  # 继承自 Pydantic 的 BaseModel,是一个数据容器
    """计算器工具允许的参数。"""

    # ConfigDict是 Pydantic v2 专门用来存放模型配置项的字典类;allow：允许；ignore：忽略；forbid：禁止;
    model_config = ConfigDict(extra="forbid")  # extra决定了模型在接收数据时，对未定义的字段的处理：禁止

    operation: Literal["add", "subtract", "multiply", "divide"]  # Literal 是 Python 的类型提示特例，它强制 operation 的值必须严格等于这四个字符串中的一个
    a: float  # float 是 Python 的类型提示特例，它强制 a 和 b 的值必须是浮点数
    b: float  # float 是 Python 的类型提示特例，它强制 a 和 b 的值必须是浮点数


class QuerySalesArgs(BaseModel):
    """销售查询工具允许的业务筛选参数。"""

    model_config = ConfigDict(extra="forbid")

    region: str | None = None  # region 的值必须是字符串或者None
    month: str | None = None  # month 的值必须是字符串或者None


def calculate(operation: str, a: float, b: float) -> dict:
    """执行指定的四则运算，不使用 eval()。"""
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
        # 此分支是防御性处理；正常情况下 Pydantic 会在调用前拦截非法操作。
        return {"ok": False, "error": f"不支持的运算：{operation}"}

    return {"ok": True, "operation": operation, "a": a, "b": b, "result": result}


def query_sales(region: str | None, month: str | None) -> dict:  # 返回值的类型必须是 dict
    """按区域和月份查询销售数据，只接受业务参数而非任意 SQL。"""
    sql = "SELECT region, month, amount FROM sales"  # 先把基础的查询语句定为 "...",这是一个SQL语句
    conditions: list[str] = []  # 存条件片段（conditions）
    parameters: list[str] = []  # 存具体的参数值（parameters）

    if region is not None:  # 如果传入了 region，就往列表里加一条 "region = ?"，并把具体的地区值存进参数列表
        conditions.append("region = ?")
        parameters.append(region)
    if month is not None:  # 如果传入了 month，就往列表里加一条 "month = ?"，并把具体的月份值存进参数列表
        conditions.append("month = ?")
        parameters.append(month)

    if conditions:  # 如果这个条件列表里有东西，就用 WHERE 把它们用 AND 连接起来，拼到基础 SQL 后面
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY month, region"  # 查询顺序;如果两个参数都没传，那这个 SQL 就是查整张表,最后返回的也是整张表

    try:
        with sqlite3.connect(DATABASE_FILE) as connection:  # 找到数据库文件，并打开一个连接
            connection.row_factory = sqlite3.Row  # 告诉 SQLite 使用字典作为行对象(数据库默认返回的是元组)
            rows = connection.execute(sql, parameters).fetchall()  # 执行这条 SQL 查询，然后把所有查到的结果行一次性赋值给 rows 变量
    except sqlite3.Error as error:
        return {"ok": False, "error": f"销售数据库查询失败：{error}"}

    sales_rows = [dict(row) for row in rows]  # 把每个 Row 对象转成 Python 的 dict
    return {"ok": True, "count": len(sales_rows), "rows": sales_rows}


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

# 仅允许此白名单中的工具被 Agent 执行。
TOOL_HANDLERS = {
    "calculate": calculate,
    "query_sales": query_sales,
}

# 每种工具名对应其参数校验模型。
ARGUMENT_MODELS = {
    "calculate": CalculateArgs,
    "query_sales": QuerySalesArgs,
}
