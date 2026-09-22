"""初始化 Day4 工具调用练习所需的 SQLite 销售数据库。"""

import sqlite3
from pathlib import Path


# 当前脚本位于 src 目录，向上一级即为项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "data" / "company.db"

# 固定主键让 INSERT OR IGNORE 在重复运行时不会创建重复记录。
SAMPLE_SALES = [
    (1, "华东", "2026-09", 128_000.0),
    (2, "华东", "2026-10", 136_500.0),
    (3, "华南", "2026-09", 96_000.0),
    (4, "华南", "2026-10", 102_300.0),
    (5, "华北", "2026-09", 88_400.0),
    (6, "华北", "2026-10", 91_700.0),
]


def initialize_database() -> None:
    """创建 sales 表并写入幂等的模拟销售数据。"""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)  # 确保在创建数据库文件前，它所在的“父级文件夹”存在

    with sqlite3.connect(DATABASE_FILE) as connection:  #  打开数据库连接，把连接对象命名为 connection
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sales(
                id INTEGER PRIMARY KEY,
                region TEXT NOT NULL,
                month TEXT NOT NULL,
                amount REAL NOT NULL
            )
            """
        )  # 定义了一张名为 sales的表格，并严格规定了它的 4 列
        connection.executemany(
            """
            INSERT OR IGNORE INTO sales (id, region, month, amount)  # 批量插入数据，且遇到重复 ID 就自动跳过
            VALUES (?, ?, ?, ?)
            """,
            SAMPLE_SALES,
        )  # 打开数据库文件


def main() -> None:
    """初始化数据库并显示数据库文件的位置。"""
    initialize_database()
    print(f"数据库已就绪：{DATABASE_FILE}")


if __name__ == "__main__":
    main()
