from pathlib import Path  # 导入 Path，用于更方便地处理文件路径

# 获取当前 Python 文件所在项目的根目录：
# __file__ 是当前脚本文件路径；
# resolve() 将路径转换为绝对路径；
# parents[1] 获取当前文件向上两级的目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 使用 / 运算符拼接路径，得到招聘信息文本文件的完整路径。
JD_FILE = PROJECT_ROOT / "data" / "jd_sample.txt"

# 定义需要统计的技能关键词列表。
KEYWORDS = [
    "Python",          # Python 编程语言
    "FastAPI",         # FastAPI Web 框架
    "RAG",             # 检索增强生成技术
    "Agent",           # 智能体
    "Prompt Engineering",  # 提示词工程
    "Embedding",       # 向量嵌入
    "向量数据库",       # 向量数据库
    "SQL",             # 结构化查询语言
    "Redis",           # Redis 缓存/数据库
    "Docker",          # Docker 容器技术
]


def count_keywords(text: str) -> list[tuple[str, int]]:
    """
    统计文本中每个预设技能关键词出现的次数，并按次数从高到低排序。

    参数：
        text (str): 待统计的原始文本内容。

    返回：
        list[tuple[str, int]]: 关键词及其出现次数组成的列表，
        例如：[("Python", 3), ("Docker", 1)]。
    """
    lower_text = text.lower()  # lower() 将文本转换为小写，便于进行不区分大小写的匹配。
    counts = []  # 创建空列表，用于保存每个关键词和对应的计数结果。

    for keyword in KEYWORDS:  # 依次遍历关键词列表中的每一个关键词。
        # keyword.lower() 将当前关键词转为小写；
        # count() 统计该关键词在 lower_text 中出现的次数。
        count = lower_text.count(keyword.lower())

        # append() 在列表末尾添加一个元素，这里添加“关键词 + 出现次数”的元组。
        counts.append((keyword, count))

    # sorted() 返回排序后的新列表，不修改原 counts；
    # key=lambda item: item[1] 表示按照元组的第二项（出现次数）排序；
    # reverse=True 表示降序，即出现次数最多的排在前面。
    return sorted(counts, key=lambda item: item[1], reverse=True)


def main() -> None:
    """
    程序主入口：读取招聘文本，统计技能关键词，并将结果打印到控制台。

    参数：
        无。

    返回：
        None：该函数只负责输出结果，不返回数据。
    """
    # read_text() 读取文件中的全部文本；
    # encoding="utf-8" 指定文件使用 UTF-8 编码，确保中文可正确读取。
    text = JD_FILE.read_text(encoding="utf-8")

    # 调用 count_keywords()，得到按出现次数排序后的统计结果。
    results = count_keywords(text)

    print("JD 技能关键词统计")  # 打印统计标题。
    print("-" * 30)  # 字符串乘法：生成 30 个 "-" 作为分隔线。

    for keyword, count in results:  # 逐个遍历统计结果，分别取出关键词和出现次数。
        # f-string 用于格式化输出；
        # <20 表示 keyword 左对齐，并预留 20 个字符宽度。
        print(f"{keyword:<20} {count}")


# __name__ 保存当前模块名称；
# 当该文件被直接运行时，__name__ 的值为 "__main__"；
# 当它被其他文件导入时，不会自动执行 main()。
if __name__ == "__main__":
    main()  # 调用主函数，启动程序。