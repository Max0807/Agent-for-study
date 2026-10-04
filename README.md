# 前三天动手练习：从 Python 脚本到 Prompt 评估

这个工程把前三天的理论变成六个按顺序完成的脚本。不要一次性运行全部文件；每完成一步，先读懂代码并修改一个地方，再进入下一步。

## 你最终会写出什么

| 步骤 | 对应学习日 | 脚本 | 能力 |
|---|---|---|---|
| 1 | Day 1 | `src/step1_jd_keywords.py` | 读取文件、列表、循环、字典式统计、排序 |
| 2 | Day 2 | `src/step2_first_api.py` | 完成第一次 LLM API 调用 |
| 3 | Day 2 | `src/step3_chat_cli.py` | 连续聊天、维护上下文、查看 Token 用量 |
| 4 | Day 3 | `src/step4_prompt_playground.py` | 加载并切换 5 个 Prompt 模板 |
| 5 | Day 3 | `src/step5_structured_output.py` | 使用 Pydantic 验证结构化输出 |
| 6 | Day 3 | `src/step6_prompt_eval.py` | 用固定测试集比较 Prompt 通过率 |

## 第 0 步：在 VS Code 中打开工程

1. 启动 VS Code。
2. 点击 `File -> Open Folder`。
3. 选择 `D:\AI大模型\AI大模型学习\llm-hands-on-days1-3`。
4. 如果右下角提示安装推荐扩展，安装 Python 和 Pylance。
5. 点击 `Terminal -> New Terminal`，确认终端路径位于本工程目录。

也可以在 PowerShell 中执行：

```powershell
code "D:\AI大模型\AI大模型学习\llm-hands-on-days1-3"
```

## 第 1 步：创建独立 Python 环境

在 VS Code 终端中逐条执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，只对当前终端临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

激活成功后，终端提示符前应出现 `(.venv)`。按 `Ctrl+Shift+P`，运行 `Python: Select Interpreter`，选择 `.venv\Scripts\python.exe`。

验证环境：

```powershell
python --version
python -c "import openai, dotenv, pydantic; print('环境安装成功')"
```

## 第 2 步：先运行不需要 API 的 Python 脚本

打开 `src/step1_jd_keywords.py`，按顺序阅读：常量、`count_keywords`、`main`、程序入口。然后运行：

```powershell
python .\src\step1_jd_keywords.py
```

你必须亲手完成的小修改：向 `KEYWORDS` 添加 `LangGraph`，再向 `data/jd_sample.txt` 增加一条包含 LangGraph 的岗位要求，重新运行并观察计数变化。

## 第 3 步：配置 API Key

在资源管理器中新建 `.env` 文件，内容参考 `.env.example`：

```dotenv
OPENAI_API_KEY=填写你自己的API密钥
OPENAI_MODEL=gpt-5.6-luna
OPENAI_BASE_URL=
```

注意：

- API Key 不能写进 `.py` 文件，也不要发到聊天、截图或代码仓库。
- ChatGPT 订阅和 API 账户额度是两套系统。
- 如果你的账户不能使用示例模型，把 `OPENAI_MODEL` 改成账户实际可用的模型 ID。
- 使用官方 OpenAI API 时，`OPENAI_BASE_URL` 保持为空。

## 第 4 步：完成第一次模型调用

先读 `src/config.py`，弄清楚 `.env -> Settings -> OpenAI client` 的流向，再运行：

```powershell
python .\src\step2_first_api.py
```

运行成功后应看到模型名、回答和 Token 用量。亲手把 `input` 修改为“为什么上下文窗口不是永久记忆？”，再次运行。

## 第 5 步：运行连续聊天 CLI

```powershell
python .\src\step3_chat_cli.py
```

依次输入：

```text
什么是 Embedding？
它在 RAG 中负责什么？
/clear
我刚才第一个问题是什么？
/exit
```

观察 `/clear` 前后模型是否还能使用原来的历史。然后把 `INSTRUCTIONS` 中的回答长度从 300 字改成 100 字，验证 Prompt 是否生效。

## 第 6 步：运行 Prompt Playground

```powershell
python .\src\step4_prompt_playground.py
```

至少测试：

1. `query_rewrite`：输入一条带寒暄的知识库问题。
2. `grounded_qa`：提供两段带 `[source_id]` 的资料，再提出问题。
3. `intent_router`：分别输入知识库、计算、SQL 和信息不足的问题。

修改一个 Prompt 文件后立即重跑同一个输入，比较修改前后的结果。一次只改一个主要规则。

## 第 7 步：验证结构化输出

```powershell
python .\src\step5_structured_output.py
```

第一次直接回车使用示例文档。第二次输入一份没有日期的文档，检查 `effective_date` 是否为 `null`、`needs_human_review` 是否合理。

重点理解：Prompt 负责说明任务，Pydantic 模型负责定义并校验程序需要的数据结构。

## 第 8 步：用测试集评估 Prompt

本步骤会调用模型 10 次，产生少量 API 费用：

```powershell
python .\src\step6_prompt_eval.py
```

比较 `baseline` 和 `improved` 的准确率。然后在 `data/route_eval_cases.json` 中增加两个边界案例，再运行一次。

## 使用 VS Code 调试

1. 打开任意脚本，在希望暂停的行号左侧单击，出现红点即断点。
2. 按 `F5`。
3. 从列表选择对应的 `Step`。
4. 在左侧 Run and Debug 面板查看变量。
5. 对 `step1_jd_keywords.py`，建议在 `count = ...` 一行打断点，观察循环中的 `keyword` 和 `count`。

## 常见错误

| 错误 | 处理方法 |
|---|---|
| `python` 找不到 | 重新安装 Python，并勾选 Add Python to PATH |
| 无法运行 `Activate.ps1` | 执行本文中的临时 `Set-ExecutionPolicy` 命令 |
| `ModuleNotFoundError` | 确认终端有 `(.venv)`，再运行安装命令 |
| 找不到 `OPENAI_API_KEY` | 确认文件名是 `.env`，不是 `.env.txt` |
| `401` | API Key 无效或读取到了多余空格 |
| `429` | 额度、账单或速率限制问题 |
| model not found | 修改 `.env` 中的模型 ID |
| 连接失败 | 检查网络；第三方服务按其文档填写 `OPENAI_BASE_URL` |

## 三天代码验收线

- 能解释 `if __name__ == "__main__"`、函数、循环、列表和字典分别在代码中的作用。
- 不看答案，重新写出一次最小 `client.responses.create(...)` 调用。
- 能说明多轮聊天历史为什么会增加 Token 消耗。
- 能修改 Prompt 并使用同一测试输入比较效果。
- 能解释普通文本 JSON 与 Pydantic 校验结果的区别。
- `.env` 未被提交或分享，源码中没有 API Key。

## Day 6：文档解析、清洗与分块

支持 PDF、Markdown、HTML 和 CSV。每个 chunk 都包含原文件名、文件类型、页码、标题和顺序编号；非分页格式的页码为 `null`。

启动 FastAPI 后，在 `http://127.0.0.1:8000/docs` 使用 `POST /documents/parse` 上传文件：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

也可以直接解析本地文件：

```powershell
.\.venv\Scripts\python.exe .\src\day06_parse_documents.py .\README.md
```

把结果保存到 JSON：

```powershell
.\.venv\Scripts\python.exe .\src\day06_parse_documents.py .\README.md --output .\data\readme_chunks.json
```

运行全部自动测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Day 7：Embedding 与向量知识库

Day7 复用 Day6 的文档加载、清洗和分块结果，再使用 Sentence Transformers
生成向量，并把文本、向量和来源元数据持久化到 Chroma。Day7 只负责检索，暂不让
DeepSeek 生成答案；检索结果会在 Day8 作为 RAG 的上下文。

如果还没有 `llm-fastapi` 环境，先创建一次；已经创建过则直接激活：

```powershell
conda create -n llm-fastapi python=3.11 -y
conda activate llm-fastapi
python -m pip install -r requirements.txt
```

项目已经在 `data/knowledge_base` 中准备了 20 份小型示例文档。首次建立索引：

```powershell
python .\src\day07_build_index.py
```

需要明确删除旧 collection 并重建时才使用：

```powershell
python .\src\day07_build_index.py --rebuild
```

命令行验证语义搜索：

```powershell
python .\src\day07_search.py "离开工位时电脑应该怎么处理？" --top-k 3
python .\src\day07_search.py "九月华东销售情况" --file-type csv
```

启动 FastAPI：

```powershell
python -m uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000/docs`，可以测试：

- `GET /knowledge/stats`：查看 collection 名称和 chunk 数量。
- `POST /knowledge/search`：执行 Top-k 语义检索和 metadata filter。

运行全部自动测试：

```powershell
python -m pytest -q
```

## Day 8：Naive RAG 问答

Day8 在 Day7 检索结果上增加上下文拼接和 DeepSeek 生成：先召回 Top-k chunks，
给来源编号为 `S1`、`S2`，再把问题与上下文交给模型，最终返回答案和引用来源。

开始前确认 Day7 索引已经存在：

```powershell
python -c "from app.vector_store import get_vector_store; print(get_vector_store().count())"
```

输出必须大于 0；否则先运行：

```powershell
python .\src\day07_build_index.py
```

命令行完成一次真实 RAG 问答（会调用 DeepSeek 并产生 Token 用量）：

```powershell
python .\src\day08_rag_cli.py "员工出差需要谁审批？" --top-k 3
```

启动 FastAPI：

```powershell
python -m uvicorn app.main:app --reload
```

在 `http://127.0.0.1:8000/docs` 测试 `POST /rag/ask`，请求体：

```json
{
  "question": "员工出差需要谁审批？",
  "top_k": 3
}
```

`source`、`file_type`、`title` 都是可选的严格过滤条件；不需要过滤时请省略或设为
`null`，不要保留 Swagger 中的占位字符串。没有检索结果时接口不会调用 DeepSeek，
而是直接返回“根据当前知识库资料无法确定”。

Day8 配置项：

```dotenv
RAG_TOP_K=5
RAG_MAX_CONTEXT_CHARS=6000
RAG_MAX_OUTPUT_TOKENS=800
```

运行自动测试不会调用真实 DeepSeek：

```powershell
python -m pytest -q
```
