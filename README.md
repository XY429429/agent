# Multi-Agent Supervisor

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多智能体任务编排示例：主管 agent 负责规划与验收，员工 agent 并行执行子任务，支持 MCP 工具调用、本地知识库检索、人工审批、断点续跑，以及 FastAPI 服务化。

> 仓库不包含任何密钥。运行前请复制 `.env.example` 为 `.env` 并填写自己的 API Key；`.env` 与本地 checkpoint 数据库均已被 `.gitignore` 排除。

## 功能

- **Supervisor 多 agent 流程**：`plan → worker（Send 扇出）→ merge → review`；验收未通过时自动回到 `plan` 修订，最多 `MAX_REVISIONS` 轮。
- **三种运行模式**
  - `llm`（默认）：由大模型负责规划、执行、汇总与验收，默认对接 DeepSeek 的 OpenAI 兼容接口。
  - `single`：单 agent + 工具循环，适合简单任务。
  - `mock`：不调用大模型，使用规则与本地检索完成流程（需启用下方的 `search_knowledge` 工具）。
- **MCP 工具**：通过 `langchain-mcp-adapters` 以 stdio 连接项目内 MCP 服务（`search_web`、`fetch_web`）和 filesystem 服务。
- **RAG 模块**：Markdown 知识库按标题/段落切块，`BAAI/bge-small-zh-v1.5` 向量检索 + CrossEncoder 重排（`src/rag.py`）。
- **持久化与恢复**：LangGraph checkpoint 默认使用 SQLite，可选 Postgres；支持按 `thread_id` 查询状态、历史并恢复执行。
- **人工审批**：通过 `interrupt` 暂停任务，审批通过或带反馈驳回。
- **FastAPI 服务**：多租户、API Key 鉴权、限流、SSE 流式进度、审批恢复。

## 目录结构

```text
.
├── src/
│   ├── cli.py            # 命令行入口
│   ├── api.py            # FastAPI 服务
│   ├── graph.py          # 构图：multi / single 模式
│   ├── agents.py         # 主管、员工、汇总、验收节点
│   ├── approval.py       # 人工审批节点
│   ├── state.py          # 状态定义与 reducer
│   ├── checkpoint.py     # SQLite / Postgres checkpointer
│   ├── config.py         # 环境变量配置
│   ├── mcp_client.py     # MCP 客户端
│   ├── mcp_server.py     # MCP 服务端（search_web / fetch_web）
│   ├── rag.py            # 本地知识库检索
│   ├── context.py        # 运行时上下文
│   └── logging_utils.py  # 日志
├── data/knowledge/       # 本地 Markdown 知识库
├── examples/             # 示例请求占位文件
├── requirements.txt
└── .env.example
```

## 环境要求

- Python 3.11+
- Node.js（filesystem MCP 默认通过 `npx` 启动；也可全局安装 `mcp-server-filesystem` 后离线使用）
- 首次使用 RAG 时需要下载 `BAAI/bge-small-zh-v1.5` 与重排模型（或提前准备本地缓存）

## 快速开始

```bash
git clone https://github.com/XY429429/agent.git
cd agent

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate   # Git Bash
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # Windows 也可用 copy .env.example .env
```

编辑 `.env`，至少填写 API Key：

```dotenv
APIKEY=你的 API Key
MODEL=deepseek-flash
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

## 命令行用法

```bash
# 默认 llm 模式
python src/cli.py run --task "介绍一下 LangGraph 的 supervisor 模式" --thread demo-001

# 人工审批：执行到审批节点后暂停
python src/cli.py run --task "写一份 MCP 集成方案" --thread demo-002 --approval

# 审批后恢复（approve / reject）
python src/cli.py resume --thread demo-002 --decision approve
python src/cli.py resume --thread demo-002 --decision reject --feedback "请补充引用"

# 查看当前状态、checkpoint 历史、可用 MCP 工具
python src/cli.py state --thread demo-001
python src/cli.py history --thread demo-001
python src/cli.py tools

# 全局参数（--mode / --db / --log-level）需写在子命令之前
python src/cli.py --mode mock run --task "梳理 RAG 的落地要点"
```

`--mode` 可选值：`llm`（默认）、`single`、`mock`；`--db` 可指定 checkpoint 的 SQLite 路径。

## API 服务

```bash
python src/cli.py serve --host 127.0.0.1 --port 8000

# 等价写法
uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（无需鉴权） |
| GET | `/tools` | MCP 工具列表 |
| POST | `/tasks` | 同步执行任务 |
| POST | `/tasks/stream` | SSE 流式执行 |
| GET | `/tasks/{thread_id}` | 查询任务状态 |
| POST | `/tasks/{thread_id}/resume` | 审批通过/驳回后恢复 |
| GET | `/tasks/{thread_id}/history` | checkpoint 历史 |

鉴权与限流：配置了 `APIKEY` 环境变量后默认开启鉴权（也可用 `REQUIRE_AUTH=1` 强制开启），请求头使用 `X-API-Key`；租户通过 `X-Tenant-Id` 指定（默认 `default`），默认每个租户每分钟 120 次请求。

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APIKEY" \
  -H "X-Tenant-Id: demo" \
  -d '{"task":"介绍一下 LangGraph 的 supervisor 模式"}'
```

响应中的 `thread_id` 由服务端按 `租户:线程` 拼接，后续查询、恢复和历史接口都需要使用完整值。

## 配置项

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_MODE` | `llm` | 运行模式：`llm` / `single` / `mock` |
| `APP_DB_PATH` | `demo.sqlite` | SQLite checkpoint 路径 |
| `MAX_REVISIONS` | `2` | 最大修订轮数 |
| `REQUIRE_APPROVAL` | `0` | 是否默认暂停等待人工审批 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `MODEL` | `deepseek-flash` | 模型名称 |
| `APIKEY` | 空 | OpenAI 兼容接口的 API Key |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | 接口地址 |
| `CHECKPOINTER` | `sqlite` | `sqlite` / `postgres` |
| `POSTGRES_URI` | 空 | Postgres 连接串（`CHECKPOINTER=postgres` 时必填） |
| `REQUIRE_AUTH` | 有 `APIKEY` 时为 `1` | 是否强制 API 鉴权 |
| `RATE_LIMIT_PER_MIN` | `120` | 每个租户每分钟请求上限 |

## 数据与安全

- `.env`、`*.sqlite`、`.venv/`、`.idea/`、`__pycache__/` 均已加入 `.gitignore`，不会提交到仓库。
- 对话与 checkpoint 数据保存在本地 SQLite（默认 `demo.sqlite`），删除前请确认无需恢复历史任务。
- 生产环境建议切换到 Postgres checkpointer，并把密钥交给环境变量或密钥管理服务。

## 当前说明

- `src/mcp_server.py` 中的 `search_knowledge` RAG 工具当前处于注释状态；启用后 `mock` 模式以及 agent 的 RAG 检索链路才会完整工作。
- `src/run_demo.py`、`examples/*.json`、`Dockerfile`、`docker-compose.yml` 目前为占位文件。
- 未声明开源许可证，如需公开复用请先补充 `LICENSE`。
