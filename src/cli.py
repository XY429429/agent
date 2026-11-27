"""命令行入口：run / resume / state / history / tools / serve。

    python src/cli.py run --task "..." --approval
    python src/cli.py state --thread demo-001
    python src/cli.py resume --thread demo-001 --decision approve
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from checkpoint import open_checkpointer
from config import Settings, load_settings, with_overrides
from graph import build_graph
from logging_utils import setup_logging
from mcp_client import load_mcp_tools


def _initial(task: str, max_revisions: int, mode: str = "llm") -> dict:
    if mode == "single":
        return {
            "messages": [HumanMessage(content=task)],
            "revision": 0,
            "max_revision": max_revisions,
            "log": [],
        }
    return {
        "task": task,
        "plan": [],
        "results": [],
        "merged": "",
        "citations": [],
        "revision": 0,
        "max_revisions": max_revisions,
        "status": "new",
        "log": [],
    }


def _message_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _print_single_summary(state: dict) -> None:
    messages = state.get("messages", [])
    ai_messages = [m for m in messages if m.type == "ai"]
    tool_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in ai_messages)
    final = next((m for m in reversed(ai_messages) if _message_text(m).strip()), None)
    print("\n=== 结果 (single) ===")
    print("revision  :", state.get("revision", 0))
    print("messages  :", len(messages))
    print("tool calls:", tool_calls)
    print("log       :", state.get("log", []))
    print("\n=== 最终回答 ===")
    print(_message_text(final) if final else "(没有最终回答)")


def _print_summary(state: dict) -> None:
    if state.get("messages"):
        _print_single_summary(state)
        return
    print("\n=== 结果 ===")
    print("status   :", state.get("status"))
    print("revision :", state.get("revision"))
    print("review   :", state.get("review"))
    print("citations:", state.get("citations"))
    print("results  :", [(r["subtask_id"], r["role"], r["ok"]) for r in state.get("results", [])])
    print("\n=== 交付物 ===")
    print(state.get("merged", ""))


@asynccontextmanager
async def _graph(settings: Settings, approval: bool):
    async with open_checkpointer(settings) as checkpointer:
        graph, ctx = await build_graph(
            checkpointer,
            mode=settings.mode,
            require_approval=approval,
        )
        yield graph, ctx


async def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    approval = args.approval or settings.require_approval
    max_revisions = args.max_revisions if args.max_revisions is not None else settings.max_revisions
    config = {"configurable": {"thread_id": args.thread}}
    async with _graph(settings, approval) as (graph, ctx):
        async for chunk in graph.astream(
            _initial(args.task, max_revisions, settings.mode),
            config,
            context=ctx,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                keys = list(update.keys()) if isinstance(update, dict) else type(update).__name__
                print(f"[{node}] {keys}")
        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            print("\n=== 等待人工审批 ===")
            for item in snapshot.interrupts:
                print("interrupt:", item.value)
            print(f"继续：python src/cli.py resume --thread {args.thread} --decision approve")
            return 0
        _print_summary(snapshot.values)
        return 0


async def cmd_resume(args: argparse.Namespace, settings: Settings) -> int:
    config = {"configurable": {"thread_id": args.thread}}
    async with _graph(settings, True) as (graph, ctx):
        snapshot = await graph.aget_state(config)
        if not snapshot.interrupts:
            print("该 thread 当前没有等待中的中断；状态：", snapshot.values.get("status"))
            return 1
        state = await graph.ainvoke(
            Command(resume={"action": args.decision, "feedback": args.feedback}),
            config,
            context=ctx,
        )
        if state.get("__interrupt__"):
            print("仍在等待审批：", state["__interrupt__"][0].value)
            return 0
        _print_summary(state)
        return 0


async def cmd_state(args: argparse.Namespace, settings: Settings) -> int:
    # 始终带 approval 节点构建：读取用，不影响已有 checkpoint；也能正确展示 next=('approval',)
    config = {"configurable": {"thread_id": args.thread}}
    async with _graph(settings, True) as (graph, _):
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            print("没有找到 thread:", args.thread)
            return 1
        print("next      :", snapshot.next)
        print("interrupts:", [item.value for item in snapshot.interrupts])
        _print_summary(snapshot.values)
        return 0


async def cmd_history(args: argparse.Namespace, settings: Settings) -> int:
    config = {"configurable": {"thread_id": args.thread}}
    async with _graph(settings, True) as (graph, _):
        count = 0
        async for snap in graph.aget_state_history(config, limit=args.limit):
            print(
                f"step={snap.metadata.get('step'):>2} "
                f"source={snap.metadata.get('source'):<5} "
                f"next={str(snap.next):<16} "
                f"status={snap.values.get('status')} "
                f"id={snap.config['configurable']['checkpoint_id'][-6:]}"
            )
            count += 1
        print("total:", count)
        return 0 if count else 1


async def cmd_tools(args: argparse.Namespace, settings: Settings) -> int:
    tools = await load_mcp_tools()
    print("mode:", settings.mode)
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    os.environ["APP_MODE"] = settings.mode
    os.environ["APP_DB_PATH"] = str(settings.db_path)
    os.environ["REQUIRE_APPROVAL"] = "1" if (args.approval or settings.require_approval) else "0"
    os.environ["MAX_REVISIONS"] = str(settings.max_revisions)
    import uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port, log_level=settings.log_level.lower())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent supervisor demo")
    parser.add_argument("--db", help="sqlite path (default: APP_DB_PATH or demo.sqlite)")
    parser.add_argument("--mode", choices=["mock", "llm","single"], help="agent mode")
    parser.add_argument("--log-level", help="DEBUG/INFO/WARNING/ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task")
    run.add_argument("--task", required=True)
    run.add_argument("--thread", default="demo-001")
    run.add_argument("--max-revisions", type=int, default=None)
    run.add_argument("--approval", action="store_true", help="pause for human approval")

    resume = sub.add_parser("resume", help="resume a paused task")
    resume.add_argument("--thread", required=True)
    resume.add_argument("--decision", choices=["approve", "reject"], default="approve")
    resume.add_argument("--feedback", default="", help="reject 时的修改意见（approve 时忽略）")

    state = sub.add_parser("state", help="show current state")
    state.add_argument("--thread", required=True)

    history = sub.add_parser("history", help="show checkpoint timeline")
    history.add_argument("--thread", required=True)
    history.add_argument("--limit", type=int, default=20)

    sub.add_parser("tools", help="list MCP tools")

    serve = sub.add_parser("serve", help="start FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--approval", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()
    if args.db:
        settings = with_overrides(settings, db_path=Path(args.db))
    if args.mode:
        settings = with_overrides(settings, mode=args.mode)
    if args.log_level:
        settings = with_overrides(settings, log_level=args.log_level.upper())
    if getattr(args, "max_revisions", None) is not None:
        settings = with_overrides(settings, max_revisions=args.max_revisions)
    setup_logging(settings.log_level)

    if args.command == "run":
        return asyncio.run(cmd_run(args, settings))
    if args.command == "resume":
        return asyncio.run(cmd_resume(args, settings))
    if args.command == "state":
        return asyncio.run(cmd_state(args, settings))
    if args.command == "history":
        return asyncio.run(cmd_history(args, settings))
    if args.command == "tools":
        return asyncio.run(cmd_tools(args, settings))
    if args.command == "serve":
        return cmd_serve(args, settings)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())