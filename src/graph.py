from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agents import (
    dispatch,
    make_route_review,
    merge_node,
    single_agent_node,
    plan_node,
    review_node,
    worker_node,
    build_tool_node
)
from approval import approval_node, single_approval_node
from context import AppContext
from mcp_client import load_mcp_tools
from state import ProjectState,OverallState



async def build_graph(
    checkpointer: Any,
    mode: str = "llm",
    require_approval: bool = False,
):
    if mode not in {"mock", "llm","single"}:
        raise ValueError(f"mode must be mock or llm or single, got {mode!r}")

    tools = await load_mcp_tools()
    model = None
    if mode == "llm":
        from dotenv import load_dotenv
        load_dotenv(verbose=True)

        model = ChatOpenAI(
            model=os.getenv("MODEL", "deepseek-flash"),
            api_key=os.getenv("APIKEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.7,
            extra_body={"thinking":{"type":"disabled"}}
        )


    elif mode == "single":
        from dotenv import load_dotenv
        load_dotenv(verbose=True)

        model = ChatOpenAI(
            model=os.getenv("MODEL", "deepseek-flash"),
            api_key=os.getenv("APIKEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.7,
            extra_body={"thinking": {"type": "disabled"}}
        )


    ctx = AppContext(tools=tools, mode=mode, model=model)

    async def multi():
        builder = StateGraph(state_schema=ProjectState, context_schema=AppContext)
        builder.add_node("plan", plan_node)
        builder.add_node(
            "worker",
            worker_node,
            retry_policy=RetryPolicy(
                max_attempts=2, initial_interval=0.5, jitter=False
            ),
        )
        builder.add_node("merge", merge_node)
        builder.add_node("review", review_node)
        if require_approval:
            builder.add_node("approval", approval_node)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", dispatch, ["worker"])
        builder.add_edge("worker", "merge")  # 等所有 worker task 完成后汇总
        builder.add_edge("merge", "review")

        route = make_route_review(require_approval)
        targets = {"plan": "plan", END: END}
        if require_approval:
            targets["approval"] = "approval"
        builder.add_conditional_edges("review", route, targets)

        return builder.compile(checkpointer=checkpointer), ctx


    async def single():
        builder=StateGraph(state_schema=OverallState,context_schema=AppContext)
        builder.add_node("agent_node",single_agent_node,retry_policy=RetryPolicy
        (
            max_attempts=3,jitter=False,initial_interval=0.5
         ))
        builder.add_node("tools",build_tool_node(ctx))
        if require_approval:
            builder.add_node("approval",single_approval_node)

        from langchain_core.messages import AIMessage
        async def route(state: OverallState) -> str:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            if require_approval:
                return "approval"
            return END
        builder.add_edge(START,"agent_node")
        path_map={"tools":"tools",END:END}
        if require_approval:
            path_map["approval"]="approval"
        builder.add_conditional_edges("agent_node",path=route,path_map=path_map)
        builder.add_edge("tools","agent_node")
        return builder.compile(checkpointer=checkpointer),ctx


    if mode=="single":
        return await single()
    return await multi()
