"""四个agent角色"""

from __future__ import annotations
import json
from typing import Any,Literal

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field

from context import AppContext
from state import ProjectState, Review, Subtask, WorkerInput, WorkerResult,OverallState


#工具函数
def mcp_text(result: Any) -> str:
    """统一返回字符串"""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(result)

def get_tool(ctx: AppContext, name: str):
    for tool in ctx.tools:
        if tool.name == name:
            return tool
    raise KeyError(f"MCP tool not found: {name}")

async def rag_search(ctx: AppContext, query: str, top_k: int = 3) -> list[dict]:
    tool = get_tool(ctx, "search_knowledge")
    raw = await tool.ainvoke({"query": query, "top_k": top_k})
    data = json.loads(mcp_text(raw))
    return data.get("hits", [])


#单agent（简单任务）

async def single_agent_node(state:OverallState,runtime:Runtime[AppContext]):
    ctx=runtime.context
    if ctx.model is None:
        raise ValueError("没有模型")
    agent = ctx.model.bind_tools(ctx.tools)
    resp = await agent.ainvoke(state["messages"])
    return {"messages": [resp]}


def build_tool_node(ctx: AppContext):
    from langgraph.prebuilt import ToolNode
    return ToolNode(tools=ctx.tools)




#主管agent
class SubtaskModel(BaseModel):
    title: str
    instruction: str
    role: str
    acceptance: str


class PlanModel(BaseModel):
    subtasks: list[SubtaskModel] = Field(min_length=1, max_length=4)


async def plan_node(state: ProjectState, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context
    revision = state.get("revision", 0)

    if ctx.mode == "llm" and ctx.model is not None:
        planner = ctx.model.with_structured_output(PlanModel, method="function_calling")
        feedback = state.get("review", {}).get("feedback", "")
        plan = await planner.ainvoke(
            [
                SystemMessage(content="""你是主管 agent，把任务拆成 1-4 个可验收，尽量可并行的子任务,
                并为每个子任务指定 role 和 acceptance。
                例如：写一个网页项目：分配两个子任务，一个前端架构师（role）负责前端，一个后端工程师（role）负责后端接口"""),
                HumanMessage(
                    content=f"任务：{state['task']}\n上一轮反馈：{feedback or '无'}\n"
                ),
            ]
        )
        subtasks = [
            {
                "id": f"r{revision}-s{i}",
                "title": item.title,
                "instruction": item.instruction,
                "role": item.role,
                "acceptance": item.acceptance,
            }
            for i, item in enumerate(plan.subtasks, 1)
        ]
        return {"plan": subtasks, "status": "planned", "log": [f"plan r{revision}: llm"]}

    if revision == 0:
        subtasks: list[Subtask] = [
            {
                "id": "r0-s1",
                "title": "资料检索",
                "instruction": "检索 LangGraph 多智能体、supervisor、checkpoint 相关资料，提炼要点。",
                "role": "researcher",
                "acceptance": "至少 2 条带引用的要点",
            },
            {
                "id": "r0-s2",
                "title": "方案分析",
                "instruction": "分析 supervisor 模式的优缺点、失败模式与适用边界。",
                "role": "analyst",
                "acceptance": "给出 2 条风险与 2 条取舍",
            },
            {
                "id": "r0-s3",
                "title": "落地建议",
                "instruction": "给出 MCP 工具集成与 RAG 的三条实施建议。",
                "role": "writer",
                "acceptance": "建议可执行、有引用支撑",
            },
        ]
        return {"plan": subtasks, "status": "planned", "log": [f"plan r{revision}: {len(subtasks)} tasks"]}

    feedback = state.get("review", {}).get("feedback", "补充缺失内容")
    fix: Subtask = {
        "id": f"r{revision}-fix",
        "title": "修订与补充",
        "instruction": f"根据验收反馈修订：{feedback}",
        "role": "writer",
        "acceptance": "覆盖反馈中的缺失项",
    }
    return {"plan": [fix], "status": "revising", "log": [f"plan r{revision}: revision"]}

def dispatch(state: ProjectState) -> list[Send]:
    return [
        Send(
            node="worker",
            arg={"task": state["task"], "subtask": subtask, "revision": state["revision"]},
        )
        for subtask in state["plan"]
    ]


#员工agent

def _mock_worker_output(subtask: Subtask, hits: list[dict]) -> str:
    if not hits:
        return f"### {subtask['title']}\n\n未检索到相关资料。"
    lines = [f"### {subtask['title']}"]
    for hit in hits:
        snippet = hit["text"].replace("\n", " ")[:140]
        lines.append(f"- {snippet}（来源：{hit['doc_id']}）")
    return "\n".join(lines)


async def _llm_worker(
    ctx: AppContext,
    task: str,
    subtask: Subtask,
) -> tuple[str, list[str]]:
    assert ctx.model is not None
    bound = ctx.model.bind_tools(ctx.tools)
    messages = [
        SystemMessage(
            content=(
                f"你是员工 agent，角色：{subtask['role']}"
                "需要工具时可以使用工具完成任务"
            )
        ),
        HumanMessage(content=f"总任务：{task}\n\n子任务：{subtask['instruction']}\n验收：{subtask['acceptance']}"),
    ]
    citations: list[str] = []
    for _ in range(4):
        ai: AIMessage = await bound.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return str(ai.content), sorted(set(citations))
        for call in ai.tool_calls:
            try:
                tool = get_tool(ctx, call["name"])
                raw = await tool.ainvoke(call["args"])
                text = mcp_text(raw)
                if call["name"] == "search_knowledge":
                    data = json.loads(text)
                    citations.extend(hit["doc_id"] for hit in data.get("hits", []))
                messages.append(ToolMessage(content=text, tool_call_id=call["id"], status="success"))
            except Exception as exc:  # 工具失败回填给模型，不炸整张图
                messages.append(
                    ToolMessage(content=f"Error: {exc}", tool_call_id=call["id"], status="error")
                )
    return "达到最大工具轮次，未能完成。", sorted(set(citations))



async def worker_node(state: WorkerInput, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context
    subtask = state["subtask"]

    try:
        if ctx.mode == "llm":
            output, citations = await _llm_worker(ctx, state["task"], subtask)
        else:
            hits = await rag_search(ctx, subtask["instruction"], top_k=3)
            output = _mock_worker_output(subtask, hits)
            citations = sorted({hit["doc_id"] for hit in hits})
        result: WorkerResult = {
            "subtask_id": subtask["id"],
            "title": subtask["title"],
            "role": subtask["role"],
            "output": output,
            "citations": citations,
            "ok": True,
            "error": "",
        }
    except Exception as exc:
        if ctx.mode == "llm":
            # 交给节点的 retry_policy；工具级错误已在 _llm_worker 里回填给模型
            raise
        result = {
            "subtask_id": subtask["id"],
            "title": subtask["title"],
            "role": subtask["role"],
            "output": "",
            "citations": [],
            "ok": False,
            "error": str(exc),
        }
    return {"results": [result]}


#对接agent
async def merge_node(state: ProjectState, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context
    results = sorted(state.get("results", []), key=lambda item: item["subtask_id"])
    if ctx.mode == "llm" and ctx.model is not None:
        prompt = [
            SystemMessage(
                content=(
                    "你是汇总编辑.你的任务是把多个员工结果合并成一份连贯交付物"
                )
            ),
            HumanMessage(
                content=f"总任务：{state['task']}\n\n员工结果：\n"
                + "\n\n".join(r["output"] for r in results)
            ),
        ]
        ai = await ctx.model.ainvoke(prompt)
        merged = str(ai.content)
    else:
        parts = [f"# 交付物：{state['task']}"]
        for result in results:
            parts.append(result["output"])
            if result["citations"]:
                parts.append("引用：" + ", ".join(result["citations"]))
        merged = "\n\n".join(parts)

    citations = sorted({cite for result in results for cite in result["citations"]})
    return {"merged": merged, "citations": citations, "status": "merged"}


#验收agent
class ReviewModel(BaseModel):
    approved: bool
    score: float = Field(ge=0.0, le=1.0)
    feedback: str
    missing: list[str]


def _rule_review(state: ProjectState) -> Review:
    issues: list[str] = []
    results = state.get("results", [])
    if not results:
        issues.append("没有任何员工结果")
    if any(not item["ok"] for item in results):
        issues.append("存在失败的子任务")
    if len(state.get("merged", "")) < 100:
        issues.append("交付物过短")
    if len({item["role"] for item in results}) < 3:
        issues.append("角色覆盖不足")
    approved = not issues
    return {
        "approved": approved,
        "score": 1.0 if approved else max(0.2, 1.0 - 0.2 * len(issues)),
        "feedback": "通过" if approved else "；".join(issues),
        "missing": issues,
    }


async def review_node(state: ProjectState, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context
    if ctx.mode == "llm" and ctx.model is not None:
        judge = ctx.model.with_structured_output(ReviewModel, method="function_calling")
        try:
            verdict = await judge.ainvoke(
                [
                    SystemMessage(content="你是主管，只做验收，验收交付物能否实现任务要求"),
                    HumanMessage(
                        content=f"任务：{state['task']}\n\n交付物：\n{state['merged']}\n\n"
                        f"引用：{state['citations']}"
                    ),
                ]
            )
            review = verdict.model_dump()
        except Exception:
            review = _rule_review(state)
    else:
        review = _rule_review(state)

    revision = state.get("revision", 0) + (0 if review["approved"] else 1)
    if review["approved"]:
        status = "approved"
    elif revision >= state.get("max_revisions", 2):
        status = "escalated"
    else:
        status = "needs_revision"
    return {
        "review": review,
        "revision": revision,
        "status": status,
        "log": [f"review: approved={review['approved']} score={review['score']} revision={revision}"],
    }


def make_route_review(require_approval: bool):
    def route(state: ProjectState) -> str:
        if state["review"]["approved"]:
            return "approval" if require_approval else END
        if state.get("revision", 0) >= state.get("max_revisions", 2):
            return END
        return "plan"

    return route
