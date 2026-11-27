from __future__ import annotations

from langgraph.graph import END
from langgraph.types import Command, interrupt

from langchain_core.messages import HumanMessage

from state import OverallState, ProjectState

def single_approval_node(state: OverallState) -> Command:
    """single 模式的审批：只依赖 messages；驳回时带着反馈回到 agent_node"""
    messages = state.get("messages", [])
    preview = str(messages[-1].content)[:800] if messages else ""
    decision = interrupt(
        {
            "type": "publish_approval",
            "merged_preview": preview,
        }
    )

    if isinstance(decision, str):
        action, feedback = decision, ""
    else:
        action = decision.get("action", "reject")
        feedback = decision.get("feedback", "")

    if action == "approve":
        return Command(goto=END, update={"log": ["approval:approve"]})

    revision = state.get("revision", 0) + 1
    if revision >= state.get("max_revision", 3):
        return Command(
            goto=END,
            update={"revision": revision, "log": ["approval:reject(escalated)"]},
        )
    return Command(
        goto="agent_node",
        update={
            "messages": [
                HumanMessage(content=f"审批未通过，请根据反馈修改后重新回答：{feedback or '请改进'}")
            ],
            "revision": revision,
            "log": ["approval:reject"],
        },
    )


def approval_node(state: ProjectState) -> Command:
    decision = interrupt(
        {
            "type": "publish_approval",
            "task": state["task"],
            "merged_preview": state.get("merged", "")[:800],
            "review": state.get("review", {}),
        }
    )

    if isinstance(decision,str):
        action=decision
        score=0.0
        feedback=""
        missing=[]

    else:
        action=decision.get("action","reject")
        score=decision.get("score",0.0)
        feedback=decision.get("feedback","请改进一下")
        missing=decision.get("missing",[])

    if action == "approve":
        return Command(
            goto=END,
            update={"status": "published", "log": ["approval:approve"]},
        )

    revision = state.get("revision", 0) + 1
    review = {
        "approved": False,
        "score": score,
        "feedback": feedback,
        "missing": missing,
    }
    if revision >= state.get("max_revisions", 3):
        return Command(
            goto=END,
            update={
                "review": review,
                "revision": revision,
                "status": "escalated",
                "log": ["approval:reject"],
            },
        )
    return Command(
        goto="plan",
        update={
            "review": review,
            "revision": revision,
            "status": "needs_revision",
            "log": ["approval:reject"],
        },
    )