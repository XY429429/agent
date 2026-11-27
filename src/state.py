from operator import add
from typing import Annotated,Literal,TypedDict
from langgraph.graph.message import MessagesState

class Subtask(TypedDict):
    id:str
    title:str
    instruction:str
    role:Literal["researcher","analyst","writer"]
    acceptance:str


class WorkerResult(TypedDict):
    subtask_id:str
    title:str
    role:str
    output:str
    citations:list[str]
    ok:bool
    error:str


class Review(TypedDict):
    approved: bool
    score: float
    feedback: str
    missing: list[str]


def merge_results(
    left: list[WorkerResult],
    right: list[WorkerResult],
) -> list[WorkerResult]:
    """按 subtask_id 去重：后完成的写入覆盖旧的同 id 结果。"""
    by_id = {item["subtask_id"]: item for item in left}
    for item in right:
        by_id[item["subtask_id"]] = item
    return [by_id[key] for key in sorted(by_id)]


def merge_unique(left: list[str], right: list[str]) -> list[str]:
    """引用去重，保持首次出现顺序。"""
    seen = set(left)
    out = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class ProjectState(TypedDict):
    task: str
    plan: list[Subtask]
    results: Annotated[list[WorkerResult], merge_results]
    merged: str
    citations: Annotated[list[str], merge_unique]
    review: Review
    revision: int
    max_revisions: int
    status: str
    log: Annotated[list[str], add]

class OverallState(MessagesState):
    """单agent时候状态"""
    revision:int
    max_revision:int
    log: Annotated[list[str], add]


class WorkerInput(TypedDict):
    """Send 给 worker 的输入：不是完整父状态。"""

    task: str
    subtask: Subtask
    revision: int
