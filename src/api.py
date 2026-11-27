"""FastAPI 服务：鉴权、多租户、限流、同步执行、SSE、审批恢复、历史。

启动：
    python src/cli.py serve --port 8000
    uvicorn api:app --app-dir src --port 8000
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from checkpoint import open_checkpointer
from config import load_settings
from graph import build_graph
from logging_utils import setup_logging

settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("multi_agent.api")

TENANT_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with open_checkpointer(settings) as checkpointer:
        app.state.graph, app.state.ctx = await build_graph(
            checkpointer,
            mode=settings.mode,
            require_approval=settings.require_approval,
        )
        yield


app = FastAPI(title="Multi-Agent Supervisor", version="0.3.0", lifespan=lifespan)


class TaskIn(BaseModel):
    task: str = Field(min_length=5, max_length=4000)
    thread_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9:_-]{1,128}$")
    max_revisions: int | None = Field(default=None, ge=0, le=5)


class ResumeIn(BaseModel):
    decision: Literal["approve", "reject"] = "approve"


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    elapsed = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s %s %.0fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        request_id,
    )
    return response


def auth(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> str:
    if settings.require_auth and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid api key")

    tenant = x_tenant_id or "default"
    if not TENANT_RE.fullmatch(tenant):
        raise HTTPException(status_code=400, detail="invalid tenant id")

    now = time.monotonic()
    bucket = _rate_buckets[tenant]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_min:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)

    request.state.tenant_id = tenant
    return tenant


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _thread_id(tenant: str, raw: str | None) -> str:
    return f"{tenant}:{raw or 'api-' + uuid4().hex[:10]}"


def _require_tenant(tenant: str, thread_id: str) -> None:
    if not thread_id.startswith(f"{tenant}:"):
        raise HTTPException(status_code=404, detail="thread not found")


def _initial(task: str, max_revisions: int) -> dict:
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


def _summary(state: dict) -> dict:
    return {
        "status": state.get("status"),
        "revision": state.get("revision"),
        "review": state.get("review"),
        "citations": state.get("citations", []),
        "results": [
            {
                "subtask_id": item["subtask_id"],
                "role": item["role"],
                "ok": item["ok"],
                "citations": len(item.get("citations", [])),
            }
            for item in state.get("results", [])
        ],
        "merged_preview": state.get("merged", "")[:1000],
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health():
    tools = [tool.name for tool in app.state.ctx.tools]
    return {
        "status": "ok",
        "mode": settings.mode,
        "approval": settings.require_approval,
        "checkpointer": settings.checkpointer,
        "tools": tools,
    }


@app.get("/tools")
async def tools(tenant: str = Depends(auth)):
    return [
        {"name": tool.name, "description": tool.description}
        for tool in app.state.ctx.tools
    ]


@app.post("/tasks")
async def create_task(req: TaskIn, tenant: str = Depends(auth)):
    thread_id = _thread_id(tenant, req.thread_id)
    max_revisions = settings.max_revisions if req.max_revisions is None else req.max_revisions
    state = await app.state.graph.ainvoke(
        _initial(req.task, max_revisions),
        _config(thread_id),
        context=app.state.ctx,
    )
    return {
        "thread_id": thread_id,
        "paused": bool(state.get("__interrupt__")),
        "interrupts": [
            {"id": item.id, "value": item.value}
            for item in state.get("__interrupt__", [])
        ],
        **_summary(state),
    }


@app.post("/tasks/stream")
async def stream_task(req: TaskIn, tenant: str = Depends(auth)):
    thread_id = _thread_id(tenant, req.thread_id)
    max_revisions = settings.max_revisions if req.max_revisions is None else req.max_revisions
    config = _config(thread_id)

    async def generate():
        try:
            async for chunk in app.state.graph.astream(
                _initial(req.task, max_revisions),
                config,
                context=app.state.ctx,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    keys = list(update.keys()) if isinstance(update, dict) else []
                    yield _sse("update", {"node": node, "keys": keys})
            snapshot = await app.state.graph.aget_state(config)
            yield _sse(
                "done",
                {
                    "thread_id": thread_id,
                    "next": list(snapshot.next),
                    "interrupts": [item.value for item in snapshot.interrupts],
                    **_summary(snapshot.values),
                },
            )
        except Exception as exc:  # 流已开始，只能把错误作为事件发出去
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/tasks/{thread_id}")
async def get_task(thread_id: str, tenant: str = Depends(auth)):
    _require_tenant(tenant, thread_id)
    snapshot = await app.state.graph.aget_state(_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="thread not found")
    return {
        "thread_id": thread_id,
        "next": list(snapshot.next),
        "interrupts": [
            {"id": item.id, "value": item.value} for item in snapshot.interrupts
        ],
        **_summary(snapshot.values),
    }


@app.post("/tasks/{thread_id}/resume")
async def resume_task(thread_id: str, req: ResumeIn, tenant: str = Depends(auth)):
    _require_tenant(tenant, thread_id)
    snapshot = await app.state.graph.aget_state(_config(thread_id))
    if not snapshot.interrupts:
        raise HTTPException(status_code=409, detail="thread is not waiting for approval")
    state = await app.state.graph.ainvoke(
        Command(resume=req.decision),
        _config(thread_id),
        context=app.state.ctx,
    )
    return {
        "thread_id": thread_id,
        "paused": bool(state.get("__interrupt__")),
        **_summary(state),
    }


@app.get("/tasks/{thread_id}/history")
async def task_history(thread_id: str, tenant: str = Depends(auth), limit: int = 20):
    _require_tenant(tenant, thread_id)
    snapshot = await app.state.graph.aget_state(_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="thread not found")
    rows = []
    async for snap in app.state.graph.aget_state_history(_config(thread_id), limit=limit):
        rows.append(
            {
                "step": snap.metadata.get("step"),
                "source": snap.metadata.get("source"),
                "next": list(snap.next),
                "status": snap.values.get("status"),
                "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
            }
        )
    return {"thread_id": thread_id, "rows": rows}


if __name__ =="__main__":
    uvicorn.run("api:app",reload=True)