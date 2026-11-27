from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from langgraph.checkpoint.base import BaseCheckpointSaver
from config import Settings


@asynccontextmanager
async def open_checkpointer(settings:Settings)->AsyncIterator[BaseCheckpointSaver]:
    if settings.checkpointer=="postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.postgres_uri)as saver:
            await saver.setup()
            yield saver
        return

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    settings.db_path.parent.mkdir(parents=True,exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.db_path))as saver:
        yield saver