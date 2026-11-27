from __future__ import annotations
import shutil
import sys
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

server=Path(__file__).absolute().with_name("mcp_server.py")

def _filesystem_server(root: str) -> dict:
    """已全局安装 mcp-server-filesystem 时直接启动，避免每次联网下载；否则回退到 npx"""
    exe = shutil.which("mcp-server-filesystem")
    if exe:
        return {"command": exe, "args": [root], "transport": "stdio"}
    npx = shutil.which("npx.cmd") or shutil.which("npx") or "npx"
    return {
        "command": npx,
        "args": ["-y", "@modelcontextprotocol/server-filesystem", root],
        "transport": "stdio",
    }

async def load_mcp_tools()->List[BaseTool]:
    client=MultiServerMCPClient(
        {
            "project":{
                "command":sys.executable,
                "args":[str(server)],
                "transport":"stdio"
            },

            "filesystem": _filesystem_server(str(Path(__file__).absolute().parent.parent)),
        }
    )
    tools=await client.get_tools()
    for tool in tools:
        tool.metadata = {**(tool.metadata or {})}
    return tools