"""运行时上下文"""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any

from langchain_core.tools import BaseTool

@dataclass
class AppContext:
    tools:list[BaseTool]
    mode:str="single"
    model:Any=None
    extra:dict[str,Any]=field(default_factory=dict)
