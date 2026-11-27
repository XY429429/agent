from __future__ import annotations
import json,sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0,str(Path(__file__).absolute().parent))
#离线缓存
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

load_dotenv(verbose=True)

from mcp.server.fastmcp import FastMCP
docs_dir=Path(__file__).absolute().parent.parent / "data" / "knowledge"
_VS=None

def get_retriever():
    """懒加载"""
    global _VS
    if _VS is None:
        from rag import VectorRetriever
        _VS=VectorRetriever(doc_dir=docs_dir)
    return _VS

mcp=FastMCP(name="agent-tool",log_level="ERROR")

# @mcp.tool()
# async def search_knowledge(query:str,top_k:int=7):
#     """检索本地知识库（仅含 LangGraph 多智能体、MCP、RAG、生产清单等技术文档）。
#     只有问题涉及这些主题时才使用
#     返回带doc_id/score/title/text...的JSON"""
#     try:
#         hits=get_retriever().search(query,top_k)
#     except Exception as exc:
#         return json.dumps({"hits":[],"error":f"knowledge base unavailable: {exc}"},ensure_ascii=False)
#     return json.dumps({"hits":hits},ensure_ascii=False)

# async def _post_tool(name:str,payload:dict)->dict:
#     """调用 DeepSeek 工具接口；失败时返回含状态码和响应片段的 {"error": ...}，不抛异常"""
#     import aiohttp
#     key=os.environ.get("APIKEY")
#     if not key:
#         return {"error":"APIKEY is not set"}
#     try:
#         async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))as session:
#             async with session.post(
#                 f"https://api.deepseek.com/v1/tools/{name}",
#                 headers={"Authorization":f"Bearer {key}"},
#                 json=payload,
#             )as response:
#                 text=await response.text()
#                 if response.status>=400:
#                     return {"error":f"HTTP {response.status}: {text[:300]}"}
#                 return json.loads(text)
#     except Exception as exc:
#         return {"error":f"{type(exc).__name__}: {exc}"}

@mcp.tool()
async def search_web(query:str):
    """按 query 搜索网页"""
    import aiohttp
    key=os.environ.get("APIKEY")
    result:dict={}
    if not key:
        result={"error":"API KEY is None"}
    else:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url="https://api.deepseek.com/v1/tools/search",
                        headers={"Authorization": f"Bearer {key}"},
                        json={"query": query, "count": 3, "need_content": True}
                ) as response:
                    text = await response.text()
                    if response.status >= 400:
                        result = {"error": f"HTTP {response.status}: {text[:300]}"}
                    else:
                        result = json.loads(text)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}

    if "error" in result:
        return f"search failed: {result['error']}"
    items=result.get("results") or []
    if not items:
        return "no results"
    return "\n\n".join(
        f"网页{i}：{it.get('title') or it.get('site_name','')}\n{it.get('url','')}\n{it.get('summary','')}"
        for i,it in enumerate(items,1)
    )

@mcp.tool()
async def fetch_web(url:str):
    """抓取单个网页"""
    import aiohttp
    key=os.getenv("APIKEY")
    if not key:
        return {"error":"API KEY is None"}
    else:
        try:
            async with aiohttp.ClientSession()as session:
                async with session.post(
                    url="https://api.deepseek.com/v1/tools/fetch",
                    headers={"Authorization":f"Bearer {key}"},
                    json={"url":url}
                )as response:
                    text=await response.text()
                    if response.status >= 400:
                        return {"error": f"HTTP {response.status}: {text[:300]}"}
                    return json.loads(text)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

if __name__=="__main__":
    mcp.run(transport="stdio")
