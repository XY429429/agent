# MCP 工具集成

## MCP 是什么

MCP（Model Context Protocol）是一套让模型应用连接外部工具、资源和提示的协议。服务端暴露 tools、resources、prompts；客户端负责发现并调用它们。传输方式包括 stdio、SSE 和 streamable HTTP。本地开发最常用 stdio：客户端启动一个子进程，通过标准输入输出交换 JSON-RPC 消息。

## 把 MCP 工具接入 LangGraph

Python 侧可以用 langchain-mcp-adapters 的 MultiServerMCPClient。配置里写命令、参数和 transport，调用 get_tools() 后得到一组 LangChain BaseTool，可以直接交给 ToolNode，也可以手动 invoke。工具名默认就是 MCP 服务端的工具名，建议加前缀避免多个服务重名。

## stdio 的三个坑

第一，服务端不要把日志写到 stdout，否则会破坏 JSON-RPC 帧；日志应写 stderr。第二，客户端要使用当前 Python 解释器启动服务端，避免 PATH 里装错环境。第三，stdio 子进程的生命周期由客户端管理，进程退出后工具调用会失败，需要在应用启动时加载工具并做健康检查。

## 工具安全

MCP 工具是不可信边界：服务端可能返回恶意内容，模型也可能编造工具名。生产上要做工具白名单、参数校验、超时、结果脱敏和审计。破坏性工具（退款、删除、发邮件）必须走人工审批，不能只靠提示词约束。

## MCP 与 RAG

RAG 可以作为 MCP 工具暴露：search_knowledge(query, top_k) 返回带 doc_id、score、text 的 JSON。员工 agent 调用它检索资料，把引用写进结果；汇总 agent 负责去重引用；主管验收时检查引用是否覆盖关键结论。