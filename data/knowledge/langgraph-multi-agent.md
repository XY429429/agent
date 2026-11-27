# LangGraph 多智能体设计

## Supervisor 模式

Supervisor 模式由一个中心节点负责规划、分派和验收。主管 agent 把用户任务拆成多个子任务，通过 Send 动态扇出给员工 agent；员工完成后把结果写回共享状态，由汇总节点合并，再回到主管校验。这个模式适合任务可以分解、角色边界清晰的场景。

Supervisor 的关键不是“多几个 agent”，而是把控制流显式化：谁决定下一跳、谁拥有写权限、失败后回到哪里。中心节点只做路由和校验，不承担具体检索或写作，避免它成为上下文瓶颈。

## Handoff 与 Swarm

Handoff 允许当前 agent 用 Command(goto=...) 把控制权直接交给另一个 agent，适合客服转人工、售前转退款等场景。跨子图交接使用 Command(graph=Command.PARENT)，把控制权交回父图。

Swarm 是平权交接：每个 agent 都能把任务交给别人。它灵活但难调试，必须加最大交接次数、owner 字段和上下文裁剪，否则两个 agent 容易互相踢皮球。

## 状态与 reducer

多 agent 共享状态时，每个字段都要明确“覆盖还是合并”。子任务结果通常用 Annotated[list[WorkerResult], merge_by_id] 按 subtask_id 去重；引用列表用 merge_unique 去重；日志用 operator.add 追加。并发写同一个 LastValue 字段会触发 InvalidUpdateError。

## Checkpoint 与恢复

每个超步结束都会写 checkpoint。主管在验收前暂停时，interrupt() 会把待恢复任务写进 checkpoint；恢复时节点从头重跑，因此中断点之前的副作用必须幂等。生产环境用 PostgresSaver，单机演示可用 SqliteSaver。

## 验收与修订

主管校验不能只看“有没有输出”，还要看覆盖率、引用、格式和验收标准。校验失败时，主管应生成修订子任务，而不是让原 agent 盲目重跑。设置 max_revisions 上限，超过后转人工，避免无限循环。