# Geant4 Agent v3 阶段性复盘

日期：2026-06-06

结论：路线没有走偏。当前推进仍然围绕 v3 first、确认门安全、context 可信、eval 可回归这四件事展开，没有提前滑向 sweep、optimization 或大范围 UI 表面改造。

## 1. 这一阶段真正完成了什么

### v3 主链路更稳了

本阶段修复了一个高优先级问题：当没有待确认行动时，用户输入 `confirm run` 不再创建新的默认运行，也不会自动生成 payload 或执行 runtime。

现在的规则是：

```text
没有 pending action + 用户说 confirm run
-> 解释当前没有可确认的运行
-> 不创建 payload
-> 不运行 Geant4
-> 不污染 runtime 结果
```

这个规则在 deterministic backend 和 in-memory backend 下保持一致。

### 内部字段不再被当成用户确认

`run_confirmed` 这类内部字段现在不会因为文本里出现 `confirm` 而被误判成确认。用户试图直接设置 `run_confirmed=true` 时，agent 会拒绝把它当成可编辑配置。

这条修复直接对应之前担心的“字典化/字段名污染 prompt”问题：内部控制字段不应该变成用户自然语言里的魔法口令。

### eval 从“能跑”推进到“能比较”

新增了 v3 safety invariant eval：

- task 文件：`eval/v3/tasks/behavior_safety.jsonl`
- runner：`tools/evaluate_v3_safety_invariants.py`
- 测试：`tests/test_v3_safety_invariants_eval.py`

它现在覆盖三类主线风险：

- confirmation safety：确认只能确认已有行动。
- prompt pollution：内部字段不能绕过确认门。
- no-run constraint：用户说“设计但不要运行”时，不能被当成取消或运行。

本轮还补了 baseline vs current compare。以后一次修改不仅要说明“当前通过”，还可以比较是否比 baseline 退化。

## 2. 当前验证结果

本轮本地验证结果：

```text
tests/test_v3_safety_invariants_eval.py
-> 5 passed

tools/evaluate_v3_safety_invariants.py --json
-> ok=True
-> 7 tasks passed
-> 12 trials passed
-> backend_invariance_failure_count=0

safety compare CLI
-> ok=True
-> task_regressions=0
-> trial_regressions=0
-> missing_current_tasks=0
-> missing_current_trials=0

v3 related pytest subset
-> 145 passed, 2 subtests passed

v3 dialogue casebank
-> default: 21/21
-> allow-in-memory: 21/21
```

这说明本轮修复没有破坏 v3 多轮对话、确认流程、response quality 相关测试。

## 3. 用简单话说当前架构

现在的 v3 可以理解成一条受控流水线：

```text
用户说一句话
-> v3 service 取出当前会话
-> context pack 整理当前事实
-> reasoner 判断下一步
-> controller 调工具或等待确认
-> tool adapter 只负责 Geant4 相关执行
-> observation 写回 state
-> response composer 把事实组织成用户能看懂的回答
```

这里最重要的边界是：

- LLM 负责理解和建议。
- 代码负责确认、校验、执行和状态更新。
- runtime 事实优先于用户目标描述。
- UI 默认只应该走 v3 API。
- legacy/strict/v2 只能作为兼容或参考，不能重新影响默认产品链路。

## 4. 路线有没有偏

没有明显偏离。证据是：

- 本轮没有新增自动执行能力。
- sweep/optimization 没有被提前做成自动闭环。
- 修改围绕确认门、pending action、prompt pollution 和 eval gate。
- 新增评测不是 smoke test，而是能捕捉具体行为退化的 invariant eval。
- backend invariance 已纳入任务集，避免 in-memory 和真实 runtime 前置决策分叉。

当前最正确的方向仍然是：

```text
先把 v3 的行为边界固定住
再把 metadata 控制信号逐步迁出
再扩大 LLM 智能化和回复体验
最后再做自动 sweep / optimization
```

## 5. 仍然存在的主要风险

### Major: metadata 仍然承担太多控制职责

现在 `run_confirmed`、`allow_in_memory`、`suppress_run`、`accept_defaults`、`config_overrides` 等信号还散落在 metadata 或 request dict 中。

短期可接受，因为兼容成本低；中期必须迁移为更明确的类型：

- `V3TurnDirectives`
- `V3PendingAction`
- `V3ExecutionAuthorization`
- `V3RuntimePolicy`
- `V3StatePatch`

不迁移的后果是：类似 `run_confirmed` 的字段污染问题可能继续以别的名字出现。

### Major: service 仍偏“大脑集中”

`V3AgentTurnService` 现在承担了会话、确认、取消、sweep 检测、controller 选择和结果包装等多种职责。

短期可以继续小步修；中期应拆出：

- pending action manager
- runtime policy resolver
- turn directive interpreter
- state transition engine

这样 agent 会更聪明，也更不容易被局部规则改坏。

### Major: eval 还没完全接入标准 run/grade/compare/calibrate

当前 safety eval 已经有任务集、grader 和 compare，但还不是完整的外部 harness 形态。

下一步应把 trial row 和 grade row 拆得更清楚，让它可以稳定进入：

```text
run -> grade -> compare -> calibrate
```

### Minor: response intelligence 还需要更强评价

目前回复质量测试能防止明显退化，但还不能充分评价“用户体验是否自然、是否解释得清楚、是否少问废话”。

这不应该靠更多关键词规则解决，而应该用：

- response rubric
- grounded evidence check
- 中文/英文对话 casebank
- LLM-as-judge 的受控复核

### Minor: 工业 runtime 闭环仍未完全闭合

已有 industrial benchmark 和 runtime evaluator，但真实 runtime、reviewed golden、官方可评价结果还没完全闭合。

这件事重要，但不应抢在确认门和 state contract 之前。

## 6. 下一阶段主线

2026-06-06 策略调整：

前几轮已经把确认门、pending action、runtime policy 和 public
`run_confirmed` 绕过风险压到可接受水平。接下来主线从“继续做鲁棒性收口”调整为
“功能性优先”。

新的优先级是：

```text
先做用户能感受到的功能闭环
-> 用现有 safety eval 做护栏
-> 遇到会破坏确认/runtime 边界的问题再局部加固
```

也就是说，鲁棒性不再是每轮默认主目标，而是功能开发的准入线和回归线。

### 功能优先队列

下一阶段优先推进这些功能：

1. 结果解释更有用：用户问“为什么这样”“下一步怎么改”时，agent 能基于最新 runtime facts 给出清楚回答。
2. 配置修改更自然：用户说“把能量改成 2 MeV 再跑”“材料换成水”时，agent 能稳定修改当前 payload，并明确是否需要重新确认。
3. UI suggestions 更像真实操作：按钮文本居中、语义明确，点击后带正确 prefill，不再像机械提示。
4. 多轮对话更顺：设计、接受默认、修改、运行、解释结果之间少重复问废话。
5. 功能 eval 更贴近用户路径：不仅测“不会出事”，还测“有没有完成用户想做的事”。

功能开发仍保留三条硬护栏：

- runtime 行动必须经过 preflight 和确认。
- result explanation 必须以 `latest_runtime_facts` 为事实来源。
- public API 不得通过内部字段直接获得 runtime 授权。

### Step 1: 把 safety eval 做成正式回归门

继续完善 `tools/evaluate_v3_safety_invariants.py`：

- 输出更接近标准 trial_result JSONL。
- 支持保存 baseline 和 current。
- compare 输出明确列出退化原因。
- 增加校准样例，确保 grader 真的能失败。

完成标准：一次主链路改动必须能回答“有没有比 baseline 退化”。

### Step 2: 把 pending action 从 metadata 迁到显式类型

先新增类型，不急着删除旧字段。

目标是让确认流程变成：

```text
V3PendingAction 存在
-> 用户确认
-> V3ExecutionAuthorization 生成
-> runtime executor 执行
```

完成标准：任何路径都不能通过普通配置字段直接得到 runtime 授权。

当前推进状态：

- 已新增 `core/agent_v3/pending_action.py`，提供 `V3PendingAction`、
  `V3ExecutionAuthorization` 和 `V3PendingActionManager`。
- `V3AgentTurnService` 的确认/取消路径已开始通过 manager 读写 pending
  action，同时保留 `metadata["pending_action"]` 作为兼容存储。
- 确认执行 runtime 前会记录 `last_execution_authorization`，后续可继续把
  runtime executor 从 `run_confirmed` 迁移到显式授权对象。
- 旧字段尚未删除；下一轮应继续迁移读取方，而不是扩大自动执行能力。

### Step 3: 引入 V3RuntimePolicy

`allow_in_memory` 这类 backend 选择必须只影响“怎么执行”，不能影响“是否该执行”。

完成标准：deterministic、in-memory、local runtime 在运行前决策上保持一致。

当前推进状态：

- 已新增 `core/agent_v3/runtime_policy.py`，提供 `V3RuntimePolicy` 和 runtime
  tool 参数 helper。
- 请求入口会生成 `runtime_policy`，同时保留旧 `allow_in_memory` 和
  `runtime_env` 字段。
- reasoner 生成 preflight/runtime tool 参数时已开始读取 runtime policy。
- pending action 确认路径会同步 runtime policy。
- 下一步应继续把 runtime executor 的授权判断迁向
  `V3ExecutionAuthorization`，减少直接依赖 `run_confirmed`。

### Step 4: 回复智能化进入可测阶段

在不牺牲安全门的前提下，提升用户体验：

- 更清楚地区分事实、假设、建议。
- 减少机械式下一步按钮。
- 针对中文用户给出更自然的说明。
- result explanation 只从 runtime facts 取权威事实。

完成标准：新增 response experience eval，而不是靠人工感觉判断。

## 7. 明确暂缓的事情

暂缓自动 sweep / optimization 闭环。

原因很简单：现在还在收紧“什么时候能运行”。在确认授权和 runtime policy 完全类型化之前，自动优化如果做成闭环，容易重新打开隐式执行风险。

暂缓大规模目录搬迁。

当前应优先迁移职责和接口，等 v3 类型边界稳定后再搬目录。否则目录看起来干净了，行为反而更难追踪。

暂缓只做 UI 表面调整。

UI 很重要，但下一轮 UI 改动应跟 v3 state、suggestions、pending action 的明确接口一起做，避免浏览器端重新承担 intent 判断。

## 8. 阶段性判断

当前路线是健康的：

- 先修真实 bug。
- 再把 bug 固化成 eval。
- 再让 eval 支持 compare。
- 再围绕 eval 推动架构拆分。

下一轮应切到功能性主线。继续保留已有 safety eval 和 confirmation guard，
但默认投入应放到“用户能完成什么”上，而不是继续把每个内部字段都优先类型化。
