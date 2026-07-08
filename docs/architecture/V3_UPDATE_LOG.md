# Geant4 Agent v3 Update Log

## 2026-06-11 Round 11 - Typed Runtime Result Facts

### What Changed

- Reworked `core/agent_v3/result_recommendations.py` around
  `RuntimeResultFacts`, a small typed fact object.
- Existing result-driven suggestions are preserved:
  - add downstream scoring when transmission metrics are missing,
  - reduce target thickness when downstream counts are all zero,
  - compare source energy when downstream counts are all zero.
- Recommendation builders now read semantic properties such as
  `has_core_source`, `downstream_metrics_missing`, and
  `downstream_counts_all_zero` instead of repeatedly indexing a dict.

### Why

This is a direct response to the non-dictionary mainline. The previous behavior
was already fact-based, but the implementation still encouraged more
`facts.get(...)` branches. That would make future material/scoring advice drift
toward dictionary piles.

The new structure keeps the public facts JSON-serializable while moving internal
reasoning to a typed object.

### Verification

```text
tests/test_agent_v3_result_recommendations.py
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_service.py
tests/test_v3_agent_intelligence_eval.py
-> 64 passed, 2 subtests passed

tools/evaluate_v3_agent_intelligence.py --json
-> ok=True passed=4 failed=0
```

### Code Review: Necessity And Reasonableness

Necessity: medium-high. It does not add a new user-visible button, but it keeps
the next user-visible additions from being built as ad hoc dict checks.

Reasonableness: good. The change is scoped to the recommendation module and
keeps the same tests passing. It does not alter runtime execution, patch
validation, or confirmation behavior.

Boundary control: strong. Facts still enter through `V3ContextPack` or runtime
observations, but recommendation logic works through a narrow typed surface.

## 2026-06-11 Round 10 - Downstream Scoring Advice

### What Changed

- Added a high-level patch field: `enable_downstream_scoring`.
- The payload builder can now turn that high-level intent into a downstream
  silicon detector plus detector/plane crossing scoring.
- Result recommendations now suggest `add downstream detector and plane scoring
  and run again` when runtime facts have target energy deposition but no
  downstream crossing metrics.
- The suggestion remains a next-turn prefill and still goes through payload
  rebuild, preflight, pending action, and confirmation.
- The intelligence harness now includes the downstream-scoring rerun path.

### Why

For a client user, a result that only reports target deposition can be hard to
act on. If the user cares about shielding or transmission, the agent should help
add the missing measurement instead of only explaining the incomplete result.

This stays on the mainline: make the agent more useful after each result while
keeping execution gated.

### Verification

```text
tests/test_agent_v3_result_recommendations.py
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_patches.py
tests/test_agent_v3_service.py
tests/test_v3_agent_intelligence_eval.py
-> 72 passed, 2 subtests passed

tools/evaluate_v3_agent_intelligence.py --json
-> ok=True passed=4 failed=0

tests/test_agent_v3_geant4_tools.py
tests/test_agent_v3_turn_understanding.py
tests/test_agent_v3_llm_reasoner.py
-> 33 passed
```

`git diff --check` reported no whitespace errors, only existing CRLF warnings.

### Code Review: Necessity And Reasonableness

Necessity: high. This closes a real product gap: the agent can now recommend and
apply a missing measurement path, not just change numeric parameters.

Reasonableness: good. `enable_downstream_scoring` is deliberately a high-level
intent, not a raw nested scoring dictionary exposed to users. That keeps the
public agent interaction cleaner and limits dictionary-style leakage.

Boundary control: acceptable. The text fallback only recognizes a narrow system
prefill style for this action. LLM paths can also emit the same high-level field
through `requested_changes`, and patch validation still rejects unsupported
internal fields.

## 2026-06-11 Round 9 - Result-Driven Thickness Advice

### What Changed

- `latest_payload` and `latest_runtime_facts` now expose `target_thickness_mm`
  as a small safe summary instead of passing raw geometry dictionaries into
  prompts.
- Runtime result recommendations now suggest a thinner target when structured
  facts show gamma/photon input, a known target thickness, and zero observed
  downstream crossings.
- Thickness recommendations carry `kind`, `rationale`, and `fact_basis`, and
  still only produce a next-turn prefill. They do not run Geant4 automatically.
- The dialogue composer keeps the result-driven thickness suggestion visible as
  the fourth runtime button when it is applicable.
- The intelligence harness now includes a thickness-rerun case: accepting the
  suggested prefill becomes a structured `target_thickness_mm` patch, clears
  stale runtime state, and returns to confirmation.

### Why

The mainline is now functional agent intelligence, not only robustness. After a
runtime result, users need concrete next actions: change thickness, rerun, then
compare. This round adds one such action while keeping the anti-dictionary rule:
the recommendation reads runtime facts, not loose user text.

### Verification

```text
tests/test_agent_v3_result_recommendations.py
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_context.py
-> 29 passed

tools/evaluate_v3_agent_intelligence.py --json
-> ok=True passed=3 failed=0
```

The eval trace now shows default runtime context carrying
`target_thickness_mm: 10.0`, and the new thickness prefill is converted into a
structured metadata-sourced patch.

### Code Review: Necessity And Reasonableness

Necessity: high. This directly improves what the user can do after a result,
which is the current product mainline.

Reasonableness: good. The change is narrow: one facts extractor, one
recommendation branch, and tests. It does not change runtime execution,
confirmation, or the LLM prompt contract.

Boundary control: acceptable. It introduces a small physical heuristic
("zero downstream counts plus known thickness means try thinner target") rather
than a growing keyword dictionary. The next step should add more result-driven
cases only when they can be expressed from structured facts and verified in the
harness.

这份日志用尽量通俗的话记录每轮主线推进。它不是提交记录，也不是完整技术设计；它回答四个问题：这轮做了什么，为什么做，怎么验证，下一步是什么。

## 2026-06-11 Round 8 - Result-Driven Suggestions 与阶段结论

### 做了什么

- 新增 `core/agent_v3/result_recommendations.py`。
- 运行结果后的建议开始读取结构化 runtime facts，而不是读用户文本关键词。
- 当 facts 表明 `gamma + G4_Pb + 下游计数为 0` 时，agent 会建议能量扫描，例如 `run sweep 0.5 1 2 MeV`。
- suggestion 会保留 `kind`、`rationale` 和 `fact_basis`，方便 harness 或人工审查知道建议从哪里来。
- 新增 result recommendation 单测和 dialogue composer 单测。
- 新增 `docs/architecture/V3_STAGE_CONCLUSION_2026-06-11.md`，总结当前阶段已经完成的功能、验证结果、剩余风险和下一阶段路线。

### 为什么做

上一轮已经建立了 intelligence harness，但 agent 的建议仍然偏“固定下一步”。这轮把建议往智能化推进一点：让它从真实 runtime facts 出发，提出一个可解释、可审查、不会自动执行的下一步。

这不是自动 optimization，也不是关键词字典。它只读取结构化事实：

```text
material
particle
source_energy_mev
detector_crossing_count
plane_crossing_count
```

### 怎么验证

本轮验证结果：

```text
tests/test_agent_v3_result_recommendations.py
tests/test_agent_v3_dialogue_composer.py
tests/test_v3_agent_intelligence_eval.py
-> 25 passed

tools/evaluate_v3_agent_intelligence.py
-> ok=True passed=2 failed=0
```

### 代码审查：必要性和合理性

必要性：中到高。现在功能主线已经进入“agent 帮用户做下一步判断”的阶段，建议不能一直停留在固定按钮。

合理性：合理。新模块只读 runtime facts，不改变执行链路；suggestion 只是下一轮 prefill，不会自动运行 sweep，也不会绕过 confirmation。

边界控制：合理。规则目前很窄，只覆盖明确的 gamma 屏蔽场景。没有把材料、能量、用户目标做成大字典。

风险：中等。建议策略还很早，覆盖面有限。后续每扩一个建议类型，都应该补 intelligence harness case。

下一步：扩展厚度、材料、能量、计分方式的 result-driven advice，并把每类建议接入 harness。

## 2026-06-11 Round 7 - Intelligence Harness 与字典化控制

### 做了什么

- 新增 `tools/evaluate_v3_agent_intelligence.py`。
- 新增 `eval/v3/tasks/agent_intelligence.jsonl`，第一批包含两类 case：
  - LLM/结构化 turn understanding 驱动参数修改和重跑；
  - 系统生成的 recommendation prefill 进入 structured metadata/patch 链路。
- 新增 `tests/test_v3_agent_intelligence_eval.py`，验证 harness 本身能跑、能保存报告、能抓到不该出现的 fallback。
- 架构文档补充了新的 eval harness 位置。
- 开发手册补充了“字典化控制原则”：LLM/结构化理解优先，系统 prefill/metadata 次之，deterministic fallback 只能小范围兜底，并且必须可观察、可评测。

### 为什么做

单靠 safety eval 只能证明“没有误运行”，不能证明 agent 真的在智能理解用户。我们需要一个更贴近主线目标的 harness，专门观察每轮到底是：

```text
LLM/结构化理解在驱动行动
还是
关键词/regex fallback 在偷偷接管行动
```

这轮的重点不是禁止所有确定性代码，而是把它关进笼子：允许它兜底，但要有来源标记、有 eval 指标、有明确边界。

### 怎么验证

本轮验证结果：

```text
tools/evaluate_v3_agent_intelligence.py
-> ok=True passed=2 failed=0

tests/test_v3_agent_intelligence_eval.py
-> 5 passed

tools/evaluate_v3_safety_invariants.py
-> ok=True passed=8 failed=0

tools/evaluate_v3_dialogue_casebank.py
-> ok=True passed=21 failed=0
```

### 代码审查：必要性和合理性

必要性：高。随着功能性推进，最容易滑坡的地方就是“为了让按钮或 case 通过，继续加关键词”。没有 harness 的话，这种退化很难被发现。

合理性：合理。新 harness 没改变产品运行链路，只读取 response 里的 state/context/dialogue 信息做评估。它还支持 mock LLM turn understanding，所以离线也能测试智能路径，不依赖真实 API。

边界控制：合理。第一批 case 没有把 deterministic fallback 直接判死刑，而是区分了“LLM 该负责的 turn”和“系统生成 prefill 可走 metadata 的 turn”。这比一刀切禁用 fallback 更符合当前工程阶段。

风险：中等。harness 还很年轻，invariant 类型有限；后续需要继续补 result-driven recommendation、material/thickness/energy 修改等 case。

下一步：把“结果解释驱动参数建议”接入这个 intelligence harness，要求 agent 给出的建议能说明依据，并能进入结构化 patch/confirmation 链路。

## 2026-06-11 Round 6 - 推荐按钮真正推动“增加事件数后重跑”

### 做了什么

- 把运行完成后的“修改参数再运行”推荐，收口成更具体的“增加事件数再运行”。
- 推荐 prefill 现在会带明确事件数，例如上次跑 2 个 events，就生成 `change event count to 20 events and run again`。
- 如果当前上下文没有可读取的事件数，默认推荐 1000 events，避免按钮发出空泛指令。
- 新增端到端测试：真实运行一次后，读取 UI 推荐 prefill，再把它作为下一轮用户输入，确认系统会重建 payload，并回到等待确认运行。

### 为什么做

上一轮修好了按钮样式和 prefill 元数据，但还有一个功能性问题：按钮看起来能“重跑”，实际上如果 prefill 不包含具体参数，agent 可能只把它当作普通 run/result 请求。

这一轮把按钮变成真正能推动流程的下一步：

```text
已有运行结果
-> 推荐增加事件数
-> 用户点击按钮
-> 生成 run_events override
-> 清掉旧 payload/preflight/runtime
-> 生成新 payload
-> 等待用户确认运行
```

### 怎么验证

本轮验证结果：

```text
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_service.py
-> 50 passed, 2 subtests passed

tools/evaluate_v3_dialogue_casebank.py
-> ok=True passed=21 failed=0

tools/evaluate_v3_safety_invariants.py
-> ok=True passed=8 failed=0

tests/test_v3_frontend_route.py
tests/test_agent_v3_response_quality.py
tests/test_v3_multi_turn_casebank.py
tests/test_agent_v3_turn_understanding.py
tests/test_agent_v3_patches.py
-> 33 passed

git diff --check
-> 无实际错误，只有 CRLF 换行提示
```

### 代码审查：必要性和合理性

必要性：高。用户点击“增加事件数再运行”时，系统必须真的知道要改成多少事件，否则按钮只是装饰，无法形成闭环。

合理性：合理。改动没有新建复杂意图字典，也没有改变确认机制；只是让已有对话建议从当前 runtime facts 里取事件数，并生成更明确的下一轮用户文本。

边界控制：合理。按钮点击后不会自动运行 Geant4，只会触发新 payload 和 pending confirmation。安全链路仍是 payload -> preflight -> confirmation -> runtime。

风险：低。当前默认是 10 倍事件数，比较保守；后续如果要更智能，可以根据结果不确定性、粒子穿越数或目标计分波动来推荐事件数。

下一步：继续做“结果解释驱动参数建议”，让 agent 不只会建议增加事件数，还能根据结果判断应该改厚度、材料、能量还是计分方式。

## 2026-06-07 Round 5 - UI 下一步按钮和运行后建议体验

### 做了什么

- 前端推荐按钮现在使用明确的 `suggestion-btn` 类，不再只套通用 ghost button 样式。
- 推荐按钮会保存 `data-prefill`，并把 prefill 放进 `title` 和 `aria-label`，方便用户悬停查看，也方便后续测试。
- CSS 让 welcome 快捷按钮、推荐按钮、pending confirmation 按钮都用 inline-flex 居中，长文字可以自然换行，不再挤偏。
- 运行完成后和结果解释后的推荐 prefill 变成更像真实下一步指令，例如“解释最新 Geant4 运行结果”“增加事件数并重新运行”，而不是简单把按钮标题原样发送。
- 新增前端静态测试和 dialogue composer 测试，锁住按钮类名、prefill 元数据、居中样式和运行后建议文案。

### 为什么做

这一轮按“功能性优先”的新主线推进。用户真正会感受到的是：agent 给出的下一步按钮能不能直接推动任务继续，而不是只像几个标签。

这次没有做新的复杂意图表，也没有把交互变成字典匹配。改动集中在两个自然位置：

```text
后端：根据当前对话阶段给出更可执行的 prefill
前端：把这些 prefill 稳定、居中、可点击地呈现出来
```

### 怎么验证

本轮验证结果：

```text
tests/test_v3_frontend_route.py
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_response_quality.py
-> 26 passed

tests/test_geant4_web_api.py
tests/test_v3_frontend_route.py
tests/test_agent_v3_dialogue_composer.py
tests/test_agent_v3_response_quality.py
tests/test_agent_v3_service.py
-> 80 passed, 2 subtests passed

tools/evaluate_v3_dialogue_casebank.py
-> ok=True passed=21 failed=0

tools/evaluate_v3_safety_invariants.py
-> ok=True passed=8 failed=0
```

### 代码审查：必要性和合理性

必要性：高。之前推荐按钮样式和实际 class 没对齐，导致用户看到的下一步按钮不够稳定；同时很多建议 prefill 只是标题复读，对多轮体验帮助有限。

合理性：合理。前端改动只影响 suggestion/pending/quick action 按钮的展示，不改变 v3 API 路径，也不改变 runtime 确认机制。后端改动只补充运行后建议的 prefill，不引入新的自动执行能力。

边界控制：合理。点击“修改并重新运行”“增加事件数并重新运行”仍只是发送下一轮用户文本，后续仍要经过 payload/preflight/confirmation，不能绕过安全链路。

风险：低到中。CSS 变化需要后续在真实浏览器里看一眼不同窗口宽度下按钮是否足够美观；功能链路已经由前端静态测试、dialogue casebank 和 safety invariant 兜住。

下一步：继续围绕“功能性”推进结果解释和配置修改，让用户从运行结果自然走到“改参数 -> 再确认 -> 再运行”的闭环。

## 2026-06-06 Round 4 - 主线策略调整为功能性优先

### 做了什么

- 把阶段复盘里的下一阶段主线，从“继续 typed control signals 收口”调整为“功能性优先”。
- 在 `ARCHITECTURE.md` 里写明当前产品策略：优先推进用户能感受到的功能闭环，安全和状态 invariant 作为护栏。
- 保留既有 confirmation、runtime、prompt pollution safety eval，不撤掉安全底线。

### 为什么做

前几轮已经处理了最危险的几个问题：

```text
无 pending action 时 confirm run 不会启动运行
public run_confirmed=true 不能绕过确认
allow_in_memory 不再改变是否允许运行
pending action 和 runtime policy 已经有结构化兼容层
```

继续无限期做鲁棒性收口，会让项目变成“内部越来越规整，但用户没有明显新能力”。现在更应该把精力转向：

- 让用户更容易改配置。
- 让结果解释更有帮助。
- 让 UI suggestion 能推动真实下一步。
- 让多轮对话少重复、少打断。

### 新主线

```text
功能性优先
-> 用现有 safety eval 做护栏
-> 功能开发中遇到真实边界问题再局部加固
```

下一阶段优先级：

1. 结果解释功能：围绕最新 runtime facts 回答“为什么”和“下一步怎么改”。
2. 配置修改功能：自然语言修改材料、能量、事件数、厚度，并能继续运行。
3. UI suggestion 功能：按钮文案、居中、prefill、下一步动作一致。
4. 多轮体验：设计 -> payload -> 修改 -> 确认 -> 运行 -> 解释结果，要像一个连续工作流。
5. 功能 eval：新增 task 不只判断安全，还判断用户目标是否完成。

### 怎么验证

本轮是策略和文档调整，没有改运行代码。验证方式：

```text
文档检查：
- V3_PHASE_REVIEW_2026-06-06.md 已写入策略调整。
- ARCHITECTURE.md 已写入 functionality-first 当前策略。
- V3_UPDATE_LOG.md 已记录本轮调整和审查。
```

下一轮功能代码推进时，仍需跑：

```text
v3 safety invariant
v3 dialogue casebank
相关功能单测
必要时加功能 eval case
```

### 策略审查：必要性和合理性

必要性：高。项目已经从“会不会误运行”的高风险阶段，进入“用户能不能顺畅完成任务”的产品阶段。继续把每轮都压在鲁棒性上，会拖慢功能闭环。

合理性：合理。不是放弃鲁棒性，而是把鲁棒性从主目标降为护栏。已有 safety eval、confirmation guard、runtime policy、pending action manager 足够支撑下一阶段功能开发。

风险：中等。功能优先容易重新引入隐式执行或状态污染。因此每个功能 PR 仍要回答三个问题：有没有绕过确认，结果解释有没有使用最新 runtime facts，UI 是否仍走 v3 API。

下一步：优先做“结果解释 + 配置修改 + UI suggestion”这一组用户可见能力，而不是继续深挖内部字段迁移。

## 2026-06-06 Round 3 - Runtime 执行授权收口

### 做了什么

- `reasoners.py` 的 runtime proposal 不再只看 `run_confirmed`，而是先看 `execution_authorization`。
- `pending_action.py` 新增了读取授权、判断授权来源的 helper。
- `build_turn_input()` 不再接受外部请求里的 `run_confirmed=true` 作为确认信号。
- 新增测试，确认 public API 传入 `run_confirmed=true` 也只能进入 pending confirmation，不能直接运行。
- safety eval 增加 `confirmation.public_run_confirmed_not_authorization.en`，把这个风险固化成回归用例。

### 为什么做

上一轮已经有了 `V3ExecutionAuthorization`，但 reasoner 仍然主要依赖 `run_confirmed`。这会留下一个隐患：如果某条外部路径把 `run_confirmed` 塞进 turn metadata，就可能把“普通字段”误当成“用户确认”。

这轮把边界往前推进了一步：

```text
外部请求不能直接确认 runtime
pending action 确认后才能生成 execution authorization
reasoner 优先读取 execution authorization
run_confirmed 只作为内部兼容兜底
```

### 怎么验证

本轮验证结果：

```text
tests/test_agent_v3_pending_action.py
tests/test_agent_v3_runtime_policy.py
tests/test_agent_v3_service.py
-> 42 passed, 2 subtests passed

tools/evaluate_v3_safety_invariants.py
-> ok=True passed=8 failed=0

tests/test_v3_safety_invariants_eval.py
-> 5 passed

tools/evaluate_v3_dialogue_casebank.py
-> default: ok=True passed=21 failed=0
-> allow-in-memory: ok=True passed=21 failed=0

v3 pytest subset
-> 160 passed, 2 subtests passed
```

### 代码审查：必要性和合理性

必要性：高。`run_confirmed` 是之前多次提到的 prompt/字段污染风险点；如果 public turn 输入能直接设置它，就和“确认只能确认已有 pending action”的主线冲突。

合理性：合理。改动没有删除旧兼容字段，而是让新授权对象优先，旧 `run_confirmed` 只在内部兼容场景下兜底。这样不会大面积震动 controller、tool registry 或老测试。

边界控制：合理。没有扩大自动运行能力，也没有改变 runtime tool 的执行逻辑；只是收紧了“谁能声明已确认”。

遗留风险：仍有少量 lower-level 测试和内部工具会直接构造 `V3TurnInput(metadata={"run_confirmed": True})`。这可以暂时保留，因为它们绕过 public service，用于底层兼容测试。后续应继续把这些测试迁到 `execution_authorization`。

### 下一步

继续减少 lower-level reasoner 和工具测试对 `run_confirmed` 的直接依赖，逐步把 runtime confirmed 状态改成只看 `V3ExecutionAuthorization`。

## 2026-06-06 Round 2 - Runtime Policy 结构化

### 做了什么

- 新增 `core/agent_v3/runtime_policy.py`。
- 引入 `V3RuntimePolicy`，把 `allow_in_memory` 和 `runtime_env` 这类 runtime backend 选择收进一个明确结构里。
- `build_turn_input()` 现在会根据请求生成 `runtime_policy`，同时继续保留旧的 `allow_in_memory` 和 `runtime_env` 字段。
- `BasicGeant4Reasoner` 生成 preflight/runtime tool 参数时，开始从 runtime policy helper 读取参数。
- pending action 确认路径会同步 runtime policy，避免确认后只有旧字段变化。
- 新增 `tests/test_agent_v3_runtime_policy.py`。

### 为什么做

之前 `allow_in_memory` 是一个裸开关，散在 request metadata、pending action tool 参数和 reasoner 里。这样容易让 backend 选择看起来像“是否允许运行”的决策信号。

这轮把它收进 `V3RuntimePolicy`，意思更清楚：

```text
runtime policy 只决定怎么执行
pending action / execution authorization 决定能不能执行
```

这正好对应当前主线：先把运行授权和运行方式拆开，再继续提升 agent 的智能化。

### 怎么验证

本轮验证结果：

```text
tests/test_agent_v3_runtime_policy.py
tests/test_agent_v3_pending_action.py
tests/test_agent_v3_service.py
-> 37 passed, 2 subtests passed

tools/evaluate_v3_safety_invariants.py
-> ok=True passed=7 failed=0

tools/evaluate_v3_dialogue_casebank.py
-> default: ok=True passed=21 failed=0
-> allow-in-memory: ok=True passed=21 failed=0

v3 pytest subset
-> 155 passed, 2 subtests passed
```

### 下一步

继续保持小步迁移：

- 让 runtime executor 更明确地检查 `V3ExecutionAuthorization`。
- 继续减少直接读取 `run_confirmed` 的地方。
- 不急着删除旧 metadata 字段，等所有读取方迁完再删。

## 2026-06-06 Round 1 - Pending Action 类型化

### 做了什么

- 新增 `core/agent_v3/pending_action.py`。
- 引入 `V3PendingAction`、`V3ExecutionAuthorization` 和 `V3PendingActionManager`。
- `V3AgentTurnService` 的确认/取消路径开始通过 manager 读写 pending action。
- 旧的 `metadata["pending_action"]` 继续保留，作为兼容存储。
- 确认 runtime 前会写入 `last_execution_authorization`。
- 新增 `tests/test_agent_v3_pending_action.py`。

### 为什么做

之前 pending action 是一个普通 dict，service 里到处直接读写。这样短期方便，但长期会让“用户确认了什么、确认的是不是当前行动、确认后能不能运行”变得难追踪。

这轮的目标是让确认链路更像这样：

```text
有一个明确的待确认行动
-> 用户确认这个行动
-> 系统生成一次明确授权
-> runtime 执行这次授权
```

### 怎么验证

本轮验证结果：

```text
tests/test_agent_v3_pending_action.py
tests/test_agent_v3_service.py
-> 33 passed, 2 subtests passed

tools/evaluate_v3_safety_invariants.py
-> ok=True passed=7 failed=0

v3 dialogue casebank
-> default: 21/21
-> allow-in-memory: 21/21

v3 pytest subset
-> 151 passed, 2 subtests passed
```

### 下一步

继续推进 `V3RuntimePolicy`，把 `allow_in_memory` 从普通 metadata 开关收进明确结构里。

## 2026-07-08 - v3 trial adapter vertical slice

这一轮把 v3 主链路接入了独立的 command-only trial adapter。它接收一个任务，
调用真实 `V3AgentTurnService`，然后只输出安全轨迹；任务里的预期答案和判分规则
不会传给 agent，也不会影响 adapter 是否完成。

新增内容：

- `eval/v3/adapters/v3_turn_adapter.py`：typed task/turn/options 输入和单 JSON 输出。
- `eval/v3/README.md`：adapter 协议、运行方式和权限边界。
- `tests/test_v3_turn_trial_adapter.py`：多轮确认、权限不可升级、CLI 单输出和 LLM
  配置要求测试。
- 修复 “Design ..., do not run” 被轨迹误记成取消操作的问题。现在只有存在待确认
  动作，或用户单独发出取消命令时，才记录 cancellation fallback。

真实 DeepSeek trial 已完成：设计工具成功，未生成 payload，未执行 runtime，配置与
密钥未进入输出。turn understanding 返回 `llm_uncertain`，明确记录为模型置信度低于
阈值，而不是伪装成确定理解。下一步是在 adapter 外增加 grader 和 baseline compare。

验证结果：`960 passed, 3 skipped, 103 subtests passed`。代码审查认为改动必要且
边界合理：adapter 不判分、不读取预期答案、不改变确认权限；service 修复只纠正轨迹
标签，不放宽执行条件。当前主要限制是 live turn understanding 仍可能因低置信度进入
保守 fallback，这应由后续 grader 按 slice 统计，而不是通过关键词补丁隐藏。

## 2026-07-08 - independent grader and baseline compare

这一轮在 trial adapter 外增加了 deterministic behavior grader、JSONL suite runner
和 baseline compare。grader 覆盖现有任务使用的 14 类结构化检查，包括最终状态、
工具证据、理解来源、参数修改和建议按钮；它不调用 agent 或 LLM。

首轮结果：behavior safety `8/8 tasks`、`13/13 trials` 通过，backend invariance
失败数为 0。相同报告经过 baseline compare 后得到 13 条可比较轨迹、0 regression、
0 missing、0 candidate failure。

代码审查重点：判分规则来自任务 invariant，不根据用户措辞猜测；adapter、grader、
compare 三层职责分开。修复了 observations 在进程内是 tuple、经过 JSON 后是 list 的
表示差异，现在两种调用方式使用同一种 JSON-native 结果。

## 2026-07-08 - merge-readiness review fixes

本轮在 `main...beta` 审查中修复了四个合并阻断问题：显式确认事件必须携带匹配的
`action_id`；空 session reset 不再清空全部会话；不安全 session ID 使用带哈希的
无碰撞文件名并校验 envelope 身份；LLM turn-understanding 不能授予 runtime 权限。
等待确认时的只读追问会保留 pending action。

同时删除未使用的 v3 intent classifier、未接线的自动 optimization/sweep 执行死代码，
以及基于 sweep 关键词的 service 分支。comprehension prompt 不再内嵌场景默认字典。

新增 grader calibration 和无 mock 的 live intelligence task。当前结果：calibration
`4/4`，live intelligence `1/1`，behavior safety `9/9 tasks`、`14/14 trials`。
标准 `pytest -q` 通过 `pytest.ini` 只收集正式 tests，不再误启动旧 UI 脚本。
