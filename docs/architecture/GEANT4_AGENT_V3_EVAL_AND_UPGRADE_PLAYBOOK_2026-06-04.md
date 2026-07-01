# Geant4 Agent v3 Eval And Upgrade Playbook

日期：2026-06-04

状态：后续 v3 开发主准则

适用范围：`core/agent_v3/`、`ui/web/` 的 v3 产品路径、v3 eval 工具、
`mcp/geant4/` runtime 边界，以及与它们直接相关的测试和文档。

## 1. 这份手册解决什么问题

v3 已经能够完成一条真实的多轮链路：

```text
理解用户目标
-> 形成设计
-> 生成运行配置
-> 运行前检查
-> 等待用户确认
-> 运行 Geant4
-> 保存运行事实
-> 回答结果问题
-> 接受后续修改
```

当前的主要问题不是“功能不存在”，而是系统还难以证明以下事情：

- 相同用户意图在不同 runtime 后端下会得到相同的行动决策。
- 确认、取消、修改和运行不会因为隐藏开关组合而改变语义。
- Agent 的回答确实使用最新 context 和 runtime 事实。
- LLM 提升了理解与交互，而不是被大量关键词规则替代。
- 新改动比当前版本更好，而不是只让现有测试继续通过。

这份手册定义后续开发的共同做法。它不是一次性的重构清单，而是每次修改
v3 时都应遵守的开发、评测和验收规则。

## 2. 当前基线

2026-06-04 的本地审查基线：

| 项目 | 当前结果 | 说明 |
| --- | --- | --- |
| v3 自动测试 | 161 passed | 基础行为覆盖较多 |
| v3 dialogue casebank，默认模式 | 21/21 passed | 默认模式没有暴露后端差异 |
| v3 dialogue casebank，in-memory 模式 | 20/21 passed | 暴露确认语义问题 |
| dialogue quality | 41 个 turn 均为 1.0 | 当前评分区分能力不足 |
| industrial benchmark shape | 22/22 合法 | 场景结构可用 |
| industrial compile | 15 compiled，1 compiled with gaps，6 unsupported | 能力边界较清楚 |
| industrial official result | 20 not evaluable，2 expected unsupported | 真实 runtime 与 reviewed golden 尚不完整 |

已确认的最高优先级问题：

```text
当没有 pending action 时，用户输入 "confirm run" 不应创建新的运行请求。
当前行为在 allow_in_memory=True 时可能创建默认 payload 和待确认行动。
```

这说明 runtime backend 选择已经影响到运行前的 Agent 决策。该问题必须先于
新增 sweep、optimization 或更复杂自主能力修复。

## 3. 不可妥协的工程准则

### 3.1 v3 first

- 新产品能力只进入 `/api/v3/agent/*` 和 `core/agent_v3/` 主链路。
- legacy、strict、v2 仅作为兼容路径或纯函数资产来源。
- v3 不得依赖旧路径的会话控制、确认状态或浏览器端意图分类。
- 对 v2 的复用必须经过明确 adapter 或 bridge。

### 3.2 LLM 负责理解和建议，确定性代码负责验证和执行

LLM 可以：

- 理解用户当前目标；
- 识别用户引用的是设计、配置、待确认行动还是运行结果；
- 提出结构化修改；
- 建议下一步；
- 解释已验证的 runtime 事实。

LLM 不可以：

- 直接设置 `run_confirmed=True`；
- 绕过 preflight；
- 自行决定 runtime 已获用户确认；
- 直接修改 session 内部状态；
- 把建议中的 sweep 或 optimization 在同一轮自动执行；
- 用目标描述替代真实运行事实。

### 3.3 确认只能确认已经存在的行动

确认是一个状态转换，不是普通运行意图的同义词。

```text
pending action 存在 + action_id 匹配 + 用户确认
-> 允许进入已确认状态

pending action 不存在 + 用户说确认
-> 不运行，不创建默认运行，向用户说明当前没有待确认行动
```

所有 runtime backend、所有语言、UI 按钮和自由文本确认都必须遵守同一规则。

### 3.4 runtime 后端不得改变运行前决策

`allow_in_memory`、local process、真实 Geant4 等选项只决定如何执行已经获准的
行动，不得改变：

- 用户意图理解；
- 是否需要确认；
- 是否存在 pending action；
- 是否应创建 payload；
- 是否允许运行；
- 是否应清除旧结果。

### 3.5 运行事实优先于用户目标

解释结果时：

- `latest_runtime_facts` 是权威事实；
- payload summary 用于说明配置；
- goal 仅用于解释用户原始意图；
- 旧 runtime observation 不得覆盖最新运行；
- 没有 runtime 结果时必须明确说明没有结果。

### 3.6 评测必须能失败

如果一个评分器长期给所有案例满分，它不能证明系统优秀，只能说明它缺少区分
能力。

每一个正式 grader 都必须至少包含：

- 一个应通过案例；
- 一个应失败案例；
- 一个边界或模糊案例；
- 一个用于检查 grader 自己是否误判的 calibration 案例。

## 4. 目标架构

目标不是一次性重写，而是逐步把职责从当前大模块中抽离。

```text
API / UI Turn
-> Turn Interpreter
-> Typed Turn Directives
-> State Transition Engine
-> Proposal Planner
-> Proposal Critic
-> Pending Action Manager
-> Workflow Executor
-> Tool Registry
-> Runtime Adapter
-> Observation / Runtime Facts
-> Response Assembler
-> Session Store
```

### 4.1 Turn Interpreter

职责：

- 结合 `V3ContextPack` 理解最新用户输入；
- 返回结构化理解；
- 标记歧义和引用对象；
- 不直接修改 state；
- 不决定 runtime 是否已获确认。

输出应逐步收口为明确类型，而不是把控制信号写入 `turn.metadata`。

建议类型：

```text
V3TurnDirectives
- dialogue_act
- user_goal
- referenced_state
- requested_changes
- confirmation_signal
- requested_action
- ambiguities
- confidence
- evidence
```

### 4.2 State Transition Engine

职责：

- 根据已验证的 directives 和 observation 更新 state；
- 应用 `V3StatePatch`；
- 清除失效 payload 和 runtime 结果；
- 维护 open questions；
- 记录状态变化原因。

它不负责调用 LLM，也不负责执行 Geant4。

### 4.3 Pending Action Manager

职责：

- 创建待确认行动；
- 生成稳定 `action_id`；
- 校验确认或取消事件；
- 拒绝过期、错误或不存在的 action；
- 确认后产生明确的 execution authorization；
- 不创建新的 payload 或运行意图。

建议类型：

```text
V3PendingAction
- action_id
- kind
- tool_call
- risk_level
- created_turn_id
- expires_at or invalidation_basis

V3ExecutionAuthorization
- action_id
- confirmed_by
- confirmed_turn_id
- authorized_tool_call_hash
```

### 4.4 Workflow Executor

职责：

- 只执行已经通过 schema、critic、preflight 和 confirmation 的行动；
- backend 选择发生在这里；
- 记录完整 observation；
- 不能推断用户意图；
- 不能把 backend 能力当作用户授权。

### 4.5 Response Assembler

职责：

- 从 state、context、observation 和建议形成稳定回答结构；
- 区分事实、假设、下一步建议；
- 保证 UI suggestions 带有明确 `text` 和 `prefill`；
- 不泄露内部控制字段；
- 可以调用 LLM naturalizer，但必须保留确定性 fallback。

### 4.6 Session Store

职责：

- 原子保存；
- schema version 迁移；
- corrupt quarantine；
- observation 压缩和保留策略；
- 并发写入保护；
- 不承担工作流决策。

## 5. 逐步移除隐形控制字典

Python 字典本身不是问题。JSON、tool schema、API payload 和 observation 使用
字典是合理的。

### 5.1 字典化控制原则

后续 v3 开发按以下顺序处理用户动作：

```text
LLM/结构化 turn understanding
-> 已验证的 requested_changes / config_overrides
-> 系统自己生成的明确 prefill 或 UI metadata
-> 受控 deterministic fallback
```

不允许把“为了让 case 通过而增加关键词/同义词表”作为主方案。确实需要 fallback
时，必须满足：

- fallback 只处理小范围、可解释、可验证的表达；
- fallback 结果要进入 `turn_understanding.source`、`last_state_patch` 或 eval metrics；
- 新增 fallback 必须有一条对应的 intelligence/safety eval；
- 不能让 fallback 直接设置 `run_confirmed` 或绕过 pending action。

新增 harness：

```text
tools/evaluate_v3_agent_intelligence.py
eval/v3/tasks/agent_intelligence.jsonl
```

它专门回答两个问题：

- 需要智能理解的 turn 是否真的走了 LLM/structured understanding；
- 系统生成的按钮 prefill 是否能进入结构化 metadata/patch 链路，而不是退回宽泛关键词匹配。

需要消除的是通过通用 `metadata` 传递主流程控制信号的做法。

当前应优先迁出的字段：

| 当前信号 | 目标归属 |
| --- | --- |
| `run_confirmed` | `V3ExecutionAuthorization` |
| `allow_in_memory` | `V3RuntimePolicy` |
| `run` / `suppress_run` | `V3TurnDirectives.requested_action` |
| `pending_action` | `V3AgentState.pending_action` |
| `accept_defaults` | `V3TurnDirectives` 或明确 UI event |
| `config_overrides` | `V3StatePatch` |
| `auto_sweep` / `optimization` | `V3SuggestedAction` 或 `V3ExperimentPlan` |
| `suggestions` | `V3AgentState.suggested_actions` |

迁移规则：

1. 新类型先与旧 metadata 并存。
2. 主流程优先读取新类型。
3. 旧字段只作为兼容输入。
4. 增加兼容读取告警或 trace。
5. 所有调用者迁移后再删除旧字段。

禁止在一次改动中整体删除 metadata。迁移必须小步、可测试、可回退。

## 6. Eval 总体结构

后续 eval 分成多条独立跑道。它们回答不同问题，不能用一个总分相互抵消。

```text
Agent Behavior Eval
Runtime And Physics Eval
Context And Grounding Eval
Response Experience Eval
```

### 6.1 Agent Behavior Eval

回答：

- Agent 是否选择了正确行动？
- 是否正确修改 state？
- 是否正确等待确认？
- 是否调用了正确工具？
- 是否拒绝了不允许的行动？

主要由确定性 grader 判断。

核心指标：

- safety invariant pass rate；
- intelligence harness pass rate；
- state transition pass rate；
- tool selection pass rate；
- tool argument fidelity；
- pending action lifecycle；
- backend invariance；
- stale action rejection。

### 6.2 Runtime And Physics Eval

回答：

- 配置是否真的能运行？
- 真实 Geant4 输出是否匹配 reviewed golden？
- runtime capability 是否被准确声明？
- 不支持能力是否被明确拒绝？

正式通过必须使用真实 Geant4。in-memory 只验证接线，不能计入工业通过率。

### 6.3 Context And Grounding Eval

回答：

- Agent 是否使用最新设计和最新 runtime？
- 修改后是否清除了旧结果？
- 回答是否引用了正确 evidence？
- 长会话恢复后是否保持一致？
- 是否出现 prompt 污染或旧示例污染？

主要检查结构化 context、state diff、evidence source 和回答中的事实一致性。

### 6.4 Response Experience Eval

回答：

- 用户是否容易理解当前状态？
- 回答是否直接、有帮助且不过度机械？
- 是否提出了合理的下一步？
- 中文、英文和中英混合输入是否自然？

确定性 grader 检查基本结构和事实一致性；校准过的 LLM grader 或人工抽样只
判断自然度、帮助程度和表达质量。

## 7. 标准 Eval Pipeline

所有正式 v3 eval 应遵守：

```text
task corpus
-> run
-> raw trajectory
-> deterministic grade
-> optional language grade
-> compare baseline vs candidate
-> calibrate graders
-> publish report
```

### 7.1 Task Corpus

每个 task 必须有稳定 ID，并描述能力而不是关键词。

建议字段：

```json
{
  "id": "confirmation.no_pending.free_text.en",
  "suite": "agent_behavior",
  "slice": "confirmation_safety",
  "lang": "en",
  "initial_state": {},
  "turns": [],
  "invariants": [],
  "expected_outcome": {},
  "forbidden_outcomes": [],
  "tags": []
}
```

任务不能只因为换了几个同义词就重复增加。新任务必须改变状态、行动、工具、
风险、context 或 runtime 结果中的至少一项。

### 7.2 Adapter

Adapter 只负责运行被测系统并输出事实，不负责判断通过或失败。

输出至少包含：

- 输入 task ID；
- 每轮 request；
- 每轮 response；
- state before / after 摘要；
- proposed action；
- tool calls；
- pending action；
- observations；
- evidence sources；
- fallback 信息；
- runtime backend；
- latency、token、cost，如果可获得。

任何 secret 都不得进入 eval 输出。

### 7.3 Deterministic Graders

建议拆分为独立 grader：

| Grader | 主要判断 |
| --- | --- |
| `SafetyInvariantGrader` | 是否出现未授权运行、确认绕过、自动 sweep |
| `StateTransitionGrader` | state 是否按预期更新或失效 |
| `ToolUseGrader` | tool 选择、参数和调用顺序 |
| `GroundingGrader` | 回答是否使用正确 runtime facts |
| `BackendInvarianceGrader` | 不同 backend 的运行前决策是否一致 |
| `SessionLifecycleGrader` | 保存、恢复、reset、过期行动 |
| `CapabilityBoundaryGrader` | 不支持能力是否诚实拒绝 |

安全 invariant 一旦失败，该 task 必须失败。不得用其他高分抵消。

### 7.4 Language Grader

Language grader 只能评价：

- 清晰度；
- 自然度；
- 是否直接回应用户；
- 是否给出有帮助的下一步；
- 是否重复或过度机械。

它不得判断：

- runtime 是否真的运行；
- 物理结果是否正确；
- 是否已获用户确认；
- tool call 是否安全；
- golden 数值是否匹配。

### 7.5 Compare

每次重要修改都应比较 baseline 和 candidate。

报告至少包含：

- 总通过率；
- 各 slice 通过率；
- 新增通过；
- 新增失败；
- safety regressions；
- fallback 变化；
- 延迟和成本变化；
- 失败案例的最小可读摘要。

合并判断不能只看总通过率。例如，总通过率上升但确认安全下降时必须阻止继续
合并。

### 7.6 Calibration

grader 自身也要测试。

Calibration 数据应包含人工标注的：

- 明确正确回答；
- 明确错误回答；
- 表达很好但事实错误的回答；
- 事实正确但表达较差的回答；
- 应拒绝却执行的轨迹；
- 正确拒绝不支持能力的轨迹。

正式使用 language grader 前，应人工复核一小批结果并记录误判类型。

## 8. 必须覆盖的高价值案例

### 8.1 确认与安全

- 无 pending action 时输入 confirm。
- 无 pending action 时输入 cancel。
- pending action 存在时自由文本确认。
- pending action 存在时 UI explicit event 确认。
- action ID 不匹配。
- action 已因配置修改而失效。
- 用户同时说“确认，但先把能量改成 2 MeV”。
- prompt injection 尝试设置 `run_confirmed`。
- 不同 runtime backend 下确认结果一致。

### 8.2 多轮 context

- 设计后修改材料。
- 运行后修改能量，旧 runtime 必须失效。
- 连续两次运行后询问“第二次为什么不同”。
- 用户引用“刚才的配置”和“最初的配置”。
- session 保存并恢复后继续修改。
- 长会话中只保留最新权威事实。

### 8.3 结果解释

- 有结果时解释指定指标。
- 无结果时拒绝编造。
- 指标缺失时明确说明。
- payload 与 runtime 结果不一致时以 runtime facts 为准。
- 当前运行是水/质子时，回答中不得出现无关的铅/gamma 模板内容。
- 建议 sweep 时只生成建议，不自动执行。

### 8.4 语言与交互

- 自然中文。
- 自然英文。
- 中英混合参数表达。
- 极短确认和取消。
- 模糊修改要求。
- 用户纠正 Agent。
- 用户要求查看当前配置而不是运行。

## 9. 分阶段推进计划

### Phase A：修复确认语义与 backend 串线

目标：任何 backend 都不能改变运行前授权语义。

实施：

- 无 pending action 的确认直接进入明确的 no-op / explain 状态。
- 将确认校验集中到 Pending Action Manager。
- backend 选项不再参与 turn understanding 或确认判断。
- 删除重复且不可达的确认 fallback 代码。
- 增加 backend invariance grader。

验收：

- 无 pending action 的确认在所有 backend 下都不创建 payload、不创建 pending、
  不运行。
- action ID 不匹配永远不能执行。
- 全部 safety cases 100% 通过。

停止条件：

- 如果 backend 仍会改变确认、payload 创建或 pending action，禁止进入 Phase B。

### Phase B：建立可比较的 Eval Harness

目标：把当前“运行并自己判分”的脚本拆成可复用评测流水线。

实施：

- 将 task corpus 从 runner 中独立出来。
- adapter 只输出轨迹。
- grader 独立读取轨迹。
- 建立 baseline / candidate compare。
- 建立第一批 calibration cases。
- 保留现有 casebank 作为迁移输入，不一次性删除。

验收：

- 一个命令可以运行稳定的 v3 behavior suite。
- 一个命令可以比较两个 eval record。
- 人工构造的错误轨迹能被对应 grader 检出。
- compare 能单独显示 safety regression。

停止条件：

- 如果 grader 无法区分至少一组明确好坏案例，不能作为合并门。

### Phase C：第一批 typed control contracts

目标：把最危险的控制信号移出 metadata。

实施顺序：

1. `V3PendingAction`
2. `V3RuntimePolicy`
3. `V3ExecutionAuthorization`
4. `V3TurnDirectives`
5. `V3SuggestedAction`

验收：

- `run_confirmed` 不再作为主流程 metadata 开关。
- `allow_in_memory` 只在 executor/runtime policy 中读取。
- pending action 生命周期有独立单元测试和 eval task。
- 旧 metadata 兼容输入仍可读取，但不会覆盖新类型。

停止条件：

- 如果新旧字段出现双重权威来源，必须先消除冲突再继续迁移。

### Phase D：拆分 Service 与 Reasoner

目标：让每个模块只承担一种可解释职责。

建议拆分：

```text
service.py
-> turn_service.py
-> pending_action_manager.py
-> state_transition.py
-> workflow_executor.py
-> response_assembler.py

reasoners.py
-> basic_reasoner.py
-> llm_reasoner.py
-> proposal_builder.py
-> result_reasoner.py
```

验收：

- confirmation 逻辑只存在于一个边界。
- runtime backend 选择只存在于 executor/runtime adapter 边界。
- reasoner 不直接修改 state。
- service 只协调，不包含 sweep 执行和复杂文本提取。

停止条件：

- 不为了缩短文件而制造无意义的小模块。
- 每次拆分必须有行为 eval 保证外部结果不变。

### Phase E：提升 Agent 回应智能化

目标：提高用户体验，同时保持事实和安全边界。

实施：

- LLM 使用 `V3ContextPack` 理解当前问题。
- 回复明确区分：当前结论、依据、假设、下一步。
- 对模糊需求优先提出最小必要问题。
- suggestion 成为可执行但仍需用户触发的下一步。
- 建立 calibrated language grader 和人工抽样流程。

验收：

- language grader 能区分事实正确但表达差、表达好但事实错。
- 中文、英文和混合语言均有真实自然案例。
- runtime 解释不会引用无关模板事实。
- 回复改进不得降低 behavior 或 safety 指标。

### Phase F：真实运行与工业能力闭环

目标：让少量核心场景先成为可信的完整能力。

优先选择 5 个代表场景：

- lead gamma shielding；
- water phantom energy deposition；
- silicon detector gamma response；
- polyethylene neutron moderation；
- multi-turn shielding thickness update。

实施：

- 固定 runtime fingerprint；
- 生成并人工 review golden；
- 接通 candidate config 到真实 runtime；
- 比较数值与关系；
- 验证结果问答只使用 structured facts。

验收：

- 5 个核心场景可正式评估；
- golden 均有 review 记录；
- 失败能够定位到 config、compile、runtime、metric 或 result QA；
- in-memory 结果不会被误报为 industrial pass。

## 10. 建议目录

第一轮不要求大规模移动目录。新增 eval 结构时建议逐步靠近：

```text
eval/
  v3/
    tasks/
      behavior.jsonl
      context.jsonl
      response.jsonl
    adapters/
      v3_turn_adapter.py
    graders/
      safety.py
      state_transition.py
      tool_use.py
      grounding.py
      backend_invariance.py
      language_quality.py
    calibration/
      language_quality.jsonl
      grader_failures.jsonl
    reports/
      README.md

core/agent_v3/
  contracts/
    turn_directives.py
    pending_action.py
    runtime_policy.py
    suggestions.py
  orchestration/
    turn_service.py
    state_transition.py
    pending_action_manager.py
    workflow_executor.py
```

是否采用这些具体目录，应以减少耦合和匹配现有导入方式为判断标准。不要为了
目录漂亮进行一次性大搬迁。

## 11. 每次改动的标准工作流

### 11.1 开始前

- 明确本次修改保护或提升哪个能力。
- 写出至少一个应通过案例和一个应失败案例。
- 标记是否影响确认、state、context、tool、runtime 或回复。
- 检查是否正在向 metadata 增加新的主流程开关。

### 11.2 实施中

- LLM 输出先验证再使用。
- state 修改通过 patch 或明确 transition。
- runtime 行动必须经过 preflight 和确认。
- 新兼容逻辑必须标明退出条件。
- 不在同一改动中顺便重构无关 legacy 模块。

### 11.3 完成前

- 运行相关单元测试。
- 运行对应 behavior slices。
- 比较 baseline 和 candidate。
- 对 LLM 输出进行人工抽样。
- 如果涉及 runtime，明确本次使用 wiring 还是真实 Geant4。
- 更新本手册、`ARCHITECTURE.md` 或实施日志中的相关状态。

## 12. 合并门

### 必须通过

- 相关单元和集成测试通过。
- safety invariants 100% 通过。
- 不出现新的未授权 runtime 行动。
- 不出现 stale runtime facts 重用。
- v3 默认 UI 不调用 legacy endpoint。
- 新类型或 API 有明确 schema version。
- secret 不进入源码、eval record 或报告。

### 需要人工审查

- prompt 或 system message 变化；
- context pack 字段变化；
- confirmation 语义变化；
- runtime tool schema 变化；
- golden metrics 变化；
- language grader 或其阈值变化；
- fallback 路径变化。

### 可以接受但必须记录

- 某些 industrial case 仍为 unsupported；
- 真实 runtime 暂不可用导致 not evaluable；
- 兼容字段暂时并存；
- language quality 小幅波动，但不得伴随 safety 或 grounding 回归。

### 不可接受

- 用总分掩盖 safety regression；
- 用 in-memory 结果宣称工业能力通过；
- 为了让 casebank 通过而增加同义词关键词；
- LLM 直接控制执行授权；
- 修改配置后继续解释旧 runtime 结果；
- 没有 calibration 就将 LLM grader 设为硬门。

## 13. 推荐命令与运行模式

当前可用的本地基线命令：

```powershell
.venv\Scripts\python.exe tools\evaluate_v3_dialogue_casebank.py `
  --casebank docs\eval\v3_dialogue_casebank.json `
  --json
```

检查 in-memory wiring 下的行为差异：

```powershell
.venv\Scripts\python.exe tools\evaluate_v3_dialogue_casebank.py `
  --casebank docs\eval\v3_dialogue_casebank.json `
  --allow-in-memory `
  --json
```

检查 industrial benchmark 当前可评估状态：

```powershell
.venv\Scripts\python.exe tools\evaluate_industrial_runtime_benchmark.py --json
```

运行 v3 测试时，PowerShell 不会像部分 shell 一样自动展开 pytest 路径通配符。
应先由 PowerShell 获取文件：

```powershell
$tests = Get-ChildItem tests\test_agent_v3_*.py | ForEach-Object FullName
.venv\Scripts\python.exe -m pytest $tests -q
```

未来 Phase B 完成后，应提供统一命令，至少支持：

```text
run behavior suite
grade trajectories
compare baseline candidate
calibrate graders
```

## 14. ARC-7 架构审查门

重大 v3 改动在合并前应从以下角度审查：

| 角度 | 必须回答的问题 |
| --- | --- |
| 架构边界 | 新逻辑是否放在正确模块？是否出现双重权威？ |
| 安全 | 是否可能绕过确认、preflight 或 schema 校验？ |
| 用户价值 | 用户能否更清楚、更少出错地完成任务？ |
| 简化 | 是否减少了隐藏状态和特殊分支？ |
| 性能与成本 | 是否增加了不必要 LLM 调用、上下文或 runtime 成本？ |
| 失败模式 | 网络失败、LLM 无效输出、session 损坏、runtime 不可用时会怎样？ |

审查结果应按严重程度记录，并绑定到具体 contract、模块或 eval case。

## 15. 风险登记

| 风险 | 当前判断 | 应对 |
| --- | --- | --- |
| 无 pending action 的确认变成新运行 | 已发生，高优先级 | Phase A 修复并建立 invariant |
| metadata 成为隐形控制总线 | 已发生，持续扩大 | Phase C 小步类型化迁移 |
| response quality 长期满分 | 已发生，grader 区分力弱 | Phase B/E 加 calibration |
| service/reasoners 职责过多 | 已发生，维护风险上升 | Phase D 按行为边界拆分 |
| session observation 持续增长 | 潜在长期问题 | 增加 compaction、retention、atomic save |
| tool schema 校验深度不足 | 已确认能力有限 | 增强嵌套对象、数组和组合规则校验 |
| v2 依赖边界不完全统一 | 已确认少量直接依赖 | 统一经 bridge/adapter |
| industrial benchmark 长期 not evaluable | 已发生 | Phase F 先闭环 5 个核心场景 |

## 16. 完成定义

v3 架构升级不是以“文件拆完”或“测试数量增加”为完成标准。

第一阶段完成的标志：

- 确认语义与 runtime backend 完全解耦；
- safety grader 能稳定捕获未授权运行；
- baseline/candidate 可以正式比较；
- 最危险的控制信号已移出 metadata；
- Agent 回答使用最新 context 和 runtime facts；
- 至少 5 个真实 Geant4 场景形成 reviewed golden 闭环。

长期完成的标志：

> Agent 能自然理解用户、谨慎提出行动、可靠保存上下文、只在授权后执行，
> 并能用真实运行证据解释结果；每次升级都可以通过独立 eval 证明没有破坏这些
> 能力。

## 17. 下一轮立即执行项

按以下顺序推进，不并行扩展自主能力：

1. 修复无 pending action 的确认语义。
2. 增加 deterministic 与 in-memory backend invariance 测试。
3. 把确认相关逻辑集中到 Pending Action Manager。
4. 建立独立 safety grader 和首批 calibration cases。
5. 将 `run_confirmed`、`allow_in_memory`、`pending_action` 迁入明确类型。
6. 完成 baseline / candidate compare 后，再开始拆分 service 和 reasoners。

## 18. 如何使用与维护本手册

### 18.1 准则优先级

当文档之间出现冲突时，按以下顺序判断：

1. 已验证的安全 invariant 和真实 runtime 证据；
2. 本手册定义的 v3 开发与验收规则；
3. `ARCHITECTURE.md` 的当前主链路和目录边界；
4. 当前阶段实施计划；
5. 历史设计文档和 legacy 行为。

历史测试仍然需要兼容时，应通过明确兼容层保留，不得让历史行为重新成为 v3
主流程的权威来源。

### 18.2 更新规则

以下变化必须更新本手册或新增带日期的决策记录：

- 改变确认、取消或 runtime 授权语义；
- 改变 context pack 的权威事实来源；
- 改变正式 eval 的 grader、阈值或合并门；
- 改变 industrial benchmark 的正式通过条件；
- 引入新的主流程控制 contract；
- 决定推迟、替换或取消某个 Phase。

仅实现手册中已经明确描述的代码细节时，不需要重复扩写手册；应更新实施日志和
对应测试。

### 18.3 每轮推进记录模板

每轮重要推进应使用下面的最小记录。可以写入实施日志、架构 review 或评测报告。

```markdown
# v3 推进记录：<主题>

日期：
负责人：
对应 Phase：

## 目标

- 本轮要解决的用户或架构问题：
- 明确不做的内容：

## 受影响边界

- Contracts：
- State / Context：
- Tools / Runtime：
- API / UI：
- Eval：

## 风险假设

- 可能破坏的已有行为：
- 需要保持不变的 safety invariant：
- 是否影响真实 runtime：

## 实施内容

- 新增：
- 修改：
- 兼容：
- 删除或弃用：

## Eval 证据

- 单元测试：
- Behavior slices：
- Baseline vs candidate：
- Live LLM 抽样：
- Real Geant4 / wiring 模式：

## 结果

- 已解决：
- 未解决：
- 新发现风险：
- 是否满足本手册合并门：

## 下一步

1.
2.
```

### 18.4 完成检查表

每轮结束前确认：

- [ ] 改动沿 v3 主链路实现，没有把新产品逻辑放入 legacy 路径。
- [ ] 没有新增未经类型约束的主流程 metadata 开关。
- [ ] LLM 建议经过确定性验证后才改变 state 或执行工具。
- [ ] runtime 行动经过 preflight 和明确确认。
- [ ] 不同 backend 不改变运行前决策。
- [ ] 修改配置后，旧 payload/runtime/context 已按规则失效。
- [ ] 新行为有通过、失败和边界 eval 案例。
- [ ] safety invariant 没有回归。
- [ ] 回答中的事实来自最新 context 或 runtime evidence。
- [ ] 已记录无法验证、not evaluable 或 unsupported 的部分。
