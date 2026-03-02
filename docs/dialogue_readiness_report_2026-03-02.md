# Geant4-Agent 阶段报告（2026-03-02）

## 1. 结论

当前项目已经完成从“配置生成内核”到“可承接对话代理的稳定底座”的过渡。

本轮之后：

- `slot-first` 仍是主配置链路。
- 共享规则（默认配置、字段标签、路径别名、phase 规则、提示语、提交后语义同步）已基本从编排层抽离。
- `strict` 链路已接入 `Dialogue Agent v1` 骨架，但目前只承担**对话动作选择与用户消息渲染**，不改动配置提交权。

因此，下一阶段可以在现有架构上继续推进真正的对话代理，而不需要再做一次大规模架构清理。

---

## 2. 本轮新增内容

### 2.1 Dialogue Agent v1 骨架

新增模块：

- `core/dialogue/types.py`
- `core/dialogue/policy.py`
- `core/dialogue/renderer.py`

职责分层：

- `types.py`
  - 定义 `DialogueAction`
  - 定义 `DialogueDecision`
  - 定义 `build_dialogue_trace(...)`
- `policy.py`
  - 决定本轮对话动作
  - 动作类型：
    - `ask_clarification`
    - `confirm_update`
    - `answer_status`
    - `finalize`
- `renderer.py`
  - 根据 `DialogueDecision` 生成用户可见文本
  - 对 `ask_clarification` 复用现有 `question_renderer`
  - 对 `confirm_update / answer_status / finalize` 走确定性文案

### 2.2 strict 链路接入 Dialogue Agent

修改：

- `core/orchestrator/session_manager.py`
- `core/orchestrator/types.py`

变更：

- `process_turn(...)` 不再直接把“问什么”结果当作最终回复。
- 现在流程变为：
  1. 先完成配置处理与校验
  2. 生成 `asked_fields`
  3. 计算 `updated_paths`
  4. 由 `dialogue.policy` 决定本轮动作
  5. 由 `dialogue.renderer` 生成 `assistant_message`

新增返回字段：

- `dialogue_action`
- `dialogue_trace`

新增会话状态字段：

- `SessionState.last_dialogue_action`

### 2.3 继续集中化共享规则

本轮在前一轮基础上继续完成三块注册表的稳定接线：

- `core/config/path_registry.py`
  - 路径别名 canonicalization
  - pattern 匹配
- `core/config/phase_registry.py`
  - phase 选择
  - phase 标题
- `core/config/prompt_registry.py`
  - “配置已完成”
  - 单字段追问 fallback
  - 追问 prompt

接入点：

- `core/config/field_registry.py`
- `planner/agent.py`
- `ui/web/legacy_api.py`
- `core/orchestrator/session_manager.py`

---

## 3. 当前架构总览

### 3.1 核心链路（配置链）

当前 `strict` 主链：

1. 用户输入文本
2. `LLM slot frame`
3. `slot validator`
4. `slot mapper`
5. `runtime extractor`（补充候选）
6. `Arbiter`
7. `Validator Gate`
8. `semantic_sync`
9. 原子提交

这条链路的核心目标仍然是：

- 理解用户需求
- 填对配置
- 保证配置一致性

### 3.2 对话链（新增）

当前 `strict` 对话链：

1. 配置链完成本轮提交/回滚
2. 计算：
   - `missing_fields`
   - `asked_fields`
   - `updated_paths`
   - `answered_this_turn`
3. `dialogue.policy` 选择动作
4. `dialogue.renderer` 生成用户文本
5. 输出：
   - `assistant_message`
   - `dialogue_action`
   - `dialogue_trace`
   - `dialogue_summary`
   - `raw_dialogue`

当前对话层定位：

- 负责“这轮怎么回应用户”
- 不负责“怎么提交配置”

也就是说，**对话代理已经有了骨架，但提交权仍完全在配置内核**。

### 3.3 raw dialogue（原始对话）

为了后续报告和回归评估，当前 `strict` 响应中已经显式提供：

- `raw_dialogue`

它是最近若干轮的原始对话记录（`role + content`），用于：

- 展示真实对话链
- 复盘用户输入与系统输出
- 和 `dialogue_trace`（处理链）配对做审查

### 3.4 可审计思考链（系统决策链）

这里的“思考链”不是模型的原始隐藏推理，而是系统可审计的决策链。

当前可观察的决策顺序如下：

1. `user_intent`
   - 由规范化/slot 层得到
2. `slot / semantic / fallback` 路由结果
   - 体现为：
     - `llm_used`
     - `fallback_reason`
     - `inference_backend`
     - `slot_debug`
3. 候选仲裁
   - 体现为：
     - `rejected_updates`
     - `applied_rules`
4. 校验结论
   - 体现为：
     - `violations`
     - `warnings`
     - `missing_fields`
5. 对话动作选择
   - 体现为：
     - `dialogue_action`
     - `dialogue_trace`

这条链是当前系统推荐暴露给用户/开发者的“思考链”。

它具备两个优点：

- 可以解释系统为什么这么回复
- 不依赖暴露模型原始 CoT

---

## 4. Dialogue Agent v1 的行为边界

当前版本 **不是完整对话代理**，而是对话代理骨架。

它当前能做：

- 配置缺失时：追问
- 有更新但未完成时：确认更新并提示剩余缺项
- 配置已完成时：结束语
- 没有新更新时：返回状态类消息

它当前还不能做：

- 主动解释复杂物理选择的原因
- 做长上下文总结
- 处理多轮自由问答中的解释/反问/推理型对话
- 生成“多步骤协商式”对话策略

这部分会是下一阶段工作。

---

## 5. 为什么现在可以推进对话代理

之前不能做，是因为核心规则还散落在多个层里：

- UI 有一套
- planner 有一套
- orchestrator 有一套
- legacy 又有一套

这会导致：

- 对话层一接入就会绑死在旧逻辑上
- 文案、phase、字段名、路径别名都可能不一致

现在已完成的关键收口包括：

- 默认配置集中化：`core/config/defaults.py`
- 路径规则集中化：`core/config/path_registry.py`
- 字段标签集中化：`core/config/field_registry.py`
- phase 规则集中化：`core/config/phase_registry.py`
- 提示语集中化：`core/config/prompt_registry.py`
- 提交后语义同步集中化：`core/orchestrator/semantic_sync.py`

因此对话代理现在可以建立在这些稳定边界之上，而不再需要直接操作杂乱的流程代码。

---

## 6. 测试结果

本轮执行：

```powershell
.\.venv\Scripts\python -m unittest tests.test_dialogue_agent tests.test_prompt_registry tests.test_path_registry tests.test_phase_registry tests.test_field_registry tests.test_question_planner tests.test_config_defaults tests.test_semantic_sync tests.test_derived_sync tests.test_smoke_no_ollama -v
```

结果：

- `Ran 35 tests`
- `OK`

新增测试：

- `tests/test_dialogue_agent.py`
- `tests/test_prompt_registry.py`
- `tests/test_path_registry.py`
- `tests/test_phase_registry.py`

覆盖重点：

- 对话动作选择
- 非 LLM 状态消息渲染
- 路径别名匹配
- phase 选择与标题
- 共享提示语
- strict 响应中 `dialogue_action` / `dialogue_trace`

---

## 7. 当前剩余问题

### 7.1 对话层仍是“动作层”，不是“智能体层”

目前只是：

- `policy`
- `renderer`

还缺：

- 对话状态摘要
- 对话动作历史
- 对话级解释策略
- 多轮确认/复述/总结机制

### 7.2 legacy 仍是兼容区

`ui/web/legacy_api.py` 仍然存在，虽然规则已明显变少，但它仍不是长期主线。

### 7.3 `knowledge/validate.py` 仍带部分旧 schema 术语

这不影响 `strict` 主链，但会影响整体术语一致性。

---

## 8. 下一阶段建议（Dialogue Agent v2）

下一步建议直接进入对话代理建设，而不是继续做大范围清理。

建议顺序：

1. 增加 `dialogue_state`
   - 记录：
     - 最近确认的事实
     - 最近用户修改
     - 最近回答的问题
     - 待确认事项

2. 扩展 `dialogue_action`
   - 在现有 4 类基础上增加：
     - `explain_choice`
     - `summarize_progress`
     - `confirm_overwrite`

3. 增加 `dialogue_summary`
   - 在每轮输出中增加：
     - 已确认
     - 本轮更新
     - 剩余缺项

4. 保持提交权不变
   - 对话层仍不能直接改配置
   - 只读取配置链结果并决定如何回应

---

## 9. Git 说明

本轮代码已本地提交。

新增本地提交：

- `e328e92` `Centralize path, phase, and prompt registries`

另外本轮对话代理骨架改动尚未单独拆成新 commit（如果需要，可以下一轮再本地提交一次）。

按当前约定：

- 后续只做本地 git 同步
- 不再尝试 `push`
