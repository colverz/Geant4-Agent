# Geant4-Agent 项目问题全景分析

> 生成时间：2026-04-21 / 最后修订：2026-04-21（结合运行时代码核实后修正）  
> 分析范围：`core/`、`nlu/`、`planner/`、`builder/`、`knowledge/`、`mcp/`、`runtime/`、`ui/`、`tests/`

---

## 目录

1. [架构层问题](#1-架构层问题)
2. [类型系统与数据建模问题](#2-类型系统与数据建模问题)
3. [NLU 与 LLM 集成问题](#3-nlu-与-llm-集成问题)
4. [对话与规划层问题](#4-对话与规划层问题)
5. [运行时与执行层问题](#5-运行时与执行层问题)
6. [知识层问题](#6-知识层问题)
7. [测试与可维护性问题](#7-测试与可维护性问题)
8. [UI 与部署问题](#8-ui-与部署问题)
9. [问题优先级汇总表](#9-问题优先级汇总表)
10. [推荐执行顺序](#10-推荐执行顺序)

---

## 1. 架构层问题

### 1.1 `session_manager.py` 是高度耦合的单体

**文件**：`core/orchestrator/session_manager.py`（约 2228 行）

**问题描述**：  
该文件同时承担了以下所有职责：
- NLU 调度（调用 BERT、LLM slot_frame、LLM semantic_frame、normalizer）
- 候选组装与 bridge 构建（`_build_v2_bridge_candidates`、`_build_interpreter_bridge_candidates`）
- 候选仲裁（调用 `arbiter.py`）
- 覆盖保护与 pending_overwrite 逻辑
- 槽记忆同步（`_merge_slot_frame_with_memory`、`_refresh_slot_memory`）
- 阶段跳转决策
- 对话动作派发
- 审计日志写入
- 回复渲染

任何一个环节的改动都需要在这 2000+ 行中定位上下文，极易引入副作用。

**影响**：可维护性极差；单元测试难以隔离；新增功能风险高。

**建议**：拆分为独立的流水线组件——`NLUDispatcher`、`CandidateAssembler`、`ConfigApplicator`、`PhaseRouter`、`DialogueDirector`，由薄层 `TurnProcessor.process()` 串联。

---

### 1.2 线性阶段 FSM 限制了用户交互自由度

**文件**：`core/orchestrator/phase_machine.py`、`core/orchestrator/types.py`

**问题描述**：  
当前阶段序列是硬编码的线性链：

```
GEOMETRY → MATERIALS → SOURCE → PHYSICS → OUTPUT → FINALIZE
```

用户无法跨阶段跳跃输入（例如在几何阶段直接说出粒子源类型），必须等前一阶段完成才能进入下一阶段。虽然 `validate_layer_c_completeness` 会做全局完整性检查，但阶段转换本身不感知跨域信息。

**影响**：用户体验受限；真实用户的输入往往是非顺序的。

**建议**：将进度建模为**槽完成度依赖图**，任何阶段的槽随时可填，缺哪个问哪个。

---

### 1.3 Legacy pipeline 与 v2 pipeline 并存，边界未文档化

**文件**：`core/pipelines/selectors.py`、`core/pipelines/geometry_legacy_pipeline.py`、`core/pipelines/geometry_v2_pipeline.py`

**问题描述**：  
系统中同时存在 legacy 和 v2 两套 pipeline，通过 `select_pipelines()` 在运行时选择。当前 legacy/v2 切换是**有意为之的功能**（支持回退和对比测试），并非需要立即消除的技术债。但两者的适用条件、行为差异以及各自的测试覆盖矩阵均缺乏文档说明。

**影响**：新维护者无法判断何时应使用哪条路径；两套路径的回归测试成本翻倍但覆盖不对称。

**建议**：不要立即删除 legacy 路径，而是：(1) 在 `core/pipelines/` 中补充说明文档，明确两者的切换条件和预期行为差异；(2) 补充 legacy 专项测试矩阵，确保两条路径都有等价覆盖。

---

### 1.4 `remove` 操作被静默忽略

**文件**：`core/orchestrator/session_manager.py`，`_apply_updates` 函数

```python
def _apply_updates(config: dict, updates: list) -> None:
    for upd in updates:
        if upd.op == "remove":
            # remove-path is intentionally omitted in v0.2 prototype
            continue
        set_path(config, upd.path, upd.value)
```

**问题描述**：`UpdateOp` 定义了 `"remove"` 操作，但实现中被注释跳过。任何产生 `op="remove"` 的生产者都会静默失效，不会报错。

**影响**：行为与契约不符；删除类用户意图（"帮我去掉材料"）无法正确执行。

---

## 2. 类型系统与数据建模问题

### 2.1 核心配置对象是裸 `dict[str, Any]`

**文件**：`core/config/defaults.py`、`core/orchestrator/types.py`（`SessionState.config`）

**问题描述**：  
整个仿真配置（geometry/materials/source/physics/output）以 `dict[str, Any]` 形式存储在 `SessionState.config` 中。所有字段读写通过字符串路径完成：

```python
get_path(c, 'geometry.structure', '')
set_path(config, "source.position", {...})
```

字段名拼写错误、路径层级错误、类型错误在运行时之前完全无法被静态检查发现。

**量化**：`session_manager.py` 中出现 **62 处** `dict[str, Any]` 类型标注。

**建议**（已修正）：**不建议一步到位**将 `SessionState.config` 全量替换为 dataclass。`arbiter.py`、`validator_gate.py`、`semantic_sync.py` 等十余个组件直接操作 config dict，全量替换是 breaking change，会产生大量连锁失败。  
更稳的策略是**在边界层逐步收口**：优先在 `SlotFrame` → typed `SimulationConfig` spec → `RuntimePayload` 的边界使用 typed 结构，内部 config dict 随时间渐进迁移，不影响当前稳定性。

---

### 2.2 大量运行时 `isinstance(x, dict)` 防御性检查

**文件**：`core/orchestrator/session_manager.py`

**问题描述**：  
由于 pipeline 元数据通过 `slot_debug: dict[str, Any]` 传递，消费方必须用 `isinstance(x, dict)` 来做运行时防御：

```python
geometry_memory = slot_memory.get("geometry") if isinstance(slot_memory.get("geometry"), dict) else {}
source_memory   = slot_memory.get("source")   if isinstance(slot_memory.get("source"),   dict) else {}
```

这种检查在 `session_manager.py` 中出现了约 **20 处以上**。每一处都是在运行时做本应由类型系统静态保证的事情。

**建议**：用 typed dataclass（`PipelineDebug`、`GeometryPipelineMeta`、`SourcePipelineMeta`）替换 `slot_debug` 裸 dict，彻底消灭防御性检查。

---

### 2.3 `source.position` / `source.direction` 的值格式是带标签的 dict

**文件**：`core/orchestrator/session_manager.py`，约第 364、369 行

**问题描述**：  
向量值在 config 中存储为：

```python
{"type": "vector", "value": [x, y, z]}
```

读取时需要嵌套两层类型检查才能取出三维坐标。本质上是应该用 `Vec3` dataclass 表达的值类型，却用了带约定结构的裸 dict。

**建议**：引入 `Vec3(x, y, z)` frozen dataclass，在 config 层直接使用。

---

### 2.4 `slot_debug` 中存在局部变量覆盖隐患

**文件**：`core/orchestrator/session_manager.py`，`_v2_compile_missing_paths` 函数

```python
spatial_meta = slot_debug.get("spatial_v2")
if isinstance(spatial_meta, dict):
    source_meta = spatial_meta.get("source_meta")   # ← 覆盖了外层 source_meta 变量
    if isinstance(source_meta, dict):
        ...
```

外层已有 `source_meta = slot_debug.get("source_v2")`，内层重新赋值为 `spatial_meta.get("source_meta")`，导致两个语义不同的变量共用同一名称。这是潜在的逻辑错误来源。

---

### 2.5 `dialogue_summary` 和 `dialogue_memory` 条目无类型约束

**文件**：`core/dialogue/state.py`

`build_dialogue_summary()` 返回 `dict[str, Any]`，存入 `state.dialogue_summary`。`dialogue_memory` 是 `list[dict[str, Any]]`。这些结构在多处被消费，但没有对应的 dataclass 定义，键名变更不会有任何静态提示。

---

## 3. NLU 与 LLM 集成问题

### 3.1 LLM 失败静默降级，不可观测

**文件**：`nlu/llm_support/ollama_client.py`、`planner/agent.py`、`nlu/llm/*.py`

**问题描述**（已核实细化）：  
`ollama_client.py` 本身已有 `timeout` 参数，这一点优于初始描述。**真正的问题在上层**：`planner/agent.py` 的 `_fallback_question_for_paths` 在 LLM 失败后直接返回硬编码字符串，**无任何 warning/error 级别日志**。`nlu/llm/` 下的多个调用点同样存在 `except Exception` 吞掉失败后静默降级的模式。

**影响**：LLM 服务不可用时系统表面上仍然运行，但实际已完全降级为规则系统，用户和运维人员均无感知，排查困难。

**建议**：在所有降级路径上补充 `logging.warning`，并在 `progress_cb` 中暴露 `llm_degraded` 状态。超时重试属于锦上添花，可后续处理。

---

### 3.2 BERT 训练数据以合成为主，真实鲁棒性未验证

**文件**：`nlu/training/bert_lab/`

**问题描述**：  
BERT 提取器的训练数据由 `bert_lab_data.py` 合成生成。README 明确说明"real-world robustness still unverified"。在真实用户输入（拼写错误、混合语言、非标准表达）下，BERT 提取器的可靠性未知。

**影响**：NLU 在生产场景下的精度缺乏保证。

---

### 3.3 多生产者置信度冲突时缺乏用户可见反馈

**文件**：`core/orchestrator/arbiter.py`

**问题描述**：  
当 BERT 和 LLM 对同一字段的提取结果置信度相近但值不同时，`arbiter.py` 按优先级静默选取胜者，用户不会得到任何提示。只有完全平局时才会触发 `E_SAME_PRIORITY_CONFLICT` 拒绝。

**影响**：低置信度情况下可能静默写入错误值，用户无感知。

**建议**：设置置信度阈值，低于阈值时升级为向用户确认。

---

### 3.4 `nlu/llm/slot_frame.py` 体量过大

**文件**：`nlu/llm/slot_frame.py`（72157 字节，约 1800+ 行）

**问题描述**：  
单文件过大，包含了槽提取的所有 prompt 构建、解析、验证逻辑，难以单独测试和维护。

---

### 3.5 LLM 知识截止问题

**问题描述**：  
系统依赖 LLM（Ollama 本地模型或 DeepSeek API）理解 Geant4 领域概念，但 LLM 的训练数据有截止日期，对最新 Geant4 版本的物理过程、几何体类型可能存在知识盲区。RAG 尚未实现，无法补充最新知识。

---

## 4. 对话与规划层问题

### 4.1 `planner/agent.py` 中存在大量硬编码的字段→问句映射

**文件**：`planner/agent.py`，`_fallback_question_for_paths` 函数（约 60 行硬编码映射）

**问题描述**：  
中英文的回退问句以硬编码字符串的形式写死在代码中，与 `core/config/field_registry.py` 的字段定义相互独立，形成知识重复。新增字段时需要同时更新两处，且极易遗漏。

```python
if "source.type" in path_set:
    return "还需要确认源类型，例如点源、束流或各向同性源。"
```

**建议**：从 `field_registry` 动态生成回退问句模板，消除硬编码。

---

### 4.2 `question_planner.py` 每轮最多追问 2 个字段，硬编码限制

**文件**：`planner/question_planner.py`，第 94 行

```python
if len(planned) >= 2:
    break
```

该限制无法通过配置调整，在某些场景（如批量初始化）下可能导致不必要的多轮交互。

---

### 4.3 对话历史截断策略过于简单

**文件**：`core/dialogue/state.py`

`build_raw_dialogue` 只保留最近 12 条历史，`dialogue_memory` 只保留最近 8 条。对于长对话或用户多次修改的场景，早期确认的信息可能丢失，导致上下文理解退化。

---

### 4.4 `naturalize_response` 的有效性验证规则有限

**文件**：`planner/agent.py`，`_is_invalid_naturalization` 函数

**问题描述**：  
LLM 改写后的合法性验证仅检查：是否包含内部字段名、是否语言不匹配、以及特定 action 的关键词检查。这些规则无法捕捉 LLM 输出中的事实幻觉（hallucination）或语义漂移。

---

## 5. 运行时与执行层问题

### 5.1 真实 Geant4 执行路径存在但未成为默认入口

**文件**：`mcp/geant4/adapter.py`、`runtime/geant4_local_app/main.cc`

> ⚠️ **原始描述已修正**：初始版本错误地说"没有真实执行路径"，经代码核实不准确。

**实际现状**：  
- `LocalProcessGeant4Adapter`（`adapter.py:197`）已完整实现：通过 `subprocess.run` 启动本地 Geant4 进程，将 `runtime_payload` 序列化为临时 JSON 文件后以 `--config` 参数传入。  
- `runtime/geant4_local_app/main.cc`（约 1105 行，`kResultSchemaVersion = "2026-04-14.v7"`）是真实的 Geant4 C++ 驱动，支持多种几何结构、粒子源和物理列表，处于活跃维护状态。  
- **真正的问题**：`Geant4McpServer.__init__` 默认仍然使用 `InMemoryGeant4Adapter()`，`LocalProcessGeant4Adapter` 没有标准化的启动入口，只在部分测试中被实例化。

**影响**：真实仿真路径已存在，但对使用者不透明，没有 live smoke test 验证其可用性。

**建议**：  
1. 为 `LocalProcessGeant4Adapter` 提供标准化配置入口（环境变量或 config 文件）；  
2. 添加 live smoke test，在 CI 或本地验证 `config → subprocess → result` 全链路可通；  
3. 在文档中说明如何切换到真实执行路径。

---

### 5.2 没有 Geant4 宏文件或 C++ 代码生成器

**问题描述**：  
系统可以收集完整的仿真配置，但无法将其转换为 Geant4 可执行的 `.mac` 宏文件或 C++ detector construction 代码。输出仅为项目内部的 JSON config dict。

**建议**：实现 `generate_macro(config) -> str` 和 `export_cpp_config(config) -> str`，作为 FINALIZE 阶段的输出产物。

---

### 5.3 会话状态无跨进程持久化

**文件**：`core/orchestrator/session_manager.py`，第 80 行

```python
SESSIONS: dict[str, SessionState] = {}
```

**问题描述**：  
所有会话状态存储在进程内存中。服务重启、进程崩溃或水平扩展（多进程部署）都会导致所有会话丢失。

**建议**：设计 `SessionStore` 协议接口，提供 SQLite（单机）和 Redis（分布式）两种后端实现。

---

### 5.4 MCP 工具集过少，功能覆盖不足

**文件**：`mcp/geant4/tools.py`

**问题描述**：  
当前只有 5 个工具：`get_runtime_state`、`apply_config_patch`、`initialize_run`、`run_beam`、`get_last_log`。缺少：
- 配置校验工具（`validate_config`）
- 几何信息查询（`list_geometries`、`describe_geometry`）
- 知识问答（`query_knowledge`，需 RAG 支持）
- 宏文件生成（`generate_macro`）

---

### 5.5 Python 侧运行时类型层与 C++ 实现层缺乏自动化 schema 对齐验证

**文件**：`core/runtime/types.py`、`core/simulation/bridge.py`、`runtime/geant4_local_app/main.cc`

> ⚠️ **原始描述已修正**：初始版本说"runtime 只有类型骨架"，经代码核实不完整。

**实际现状**：  
- `core/runtime/types.py` 是 Python 侧接口契约层（`Geant4RuntimePhase`、`RuntimeStateSnapshot`、`ExecutionObservation`），设计上不需要"实现"。  
- 真实运行时逻辑分布在两处：`mcp/geant4/adapter.py`（Python 进程管理）和 `runtime/geant4_local_app/main.cc`（C++ 执行引擎）。  
- **真正的问题**：Python 的 `build_runtime_payload()` 输出 schema 与 C++ `main.cc` 中的 `RuntimeConfig` struct 字段（如 `geometry_structure`、`source_type`、`energy_mev`）的一致性完全依赖人工维护，没有自动化对齐验证。

**影响**：任何一侧的字段重命名或新增都可能导致静默的运行时不匹配，仅在实际执行 Geant4 时才能发现。

**建议**：增加 schema 兼容性测试，对比 `build_runtime_payload()` 的输出键集合与 `main.cc` 中已知解析字段，产生不匹配时测试失败。

---

## 6. 知识层问题

### 6.1 RAG 是占位符，无法回答超出硬编码范围的领域问题

**文件**：`knowledge/rag/`（目录为空占位）

**问题描述**：  
`knowledge/rag/` 目录存在但没有任何检索实现。系统无法回答"FTFP_BERT 适用于哪些场景？"、"G4_WATER 的密度是多少？"等超出硬编码提示词范围的问题，这类问题完全依赖 LLM 的参数记忆，存在幻觉风险。

**建议**：基于现有 `knowledge/data/` 和 Geant4 官方文档构建 FAISS 或 BM25 检索索引。

---

### 6.2 物理列表是参考列表，非完整集合

**文件**：`knowledge/data/physics_lists.json`

README 明确说明"Physics lists are reference-only: fetched from official reference list, not a complete superset"。用户输入的合法物理列表名可能被误判为无效。

---

### 6.3 材料→体积映射需手工指定 `volume_names`

**文件**：`core/validation/validator_gate.py`，`validate_layer_b_consistency`

**问题描述**：  
材料与体积的映射关系（`volume_material_map`）需要用户明确提供体积名称，系统无法从几何配置中自动推断所有可用体积名。对于复杂几何（ring、grid、nest），体积命名规则不透明。

---

### 6.4 输出格式枚举中的 `json` 是项目自定义扩展

**文件**：`knowledge/data/output_formats.json`

`json` 格式不是 Geant4 官方分析框架支持的格式，是项目本地扩展。没有文档说明该格式的实际写入实现是否存在，也没有与 `root/csv/hdf5/xml` 统一处理的代码路径。

---

## 7. 测试与可维护性问题

### 7.1 端到端集成测试依赖 LLM 可用性

**文件**：`tests/test_smoke_no_ollama.py`（59881 字节）、`tests/test_pipeline_selector_integration.py`（42318 字节）

**问题描述**：  
大型集成测试文件以 mock 方式绕过 LLM 调用，但 mock 逻辑本身占用了大量测试代码。真正端到端的 LLM 路径没有自动化回归覆盖，只能依赖手动测试。

---

### 7.2 测试文件体量过大，职责不清晰

`test_smoke_no_ollama.py`（60KB）、`test_pipeline_selector_integration.py`（42KB）、`test_llm_slot_frame.py`（47KB）单文件过大，难以定位失败原因，CI 运行时间长。

---

### 7.3 没有性能基准和响应时延 SLA 定义

**问题描述**：  
系统每轮调用可能触发多次 LLM 请求（slot_frame + normalizer + recommender + naturalize_response + interpreter），总延迟可能超过 10 秒，但没有任何性能基准测试或超时 SLA。用户体验无法量化评估。

---

### 7.4 `remove` 操作无测试覆盖

如前文第 1.4 节所述，`op="remove"` 被静默跳过，但测试中没有专门验证删除意图被正确拒绝或有明确警告的用例。

---

### 7.5 `audit_trail` 只写不读，审计数据无消费路径

**文件**：`core/audit/audit_log.py`、`core/orchestrator/session_manager.py`

每轮结束都会调用 `append_audit_entry`，但没有任何代码读取或展示 `audit_trail` 的内容，也没有导出接口。审计数据处于"只写"状态。

---

## 8. UI 与部署问题

### 8.1 Chromium 桌面壳处于 Stage-1 阶段，未完成

**文件**：`ui/desktop/`

README 描述其为"Stage-1 desktop shell migration target"，实际代码尚未完成 Python runtime bridge 的集成。`ui/launch/runtime_bridge.py` 的实现状态未知。

---

### 8.2 Web UI 与后端通过 HTTP 轮询，无流式推送

**文件**：`ui/web/`

**问题描述**：  
当前 web UI 通过普通 HTTP 请求与后端交互。每次 LLM 调用可能有数秒延迟，用户侧无进度反馈（除了 `progress_cb` 机制，但 UI 层是否实际消费尚不确定）。

**建议**：引入 SSE（Server-Sent Events）或 WebSocket，将 `progress_cb` 的各阶段进度实时推送到前端。

---

### 8.3 `SESSIONS` 字典不线程安全

**文件**：`core/orchestrator/session_manager.py`，第 80 行

```python
SESSIONS: dict[str, SessionState] = {}
```

在多线程 Web 服务器（如 gunicorn + threaded workers）下，对 `SESSIONS` 的并发读写没有任何锁保护，存在竞态条件风险。

---

## 9. 问题优先级汇总表

> 表格已按"先修低成本真 bug，再推进 typed boundary，大型重构最后"重新排序。

| # | 问题 | 分类 | 严重程度 | 修复难度 | 执行阶段 |
|---|---|---|---|---|---|
| 1.4 | `remove` 操作静默跳过（**真 bug**） | 架构 | 🔴 高 | **极低** | 阶段一 |
| 7.4 | `remove` 操作无测试覆盖 | 测试 | 🔴 高 | **极低** | 阶段一（与上条同步）|
| 2.4 | `source_meta` 变量名覆盖（**真 bug**） | 类型系统 | 🔴 高 | **极低** | 阶段一 |
| 3.1 | LLM 失败静默降级，不可观测 | NLU | 🔴 高 | 低 | 阶段一 |
| 8.3 | `SESSIONS` 不线程安全 | 部署 | � 高 | 低 | 阶段一 |
| 2.2 | `slot_debug` 裸 dict 防御检查扩散 | 类型系统 | 🟠 中 | 中 | 阶段二 |
| 5.1 | 真实执行路径未成为默认入口（已修正） | 运行时 | 🟠 中 | 低 | 阶段二 |
| 5.5 | Python/C++ schema 对齐无验证（已修正） | 运行时 | 🟠 中 | 低 | 阶段二 |
| 5.3 | 会话无跨进程持久化 | 运行时 | 🟠 中 | 中 | 阶段二 |
| 2.1 | 核心配置是裸 `dict`（渐进迁移） | 类型系统 | 🟠 中 | 高 | 阶段二~三 |
| 3.3 | 低置信度冲突静默选取 | NLU | 🟠 中 | 中 | 阶段二 |
| 4.1 | 硬编码字段→问句映射 | 对话规划 | 🟠 中 | 低 | 阶段二 |
| 5.2 | 无宏文件/C++ 代码生成 | 运行时 | 🟠 中 | 高 | 阶段四 |
| 6.1 | RAG 是占位符 | 知识层 | 🟠 中 | 高 | 阶段四 |
| 1.1 | `session_manager.py` 单体架构 | 架构 | 🟠 中 | 高 | 阶段三 |
| 1.3 | Legacy/v2 并存，边界未文档化 | 架构 | 🟡 低 | 低 | 阶段二 |
| 1.2 | 线性 FSM 限制跨阶段输入 | 架构 | 🟡 低 | 高 | 阶段四 |
| 2.3 | 向量值用带标签 dict 表示 | 类型系统 | 🟡 低 | 低 | 阶段三 |
| 2.5 | `dialogue_summary` 条目无类型约束 | 类型系统 | 🟡 低 | 低 | 阶段三 |
| 3.2 | BERT 合成数据鲁棒性未验证 | NLU | 🟡 低 | 高 | 阶段四 |
| 3.4 | `slot_frame.py` 单文件过大 | NLU | 🟡 低 | 中 | 阶段三 |
| 4.2 | 追问字段数硬编码上限 | 对话规划 | 🟡 低 | 低 | 阶段二 |
| 4.3 | 对话历史截断策略简单 | 对话规划 | 🟡 低 | 低 | 阶段三 |
| 5.4 | MCP 工具集过少 | 运行时 | 🟡 低 | 中 | 阶段四 |
| 6.2 | 物理列表非完整集合 | 知识层 | 🟡 低 | 低 | 阶段三 |
| 6.3 | 材料→体积映射需手工指定 | 知识层 | 🟡 低 | 中 | 阶段三 |
| 7.3 | 无性能基准和响应延迟 SLA | 测试 | 🟡 低 | 中 | 阶段三 |
| 7.5 | `audit_trail` 只写不读 | 测试 | 🟡 低 | 低 | 阶段三 |
| 8.1 | 桌面壳未完成 | UI | 🟡 低 | 高 | 阶段四 |
| 8.2 | Web UI 无流式推送 | UI | 🟡 低 | 中 | 阶段三 |

---

## 10. 推荐执行顺序

> 原则：**低成本真 bug 优先 → typed boundary 渐进推进 → workflow 回归稳住 → 大型重构最后**。  
> 当前主线：`typed boundary + workflow regression + runtime bridge 可真实执行`，不被"大而全重构"带偏。

### 阶段一：立即处理（成本极低，收益立竿见影）

1. **修复 `remove` 操作静默跳过**（1.4 / 7.4）  
   `_apply_updates` 中将 `continue` 替换为 `remove_path(config, upd.path)`；补充对应测试用例。

2. **修复 `source_meta` 变量名覆盖**（2.4）  
   `_v2_compile_missing_paths` 中将内层变量重命名为 `spatial_source_meta`，消除逻辑歧义。

3. **补充 LLM 降级日志**（3.1）  
   所有 `except Exception` 降级路径加 `logging.warning`；`progress_cb` 中暴露 `llm_degraded` 状态字段。

4. **`SESSIONS` 加锁**（8.3）  
   为读写加 `threading.Lock`；或改为 per-request context，为后续持久化做准备。

### 阶段二：近期处理（配合当前 workflow 回归方向）

5. **固化 workflow 回归测试**  
   `config → SimulationSpec → RuntimePayload → run_summary → SimulationResult → MCP payload` 全链路有确定性测试覆盖。

6. **`slot_debug` 收成 typed meta**（2.2）  
   定义 `PipelineDebug` dataclass，停止向 `slot_debug` 裸 dict 写入新键。

7. **`LocalProcessGeant4Adapter` 标准化入口 + live smoke test**（5.1）  
   增加环境变量配置入口；补充 `config → subprocess → result` 集成测试。

8. **Python/C++ schema 对齐测试**（5.5）  
   对比 `build_runtime_payload()` 输出键集合与 `main.cc` 中 `RuntimeConfig` 字段，加自动化断言。

9. **文档化 legacy/v2 切换边界**（1.3）  
   在 `core/pipelines/` 中补充说明文档，补充 legacy 专项测试矩阵。

### 阶段三：中期处理（需要设计工作，workflow 稳定后推进）

10. **`session_manager.py` 拆分**（1.1）  
    逐步提取 `NLUDispatcher`、`CandidateAssembler`、`ConfigApplicator` 等组件，由薄层 `TurnProcessor` 串联。

11. **config dict 边界层收口**（2.1）  
    在 `SlotFrame → SimulationConfig → RuntimePayload` 边界优先引入 typed spec，内部 config 随时间渐进迁移。

12. **`SessionStore` 持久化协议**（5.3）  
    设计协议接口，先实现 SQLite 后端，为多进程部署做准备。

### 阶段四：长期目标

- RAG 知识检索（6.1）
- 宏文件 / C++ 代码生成（5.2）
- 线性 FSM 改为槽依赖图（1.2）
- 桌面壳完成（8.1）

---

*本文档由 Cascade 基于代码库静态分析生成，已结合运行时代码核实进行修订（2026-04-21）。  
建议将第 9 节优先级表作为 backlog 参考，第 10 节作为实际执行路线图。*
