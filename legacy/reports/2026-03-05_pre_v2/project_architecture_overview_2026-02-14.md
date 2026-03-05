# Geant4-Agent 项目架构总览（2026-02-14）

## 1. 项目目标
本项目用于把用户自然语言需求转换为可审计的 Geant4 最小配置草案，并通过理论规则做一致性与闭环校验。当前重点是：
- 减少 LLM 幻觉导致的参数漂移。
- 在多轮对话中保持已确认约束不被覆盖。
- 让配置生成过程可解释、可回溯、可复现。

## 2. 当前架构主线
当前已实现双路径：
- `Legacy Path`：旧版 `ui/web/server.py::step()` 逻辑（可回退）。
- `V0.2 Strict Path`：新 Orchestrator 逻辑（默认开启，`strict_mode=true`）。

推荐始终使用 `V0.2 Strict Path`。

## 3. 目录分层
```text
core/
  orchestrator/
    types.py
    path_ops.py
    phase_machine.py
    constraint_ledger.py
    intent_guard.py
    arbiter.py
    session_manager.py
  validation/
    error_codes.py
    minimal_schema.py
    geometry_registry.py
    validator_gate.py
  audit/
    audit_log.py
  schema/
    geant4_min_config.schema.json

nlu/
  llm/
    normalizer.py
    recommender.py
  bert/
    extractor.py
  bert_lab/
    semantic.py
    postprocess.py
    llm_bridge.py
    ...

ui/
  web/
    server.py
    app.js
    index.html
```

## 4. 核心设计原则
1. 单一真值源：会话状态只由 `SessionState` 持有并由 Orchestrator 写入。  
2. 提案与落盘分离：LLM/BERT 只产出候选更新，不直接改配置。  
3. 约束优先：锁字段默认不可被非显式改写覆盖。  
4. 门禁优先：所有更新先仲裁，再过三层校验，最后才提交。  
5. 审计优先：每轮保留前后差异、接受与拒绝原因。

## 5. V0.2 关键模块职责
### 5.1 Orchestrator 层
- `session_manager.py`
  - 统一处理每轮输入。
  - 调用 Normalizer/BERT/Recommender 生成候选。
  - 调用 Arbiter + Validator。
  - 执行提交、回滚、追问、审计。
- `phase_machine.py`
  - 管理 phase 推进规则（geometry -> materials -> source -> physics -> output -> finalize）。
- `constraint_ledger.py`
  - 管理锁约束（路径、值、scope、reason_code）。
- `intent_guard.py`
  - 判定锁字段是否允许覆盖。
- `arbiter.py`
  - 执行冲突裁决与候选筛选。

### 5.2 Validation 层
- `minimal_schema.py`
  - 唯一最小闭环 required 权威源。
- `geometry_registry.py`
  - 结构族作用域白名单与 required 规则。
- `validator_gate.py`
  - Layer A 参数合法性。
  - Layer B 结构一致性与命名绑定。
  - Layer C 最小闭环完整性。

### 5.3 NLU 层
- `nlu/llm/normalizer.py`
  - 把用户输入转为受控归一化文本与结构化 intent/target_paths。
- `nlu/bert/extractor.py`
  - 从归一化文本提取几何/源/材料等候选更新。
- `nlu/llm/recommender.py`
  - 物理列表推荐候选（主选、备选、理由、覆盖过程）。
- `nlu/bert_lab/*`
  - 保留现有模型能力与工具链。

### 5.4 审计层
- `core/audit/audit_log.py`
  - 记录 `accepted_updates`、`rejected_updates`、`applied_rules`、`violations`、`config_diff`。

## 6. 运行时调用序列
```text
UI /api/step
  -> Orchestrator.process_turn
    -> LLM Normalizer (intent + normalized_text)
    -> BERT Extractor (candidate updates)
    -> LLM Recommender (physics candidate updates)
    -> Arbiter (accept/reject by deterministic rules)
    -> Validator Gate (A/B/C)
    -> Commit or Rollback
    -> Audit append
  -> response(config + missing + violations + audit metadata)
```

## 7. 状态模型
`SessionState` 主要字段：
- `phase`：当前阶段。
- `config`：当前生效配置（真值）。
- `constraint_ledger`：锁字段约束表。
- `field_sources`：字段来源追踪（谁写入）。
- `audit_trail`：每轮审计日志。
- `history`：对话历史。

## 8. API 入口
- `POST /api/step`
  - 默认走 `strict_mode=true`（V0.2）。
- `POST /api/solve`
  - 旧单轮求解接口（保留兼容）。
- `POST /api/reset`
  - 重置会话。
- `POST /api/audit`
  - 导出指定会话审计轨迹。
- `GET /api/runtime`
  - 查询当前 Ollama 配置与可选配置。
- `POST /api/runtime`
  - 切换运行时 Ollama 配置。

## 9. 错误码体系（核心）
- `E_LOCKED_FIELD_OVERRIDE`
- `E_SAME_PRIORITY_CONFLICT`
- `E_SCOPE_LEAK`
- `E_NAME_BINDING`
- `E_REQUIRED_MISSING`
- `E_TYPE_INVALID`
- `E_RANGE_INVALID`
- `E_SOURCE_TYPE_MISSING`
- `E_PHASE_WRITE_FORBIDDEN`
- `E_CANDIDATE_REJECTED_BY_GATE`

## 10. 当前已知差距
1. Legacy 脚本与新 Orchestrator 评测仍并存，尚未完全统一。  
2. 结构族注册表仍需扩充到更多 Geant4 常见几何族。  
3. 物理列表推荐可进一步加入规则裁决层（更强工程约束）。  
4. 对抗测试集（golden conversation set）需要补全并纳入 CI。

## 11. 推荐使用方式
1. 启动服务：`.\.venv\Scripts\python ui/web/server.py`  
2. 打开：`http://127.0.0.1:8088`  
3. 保持 `strict_mode=true`（前端已默认）。  
4. 使用 `/api/audit` 审查每轮配置变更与拒绝原因。
