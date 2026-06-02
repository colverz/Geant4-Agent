# v2 Legacy Candidates

Date: 2026-05-21

These files or areas should not drive the v3 agent. They may remain temporarily for compatibility, tests, or UI fallback.

## Freeze First

| Path | Reason | Action |
| --- | --- | --- |
| `core/orchestrator/session_manager.py` | Too many responsibilities: dialogue, NLU, state mutation, confirmation, runtime, and answer generation are mixed. | Keep as v2 compatibility entry point until v3 API is ready. |
| `core/orchestrator/candidate_pipeline.py` | Coupled to v2 pipeline behavior. | Freeze, extract only reusable pure logic if needed. |
| `core/orchestrator/pipeline_debug.py` | Debug surface for old pipeline. | Keep for v2 diagnostics only. |
| `core/orchestrator/graph_override_policy.py` | Old override policy. | Review after v3 graph is stable. |
| `core/orchestrator/slot_memory.py` | Slot-memory pattern contributes to non-agentic behavior. | Do not use as v3 state authority. |
| `planner/agent.py` | Hardcoded planner behavior. | Replace with v3 workflow planning. |
| `planner/question_planner.py` | Slot-fill questioning behavior. | Extract wording only if useful. |

## Move Later

| Path | Reason | Move target |
| --- | --- | --- |
| `nlu/bert_lab/*` | Compatibility/shim area, not v3 primary NLU. | `legacy/nlu/bert_lab/` or archive after dependency check. |
| `nlu/training/bert_lab/*` | Training/model assets, not runtime source. | Archive or local asset area after dependency check. |
| `ui/web/legacy_*` | Legacy web entry points. | `legacy/ui/web/` after v3 endpoint is default. |
| `ui/desktop/*` | Compatibility desktop UI path. | `legacy/ui/desktop/` if no active workflow depends on it. |
| Old benchmark/report docs superseded by industrial runtime benchmark | Can confuse acceptance criteria. | `legacy/docs/` after marking superseded. |

## Generated or Local Assets

| Asset | Rule |
| --- | --- |
| `docs/eval/golden/industrial_runtime/*.golden.json` | Keep as evaluation artifacts. Do not treat as source code. |
| Runtime reports under `docs/reports/` | Keep generated reports separate from architecture source. |
| Local model outputs or temporary files | Do not migrate into v3 source tree. |

## Migration Conditions

Before moving any candidate into `legacy/`:

1. `rg` confirms current imports and references.
2. v3 replacement exists for the user-facing behavior.
3. compatibility shims are added where public imports may break.
4. targeted tests pass.
5. agent smoke and relevant casebank regression pass.

No deletion should happen during the first v3 migration pass.
