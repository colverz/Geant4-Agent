# Legacy Archive

This directory stores archived utilities that are not part of the active strict runtime architecture.

Current active runtime layers remain in:

- `core/`
- `nlu/llm/`
- `nlu/bert/`
- `planner/` for agent/result planning helpers
- `core/geometry/`
- `ui/web/`
- `ui/launch/`
- `runtime/`
- `mcp/`

Archived experimental or evaluation-oriented scripts have been moved to:

- `legacy/nlu_bert_lab_tools/`

These files are preserved for reference and reproducibility, but they are not on the active execution path for the strict multi-turn orchestrator.

## Maintenance Rule

Do not add new product behavior here. If a legacy file is needed by an active
import, keep it stable and document the compatibility reason. New geometry,
runtime, prompt, or web work should land in the active layers above.
