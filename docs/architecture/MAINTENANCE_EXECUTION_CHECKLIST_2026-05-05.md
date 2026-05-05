# Maintenance Execution Checklist 2026-05-05

This checklist is the short operational version of
`MAINTENANCE_REDUCTION_PLAN_2026-05-05.md`.

## Current Checkpoint

- GitHub `main` has the security and maintenance checkpoint.
- Local secret config files are ignored by Git.
- `tools/check_secrets.ps1` is the required pre-commit/pre-push manual scan.
- `tools/install_git_hooks.ps1` can install hooks, but this Windows Git setup may
  report a `sh.exe` signal pipe error. If that happens, run the secret scanner
  manually before commit and push.
- Phase 1 orchestrator extraction is complete:
  `confirmation_policy.py`, `slot_memory.py`, `graph_override_policy.py`, and
  `candidate_pipeline.py` now own their respective policies/helpers.
- Phase 2 prompt consolidation is mostly complete for low-risk producers:
  response naturalization, physics recommendation, normalization, and
  interpreter prompts now route through `PromptProfile`.
- Phase 3 BERT compatibility audit has started. Active `ui/`, `core/`, `nlu/`,
  `legacy/runtime/`, and `tests/` Python paths no longer import
  `nlu.bert_lab` directly.

## Hard Rules

- Never commit real API keys.
- Keep real keys only in ignored `*.local.json`, `.env.*`, or `secrets/`.
- Do not add new runtime behavior under `legacy/` or `nlu/bert_lab/`.
- Do not reintroduce dictionary/corpus expansion as the main NLU strategy.
- Do not trigger Geant4 run/viewer from normal chat.
- Do not remove safety guards; centralize them so they are easier to audit.

## Active Mainline

- `core/`: typed business logic, geometry/source/simulation bridge, orchestration.
- `nlu/`: LLM/BERT extraction and normalization.
- `runtime/`: local Geant4 runtime app.
- `mcp/`: runtime adapter boundary.
- `ui/web/` and `ui/launch/`: active user-facing UI path.
- `tests/`: regression contracts.
- `tools/`: scanners, evaluators, maintenance scripts.

## Cleanup Priority

1. Keep remaining strict slot/semantic prompt behavior stable; do not rewrite it
   unless tests prove a real maintenance benefit.
2. Clarify or retire compatibility paths after import audit.
3. Audit legacy tool scripts separately before touching `legacy/nlu_bert_lab_tools`.
4. Revisit `session_manager.py` only for clearly bounded extraction, not broad
   opportunistic cleanup.

## Completed Refactor Targets

- `core/orchestrator/confirmation_policy.py`
- `core/orchestrator/slot_memory.py`
- `core/orchestrator/graph_override_policy.py`
- `core/orchestrator/candidate_pipeline.py`

## Current Refactor Target

Continue Phase 2 by migrating low-risk prompt producers to `PromptProfile`.
Prefer response naturalization, runtime/result explanation, clarification, and
recommender prompts before touching strict slot/semantic JSON extraction prompts.

Current follow-up target:

Audit and document `nlu/bert_lab` compatibility shims. Do not delete them while
archived tools still import them.

## Required Checks Before Commit

```powershell
powershell -ExecutionPolicy Bypass -File tools\check_secrets.ps1
```

For confirmation/session refactors:

```powershell
pytest tests/test_pending_overwrite_flow.py tests/test_workflow_guard_contract.py tests/test_config_summary_api.py -q
```

For final checkpoint:

```powershell
pytest -q
```

## Next Decision

The next coding phase should finish the BERT compatibility audit and then move
to directory cleanup only where imports prove a path is compatibility-only.
