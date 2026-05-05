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

1. Extract confirmation and overwrite policy from `core/orchestrator/session_manager.py`.
2. Extract slot memory merge helpers from `session_manager.py`.
3. Extract graph override protection from `session_manager.py`.
4. Move low-risk prompt strings into `PromptProfile`.
5. Clarify or retire compatibility paths after import audit.

## First Refactor Target

Create `core/orchestrator/confirmation_policy.py` and move only behavior-preserving
logic related to:

- implicit overwrite detection
- pending overwrite extraction
- delete confirmation
- low-confidence confirmation boundaries

Do not change user-facing behavior in this step.

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

The next coding phase should begin with `confirmation_policy.py`, not directory
moves. Directory cleanup should wait until behavior-preserving extraction has
reduced the size and ambiguity of `session_manager.py`.
