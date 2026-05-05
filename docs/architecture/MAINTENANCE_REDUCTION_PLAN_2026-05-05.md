# Maintenance Reduction Plan 2026-05-05

## Summary

The project direction is still valid: natural language configuration flows into typed
simulation boundaries, guarded runtime actions, Geant4 execution, and structured
results. The current maintenance risk is not a single broken feature. The risk is
that too many historical paths, compatibility layers, prompt locations, and guard
checks now coexist without a clear ownership map.

This plan keeps the mainline intact and reduces maintenance cost gradually. It
prioritizes no-behavior-change cleanup, boundary clarification, and small
extractive refactors before any directory move or legacy removal.

## Mainline To Preserve

- `core/`: business logic, typed contracts, orchestration, validation, geometry,
  source, dialogue, and simulation bridge.
- `nlu/`: LLM/BERT extraction, normalization, and runtime semantic helpers.
- `runtime/`: local Geant4 application and runtime-side execution assets.
- `mcp/`: MCP/runtime adapter boundary.
- `ui/web/`: active browser UI and HTTP handlers.
- `ui/launch/`: active local launch wrappers.
- `tests/`: regression and workflow contract tests.
- `docs/`: design notes, evaluation casebanks, and reports.
- `tools/`: evaluation and maintenance scripts.

## Directory Issues

### Keep As Active

- `core/`
- `nlu/`
- `runtime/`
- `mcp/`
- `ui/web/`
- `ui/launch/`
- `tests/`
- `docs/`
- `tools/`

### Reclassify Or Shrink

- `builder/`: early geometry-builder work. New geometry work should target
  `core/geometry/`. Keep only if an active import still depends on it; otherwise
  mark as prototype or move under `legacy/` in a later phase.
- `knowledge/`: historical knowledge/RAG-oriented material. Since the project
  should not drift back into dictionary/large-corpus behavior, keep this out of
  the main runtime path and mark it experimental unless there is a concrete
  active dependency.
- `planner/`: still useful for agent/result intent work, but overlaps with
  `core/orchestrator/`. Long term, either define it as the agent planning layer
  or move stable components under `core/planner/`.
- `legacy/`: should remain available for reference and compatibility, but must
  not be treated as a new development entry point.

### Local Noise Only

These paths should remain ignored and may be cleaned locally when needed:

- `.venv/`
- `.pytest_cache/`
- `runtime_artifacts/`
- `ui/desktop/node_modules/`
- `runtime/geant4_local_app/build/`
- `__pycache__/`

## Structural Duplication

### Orchestration

`core/orchestrator/session_manager.py` has become the largest maintenance risk.
It currently mixes:

- session state access
- LLM slot extraction flow
- semantic extraction flow
- legacy/v2 switch behavior
- slot memory merge
- overwrite detection
- pending confirmation
- low-confidence staging
- source/geometry dependent staging
- graph override policy
- final response assembly

This should be split by policy, not rewritten. The first extraction targets are:

- `core/orchestrator/confirmation_policy.py`
- `core/orchestrator/slot_memory.py`
- `core/orchestrator/graph_override_policy.py`
- `core/orchestrator/candidate_pipeline.py`

Each extraction must preserve existing behavior and be covered by existing
regression tests before any further changes.

### BERT And Runtime Paths

There are several BERT-related paths:

- `nlu/bert/`: current lightweight runtime extractor path.
- `nlu/bert_lab/`: compatibility shims and old lab-facing imports.
- `nlu/training/bert_lab/`: training-side code and small examples.
- `legacy/runtime/bert_lab/`: old runtime implementation.

The project should define one supported runtime path and mark the others as
compatibility, training, or archive paths. Do not delete these paths until active
imports are checked.

### Prompt Locations

`core/config/prompt_profiles.py` is the desired prompt boundary, but prompt text
still exists in multiple places:

- `core/config/llm_prompt_registry.py`
- `core/config/prompt_registry.py`
- `nlu/llm_support/llm_bridge.py`
- `planner/agent.py`
- `nlu/llm/recommender.py`
- `planner/flows/min_config_flow.py`

Prompt engineering remains part of the mainline, but prompts must be task
profiles with validators, not ad-hoc strings or phrase dictionaries.

### UI And Launch Paths

The active UI path is `ui/web/` plus `ui/launch/`. The project still contains
desktop and legacy wrappers:

- `ui/desktop/`
- top-level UI shims such as `ui/browser_shell.py`
- `legacy/ui_desktop/`
- web modules prefixed with `legacy_`

These should be documented as compatibility paths before any removal or move.

## Guard And Safety Review

The safety strategy is correct:

- asking about current config must be read-only
- asking about last result must be read-only
- Geant4 run/viewer actions must require explicit user intent or UI controls
- low-confidence writes must require confirmation
- deletes and overwrites must require confirmation

The maintenance problem is that these protections are distributed across many
modules. The goal is not to remove guardrails. The goal is to centralize policy
ownership so future changes are auditable.

Candidate consolidation targets:

- runtime action safety: keep backend authoritative; simplify frontend to UI
  affordance only
- overwrite/delete confirmation: move policy into `confirmation_policy.py`
- source/geometry dependent pending logic: keep behavior, isolate policy
- graph override protection: move to `graph_override_policy.py`

## Phased Implementation

### P0: Documentation Checkpoint

- Add this maintenance plan.
- Keep behavior unchanged.
- Confirm git status and existing tests remain untouched.

### P1: Local And Directory Hygiene

- Add a local cleanup script for ignored artifacts only.
- Add README/deprecation notes to compatibility-heavy directories.
- Do not delete tracked source files.
- Do not move imports yet.

### P2: Orchestrator Boundary Extraction

- Extract confirmation and overwrite policy from `session_manager.py`.
- Extract slot memory merge helpers.
- Extract graph override policy.
- Keep public behavior and tests unchanged.

### P3: Prompt Boundary Consolidation

- Move low-risk prompts into `PromptProfile` first.
- Start with runtime result explanation, normalization, recommender, and
  clarification prompts.
- Keep strict slot/semantic JSON schemas stable.

### P4: Legacy Path Clarification

- Mark `nlu/bert_lab/` compatibility shims explicitly.
- Mark `legacy/` as reference-only.
- Check active imports before moving or deleting anything.

### P5: UI Entry Consolidation

- Define canonical UI entry points in docs and README.
- Keep legacy wrappers only where active scripts still depend on them.
- Avoid UI redesign during this phase.

## Regression Policy

Every behavior-affecting phase must run targeted tests first, then full tests.

Suggested targeted set:

```powershell
pytest tests/test_runtime_intent.py tests/test_workflow_guard_contract.py tests/test_config_summary_api.py tests/test_geant4_web_api.py -q
```

For prompt and LLM boundary changes:

```powershell
pytest tests/test_prompt_profiles.py tests/test_llm_slot_frame.py tests/test_llm_semantic_frame.py tests/test_v2_real_prompt_regression.py -q
```

For final validation:

```powershell
pytest -q
```

Live Geant4 and live LLM tests remain opt-in and must not run implicitly.

## Non-Goals

- No immediate large directory move.
- No legacy deletion without import audit.
- No UI redesign.
- No BERT retraining.
- No RAG reintroduction.
- No dictionary-style corpus expansion.
- No automatic high-cost Geant4 run from normal chat.

## Success Criteria

- New contributors can identify the active runtime/UI/NLU paths quickly.
- `session_manager.py` becomes smaller through behavior-preserving extraction.
- Prompt ownership is visible through `PromptProfile`.
- Guard policy remains strict but easier to inspect.
- Legacy paths are available for reference without polluting the mainline.
