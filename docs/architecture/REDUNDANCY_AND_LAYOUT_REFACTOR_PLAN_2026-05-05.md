# Redundancy And Layout Refactor Plan 2026-05-05

## Summary

The project should not start with a large directory move. The current problem is
not merely that there are many folders. The deeper problem is that several
folders and modules now express overlapping responsibilities:

- orchestration policy is concentrated in one large file
- BERT runtime, BERT lab, BERT training, and legacy BERT paths overlap
- prompt text lives in several registries and runtime helpers
- active UI paths coexist with legacy UI wrappers
- safety guards are correct, but policy ownership is scattered

This plan reduces redundancy first, then performs directory cleanup only after
imports and behavior are stable.

## Goals

- Make the active execution path obvious.
- Reduce duplicated concepts before moving files.
- Keep legacy paths available for reference but out of new development.
- Preserve current safety behavior.
- Keep every phase regression-testable.
- Avoid a cosmetic folder reshuffle that breaks imports without improving
  maintainability.

## Non-Goals

- No immediate repository history rewrite.
- No deletion of legacy files before import audit.
- No UI redesign.
- No RAG or dictionary/corpus expansion.
- No BERT retraining.
- No removal of guarded runtime actions.
- No automatic Geant4 run/viewer from normal chat.

## Current Active Path

```text
User text
  -> ui/web or API
  -> planner/runtime intent for read-only or guarded runtime actions
  -> core/orchestrator/session_manager.py
  -> nlu/llm and nlu/bert extraction
  -> core typed config / geometry / source / simulation bridge
  -> mcp/geant4 adapter
  -> runtime/geant4_local_app
  -> structured result / runtime smoke report
```

## Directory Classification

### Active Mainline

These directories remain first-class development targets:

- `core/`
- `nlu/`
- `planner/`
- `runtime/`
- `mcp/`
- `ui/web/`
- `ui/launch/`
- `tests/`
- `tools/`
- `docs/`

### Compatibility Or Legacy

These directories should not receive new product behavior unless there is a
specific compatibility reason:

- `legacy/`
- `nlu/bert_lab/`
- `ui/desktop/`
- `legacy/ui_desktop/`

### Reclassify Before Moving

These directories need import and responsibility review before any move:

- `builder/`
- `knowledge/`
- selected `planner/flows/` modules

## Redundancy Map

### 1. Orchestration Redundancy

Main risk:

- `core/orchestrator/session_manager.py` is responsible for too many policies.

Responsibilities currently mixed together:

- session state lifecycle
- slot-frame merge
- semantic-frame merge
- legacy/v2 switching
- overwrite detection
- delete confirmation
- low-confidence confirmation
- source/geometry dependent staging
- graph override protection
- response rendering

Target split:

- `core/orchestrator/confirmation_policy.py`
- `core/orchestrator/slot_memory.py`
- `core/orchestrator/graph_override_policy.py`
- `core/orchestrator/candidate_pipeline.py`

Rule:

- Extract only. Do not change behavior during the first pass.

### 2. BERT Path Redundancy

Current paths:

- `nlu/bert/`: active lightweight runtime extractor.
- `nlu/bert_lab/`: compatibility and lab-facing re-export path.
- `nlu/training/bert_lab/`: training utilities.
- `legacy/runtime/bert_lab/`: old runtime reference.

Target state:

- Runtime work lands in `nlu/bert/`.
- Training work lands in `nlu/training/bert_lab/`.
- `nlu/bert_lab/` stays compatibility-only until imports are retired.
- `legacy/runtime/bert_lab/` stays archive/reference-only.

First action:

- Run import audit for `nlu.bert_lab`.
- Do not remove shims until tests prove they are unused.

### 3. Prompt Redundancy

Current prompt owners:

- `core/config/prompt_profiles.py`
- `core/config/llm_prompt_registry.py`
- `core/config/prompt_registry.py`
- `nlu/llm_support/llm_bridge.py`
- `planner/agent.py`
- `nlu/llm/recommender.py`
- `planner/flows/min_config_flow.py`

Target state:

- `PromptProfile` owns task prompts.
- Validators own output acceptance.
- Runtime owns facts.
- LLM owns explanation, extraction assistance, routing, and rewrite.

First migration candidates:

- runtime result explanation
- clarification
- normalization
- recommender prompts

Do not migrate the most complex slot/semantic JSON prompts first.

### 4. UI And Launch Redundancy

Active path:

- `ui/web/`
- `ui/launch/`

Compatibility path:

- `ui/desktop/`
- `legacy/ui_desktop/`
- web modules prefixed with `legacy_`

Target state:

- README documents only the active UI launch path.
- Legacy UI wrappers are explicitly marked compatibility-only.
- No UI redesign during this cleanup.

### 5. Guard Redundancy

Safety policy is necessary and should remain strict:

- read config is read-only
- read result is read-only
- run/viewer is explicit and guarded
- overwrite/delete requires confirmation
- low-confidence writes require confirmation

Maintenance issue:

- these protections are implemented across frontend, API, planner, and
  session-manager code.

Target state:

- backend remains authoritative
- frontend remains a UX affordance
- confirmation policy has one owner
- runtime action guard has one owner

## Refactor Phases

### Phase 0: Freeze The Map

Status: mostly done.

- Add maintenance plan.
- Add secret scanner.
- Add local cleanup script.
- Add compatibility notes.

Exit criteria:

- `tools/check_secrets.ps1` passes.
- Git status is clean except ignored local config files.

### Phase 1: Orchestrator Extraction

Scope:

- Extract confirmation and overwrite policy.
- Extract slot memory helpers.
- Extract graph override policy.

No behavior change allowed.

Suggested files:

- `core/orchestrator/confirmation_policy.py`
- `core/orchestrator/slot_memory.py`
- `core/orchestrator/graph_override_policy.py`

Required tests:

```powershell
pytest tests/test_pending_overwrite_flow.py tests/test_workflow_guard_contract.py tests/test_config_summary_api.py -q
```

### Phase 2: Prompt Consolidation

Scope:

- Move low-risk prompts into `PromptProfile`.
- Keep current schemas and validators stable.

Required tests:

```powershell
pytest tests/test_prompt_profiles.py tests/test_dialogue_naturalization.py tests/test_runtime_result_explanation.py -q
```

### Phase 3: BERT Compatibility Audit

Scope:

- Identify active imports from `nlu/bert_lab`.
- Replace active imports with `nlu/bert` or `nlu/training/bert_lab` where safe.
- Keep compatibility shims until all tests pass.

Required checks:

```powershell
rg "nlu\\.bert_lab|from nlu.bert_lab|import nlu.bert_lab" .
pytest -q
```

### Phase 4: Directory Layout Cleanup

Scope:

- Move or archive only paths proven inactive.
- Update README and architecture docs.
- Keep compatibility wrappers when external scripts still depend on them.

Candidate decisions:

- `builder/geometry/` -> archive or keep as prototype
- `knowledge/` -> experimental or archive
- `ui/desktop/` -> compatibility-only

No move should happen without import audit.

### Phase 5: Final Verification

Required checks:

```powershell
powershell -ExecutionPolicy Bypass -File tools\check_secrets.ps1
pytest -q
```

Live LLM and live Geant4 remain opt-in.

## Immediate Next Step

Start with Phase 1.

The first implementation target should be:

```text
core/orchestrator/confirmation_policy.py
```

Move only the policy logic that decides whether a proposed update needs
confirmation. Keep mutation application and session persistence in
`session_manager.py` for now.

## Success Criteria

- `session_manager.py` becomes smaller without changing user-visible behavior.
- New code has one obvious place for overwrite/delete confirmation policy.
- Prompt text becomes easier to locate and validate.
- BERT runtime/training/legacy roles are clear.
- Directory count feels less noisy because inactive paths are explicitly labeled
  before they are moved.
