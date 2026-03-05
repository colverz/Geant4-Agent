# Geant4-Agent Code Review & Feasibility (2026-03-05)

## Scope
- Review target: strict orchestration path (`core/orchestrator/session_manager.py`) + NLU fallback path (`nlu/llm/normalizer.py`, `nlu/bert/extractor.py`) + dialogue rendering (`core/dialogue/renderer.py`).
- Goal: robustness for real multi-turn usage, especially Chinese input and no-Ollama fallback.

## Findings (Ordered by Severity)

1. `High` - Hidden LLM dependency in fallback path (fixed)
- Symptom: when `llm_router=false`, system still called `normalize_user_turn()->chat()` implicitly.
- Risk: offline instability and inconsistent behavior.
- Fix:
  - Added explicit switch `enable_llm` in `normalize_user_turn`.
  - `session_manager` now passes `enable_llm=bool(normalize_input and llm_router)`.

2. `High` - Intent false positive to `QUESTION` for payload text with `?` (fixed)
- Symptom: payload-like inputs containing `?` could be misrouted as question-only turns.
- Risk: repeated clarifications and missed parameter commits.
- Fix:
  - Split question detection into punctuation and semantic patterns.
  - Added payload signal patterns (units/materials/particles/vectors); punctuation question only applies when payload signal is absent.

3. `Medium` - Chinese target-path inference under-covered geometry/material fields (fixed)
- Symptom: Chinese inputs could set only partial target paths (e.g., `module_z` via `+z` false hit), causing extractor updates to be filtered out.
- Fix:
  - Expanded Chinese hints for geometry families/materials.
  - Removed axis-direction hints from `module_x/y/z` keys.
  - Relaxed 3D size regex boundaries.

4. `Medium` - Tubs parameter extraction missing (`child_rmax`, `child_hz`) in Chinese (fixed)
- Symptom: `半径` / `半长` text did not consistently fill Tubs family required params.
- Fix:
  - Added named-length parsing and fallback writes for `geometry.params.child_rmax` and `geometry.params.child_hz`.

5. `Medium` - User-facing text still leaked internal field names in some summaries (fixed)
- Symptom: `materials.selection_source`-like text surfaced in user replies.
- Fix:
  - Added friendly labels for metadata fields in `core/config/field_registry.py`.
  - Added field-name normalization in explanation rendering.

6. `Medium` - Model readiness lacked startup-level visibility (fixed)
- Symptom: model asset problems surfaced late.
- Fix:
  - Added `nlu/runtime_components/model_preflight.py`.
  - Runtime API payload now includes `model_preflight` report (`ready`, `structure`, `ner`, missing assets/warnings).
  - `infer._require_local_model_dir` now reports missing tokenizer assets explicitly.

## Validation
- Unit/integration tests:
  - Full suite: `108 passed`.
- Regression workflow:
  - `scripts/run_multiturn_regression_zh.py`
  - Result summary:
    - total cases: 10
    - complete match rate: 100%
    - overwrite flow pass rate: 100%
    - natural reply no internal-field leak rate: 90%
  - Artifacts:
    - `docs/multiturn_regression_zh_2026-03-05.json`
    - `docs/multiturn_regression_zh_2026-03-05.md`

## Feasibility Assessment
- Current status: **ready for controlled user testing** (not final production-hardening).
- Why feasible now:
  - deterministic no-ollama path works and is observable.
  - overwrite confirmation guard is enforced.
  - Chinese geometry/source extraction significantly improved.
  - model readiness is now detectable before run-time failures.

## Residual Risks
- Tokenizer asset checks are file-level; they do not validate semantic compatibility between config and checkpoint.
- Naturalized user-facing responses still depend on LLM quality when enabled.
- Extreme long-context sessions may need explicit dialogue memory pruning policy tuning.

