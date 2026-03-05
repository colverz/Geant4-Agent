# Docs Hygiene And Readiness Report (2026-03-04)

## What Was Checked
- Verified the top-level strict-contract reports in `docs/`
- Verified whether `docs/multiturn_dialogue_eval_strict_contract_2026-03-03.*` was still a partial subset run
- Cleaned the docs index so the current canonical baseline is explicit

## Docs State
### Canonical Reports Kept At Top Level
- `docs/bilingual_dialogue_eval_strict_contract_2026-03-03.*`
  - scope: full bilingual suite
  - status: canonical
  - latest validated result: `10/10`

- `docs/multiturn_dialogue_eval_strict_contract_2026-03-03.*`
  - scope: full multiturn suite
  - status: canonical
  - latest validated result: `8/8`

### Archived / Non-Baseline Reports
- Old exploratory and superseded reruns remain under:
  - `legacy/reports/2026-03-03_pre_rerun/`
  - `legacy/reports/2026-03-03_full_contract_attempt/`

This keeps the top-level `docs/` directory reserved for the current baseline and core design documents.

## Validation Performed
### Report Verification
I inspected the current multiturn JSON directly.

Result:
- `8` cases present
- `8/8` marked as passed

That means the current top-level multiturn report is already the full canonical report, not a stale subset file.

### Current Iteration Test Passes
Local:
```powershell
.\.venv\Scripts\python -m unittest tests.test_llm_prompt_registry tests.test_llm_slot_frame tests.test_normalizer_controls tests.test_dialogue_agent tests.test_smoke_no_ollama -v
```

Result:
- `42` tests passed

Remote:
```powershell
.\.venv\Scripts\python scripts/run_strict_contract_eval.py --suite bilingual
```

Result:
- `10/10`

Remote targeted subset:
```powershell
.\.venv\Scripts\python scripts/run_strict_contract_eval.py --suite multiturn --case-id MT3_EN --case-id MT3_ZH
```

Result:
- `2/2`

## Readiness Assessment
### Can The Current Program Be Put In Front Of Test Users?
Yes, for controlled pilot testing.

### Why
- The strict-contract bilingual baseline is fully green
- The multiturn canonical baseline remains fully green (`8/8`) in the current top-level report
- The latest prompt iteration (`strict_slot_v2`) passed local regression and did not regress the bilingual baseline
- Overwrite confirmation and field-scoped blocking are already in place

### Limits
- Remote inference latency remains high
- The system should still be presented as a first-pass configuration assistant, not a final authority
- Real user sessions should be logged for post-run review:
  - `raw_dialogue`
  - `dialogue_trace`
  - `slot_debug`
  - final config

## Recommendation
Proceed with controlled user feedback testing now.

Do not start `BERT` retraining yet. Current evidence still points to:
- prompt/agent behavior
- control-layer correctness
- remote model behavior

as the active levers, not a demonstrated `BERT` bottleneck.
