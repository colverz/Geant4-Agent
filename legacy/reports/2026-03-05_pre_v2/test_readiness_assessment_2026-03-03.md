# Test Readiness Assessment (2026-03-03)

## Scope
- Runtime: `strict slot-first + dialogue agent`
- Model: `qwen3:14b`
- Prompt profile: `strict_slot_v2`
- Environment: remote Ollama at `http://114.212.130.6:11434`

## Changes In This Iteration
- Centralized strict LLM prompts in `core/config/llm_prompt_registry.py`
- Switched slot prompt baseline from `strict_slot_v1` to `strict_slot_v2`
- Kept semantic-frame prompt versioned under the same registry
- Cleaned `nlu/llm/normalizer.py` control hints to remove corrupted Chinese literals

## Validation Performed
### Local Automated Tests
Command:
```powershell
.\.venv\Scripts\python -m unittest tests.test_llm_prompt_registry tests.test_llm_slot_frame tests.test_normalizer_controls tests.test_dialogue_agent tests.test_smoke_no_ollama -v
```

Result:
- `42` tests passed
- `0` failures

Coverage in this batch:
- prompt profile versioning
- slot-frame parsing and backfill
- explicit control extraction
- dialogue policy
- no-Ollama smoke path

### Remote Strict-Contract Evaluation
Commands:
```powershell
.\.venv\Scripts\python scripts/run_strict_contract_eval.py --suite bilingual
.\.venv\Scripts\python scripts/run_strict_contract_eval.py --suite multiturn --case-id MT3_EN --case-id MT3_ZH
```

Results:
- `bilingual: 10/10`
- `multiturn subset (MT3): 2/2`

Notes:
- Full multiturn suite was previously green (`8/8`) before this prompt-only iteration.
- After `strict_slot_v2`, a targeted multiturn subset covering recommendation/explanation still passed.
- Full multiturn rerun after this prompt-only change was not completed due runtime cost; this should still be run before wider external exposure if time permits.

## Readiness Judgment
### Can It Be Put In Front Of Test Users?
Yes, for a controlled pilot.

### Recommended Pilot Scope
- Use the current strict-contract workflow only
- Keep the interaction framed as a guided configuration assistant, not open-ended chat
- Collect feedback specifically on:
  - missing-field clarification quality
  - overwrite confirmation behavior
  - recommendation explanation usefulness
  - bilingual prompt compliance

### Constraints
- Remote inference latency is still high; long evaluation runs are expensive
- This is not yet a production deployment
- User testing should be framed as feedback collection, not final configuration authority

## Risks Still Present
- Full multiturn matrix after `strict_slot_v2` has not been rerun end-to-end in this iteration
- The system still depends on a remote Ollama endpoint; availability and latency affect UX
- Broader unconstrained user phrasing outside the strict-contract assumptions may still produce regressions

## Recommendation
Proceed with limited user feedback testing now, under these conditions:
- keep strict mode enabled
- keep overwrite confirmation enabled
- keep the current `qwen3:14b` baseline fixed
- log `raw_dialogue`, `dialogue_trace`, `slot_debug`, and final config for every user session

Before broader rollout, do:
1. rerun full multiturn strict-contract matrix after `strict_slot_v2`
2. compare against one stronger model once Ollama supports it
3. review real user logs for prompt-contract violations
