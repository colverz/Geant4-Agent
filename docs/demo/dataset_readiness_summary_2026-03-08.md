# Dataset Readiness Summary 2026-03-08

## Scope

This note classifies the current evaluation datasets by demo readiness and recommended usage.
All files listed here are intended to sit beside the main demo package in `docs/demo`.

## Recommendation Summary

| Dataset | Latest Referenced Result | Pass Rate | Readiness | Recommended Use |
|---|---|---:|---|---|
| `eval_casebank_v1_20.json` | archived `2026-03-05_234917` | 1.0000 | Demo-ready | Small static baseline, quick overview |
| `eval_casebank_missing_multiturn_v1_10.json` | archived `2026-03-06_124519` | 1.0000 | Demo-ready | Missing-parameter closure demo |
| `eval_casebank_v2_50.json` | archived `2026-03-06_134702` | 0.9200 | Internal-only | Broad regression, not stable enough for external demo |
| `eval_casebank_multiturn_live_v2_12.json` | archived `2026-03-07_145107` | 1.0000 | Demo-ready | Compact live multi-turn demo |
| `eval_casebank_multiturn_live_v3_24.json` | current `2026-03-08_001313` | 1.0000 | Primary demo baseline | Main external demonstration set |
| `eval_casebank_multiturn_live_v4_24_colloquial_materials.json` | current `2026-03-08_102153` | 0.8333 | Stress-only | Colloquial material robustness appendix |

## Interpretation

### Primary demo sets

Use these externally without caveat:

- `docs/datasets/casebank/eval_casebank_v1_20.json`
- `docs/datasets/casebank/eval_casebank_missing_multiturn_v1_10.json`
- `docs/datasets/casebank/eval_casebank_multiturn_live_v2_12.json`
- `docs/datasets/casebank/eval_casebank_multiturn_live_v3_24.json`

These sets already support a coherent story:

1. small structured baseline
2. missing-parameter closure
3. compact live multi-turn
4. full live external baseline

### Internal-only set

- `docs/datasets/casebank/eval_casebank_v2_50.json`

This set is useful because it is broad, but its latest retained pass rate is `0.92`, so it is better treated as an engineering regression bank rather than polished demo evidence.

### Stress set

- `docs/datasets/casebank/eval_casebank_multiturn_live_v4_24_colloquial_materials.json`

This set should be shown only as a robustness appendix. It is useful because it replaces explicit Geant4 material tags with colloquial material names, but the current retained result is `20/24`, so it still exposes unresolved material canonicalization gaps.

## Suggested demo folder contents

For a clean external handoff, the recommended files in `docs/demo` are:

- `external_demo_package_2026-03-08.md`
- `external_demo_package_bilingual_2026-03-08.pdf`
- `dataset_readiness_summary_2026-03-08.md`
- `dataset_readiness_summary_bilingual_2026-03-08.pdf`

## Next promotion target

The next dataset to promote is `v4_24 colloquial materials`. It becomes demo-ready once the four remaining failures are eliminated and a fresh live regression reaches the same reliability band as `v3_24`.
