# Strict Contract Evaluation Summary

Generated: 2026-03-03

## Scope

This rerun used the new strict turn contract rather than the archived wide-fill prompts.

It covered a focused subset of the updated suites:

- Bilingual subset: `S1_EN`, `S1_ZH`, `S4_EN`, `S4_ZH`
- Multiturn subset: `MT1_EN`, `MT1_ZH`, `MT2_EN`, `MT2_ZH`

Generated reports:

- `docs/bilingual_dialogue_eval_strict_contract_2026-03-03.json`
- `docs/bilingual_dialogue_eval_strict_contract_2026-03-03.md`
- `docs/multiturn_dialogue_eval_strict_contract_2026-03-03.json`
- `docs/multiturn_dialogue_eval_strict_contract_2026-03-03.md`

## Results

- Bilingual strict-contract subset: `3 / 4`
- Multiturn strict-contract subset: `4 / 4`

## Main Finding

The control-layer changes are holding under the new strict contract:

1. Field-scoped pending overwrite works.
2. Narrow-turn supplementation works during pending overwrite.
3. Multiturn explicit completion works when each turn stays within one explicit scope.

The main remaining failure in this subset is:

- `S1_EN`: explicit one-turn English `single_box` setup remained incomplete because `source.position` was not extracted, even though `source.direction` was extracted from the same sentence.

## Interpretation

This shifts the bottleneck again:

- The primary blocker is no longer overwrite gating.
- The primary remaining weakness is English one-turn vector extraction / backfill.

This points first to:

1. `slot` raw-text backfill for English vector phrases
2. `runtime_semantic` vector parsing consistency

It does **not** yet justify immediate `BERT` retraining. The current evidence is too narrow and the dominant failure still looks like prompt/backfill extraction, not a broad classifier failure.

## Recommended Next Step

1. Tighten English vector backfill for:
   - `at (x,y,z)`
   - `position (x,y,z)`
   - `pointing (x,y,z)`
   - `pointing +z` / `along +z`
2. Re-run the same strict-contract subset.
3. If English vector extraction still fails after deterministic backfill tightening, then run a targeted `BERT/runtime_semantic` diagnostics pass before deciding on retraining.
