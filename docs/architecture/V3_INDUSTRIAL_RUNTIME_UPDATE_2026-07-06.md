# V3 Industrial Runtime Update - 2026-07-06

## What This Round Was For

This round closed the existing eight-case real Geant4 baseline and added a
v3-native path for evaluating LLM-generated runtime candidates. It did not add
material recommendation logic or automatic optimization.

## What Is Complete

- Eight industrial cases were run three times each with real Geant4: 24 runs.
- All three runs for every case had identical runtime fingerprints and metrics.
- The eight golden files are reviewed by `project-owner` and use unchanged,
  zero-width tolerances.
- The scoped deterministic official gate passes 8/8.
- Candidate generation, review, and promotion are separate operations. Candidate
  generation cannot overwrite official golden files.
- The v3 industrial runner checks the LLM candidate against the expected runtime
  contract before preflight. A mismatch cannot reach confirmation or runtime.
- Confirmation uses the exact pending `action_id`; in-memory execution is always
  disabled for this gate.
- LLM understanding, planning, design, result analysis, and naturalization are
  represented by the typed `V3LlmPolicy`. Compatibility metadata is only a mirror.

Tracked reproducibility evidence:

`docs/eval/golden/industrial_runtime/REPRODUCIBILITY_2026-07-06.json`

## Verification

```text
real Geant4 reproducibility: 24/24 runs completed, 8/8 cases exact
scoped deterministic industrial gate: 8 passed, 0 failed
v3 tests: 161 passed, 2 subtests passed
industrial tests: 41 passed
safety harness: 13/13 trials passed
```

## What Is Not Complete

- The live DeepSeek-backed v3 gate did not complete within a practical timeout.
  The first attempt exposed fallback use; the runner now rejects fallback as
  `llm_unavailable` instead of treating it as an LLM candidate.
- Repeated LLM work was found in understanding, planning, and design. The runner
  now enables only the design call, but the external request still remained slow.
- `shielding_concrete_gamma_transmission` is a paired-run case. The deterministic
  runtime supports it, but the v3 conversational workflow does not yet have a
  first-class multi-run experiment contract. The v3 runner reports this honestly
  instead of silently using compiler-generated variants.
- Seven additional compilable benchmark cases still have no reviewed golden.
  They are outside this round's approved eight-case scope.

## Code Review

The new code is necessary because the previous industrial LLM runner entered the
legacy v2 orchestration path and could not prove v3 behavior. Sharing only the
pure runtime-contract comparator avoids copying policy logic into the new runner.

The design remains reasonable because execution authority is unchanged:

```text
LLM candidate -> exact contract check -> preflight -> pending action
-> exact action confirmation -> local-process Geant4 -> metric comparison
```

The main remaining architecture risk is LLM call latency. Do not weaken contract
checks or use deterministic fallback to make the live gate appear green.

## Next Mainline

1. Add observable total-deadline and timing records around each LLM phase.
2. Make the v3 reasoner reuse one validated design result instead of allowing
   overlapping planner and design responsibilities in ordinary UI turns.
3. Add a typed multi-run experiment plan for paired cases, with one confirmation
   covering an immutable set of preflighted runtime actions.
4. Re-run the live v3 gate on one case, inspect the actual LLM candidate, then
   expand to the remaining single-run cases.
5. After the live gate is usable, continue with user-visible result-driven advice.
