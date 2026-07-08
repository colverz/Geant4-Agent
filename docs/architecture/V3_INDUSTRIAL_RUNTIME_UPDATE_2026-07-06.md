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
real DeepSeek health check: request id, model, usage, and nonce verified
v3 live candidate: one real design call; semantic contract and local runtime passed
v3 tests: 161 passed, 2 subtests passed
industrial tests: 41 passed
safety harness: 13/13 trials passed
```

## What Is Not Complete

- Real DeepSeek connectivity is now verified through the optional local proxy
  setting. The check requires a provider request id, model, token usage, and a
  returned random nonce; fallback cannot pass it.
- Repeated LLM design work was removed. Accepting defaults now builds the payload
  from the current design instead of calling the design model two more times.
- The live v3 case is physically valid but intentionally not canonical. Its
  names, transverse dimensions, source distance, detector distance, and added
  scoring plane differ from the deterministic golden setup.
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
LLM candidate -> physical semantic contract -> preflight -> pending action
-> exact action confirmation -> local-process Geant4 -> metric completeness

Canonical compiler payload -> exact contract -> local-process Geant4
-> reviewed golden numeric comparison
```

The two gates must remain separate. A freely designed LLM candidate cannot be
compared numerically with a golden produced from different geometry. Conversely,
the semantic gate must not replace exact canonical regression coverage.

## 2026-07-08 Follow-up

The first real DeepSeek and Geant4 v3 loop now passes end to end:

```text
one DeepSeek design call
-> semantic candidate contract passed
-> runtime preflight passed
-> pending action created
-> exact action_id confirmed
-> local-process Geant4 completed
-> required metrics extracted
```

Observed metrics for the non-canonical 10 mm lead candidate:

```text
detector_crossing_count = 5976
detector_edep_total_mev = 83.8698
transmission_factor = 0.5976
```

The canonical golden comparison was not performed because the LLM chose a
different but physically valid transverse geometry and placement. The report
records `comparison_scope=semantic_contract_and_real_runtime`.

## Next Mainline

1. Run the semantic live gate across the remaining single-run reviewed cases and
   inspect where the typed physical requirements need richer geometry semantics.
2. Add observable timing records around each LLM phase.
3. Add a typed multi-run experiment plan for paired cases, with one confirmation
   covering an immutable set of preflighted runtime actions.
4. Keep canonical golden regression and free-design semantic evaluation as two
   named report scopes.
5. After the live gate is usable, continue with user-visible result-driven advice.
