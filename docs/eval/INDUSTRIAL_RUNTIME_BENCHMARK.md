# Industrial Runtime Benchmark

Status: new acceptance benchmark design.

This benchmark is the project acceptance standard. It is intentionally stronger
than the legacy NLU/config benchmarks.

The benchmark evaluates whether Geant4Agent can complete concrete industrial or
scientific simulation tasks end to end:

`raw user scenario -> LLM candidate config -> typed simulation spec -> real Geant4 runtime -> structured result -> golden numeric comparison -> grounded result QA`

## Non-Negotiable Rules

1. A formal benchmark pass requires real Geant4 execution.
2. `InMemoryGeant4Adapter` may only test evaluator wiring; it cannot contribute
   to the industrial score.
3. Every official case must have fixed golden numeric metrics.
4. Golden metrics must be generated from a pinned runtime environment, not by an
   LLM and not by manual guesswork.
5. LLM output is only a candidate. The judge is typed config fidelity, runtime
   execution, and numeric comparison against golden results.
6. If a case has no golden metrics or cannot launch real Geant4, the result is
   `not_evaluable`, not `passed`.
7. Unsupported industrial scenarios stay in the benchmark as capability gaps.
   They must not be weakened into toy cases.

## Runtime Fingerprint

Every golden result must record:

- Geant4 version.
- Local runtime wrapper version or build hash.
- C++ source build hash when available.
- Physics list.
- Random engine and seed.
- Event count.
- Thread count. Industrial benchmark defaults to single-thread for
  reproducibility.
- Platform metadata.
- Runtime command fingerprint without secrets.

## Case Families

The benchmark is organized by practical use:

- `industrial_ndt`: radiography, step wedges, void/inclusion contrast, weld and
  pipe inspection.
- `shielding`: gamma, electron, and neutron shielding with transmission and
  detector response.
- `medical_phantom`: water phantom energy deposition, proton/electron/gamma
  simplified dose-like metrics.
- `detector_response`: silicon and scintillator detector response, hit counts,
  detector energy deposition.
- `beam_source`: Gaussian beams, collimation, isotropic source acceptance, plane
  scorer spread.
- `multi_turn_engineering`: modify energy/material/thickness/detector position,
  rerun, and compare golden changes.
- `unsupported_boundary`: CAD import, rotating CT gantry, moving geometry, and
  complex detector arrays. These expose missing capabilities rather than being
  silently simplified.

## Required Output Per Case

Each evaluated case must produce a record containing:

- raw dialogue
- LLM candidate config
- normalized typed config/spec
- runtime payload
- runtime command fingerprint
- generated macro/config artifacts
- run summary path
- structured result summary
- actual metrics
- golden metrics
- metric diff
- pass/fail/not-evaluable status
- failure category

Failure categories:

- `llm_config_error`
- `spec_compile_error`
- `runtime_unavailable`
- `runtime_error`
- `missing_golden`
- `missing_metric`
- `metric_mismatch`
- `result_qa_hallucination`
- `unsupported_capability`

## Golden Metric Policy

Weak health checks such as `edep >= 0` are not acceptance criteria.

Accepted metric forms:

- Exact integer counts, for example `events_completed`, `detector_crossing_count`,
  `plane_crossing_count`.
- Exact floating point comparison when the pinned runtime is deterministic enough.
- Narrow tolerance around a golden value when Geant4/Monte Carlo numerical
  variation requires it.
- Paired-case numeric deltas only when both sides also have golden values.

Examples:

- Good: `target_edep_total_mev = 84.215 +/- 0.050`
- Good: `detector_crossing_count = 1273`
- Good: `transmission_ratio = 0.318 +/- 0.010`
- Bad: `target_edep_total_mev >= 0`
- Bad: `detector_crossing_count is plausible`
- Bad: `LLM says shielding reduced flux`

## Evaluation Modes

`official`

- Requires real Geant4.
- Requires all golden metrics.
- Produces pass/fail/not-evaluable.
- Used for industrial readiness.

`golden_generation`

- Requires real Geant4.
- Creates or refreshes golden metrics.
- Must be reviewed before committing golden files.

`wiring`

- May use fake or in-memory runtime.
- Only tests evaluator shape and report plumbing.
- Cannot count as industrial pass.

## Full Scenario Set

The current scenario manifest is:

- `docs/eval/industrial_runtime_benchmark.json`

It contains the full intended industrial benchmark surface, not a toy subset.
Some cases may initially expose unsupported functionality. That is expected and
useful: the benchmark is allowed to fail when the project is not ready.

## Immediate Implementation Tasks

1. Add a shape validator for `industrial_runtime_benchmark.json`.
2. Add an official evaluator that refuses to pass without real Geant4.
3. Add a golden generation tool that runs real Geant4 and writes reviewed golden
   files under `docs/eval/golden/industrial_runtime/`.
4. Add failure analysis that groups failures by LLM, spec, runtime, metric, and
   unsupported-capability causes.
5. Stop treating legacy NLU/config benchmarks as acceptance evidence.
