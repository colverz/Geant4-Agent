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

The execution contract is:

- `docs/eval/INDUSTRIAL_RUNTIME_EXECUTION_CONTRACT.md`

That contract defines the only official acceptance path: LLM candidate,
deterministic typed compilation, real Geant4 runtime, structured metrics, and
golden numeric comparison.

## Immediate Implementation Tasks

1. Add a shape validator for `industrial_runtime_benchmark.json`.
   Status: implemented in `tools/evaluate_industrial_runtime_benchmark.py`.
2. Add an official evaluator that refuses to pass without real Geant4.
   Status: implemented. The current evaluator reports formal cases as
   `not_evaluable` when real runtime or golden metrics are missing.
3. Add a golden generation tool that runs real Geant4 and writes reviewed golden
   files under `docs/eval/golden/industrial_runtime/`.
   Status: implemented as a strict gate in
   `tools/create_industrial_golden.py`. It intentionally refuses to write
   goldens until real runtime opt-in and scenario-to-runtime compilation are
   available.
4. Add failure analysis that groups failures by LLM, spec, runtime, metric, and
   unsupported-capability causes.
   Status: implemented in `tools/analyze_industrial_benchmark_failures.py`,
   including runtime/golden blockers and deterministic compile blockers.
5. Stop treating legacy NLU/config benchmarks as acceptance evidence.
   Status: documented in `docs/eval/LEGACY_BENCHMARK_ARCHIVE.md`.

6. Add deterministic scenario-to-runtime compiler coverage.
   Status: initial compiler implemented in `tools/industrial_runtime_compiler.py`.
   It compiles the currently expressible single-volume detector/plane cases and
   reports explicit geometry/source/scoring/metric gaps for the rest. This is
   not a pass condition by itself; official pass still requires real Geant4 and
   reviewed golden metrics.

7. Add runtime execution and metric comparison bridge.
   Status: implemented in `tools/industrial_runtime_executor.py`. It can execute
   compiled cases through the existing Geant4 MCP adapter, extract structured
   metrics, generate unreviewed golden files, and compare actual metrics against
   golden tolerances. Official scoring still requires a real local-process
   Geant4 runtime and reviewed goldens.

8. Add live LLM-to-runtime stage runner.
   Status: implemented in `tools/run_industrial_llm_runtime_stage.py`. This is
   the first entrypoint that actually tests the LLM as the candidate
   configuration producer before Geant4 execution. It does not let the LLM judge
   results: the LLM output must match the typed runtime contract, then the
   candidate config is executed by the local-process Geant4 adapter and compared
   with reviewed golden metrics.

## Current Evaluator

The current evaluator is intentionally strict:

```powershell
.venv\Scripts\python.exe tools\evaluate_industrial_runtime_benchmark.py --json
```

Without real runtime opt-in and golden metrics, the evaluator must return:

- `ok=false`
- official cases as `not_evaluable`
- unsupported boundary cases as `unsupported_capability`
- `passed=0`

This is the desired behavior. The industrial benchmark must not pass by using
in-memory runtime, weak metric checks, or config-only validation.

Official runtime evaluation will require:

```powershell
$env:GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK="1"
$env:GEANT4_RUNTIME_COMMAND_JSON='["<path-to-real-geant4-wrapper>"]'
.venv\Scripts\python.exe tools\evaluate_industrial_runtime_benchmark.py `
  --golden-dir docs\eval\golden\industrial_runtime `
  --outdir docs\reports\eval `
  --run-id industrial-runtime-latest `
  --json
```

The next implementation step is not to weaken the evaluator. It is to add golden
generation and scenario-to-runtime compilation until these cases become
evaluable.

Golden generation currently uses a hard fail-safe:

```powershell
.venv\Scripts\python.exe tools\create_industrial_golden.py --json
```

If real runtime opt-in or the deterministic scenario compiler is missing, the
tool reports `blocked` or `not_evaluable` and writes no golden files.

When a case is compiled, a real local-process runtime is configured, and all
required metrics are extractable, the golden tool writes:

```text
docs/eval/golden/industrial_runtime/<case-id>.golden.json
```

Generated files are marked `review.status="unreviewed"` and must be reviewed
before they are treated as official baselines.

Golden review is explicit and auditable. Review does not regenerate metrics and
does not change expected values; it only marks an already generated real-runtime
golden as accepted after the reviewer has checked the scenario, runtime
fingerprint, artifacts, and metric reasonableness.

```powershell
.venv\Scripts\python.exe tools\review_industrial_golden.py `
  --case-id shielding_lead_gamma_transmission `
  --golden-dir docs\eval\golden\industrial_runtime `
  --reviewer "<name>" `
  --notes "Checked runtime payload, Geant4 version, seed, event count, artifacts, and metrics." `
  --json
```

Use `--dry-run` first when checking a newly generated file. Dry-run validates
the file and reports the metrics hash without changing `review.status`.

The review tool refuses to approve files with missing numeric metrics, missing
runtime fingerprint fields, missing reviewer identity, or an already reviewed
status unless `--force` is used to update review metadata intentionally.

Official evaluator policy:

- Default: only `review.status="reviewed"` golden files can be used for a pass.
- Development/wiring mode: `--allow-unreviewed-goldens` may be used to verify
  runtime plumbing, but the result is not an official benchmark pass.
- Manifest placeholders with `expected=null` are never passable.

Failure analysis is available via:

```powershell
.venv\Scripts\python.exe tools\analyze_industrial_benchmark_failures.py --json
```

This groups failures by category and domain so the next engineering work is
driven by hard blockers rather than by easier config-only cases.

Stage workflow runner:

```powershell
.venv\Scripts\python.exe tools\run_industrial_runtime_stage.py --json
```

For real local runtime + golden generation:

```powershell
$env:GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK="1"
$env:GEANT4_RUNTIME_COMMAND_JSON='["<path-to-real-geant4-wrapper>"]'
.venv\Scripts\python.exe tools\run_industrial_runtime_stage.py `
  --case-id shielding_lead_gamma_transmission `
  --generate-goldens `
  --golden-dir docs\eval\golden\industrial_runtime `
  --json
```

This runner is the preferred local entrypoint for this stage because it returns
one compact `stage_summary` covering compile coverage, runtime readiness,
golden generation, evaluator status, and top blockers.

For wiring-only checks after generating unreviewed goldens:

```powershell
.venv\Scripts\python.exe tools\run_industrial_runtime_stage.py `
  --allow-unreviewed-goldens `
  --json
```

Do not use this flag for official readiness claims.

Live LLM full-chain runner:

```powershell
$env:GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK="1"
$env:GEANT4_RUNTIME_COMMAND_JSON='["<path-to-real-geant4-wrapper>"]'
.venv\Scripts\python.exe tools\run_industrial_llm_runtime_stage.py `
  --live-llm `
  --llm-config nlu\llm_support\configs\deepseek_api.local.json `
  --case-id shielding_lead_gamma_transmission `
  --golden-dir docs\eval\golden\industrial_runtime `
  --json
```

This runner has a stricter role split:

- The LLM receives the raw industrial dialogue plus the benchmark scenario brief
  and proposes a config.
- Deterministic code converts that config into `SimulationSpec` and
  `RuntimePayload`.
- A contract check compares critical geometry/source/detector/scoring/runtime
  fields against the benchmark runtime requirement.
- Only a contract-passing candidate config is sent to Geant4.
- The result is judged by structured metrics and reviewed golden values, never
  by an LLM.

By default, the runner refuses to call the live LLM if real runtime opt-in is
missing. Use `--allow-llm-without-runtime` only for parser debugging; it is not a
full-chain benchmark.

The evaluator also includes a `compile_summary`:

- `compiled`: current runtime payload can represent the case and all declared
  metrics are structurally supported.
- `compiled_with_gaps`: a runtime payload can be generated, but one or more
  required metrics are not yet available from structured runtime results.
- `unsupported_capability`: the case requires geometry/source/scoring/runtime
  behavior that the current deterministic compiler must not simplify away.

If reviewed golden files exist, the evaluator uses them instead of the manifest
placeholders. It then runs the compiled case, extracts actual metrics, and
returns `passed`, `missing_metric`, or `metric_mismatch` from numeric comparison.
