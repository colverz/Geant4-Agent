# Industrial Runtime Execution Contract

Status: acceptance contract for the industrial benchmark.

This document defines the non-negotiable execution path for the industrial
benchmark. It is deliberately stricter than earlier NLU/config benchmarks.

## Acceptance Pipeline

Every official benchmark case must pass this full path:

`raw dialogue -> LLM candidate config -> deterministic validation -> typed SimulationSpec -> RuntimePayload -> real Geant4 run -> structured metrics -> golden comparison -> grounded answer check`

No stage may be replaced by an LLM judgment. The LLM can propose and explain,
but the benchmark judge is deterministic code plus real runtime output.

## Stage Contracts

### 1. Raw Dialogue

Input is the user-facing industrial task. It can be multi-turn, mixed language,
and partially underspecified in the same way a real engineer would ask.

Required record fields:

- `raw_dialogue`
- `domain`
- `task`
- `capability_pressure`

### 2. LLM Candidate Config

The LLM may produce a candidate interpretation, but it is not trusted as a
source of physical truth.

Required record fields:

- `candidate_config`
- `prompt_profile_id`
- `prompt_validation`
- `llm_model`
- `llm_latency_ms`
- `llm_raw_response_path`

Failure categories:

- `llm_config_error`
- `result_qa_hallucination`

### 3. Deterministic Validation

The candidate config must pass schema, allowlist, unit, material, source, and
scoring validation before any runtime launch.

Required record fields:

- `validation_result`
- `normalized_config`
- `rejected_fields`
- `confirmation_required`

Failure categories:

- `llm_config_error`
- `spec_compile_error`

### 4. Typed SimulationSpec

The normalized config must be compiled into a typed simulation spec. This is
where industrial scenario semantics become explicit runtime geometry/source/
scoring requirements.

Required record fields:

- `simulation_spec`
- `spec_schema_version`
- `spec_compile_warnings`

Failure categories:

- `spec_compile_error`
- `unsupported_capability`

The deterministic compiler must prefer explicit rejection over silent
simplification. For example, if a case asks for a step wedge, embedded void,
multi-layer shield, depth-binned scorer, paired rerun, or isotropic source and
the current runtime cannot represent it, the compiler must return a structured
gap instead of producing a weaker single-box approximation.

### 5. RuntimePayload

The typed spec must compile into the exact payload consumed by the Geant4
runtime wrapper. This stage must not silently drop unsupported geometry,
source, scoring, detector, or run fields.

Required record fields:

- `runtime_payload`
- `runtime_payload_schema_version`
- `payload_compile_warnings`
- `runtime_command_fingerprint`

Failure categories:

- `spec_compile_error`
- `unsupported_capability`

### 6. Real Geant4 Run

Official benchmark scoring requires a real Geant4 runtime. The in-memory adapter
can test wiring, but it cannot produce an official pass.

Required runtime gate:

- `GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK=1`
- `GEANT4_RUNTIME_COMMAND_JSON` or `GEANT4_RUNTIME_COMMAND`
- single-thread default unless a case explicitly requires otherwise
- pinned seed and event count

Required record fields:

- `runtime_phase`
- `runtime_stdout_path`
- `runtime_stderr_path`
- `artifact_dir`
- `run_summary_path`
- `runtime_fingerprint`

Failure categories:

- `runtime_unavailable`
- `runtime_error`

### 7. Structured Metrics

The runtime must produce structured metrics matching the case's required golden
metric names. A human-readable log is not enough.

Required record fields:

- `actual_metrics`
- `missing_metrics`
- `result_summary`
- `runtime_smoke_report`

Failure categories:

- `missing_metric`
- `runtime_error`

### 8. Golden Comparison

Official cases require reviewed golden numeric metrics. Goldens are generated
from a pinned real runtime, not by manual guesses or LLM outputs.

Required record fields:

- `golden_metrics`
- `metric_diff`
- `comparison_status`
- `golden_file`
- `golden_fingerprint`

Failure categories:

- `missing_golden`
- `metric_mismatch`

### 9. Grounded Result QA

If the user asks what happened, the answer must be grounded in actual runtime
metrics and comparison status. It must not invent physics conclusions.

Required record fields:

- `answer`
- `answer_sources`
- `answer_validation`

Failure categories:

- `result_qa_hallucination`

## Golden File Contract

Golden files live under:

`docs/eval/golden/industrial_runtime/`

Each file must use this shape:

```json
{
  "schema_version": "geant4_agent_industrial_golden.v1",
  "case_id": "shielding_lead_gamma_transmission",
  "created_at_utc": "2026-05-13T00:00:00Z",
  "runtime_fingerprint": {
    "geant4_version": "...",
    "runtime_wrapper_hash": "...",
    "physics_list": "FTFP_BERT",
    "seed": 1337,
    "events": 10000,
    "threads": 1,
    "platform": "..."
  },
  "runtime_payload_hash": "...",
  "metrics": {
    "detector_crossing_count": {"expected": 1234, "tolerance": 0},
    "detector_edep_total_mev": {"expected": 84.215, "tolerance": 0.05}
  },
  "artifact_dir": "...",
  "run_summary_path": "...",
  "review": {
    "status": "reviewed",
    "reviewer": "...",
    "notes": "..."
  }
}
```

Unreviewed or manually invented numbers must not be committed as official
golden metrics.

## Failure Taxonomy

- `llm_config_error`: LLM candidate cannot be validated into the requested task.
- `spec_compile_error`: deterministic compiler cannot map scenario into typed
  spec or runtime payload.
- `runtime_unavailable`: real Geant4 opt-in or runtime command is missing.
- `runtime_error`: real Geant4 launched but failed.
- `missing_golden`: official case has no reviewed golden numeric metrics.
- `missing_metric`: runtime output lacks a required metric.
- `metric_mismatch`: actual metric differs from golden beyond tolerance.
- `result_qa_hallucination`: result explanation adds unsupported claims.
- `unsupported_capability`: scenario intentionally exposes a missing capability.

## Tooling

Strict evaluator:

```powershell
.venv\Scripts\python.exe tools\evaluate_industrial_runtime_benchmark.py --json
```

Golden generation gate:

```powershell
$env:GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK="1"
$env:GEANT4_RUNTIME_COMMAND_JSON='["<path-to-real-geant4-wrapper>"]'
.venv\Scripts\python.exe tools\create_industrial_golden.py --case-id shielding_lead_gamma_transmission --json
```

Failure analysis:

```powershell
.venv\Scripts\python.exe tools\analyze_industrial_benchmark_failures.py --json
```

Current deterministic compiler boundary:

```powershell
.venv\Scripts\python.exe tools\evaluate_industrial_runtime_benchmark.py --json
```

The report's `compile_summary` is part of the acceptance feedback. It separates
cases that are runtime-payload-ready from cases blocked by missing geometry,
source, scorer, multi-run, or metric extraction capability.

## What This Contract Forbids

- Passing an official case with `InMemoryGeant4Adapter`.
- Replacing numeric comparison with a plausibility judgment.
- Treating config-only correctness as industrial readiness.
- Weak checks such as `edep >= 0`.
- Hiding unsupported capabilities by simplifying the scenario.
- Letting the LLM decide whether a physics result is correct.

## Current Expected State

At the current project stage, the industrial evaluator is expected to fail or
report official cases as `not_evaluable` until these are implemented:

- deterministic scenario-to-runtime compilation for each industrial case family
- real Geant4 runtime command opt-in
- structured metric extraction for all required metrics
- reviewed golden files

This is not a benchmark weakness. It is the acceptance target forcing the
project to become a real simulation agent instead of a config generator.
