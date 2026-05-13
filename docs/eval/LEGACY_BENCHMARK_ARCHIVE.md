# Legacy Benchmark Archive

Status: archived for regression only.

The previous benchmark files remain in the repository because existing tests and
tools still use them as wiring, NLU, guard, and report-regression checks. They no
longer define whether Geant4Agent is ready for industrial use.

## Archived Scope

These files are legacy regression assets:

- `docs/eval/agentic_benchmark_v1.json`
- `docs/eval/llm_scenario_live_casebank.json`
- `docs/eval/agentic_behavior_casebank.json`
- `docs/eval/simulation_scenario_casebank.json`
- `docs/eval/workflow_guard_casebank.json`
- `docs/eval/multiturn_guard_casebank.json`
- `docs/eval/session_behavior_casebank.json`
- `docs/eval/runtime_result_qa_casebank.json`
- `docs/eval/nlu_agentic_adversarial_casebank.json`
- `docs/eval/GEANT4_AGENT_BENCHMARK.md`

## What They Are Still Good For

- Detecting accidental regressions in prompt profile routing.
- Checking workflow guards, confirmation policy, and read-only behavior.
- Verifying that config-to-runtime payload wiring has not broken.
- Preserving historical context while the industrial runtime benchmark is built.

## What They Must Not Claim

- They do not prove industrial readiness.
- They do not prove physical correctness.
- They do not prove Geant4 runtime correctness.
- They do not replace golden numeric result comparison.
- They must not be used as the final project acceptance benchmark.

## New Authority

The new acceptance benchmark is:

- `docs/eval/INDUSTRIAL_RUNTIME_BENCHMARK.md`
- `docs/eval/industrial_runtime_benchmark.json`

Those files define the target standard:

`industrial scenario -> LLM candidate config -> typed spec -> real Geant4 runtime -> golden numeric comparison -> grounded result QA`

If a case cannot run on real Geant4 or does not have fixed golden metrics, it is
not evaluable as an industrial benchmark case.
