# Geant4-Agent Rebuild Progress

Date: 2026-05-21

This progress note tracks implementation against
`docs/architecture/GEANT4_AGENT_REBUILD_ROADMAP_2026-05-21.html`.

## Completed In This Pass

### P0: User-Visible Agent State

Implemented `agent_state_summary` as a stable response contract.

Current coverage:

- main `process_turn` response
- simulation-design read-only response
- config summary API response
- `/api/agent/state` cached latest summary

The summary exposes:

- intent and safety class
- terminal state
- understood/applied/rejected paths
- missing fields and asked fields
- pending confirmation
- runtime readiness
- runtime/tool guard state
- LLM degraded/fallback status
- next action
- user-visible status text

Primary files:

- `core/agent/state_summary.py`
- `core/orchestrator/session_manager.py`
- `ui/web/runtime_state.py`
- `ui/web/request_router.py`
- `ui/web/strict_api.py`

### M1: Minimal Natural-Language Runtime Smoke Entry

Implemented a script that tests the first real closed-loop shape:

```text
natural language
-> process_turn
-> agent config
-> validate_config
-> apply_config_patch
-> initialize_run
-> run_beam
-> summarize_last_result
-> runtime smoke report
```

The tool refuses to count in-memory runtime as real success unless explicitly
asked for wiring checks.

Primary files:

- `tools/run_minimal_agent_smoke.py`
- `tests/test_minimal_agent_smoke_tool.py`

Manual command:

```powershell
.venv\Scripts\python.exe tools\run_minimal_agent_smoke.py --events 2 --json
```

Current local result without `GEANT4_RUNTIME_COMMAND_JSON`:

```text
status=not_evaluable
reason=local_process_runtime_required
agent_state_summary.status=ready_to_run
```

This is expected: the agent can produce a runtime-ready config, but a real
local-process Geant4 runtime still needs to be configured for formal execution.

### M2: DesignAdvisor Contract

Implemented `DesignAdvisor` as a structured layer above simulation design
candidates.

Purpose:

- turn a raw simulation-design candidate into a user-facing plan
- expose primary option, alternatives, assumptions, tradeoffs, unsupported
  capabilities, required decisions, and next steps
- keep design advice read-only until user acceptance

Primary files:

- `core/agent/design_advisor.py`
- `tests/test_design_advisor.py`

The `/api/simulation/design` path now returns `design_advice`.

### M2.1: Design Acceptance Patch

Implemented a first audit bridge from read-only design advice to config
mutation.

Current behavior:

- existing `/api/simulation/accept` still commits the recommended config for
  compatibility
- the response now also includes `design_acceptance_patch`
- the patch is represented as a `CandidatePatchEnvelope`-compatible structure
- operations include path, op, confidence, evidence, and confirmation markers
- evidence is sourced from `design_advice`

This is intentionally an intermediate step. It makes design acceptance visible
and auditable before replacing direct commit with the full candidate patch
pipeline.

Primary files:

- `core/agent/design_acceptance.py`
- `tests/test_design_acceptance.py`

## Verification

Focused verification run:

```text
83 passed, 11 subtests passed
```

Additional DesignAdvisor and design-acceptance regression:

```text
88 passed, 11 subtests passed
```

## Next Implementation Target

The next step is to make LLM design output operational without weakening safety:

1. Keep `SimulationDesignCandidate` and `DesignAdvice` read-only by default.
2. Move `/api/simulation/accept` from direct commit to applying the generated
   design acceptance patch.
3. Require evidence and capability references for design-derived config paths.
4. Run grounding and typed validation before committing accepted design configs.
5. Add UI/API affordance for "accept design assumptions" versus "revise design".

This moves LLM from "field extractor" toward "design collaborator" while still
keeping deterministic code responsible for state mutation and runtime execution.

## 2026-05-22: v3 Clean-Room Agent Kernel Started

Started the v3 implementation as a new clean-room package instead of extending
the v2 session manager.

Implemented first-pass v3 kernel contracts:

- `V3TurnInput`
- `V3AgentState`
- `V3ActionProposal`
- `V3ToolCall`
- `V3Observation`
- `V3Answer`
- `V3AgentResult`

Implemented the first `AgentController` loop:

```text
perceive
-> reason
-> review_constraints
-> commit_gate
-> act
-> observe
-> answer
```

Important boundaries now covered by tests:

- read-only tools can run without commit gate
- runtime execution is blocked until confirmed
- confirmed runtime execution without a registered Geant4 tool returns
  `not_evaluable` instead of fake success
- constraint review can block an uncommittable action before tool execution

Primary files:

- `core/agent_v3/contracts.py`
- `core/agent_v3/controller.py`
- `core/agent_v3/tool_registry.py`
- `core/agent_v3/trace.py`
- `tests/test_agent_v3_contracts.py`
- `tests/test_agent_v3_controller.py`

Focused verification:

```text
6 passed
```

Next v3 implementation target:

1. Add Geant4-specific tools to `core/agent_v3/tools/`.
2. Start with read-only `geant4_capability_tool`.
3. Add `design_template_tool` for shielding, detector response, medical dose,
   and NDT draft designs.
4. Add `payload_builder_tool` and `geant4_runtime_tool` adapters after the
   draft design path is stable.
5. Add `tools/run_v3_agent_smoke.py` as the first v3 end-to-end command.

## 2026-05-22: v3 Geant4 Capability and Design Draft Tools

Implemented the first Geant4-specific v3 tools and a deterministic smoke
reasoner.

New v3 tools:

- `geant4_capability_tool`
  - risk level: `read_only`
  - exposes runtime DSL capabilities and design capabilities
  - lists supported scenario families: shielding, detector response, medical
    dose, and NDT contrast
- `geant4_design_template_tool`
  - risk level: `draft_only`
  - converts an open-ended physics goal into a `SimulationDesign` draft
  - reuses existing simulation design knowledge assets without entering the v2
    session manager

New smoke reasoner:

```text
inspect_geant4_capability
-> draft_geant4_design
-> present_geant4_design
```

This reasoner is intentionally deterministic. It is the harness for v3
behavior tests before live LLM reasoning is wired in.

Primary files:

- `core/agent_v3/tools/geant4_tools.py`
- `core/agent_v3/reasoners.py`
- `tools/run_v3_agent_smoke.py`
- `tests/test_agent_v3_geant4_tools.py`
- `tests/test_v3_agent_smoke_tool.py`

Focused verification:

```text
10 passed
```

Manual smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py
```

Current result:

```text
Chinese open-ended shielding request
-> Geant4 capability observation
-> SimulationDesign draft
-> user-facing Geant4 design answer
```

Next v3 implementation target:

1. Convert `SimulationDesign` draft into a typed `SimulationSpec` proposal.
2. Add `payload_builder_tool` as `draft_only` first, then allow committed
   payload persistence through `state_mutation`.
3. Add `runtime_preflight_tool` and `geant4_runtime_tool`.
4. Extend `run_v3_agent_smoke.py` with `--accept-defaults` and `--run` modes.

## 2026-05-22: v3 Design Draft to Runtime Payload Draft

Implemented the first v3 path from an open-ended Geant4 design draft to a
runtime payload draft, still without writing session state or executing
Geant4.

New v3 tool:

- `geant4_payload_builder_tool`
  - risk level: `draft_only`
  - input: `SimulationDesign` draft and event count
  - output: `recommended_config`, `SimulationSpec` summary, and
    `runtime_dsl.v1` payload
  - uses existing `core.simulation.build_simulation_spec` and
    `mcp.geant4.runtime_payload.build_runtime_payload`

Extended smoke reasoner:

```text
inspect_geant4_capability
-> draft_geant4_design
-> draft_geant4_runtime_payload   # only when accept_defaults is set
-> present_geant4_payload_draft
```

Extended CLI:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --accept-defaults --events 15
```

Current result:

```text
open-ended shielding goal
-> SimulationDesign draft
-> recommended_config
-> SimulationSpec summary
-> runtime_dsl.v1 payload
-> user-facing answer that explicitly says Geant4 has not been executed yet
```

Focused verification:

```text
13 passed
```

Next v3 implementation target:

1. Add `runtime_preflight_tool` as a read-only/draft runtime readiness check.
2. Add `geant4_runtime_tool` behind `runtime_execution` risk and confirmation.
3. Extend `run_v3_agent_smoke.py` with `--run`, keeping no-runtime paths
   explicitly `not_evaluable`.

## 2026-05-22: v3 Runtime Preflight and Runtime Execution Boundary

Implemented the v3 runtime boundary. The agent can now move from open-ended
goal to design draft, runtime payload draft, runtime preflight, and only then
runtime execution when a real local-process Geant4 runtime exists.

New v3 tools:

- `geant4_runtime_preflight_tool`
  - risk level: `draft_only`
  - validates the recommended config through `Geant4McpServer.validate_config`
  - checks whether the adapter is a real `local_process`
  - returns `not_evaluable` with `local_process_runtime_required` when only the
    in-memory adapter is available
- `geant4_runtime_tool`
  - risk level: `runtime_execution`
  - requires controller confirmation before execution
  - refuses in-memory runtime as a real result unless explicitly allowed for
    wiring tests
  - executes validate/apply/init/run/summarize through the MCP server when
    runtime is available

Extended smoke reasoner:

```text
inspect_geant4_capability
-> draft_geant4_design
-> draft_geant4_runtime_payload
-> preflight_geant4_runtime
-> run_geant4_runtime       # only after preflight ok and confirmed
-> present runtime observation / not_evaluable reason
```

Extended CLI:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --run --events 7
```

Current local result without `GEANT4_RUNTIME_COMMAND_JSON`:

```text
preflight status: not_evaluable
reason: local_process_runtime_required
adapter: in_memory
```

Focused verification:

```text
17 passed
```

Next v3 implementation target:

1. Add a reviewed local-process runtime smoke fixture or adapter injection path
   for CI-safe execution tests.
2. Add result interpretation for real `Geant4Observation`.
3. Add v3 casebank records for design-only, payload-only, preflight, and
   runtime-not-evaluable paths.

## 2026-05-22: Real Local Geant4 Runtime Verified

Verified that the compiled local Geant4 runtime exists and can execute through
the Python MCP adapter when explicitly configured.

Runtime executable:

```text
runtime/geant4_local_app/build/Release/geant4_local_app.exe
```

Verification command:

```powershell
$env:GEANT4_RUNTIME_COMMAND_JSON='["runtime/geant4_local_app/build/Release/geant4_local_app.exe"]'
$env:GEANT4_ROOT='F:\Geant4'
.venv\Scripts\python.exe tools\local_geant4_smoke.py --events 1 --require-runtime --json
```

Result:

```text
ok=true
events_requested=1
events_completed=1
completion_fraction=1.0
artifact_dir=F:\geant4agent\runtime_artifacts
run_summary_path=F:\geant4agent\runtime_artifacts\run_summary.json
```

Verified the v3 clean-room path against the same real runtime:

```powershell
$env:GEANT4_RUNTIME_COMMAND_JSON='["runtime/geant4_local_app/build/Release/geant4_local_app.exe"]'
$env:GEANT4_ROOT='F:\Geant4'
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --run --events 2 --json
```

Result:

```text
terminated_reason=observed
adapter=local_process
events_requested=2
events_completed=2
target_edep_total_mev=0.342633
detector_crossing_count=1
artifact_dir=F:\geant4agent\runtime_artifacts
run_summary_path=F:\geant4agent\runtime_artifacts\run_summary.json
```

Important note:

- Without `GEANT4_RUNTIME_COMMAND_JSON` or `GEANT4_RUNTIME_COMMAND`, the runtime
  correctly falls back to `in_memory`.
- v3 correctly reports that path as `not_evaluable` for real simulation
  results.
- This is the right safety boundary: in-memory is acceptable for wiring tests,
  but not for claiming real Geant4 results.

## 2026-05-22: Main Flow Dialogue-Driven Audit

Current UI flow is text-entry driven, but not yet fully backend-agent driven.

Observed current path:

```text
ui/web/app.js sendPrompt()
-> /api/geant4/intent
-> frontend decides branch:
   - read_summary -> /api/geant4/summary
   - read_config -> /api/config/summary
   - run_requested -> maybe /api/simulation/design, /api/simulation/accept,
     /api/geant4/validate, /api/geant4/run
   - viewer_requested -> /api/geant4/viewer/open
   - otherwise -> /api/simulation/design
```

This means the user interface is conversational at the surface, but the main
workflow is still orchestrated by frontend branch logic and specialized API
buttons/endpoints.

Specific non-v3-agent signs:

- `sendPrompt()` in `ui/web/app.js` classifies intent and chooses multiple API
  calls itself.
- `ensureCandidateCommitted()` auto-calls `/api/simulation/accept` before
  runtime.
- Runtime execution goes through `/api/geant4/run`, not through a single v3
  `AgentController` turn.
- `/api/simulation/design` and `/api/simulation/accept` still delegate to v2
  session manager compatibility code.

Desired v3 path:

```text
ui sends only user text and session id
-> /api/v3/agent/turn
-> AgentController decides design / ask / payload / preflight / run / answer
-> UI renders returned trace, artifacts, questions, and observations
```

Conclusion:

- The current UI is not purely button-driven, because the user starts with
  natural language.
- It is also not fully dialogue-driven, because the frontend still acts as a
  workflow router and silently performs accept/validate/run steps.
- v3 should replace this with a single dialogue turn endpoint before becoming
  the default product path.

## 2026-05-22: Runtime Auto-Discovery and Language-Driven Plan

Added a small runtime discovery layer so local Geant4 execution can be enabled
explicitly without manually setting environment variables for every smoke run.

New discovery module:

```text
mcp/geant4/runtime_discovery.py
```

Default discovered executable:

```text
runtime/geant4_local_app/build/Release/geant4_local_app.exe
```

Verification commands:

```powershell
.venv\Scripts\python.exe tools\local_geant4_smoke.py --auto-discover-runtime --events 1 --require-runtime --json
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2 --json
```

v3 smoke result:

```text
ok=true
terminated_reason=observed
adapter=local_process
events_requested=2
events_completed=2
target_edep_total_mev=0.342633
detector_crossing_count=1
artifact_dir=F:\geant4agent\runtime_artifacts
run_summary_path=F:\geant4agent\runtime_artifacts\run_summary.json
```

Important boundary:

- Runtime auto-discovery is opt-in through `--auto-discover-runtime`.
- Default in-memory mode remains available for wiring tests.
- v3 must not claim a scientific Geant4 result from in-memory execution.
- A real result requires `adapter=local_process` and a runtime observation.

Added a dedicated language-driven architecture note:

```text
docs/architecture/GEANT4_AGENT_V3_LANGUAGE_DRIVEN_FLOW_ANALYSIS_2026-05-22.html
```

Pure language-driven difficulty analysis:

1. Intent ambiguity: one utterance may mean design, modify, run, inspect
   previous result, or ask for explanation.
2. Hidden multi-step workflow: "run it" implies design, payload, preflight,
   confirmation, execution, observation, and interpretation.
3. Memory requirements: users will say "the previous plan", "change energy to
   2 MeV", or "do not change the material".
4. Runtime risk: natural language can imply expensive or state-changing actions,
   so execution must pass through an explicit proposal/confirmation policy.
5. Observation grounding: the agent must never invent Geant4 outputs. Without a
   runtime observation, the answer must be design-only or `not_evaluable`.
6. Frontend bypass risk: if the UI still calls accept/validate/run endpoints
   directly, the product remains frontend-orchestrated rather than agentic.
7. Evaluation difficulty: field-level tests are not enough. v3 needs multi-turn
   behavior casebanks that verify memory, confirmation, execution, and result
   interpretation.

Required v3 guarantees:

- One backend turn endpoint: `POST /api/v3/agent/turn`.
- UI sends user text and session id; it does not choose workflow branches.
- Buttons may remain only as text shortcuts or proposal confirmations.
- Runtime execution is a high-risk tool call requiring explicit confirmation.
- Every answer that discusses simulation results must cite a tool observation.
- Trace and action proposals must be visible to the UI.
- Regression casebank must cover design-only, modify-and-run, no-run, result
  question, runtime unavailable, and error recovery flows.

Next implementation target:

1. Promote the v3 smoke flow into a reusable backend turn service.
2. Add `POST /api/v3/agent/turn`.
3. Add pending action confirmation state for runtime execution.
4. Convert the UI to call the single v3 turn endpoint before moving old
   frontend-orchestrated endpoints into legacy.

## 2026-05-22: First Backend v3 Turn Endpoint

Implemented the first backend-only v3 language turn service.

New files:

```text
core/agent_v3/service.py
ui/web/v3_agent_api.py
tests/test_agent_v3_service.py
```

Updated router:

```text
ui/web/request_router.py
```

New endpoints:

```text
POST /api/v3/agent/turn
POST /api/v3/agent/reset
```

Behavior now covered by tests:

- UI/backend callers can send plain text to a single v3 endpoint.
- The v3 service keeps per-session `V3AgentState` in memory.
- A language request such as "请按默认参数运行一个铅屏蔽 gamma 模拟" reaches
  the runtime proposal path and stops at `waiting_confirmation`.
- A follow-up language turn such as "确认运行" can execute the pending runtime
  action when the session state already contains the draft payload and preflight
  observation.
- "不要运行" / "只给方案" is treated as a no-run intent, not as a runtime
  request merely because the text contains the word "运行".

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agent_v3_service.py tests\test_geant4_runtime_discovery.py tests\test_local_geant4_smoke_tool.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
18 passed
```

Real runtime smoke was re-run after the endpoint/service changes:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

## 2026-05-23: v3 Dialogue Composer

Implemented a dedicated user-facing dialogue layer for the v3 route.

Updated files:

```text
core/agent_v3/dialogue_composer.py
core/agent_v3/service.py
tools/run_v3_live_llm_dialogue.py
ui/web/app.js
tests/test_agent_v3_dialogue_composer.py
tests/test_v3_live_llm_dialogue_tool.py
```

New response fields:

```json
{
  "display_message": "Human-facing collaborative answer.",
  "raw_message": "Original controller/reasoner message.",
  "dialogue_act": "design_presented | action_needs_confirmation | action_cancelled | runtime_observed",
  "evidence_used": [{"source": "geant4_payload_builder_tool", "status": "ok"}],
  "dialogue": {}
}
```

Why this matters:

- The agent no longer exposes protocol-like controller text as the primary
  answer.
- Raw protocol text is still preserved for audit, debugging, and raw dialogue
  inspection.
- The frontend and live dialogue tool now prefer `display_message`.
- Confirmation and cancellation turns read like collaborative dialogue while
  still carrying `pending_action`, `raw_message`, trace, and observations.
- A language-intent boundary issue was fixed: English `setup` is no longer
  misread as the modification verb `set`, so "design a setup, do not run" stays
  design-only.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agent_v3_dialogue_composer.py tests\test_v3_live_llm_dialogue_tool.py tests\test_agent_v3_service.py tests\test_v3_frontend_route.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
32 passed
```

Real runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

Live LLM dialogue smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_live_llm_dialogue.py --live-llm --llm-config nlu\llm_support\configs\your_provider.local.json --compact --json
```

Result:

```text
ok=true; raw_dialogue now includes content/display_message, raw_message,
dialogue_act, pending_action, and observation summaries.
```

## 2026-05-23: Prompt-Level Dialogue Strategy

Refined the live simulation design prompt profile from:

```text
simulation_design_live_v1
```

to:

```text
simulation_design_live_v2_human_collab
```

Updated files:

```text
core/agent/simulation_design_llm.py
core/agent_v3/dialogue_composer.py
tools/evaluate_simulation_design_live.py
tests/test_simulation_design.py
tests/test_agent_v3_dialogue_composer.py
```

Prompt policy changes:

- Natural-language fields should sound like a concise senior collaborator, not
  a validation report.
- Keep `assumptions` to at most 3 short items.
- Keep `design_rationale` to one or two short practical sentences.
- Keep `alternatives_considered` as a compact JSON array.
- Avoid phrases such as "the system", "the user requirement", or long nested
  report clauses.
- For Chinese strings, prefer plain collaborative Chinese and avoid duplicated
  punctuation.

Engineering safeguards:

- Normalization now trims duplicate sentence punctuation from human-facing
  fields.
- Normalization caps assumption/user-decision lists so LLM verbosity cannot
  leak into the dialogue layer unchecked.
- `Dialogue Composer` uses `user_explanation` or `design_rationale` when
  available, but still keeps raw controller output in `raw_message`.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_simulation_design.py tests\test_simulation_design_live_evaluator.py tests\test_agent_v3_dialogue_composer.py tests\test_v3_live_llm_dialogue_tool.py tests\test_agent_v3_geant4_tools.py tests\test_agent_v3_service.py
```

Result:

```text
49 passed
```

Runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

Live LLM smoke confirmed the new prompt profile in raw output:

```text
prompt_profile_id=simulation_design_live_v2_human_collab
```

## 2026-05-23: v3 Runtime Result Follow-Up

Added a v3-native result explanation and follow-up path.

Updated files:

```text
core/agent_v3/result_explainer.py
core/agent_v3/reasoners.py
core/agent_v3/service.py
core/agent_v3/dialogue_composer.py
core/config/prompt_profiles.py
ui/web/app.js
tools/run_v3_live_llm_dialogue.py
tests/test_agent_v3_service.py
tests/test_runtime_result_explanation.py
tests/test_geant4_web_api.py
```

Behavior changes:

- A user follow-up like "What does detector crossing mean in the result?" now
  reads the latest v3 runtime observation before payload/design branches.
- If no runtime result exists, v3 answers that there is no explainable result
  yet instead of starting a new design flow.
- `Dialogue Composer` marks these turns as `runtime_result_answered`.
- The frontend sends `llm_result_enabled` when the model config is ready.
- The live dialogue tool also passes `llm_result_enabled` so result follow-up
  tests can opt into the same prompt path.

Prompt profile changes:

```text
runtime_result_explain_zh_v2_human_collab
runtime_result_explain_en_v2_human_collab
runtime_result_qa_zh_v2_human_collab
runtime_result_qa_en_v2_human_collab
```

The v2 result prompts keep the existing grounded-rewrite validator but add a
more agentic style contract: answer like a concise research collaborator,
answer directly, name the recorded metric used as evidence, and avoid report
tone. The validator still rejects new numbers and ungrounded facts.

Important language-boundary fix:

- "透射/transmission" by itself no longer triggers a result question. This
  prevents "设计一个透射方案，不要运行" from being misrouted away from design.
- Result routing now prefers explicit result/question language or specific
  recorded metrics such as `detector_crossing_count`, `edep`, dose, or hit
  questions.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agent_v3_dialogue_composer.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py tests\test_v3_frontend_route.py tests\test_simulation_design.py tests\test_simulation_design_live_evaluator.py tests\test_agent_v3_service.py tests\test_runtime_result_explanation.py
```

Result:

```text
65 passed
```

Runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

Live LLM smoke:

```text
ok=true; first turn remains design_presented for "透射方案，不要运行";
prompt_profile_id=simulation_design_live_v2_human_collab
```

## 2026-05-23: Real Geant4 Metric Smoke

Added a real local-process metric smoke tool:

```text
tools/run_v3_geant4_metric_smoke.py
tests/test_v3_geant4_metric_smoke_tool.py
```

The tool runs multiple small Geant4 cases through the MCP runtime adapter:

- copper target `target_edep` at 0.5, 1.0, and 2.0 MeV gamma source energy
- downstream silicon detector crossing count
- scoring-plane crossing count

It reports:

```text
events_completed
completion_fraction
source.energy_mev
source.primary_count
target_edep_total_mev
target_edep_mean_mev_per_event
detector_crossing_count
detector_crossing_mean_per_event
plane_crossing_count
plane_crossing_mean_per_event
run_summary_path
```

Also fixed a trace/readability issue in `mcp/geant4/adapter.py`: local-process
`apply_config_patch` now builds preview runtime payloads with the configured
`run.events` instead of the default `1`. Actual execution already used the
`run_beam --events` argument correctly; this fix prevents misleading preview
payloads in trace/UI evidence.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_geant4_mcp_adapter.py tests\test_v3_geant4_metric_smoke_tool.py tests\test_agent_v3_service.py tests\test_agent_v3_geant4_tools.py
```

Result:

```text
39 passed
```

Real metric smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_geant4_metric_smoke.py --events 20 --json
```

Observed result:

```text
ok=true
target_edep_copper_1mev: events_completed=20, target_edep_total_mev=6.73007
target_edep_copper_2mev: events_completed=20, target_edep_total_mev=6.47295
target_edep_copper_0p5mev: events_completed=20, target_edep_total_mev=3.08563
detector_crossing_vacuum_gamma: events_completed=20, detector_crossing_count=20
plane_crossing_vacuum_gamma: events_completed=20, plane_crossing_count=20
```

v3 agent main-path runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 5 --json
```

Observed result:

```text
terminated_reason=observed
target_edep_total_mev=1.73157
detector_crossing_count=2
events_completed=5
run_summary_path=F:\geant4agent\runtime_artifacts\run_summary.json
```

## 2026-05-22: v3 Live LLM Raw Dialogue Smoke

Added an opt-in live LLM dialogue smoke for v3.

New files:

```text
tools/run_v3_live_llm_dialogue.py
tests/test_v3_live_llm_dialogue_tool.py
```

Updated file:

```text
core/agent_v3/tools/geant4_tools.py
```

New behavior:

- `geant4_llm_design_tool` now records the model `raw_response` inside the LLM
  design observation.
- `tools/run_v3_live_llm_dialogue.py` can run a multi-turn v3 dialogue and emit:
  - raw user turns,
  - raw agent responses,
  - pending actions,
  - observation summaries,
  - raw LLM JSON responses.
- Live calls are opt-in through `--live-llm`.
- `--compact` omits deeply nested full responses so the raw dialogue remains
  readable.

Command:

```powershell
.venv\Scripts\python.exe tools\run_v3_live_llm_dialogue.py --live-llm --llm-config nlu\llm_support\configs\your_provider.local.json --compact --json
```

Observed live dialogue:

```text
User: 我想评估 1 MeV gamma 穿过铅屏蔽后的透射效果，请先给出 Geant4 方案，不要运行。
Agent: 已生成 Geant4 方案草案，使用 single_box / G4_Pb / beam，并包含 target_edep、detector_crossing_count、detector_edep、transmission_factor。

User: 把刚才方案改成 2 MeV，再跑 5 events。
Agent: 行动 `run_geant4_runtime` 的风险等级为 runtime_execution，需要确认后执行。

User: 取消运行，只保留方案。
Agent: 保留 2 MeV、5 events 的 runtime payload 草案，不执行 Geant4。
```

Live LLM raw response was captured from `geant4_llm_design_tool`; it proposed:

```json
{
  "geometry": "single_box",
  "material": "G4_Pb",
  "environment_material": "G4_Galactic",
  "source": "beam",
  "source_particle": "gamma",
  "source_energy_mev": 1.0,
  "detector": {
    "material": "G4_Si"
  },
  "scoring": [
    "target_edep",
    "detector_crossing_count",
    "detector_edep",
    "transmission_factor"
  ],
  "next_action": "build_candidate_config"
}
```

Important v3 boundary:

- The LLM raw response is only a design draft.
- The second turn's `2 MeV` and `5 events` are applied by v3 controlled
  overrides.
- Runtime still stops at `pending_action`; no execution happens without user
  confirmation.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_v3_live_llm_dialogue_tool.py tests\test_v3_frontend_route.py tests\test_agent_v3_service.py tests\test_geant4_runtime_discovery.py tests\test_local_geant4_smoke_tool.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
32 passed
```

Real runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

## 2026-05-22: v3 Multi-Turn Parameter Overrides Expanded

Expanded the v3-native multi-turn modification path beyond source energy.

New supported language examples:

```text
把材料换成铜，厚度改成 20 mm，再跑 15 events
把刚才方案改成 20 x 20 x 5 mm，再跑
```

Supported structured overrides:

```json
{
  "source_energy_mev": 2.0,
  "run_events": 15,
  "target_material": "G4_Cu",
  "target_thickness_mm": 20.0,
  "geometry_dimensions_mm": [20.0, 20.0, 5.0]
}
```

Updated files:

```text
core/agent_v3/service.py
core/agent_v3/tools/geant4_tools.py
tests/test_agent_v3_service.py
```

Behavior:

- `service.py` extracts modification intent and converts language into
  structured `config_overrides`.
- `run_events` updates the turn event count before payload generation.
- `geant4_payload_builder_tool` applies overrides to the draft recommended
  config only.
- Runtime state is not mutated directly.
- Modified reruns still return `pending_action` and wait for confirmation.

This keeps parameter editing inside the v3 route:

```text
language modification
-> V3AgentTurnService config_overrides
-> BasicGeant4Reasoner draft payload
-> geant4_payload_builder_tool applied_overrides
-> runtime preflight
-> pending_action(run_simulation)
```

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_v3_frontend_route.py tests\test_agent_v3_service.py tests\test_geant4_runtime_discovery.py tests\test_local_geant4_smoke_tool.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
29 passed
```

Real runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

## 2026-05-22: v3 Multi-Turn Modification Flow

Implemented the first v3-native multi-turn modification path.

Supported language pattern:

```text
把刚才方案改成 2 MeV 再跑
```

Updated files:

```text
core/agent_v3/service.py
core/agent_v3/reasoners.py
core/agent_v3/tools/geant4_tools.py
tests/test_agent_v3_service.py
```

Behavior:

- `V3AgentTurnService` detects language overrides such as `2 MeV` when the user
  phrases the turn as a modification.
- The service clears stale downstream observations from the previous run:
  payload draft, runtime preflight, runtime result, and commit gate.
- The original design/capability observations remain in session state.
- The reasoner drafts a new payload from the existing design and applies the
  language override.
- The runtime path then goes through preflight and stops at `pending_action`
  again, rather than silently running.

Current supported override:

```json
{
  "source_energy_mev": 2.0
}
```

Example validated flow:

```text
User: 请按默认参数运行一个铅屏蔽 gamma 模拟
Agent: pending_action(run_simulation)
User: 确认
Agent: observed(runtime)
User: 把刚才方案改成 2 MeV 再跑
Agent: new payload with source.energy=2.0 MeV, then pending_action(run_simulation)
```

Why this is v3-aligned:

- The frontend does not inspect the sentence or patch config directly.
- The old v2 accept/validate/run chain is not used.
- The modification is represented as turn metadata and applied by the v3 payload
  builder.
- A modified rerun still requires the same runtime confirmation gate.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_v3_frontend_route.py tests\test_agent_v3_service.py tests\test_geant4_runtime_discovery.py tests\test_local_geant4_smoke_tool.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
27 passed
```

Real runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

## 2026-05-22: v3 LLM-Assisted Design Tool

Added an LLM-assisted design path without breaking the v3 safety boundary.

Updated files:

```text
core/agent_v3/tools/geant4_tools.py
core/agent_v3/tools/__init__.py
core/agent_v3/reasoners.py
core/agent_v3/service.py
ui/web/v3_agent_api.py
ui/web/app.js
tests/test_agent_v3_geant4_tools.py
tests/test_agent_v3_service.py
tests/test_v3_frontend_route.py
```

New v3 tool:

```text
geant4_llm_design_tool
```

Design principle:

- LLM is allowed to help with simulation design only inside a `draft_only`
  tool boundary.
- LLM output is treated as a `SimulationDesign` draft, not as runtime truth.
- The design still flows through payload builder, runtime preflight, pending
  action confirmation, and Geant4 runtime observation.
- If the LLM call fails, returns non-JSON, or fails prompt validation, v3 falls
  back to deterministic design and records the fallback reason in the
  observation.

Reasoner behavior:

- Default path remains deterministic unless `llm_design_enabled` and
  `llm_config_path` are provided.
- When enabled, the reasoner proposes `draft_geant4_design_with_llm` after the
  Geant4 capability observation.
- The LLM design tool is registered in the same tool registry as the rest of v3,
  so it cannot bypass risk policy or controller trace.

Frontend behavior:

- The browser still sends user text to `/api/v3/agent/turn`.
- It enables LLM design only when the runtime model preflight reports ready.
- The frontend does not call the LLM directly and does not route around the v3
  controller.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_v3_frontend_route.py tests\test_agent_v3_service.py tests\test_geant4_runtime_discovery.py tests\test_local_geant4_smoke_tool.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
25 passed
```

Real runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```

Remaining gap:

- The frontend still calls the legacy/v2-orchestrated endpoint chain.
- The next iteration should switch `ui/web/app.js` to send user text to
  `/api/v3/agent/turn` and render answer/trace/proposals from that response.

## 2026-05-22: Frontend Default Path Switched to v3 Turn

Changed the browser conversation path so `sendPrompt()` now defaults to v3.

Updated file:

```text
ui/web/app.js
```

New default behavior:

```text
user text
-> requestV3AgentTurn()
-> POST /api/v3/agent/turn
-> applyV3Response()
-> render answer + v3 observations + trace
```

What deliberately no longer happens on the default path:

- No frontend call to `/api/geant4/intent` before the agent turn.
- No frontend branch deciding design vs run vs summary.
- No frontend auto-call to `/api/simulation/accept`.
- No frontend direct call to `/api/geant4/validate` or `/api/geant4/run`.

Compatibility note:

- The old functions remain in `app.js` for now, but they are behind the
  non-default compatibility path.
- `state.agentMode` is set to `v3`.
- Buttons are still UI controls, but the main user prompt path is now language
  turn driven.

Added guard tests:

```text
tests/test_v3_frontend_route.py
```

These tests assert that the default send path calls `/api/v3/agent/turn` and
does not call legacy intent/design/run helpers before the v3 branch returns.

## 2026-05-22: v3 Pending Action Protocol

Implemented explicit pending action support in the v3 turn service.

Updated files:

```text
core/agent_v3/service.py
core/agent_v3/reasoners.py
ui/web/app.js
tests/test_agent_v3_service.py
```

New response field:

```json
{
  "pending_action": {
    "schema_version": "geant4_agent_v3_pending_action.v1",
    "kind": "run_simulation",
    "intent": "run_geant4_runtime",
    "risk_level": "runtime_execution",
    "requires_confirmation": true,
    "expected_observation": "Geant4 runtime observation",
    "tool_call": {}
  }
}
```

Why this matters:

- The frontend no longer needs to infer a pending runtime action from raw
  `commit_gate` observations.
- The backend now remembers the pending action in `V3AgentState.metadata`.
- A follow-up turn that only says "确认" can execute the previously proposed
  runtime action.
- A cancellation turn such as "取消运行，只保留方案" clears the pending action and
  suppresses runtime intent, even though the sentence contains the word "运行".

This keeps the v3 route agentic: proposal, confirmation, cancellation, and
execution are owned by the v3 service/controller, not by frontend workflow
branches.

Verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_v3_frontend_route.py tests\test_agent_v3_service.py tests\test_geant4_runtime_discovery.py tests\test_local_geant4_smoke_tool.py tests\test_agent_v3_geant4_tools.py tests\test_v3_agent_smoke_tool.py
```

Result:

```text
22 passed
```

Real runtime smoke:

```powershell
.venv\Scripts\python.exe tools\run_v3_agent_smoke.py --auto-discover-runtime --run --events 2
```

Result:

```text
Geant4 runtime execution completed and produced a runtime observation.
```
