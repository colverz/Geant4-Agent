# NLU Agent Workflow Rebuild Plan

Date: 2026-05-05

## Executive Summary

Do not rebuild NLU by deleting the current pipeline. Rebuild the NLU *role*.

The old mental model is:

```text
natural language -> slot/dictionary extraction -> config fields
```

The target agentic model is:

```text
user turn
-> intent router
-> LLM interpretation
-> candidate patch
-> typed validator
-> confirmation policy
-> session apply or read-only answer
-> runtime/result loop
-> trajectory evaluation
```

This keeps the useful parts we already have, but moves the project away from
phrase accumulation. LLMs should handle interpretation, disambiguation, planning,
and explanation. Deterministic code should own facts, state transitions, safety,
and runtime execution.

## External Design Signals

The plan is based on public agent systems and documentation, not leaked source.

- OpenAI Agents guardrails separate input/output/tool checks and support tripwire
  behavior before unsafe or expensive steps.
- OpenAI Agents tracing records LLM generations, tool calls, handoffs, guardrails,
  and custom events, which matches our need to evaluate trajectories.
- LangGraph durable execution emphasizes resumable, stateful workflows and warns
  that side effects should be wrapped so they are not repeated during replay.
- Microsoft Agent Framework tool approvals model high-risk function calls as
  explicit human-in-the-loop approval requests.
- Claude Code hooks/subagents show two useful concepts for us: pre/post action
  boundaries and task-specialized agents with separate context.
- SWE-agent and mini-SWE-agent demonstrate that mature agent evaluation is
  trajectory/tool/result oriented, not just final-text matching.
- DeepSeek API docs list `deepseek-v4-flash` and `deepseek-v4-pro`; Flash is the
  default low-cost live smoke model for this project, Pro is escalation-only.

## Current Diagnosis

### What Is Working

- Legacy/v2 switch gives rollback safety.
- `PromptProfile` gives prompt ownership and language/task separation.
- Runtime bridge is typed enough to validate `SimulationSpec`, runtime payload,
  smoke report, and result summary.
- Guard layer already blocks accidental run/viewer from normal chat.
- Live LLM evaluator now confirms `llm_used=true`, no silent fallback, and runtime
  payload readiness.
- DeepSeek V4 Flash has passed the current 5-case live scenario smoke.

### What Is Still Weak

- NLU is still too centered on extraction output rather than agent state.
- Casebanks are still partly "advanced dictionaries" unless they check behavior
  boundaries and trajectories.
- `session_manager.py` still mixes intent, interpretation, candidate merging,
  confirmation, session mutation, and response generation.
- BERT/local deterministic extraction remains useful as fallback, but should not
  be the main authority for complex geometry/source interpretation.
- Multi-turn memory exists, but the boundary between stable slots, user intent,
  and mutable candidate patches is not clean enough.
- Live LLM tests are too small and too "happy path" to prove robust user behavior.

## Target Architecture

### Layer 1: User Turn Router

Purpose: classify what kind of turn this is before any expensive or mutating work.

Outputs:

- `read_config`
- `read_summary`
- `config_mutation`
- `clarification_answer`
- `run_requested`
- `viewer_requested`
- `normal_chat`

Rules:

- `read_config` and `read_summary` are always read-only.
- `run_requested` and `viewer_requested` return guarded runtime actions, not direct
  execution from chat.
- Only `config_mutation` and `clarification_answer` can enter candidate generation.

### Layer 2: LLM Interpretation Agent

Purpose: convert a user turn plus compact context into an interpretation frame.

Responsibilities:

- infer task intent
- normalize language-specific phrasing
- identify candidate geometry/source/physics/output/scoring changes
- identify ambiguity and ask clarification
- preserve known stable slots unless the user clearly changes them

Non-responsibilities:

- no direct session mutation
- no runtime execution
- no unsupported physics/geometry invention
- no hidden fallback success

Output contract:

```json
{
  "intent": "config_mutation",
  "confidence": 0.0,
  "candidate_patch": {},
  "stable_slots_referenced": [],
  "ambiguities": [],
  "requires_confirmation": [],
  "unsupported_requests": [],
  "evidence": [
    {"text_span": "...", "maps_to": "source.energy"}
  ]
}
```

### Layer 3: Candidate Patch Normalizer

Purpose: convert LLM interpretation into internal update candidates.

Rules:

- accept only allowed config paths
- preserve units and explicit user values
- attach evidence/source metadata to each update
- mark deletes/overwrites as confirmation-required
- downgrade low-confidence updates to clarification

### Layer 4: Typed Validator

Purpose: decide whether a candidate can become a simulation contract.

Validation targets:

- schema allowed paths
- unit consistency
- material mapping
- geometry/source compatibility
- source direction/position validity
- scoring/detector compatibility
- runtime payload readiness

This layer owns "truth" for executable configuration. The LLM only proposes.

### Layer 5: Confirmation Policy

Purpose: make unsafe or ambiguous state changes visible to the user.

Confirmation triggers:

- overwrite known stable slot
- delete existing value
- low confidence update
- unsupported geometry/source/scoring request
- run/viewer/batch/high-cost operation
- LLM output introduces numbers not grounded in user text/context

### Layer 6: Session Apply

Purpose: mutate session state only after routing, interpretation, validation, and
confirmation have agreed.

Rules:

- one session turn mutation boundary
- write audit event for every applied update
- store previous value, new value, confidence, evidence, profile id
- read-only paths cannot change `turn_id`

### Layer 7: Runtime/Result Loop

Purpose: connect typed config to Geant4 and make results available for follow-up.

Rules:

- runtime run/viewer require explicit UI/runtime action
- `/api/geant4/summary` is read-only
- LLM result explanation is grounded rewrite only
- result Q&A must cite `RuntimeSmokeReport` fields or say not available

### Layer 8: Agentic Evaluation

Purpose: evaluate behavior, not just field matching.

Required trace fields:

- `turn_id_before`, `turn_id_after`
- `intent`
- `action_safety_class`
- `prompt_profile_id`
- `llm_used`
- `fallback_reason`
- `candidate_patch_paths`
- `confirmation_required`
- `applied_paths`
- `rejected_paths`
- `runtime_payload_ready`
- `tool_calls_allowed`
- `tool_calls_blocked`

## Migration Plan

### P0: Freeze Current Good Runtime Boundary

Status: mostly complete.

Exit criteria:

- default runtime is in-memory
- real Geant4 is opt-in
- viewer is guarded
- summary is read-only
- full tests and secrets scan pass

### P1: Add NLU Agent Trace Object

Goal: introduce a typed `NluTurnTrace` without changing behavior.

Tasks:

- create `core/nlu_trace.py` or `core/agent/turn_trace.py`
- populate trace from existing `process_turn`
- include intent, profile ids, LLM/fallback, candidate paths, confirmation status,
  applied/rejected paths, runtime readiness
- return trace in debug/API response under a stable field

Tests:

- read-only turns have no applied paths
- config mutation records candidate and applied paths
- live LLM turn records `llm_used=true`
- fallback turn records fallback reason

### P2: Extract Intent Router Boundary

Goal: separate turn routing from extraction.

Tasks:

- move intent classification into a small module, e.g. `core/agent/intent_router.py`
- keep deterministic router first, optional LLM router later
- make router return `IntentDecision`
- ensure `/api/geant4/intent` and chat use the same router

Tests:

- config questions never call `step_async`
- result questions never call run/viewer
- run/viewer requests return guarded action only
- config mutations are the only chat path allowed into candidate generation

### P3: Build LLM Interpretation Frame

Goal: replace "LLM returns fields" with "LLM returns interpretation + evidence."

Tasks:

- add `PromptTask.INTERPRET_USER_TURN_V2`
- define strict JSON output with `candidate_patch`, `ambiguities`,
  `requires_confirmation`, `evidence`
- validate output with allowlist and evidence checks
- keep existing slot/semantic extractors as compatibility adapters

Tests:

- non-JSON output rejects
- internal fields reject
- ungrounded numbers reject
- unsupported request routes to clarification/unsupported, not hallucinated config

### P4: Candidate Patch Normalizer

Goal: centralize path-level update rules.

Tasks:

- create `core/agent/candidate_patch.py`
- convert interpreter frame into typed update candidates
- deduplicate current scattered merge/shadow/drop logic
- preserve existing slot static behavior

Tests:

- static known slot is preserved across unrelated turns
- explicit overwrite requires confirmation
- delete requires confirmation
- same update from LLM and deterministic extractor merges without duplication

### P5: Confirmation Policy Module

Goal: stop spreading confirmation logic through session orchestration.

Tasks:

- create `core/agent/confirmation_policy.py`
- define confirmation reasons as enum/string constants
- return user-visible confirmation payload
- support approve/reject/keep-original paths

Tests:

- low confidence mutation pauses
- overwrite pauses
- delete pauses
- approval applies exact staged patch
- rejection leaves state unchanged

### P6: Agentic Casebank V2

Goal: replace dictionary-like live casebank with behavior cases.

Case types:

- read config after previous configuration
- ask result after run summary
- ambiguous geometry request
- overwrite source particle but preserve source direction
- delete detector/scoring
- mixed-language correction
- request unsupported complex geometry
- request run/viewer from chat
- result question asking for unavailable dose metric

Expected fields:

```json
{
  "expected_intent": "config_mutation",
  "expected_safety": "config_mutation",
  "expected_trace": {
    "must_use_llm": true,
    "forbid_fallback": true,
    "must_require_confirmation": false,
    "must_have_runtime_payload": true,
    "must_not_call_runtime": true
  },
  "expected_state": {
    "turn_delta": 1,
    "applied_paths": ["source.energy"],
    "preserved_paths": ["source.direction"]
  }
}
```

### P7: Live LLM A/B Evaluation

Goal: compare models by trajectory quality, not vibes.

Default matrix:

- `deepseek-v4-flash`: default live smoke
- `deepseek-v4-pro`: escalation when Flash fails
- existing local/Ollama model: offline/local comparison only

Metrics:

- scenario accuracy
- trajectory pass rate
- fallback rate
- confirmation correctness
- runtime payload readiness
- hallucinated value rejection rate
- average latency
- token/cost if available

Rules:

- never edit `.local.json` for model tests
- use `--model-override`
- never commit API keys
- full live run is opt-in

### P8: Session Manager Decomposition

Goal: reduce maintenance risk after boundaries are proven.

Extract in this order:

1. `intent_router.py`
2. `turn_trace.py`
3. `candidate_patch.py`
4. `confirmation_policy.py`
5. `session_apply.py`

Do not split everything at once. Each extraction needs tests before and after.

### P9: BERT Role Reassignment

Goal: keep BERT useful without making it the complex-geometry authority.

New role:

- low-cost local hinting
- fallback extraction when LLM unavailable
- confidence signal
- regression comparison target

Not its role:

- final complex geometry modeling
- unsupported source/scoring invention
- replacing LLM interpretation for ambiguous user intent

## Testing Strategy

### Always-On Tests

- prompt profile contracts
- intent/action guard contracts
- candidate patch normalization
- confirmation policy
- runtime payload contracts
- read-only API invariants
- secrets scan

### Opt-In Tests

- live LLM NLU scenario
- live LLM agentic casebank
- real Geant4 smoke
- model A/B comparison

### Required Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/test_prompt_profiles.py tests/test_runtime_intent.py tests/test_workflow_guard_contract.py -q
.venv\Scripts\python.exe -m pytest tests/test_llm_scenario_parsing_benchmark.py -q
.venv\Scripts\python.exe tools\evaluate_llm_scenario_parsing.py --json --max-cases 2
powershell -ExecutionPolicy Bypass -File tools\check_secrets.ps1
```

Opt-in live:

```powershell
.venv\Scripts\python.exe tools\evaluate_llm_scenario_parsing.py `
  --live-llm `
  --casebank docs\eval\llm_scenario_live_casebank.json `
  --llm-config nlu\llm_support\configs\deepseek_api.local.json `
  --model-override deepseek-v4-flash `
  --json
```

## Stop Conditions

Stop and reassess if any of these happen:

- NLU refactor breaks legacy/v2 switch.
- Casebank growth becomes phrase enumeration again.
- LLM can mutate session without validator approval.
- Read-only questions change session state.
- Runtime actions can be triggered from normal chat.
- Live LLM fallback is counted as success.
- Secrets scan fails.

## Near-Term Execution Recommendation

Start with P1 and P2 only.

Reason:

- They create observability and routing boundaries without destabilizing the parser.
- They make future failures diagnosable.
- They let us prove whether a deeper rebuild is necessary before touching the
  high-risk candidate merge logic.

After P1/P2 pass full regression, move to P3 with a small interpreter-frame pilot
behind a feature flag.

## References

- OpenAI Agents guardrails: https://openai.github.io/openai-agents-js/guides/guardrails/
- OpenAI Agents tracing: https://openai.github.io/openai-agents-python/tracing/
- LangGraph durable execution: https://docs.langchain.com/oss/python/langgraph/durable-execution
- LangGraph overview: https://docs.langchain.com/oss/javascript/langgraph/overview
- Claude Code hooks: https://docs.anthropic.com/en/docs/claude-code/hooks
- Claude Code subagents: https://docs.claude.com/en/docs/claude-code/subagents
- Microsoft Agent Framework tool approvals: https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval
- SWE-agent repository: https://github.com/SWE-agent/SWE-agent
- DeepSeek model list: https://api-docs.deepseek.com/api/list-models/
