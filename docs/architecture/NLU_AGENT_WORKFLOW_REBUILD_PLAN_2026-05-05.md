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

## 2026-05-06 Review Addendum

A follow-up recommendation reviewed on 2026-05-06 keeps this plan's direction but
strengthens it from a layered NLU design into a production-grade agent workflow.
The accepted upgrades are:

- define an explicit `WorkflowGraphSpec` before large behavior changes
- use one shared `ActionSafetyClass` taxonomy across routing, confirmation,
  runtime guards, and evaluation
- build a knowledge-aware `ContextPack`, but never let knowledge snippets directly
  authorize configuration changes
- add an `EvidenceGroundingChecker` before LLM-proposed values reach normalization
- treat confirmation as staged interrupt/resume through `StagedPatchStore`
- add `IdempotencyReplayPolicy` before expanding runtime or batch side effects
- expand casebanks toward adversarial trajectory behavior, not phrase coverage

The upgraded workflow target is:

```text
IntentRouter
-> Knowledge-Aware ContextPack
-> LLM InterpretationFrame
-> EvidenceGroundingChecker
-> CandidatePatchNormalizer
-> TypedValidator
-> ConfirmationPolicy / InterruptResume
-> SessionApply
-> RuntimeActionGuard
-> ResultGroundedAnswer
-> TrajectoryEvaluation
```

This does not require adopting LangGraph or another framework immediately. First
define the internal graph contract, trace it, and prove the invariants with tests.

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

The router should return a typed `IntentDecision` rather than a loose string:

```python
@dataclass
class IntentDecision:
    intent: str
    confidence: float
    safety_class: str
    requires_kb: bool
    allowed_next_nodes: list[str]
```

### Layer 1.5: Workflow Graph Spec

Purpose: make the intended node sequence testable before decomposing
`session_manager.py`.

Initial nodes:

```text
start
route_intent
build_context
interpret
check_grounding
normalize_patch
validate
confirmation_policy
wait_confirmation
apply_session
runtime_guard
answer
end
```

Terminal states:

```text
read_only_answer
mutation_applied
waiting_confirmation
rejected
unsupported
runtime_action_guarded
error
```

The first implementation should be no-behavior-change: record the path in trace
and assert path invariants.

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
  "candidate_updates": [
    {
      "path": "source.energy",
      "operation": "set",
      "value": 10,
      "unit": "MeV",
      "confidence": 0.94,
      "evidence": [
        {"source": "explicit_user_text", "text_span": "10 MeV"}
      ]
    }
  ],
  "stable_slots_referenced": [],
  "ambiguities": [],
  "requires_confirmation": [],
  "unsupported_requests": [],
  "evidence": [
    {"text_span": "...", "maps_to": "source.energy"}
  ]
}
```

### Layer 2.5: Knowledge-Aware Context Pack

Purpose: give the LLM compact, auditable context without turning the knowledge base
into an unchecked config generator.

The context pack should include:

- user turn
- routed intent
- current session summary
- stable slots
- staged patch summary
- allowed config paths
- supported and unsupported capabilities
- retrieved knowledge snippets
- latest runtime summary when relevant
- context pack hash

Knowledge categories:

- capability KB can ground supported enum/capability decisions
- domain explanation KB can support user-facing explanation, not mutation
- implementation contract KB can support developer/runtime interpretation
- deprecated or unsupported KB can block or explain, never ground patches

Retrieval happens after routing. `read_config` needs current config and schema
labels, while `config_mutation` needs capability KB and allowed paths. Normal chat
should not receive Geant4 KB unless explicitly needed.

### Layer 2.6: Evidence Grounding Checker

Purpose: reject LLM-proposed values, paths, or assumptions that are not grounded.

Every candidate update must be grounded in at least one approved source:

- explicit user text
- existing session value
- capability KB
- implementation contract
- deterministic default policy
- derived value with formula trace

Numeric values are stricter: they must come from explicit user text, existing
session state, deterministic default policy, or a transparent derivation.
Ungrounded numbers are rejected, not merely confirmed.

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
- include node sequence, intent, action safety class, profile ids, LLM/fallback,
  candidate paths, confirmation status, applied/rejected paths, context pack hash,
  patch hash, grounding status, interrupt status, idempotency key/action id, and
  runtime readiness
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
- attach shared `ActionSafetyClass`
- attach `requires_kb` and allowed next workflow nodes
- ensure `/api/geant4/intent` and chat use the same router

Tests:

- config questions never call `step_async`
- result questions never call run/viewer
- run/viewer requests return guarded action only
- config mutations are the only chat path allowed into candidate generation

### P2.5: Add WorkflowGraphSpec

Goal: define graph nodes, transitions, terminal states, and path contracts without
changing behavior.

Tasks:

- add `core/agent/workflow_graph.py`
- define workflow nodes and terminal states
- map current `process_turn` branches onto graph paths
- write graph path into `NluTurnTrace`

Tests:

- `read_config` path never reaches session apply
- `normal_chat` path never reaches validation
- `config_mutation` path passes validation before session apply
- `run_requested` ends as guarded runtime action
- unsupported request ends as unsupported or clarification

### P3a: Add Knowledge-Aware ContextPackBuilder

Goal: build minimal, typed, auditable LLM context.

Status as of 2026-05-06:

- implemented a read-only `ContextPack` and `KnowledgeSnippet`
- exposed `context_pack` and `context_pack_hash` through `process_turn`
- added capability and unsupported capability visibility
- added adversarial context evaluation for unsupported CT scanner, implicit runtime
  requests, and LET scoring authority
- not yet connected to LLM prompt construction or config mutation authority

Tasks:

- add `core/agent/context_pack.py`
- define `ContextPack` and `KnowledgeSnippet`
- add a minimal capability KB loader
- separate supported, unsupported, deprecated, domain explanation, and implementation
  contract snippets

Tests:

- supported capability can ground enum mapping
- unsupported capability blocks fake config
- deprecated snippet cannot ground a candidate patch
- domain explanation can be used in answer but not mutation

### P3a.5: Composite Intent Policy

Goal: define how to handle turns that combine mutation and high-cost runtime intent
before the interpretation frame starts producing path-level patches.

Observed issue:

`Change source energy to 10 MeV and run 10 events now` is currently routed as
`run_requested`, so the safe runtime guard wins and mutation is not applied. This is
safe, but the long-term desired behavior is more nuanced:

```text
stage/validate the mutation
-> require or apply confirmation if needed
-> return guarded runtime action
-> never execute runtime from chat
```

Required policy:

- `mutation + run/viewer` must never execute runtime directly
- mutation can only apply after normal validation and confirmation policy
- runtime request is returned as a separate guarded action after mutation state is
  resolved
- if mutation is ambiguous or unsupported, runtime action is not offered yet
- trace must record both the mutation path and guarded runtime intent

This should be implemented before P3b if we want the LLM interpreter to support
composite user turns without losing safety.

### P3b: Build LLM Interpretation Frame V2

Goal: replace "LLM returns fields" with "LLM returns interpretation + evidence."

Status as of 2026-05-09:

- added `PromptTask.INTERPRET_USER_TURN_V2`
- added a path/evidence prompt profile for `candidate_updates`, `ambiguities`,
  `unsupported_requests`, and `guarded_actions`
- added validator checks for JSON-only output, allowed update paths, allowed ops,
  evidence source allowlist, guarded runtime actions, and grounded numeric values
- exposed `build_interpreter_v2_prompt()` for pilot usage
- added `run_interpreter_v2()` and an opt-in live LLM evaluator for checking real
  model output against the path/evidence contract
- not yet connected to session mutation or runtime execution

Tasks:

- add `PromptTask.INTERPRET_USER_TURN_V2`
- define strict JSON output with path-level `candidate_updates`, `ambiguities`,
  `requires_confirmation`, `evidence`
- validate output with allowlist and evidence checks
- keep existing slot/semantic extractors as compatibility adapters

Tests:

- non-JSON output rejects
- internal fields reject
- ungrounded numbers reject
- unsupported request routes to clarification/unsupported, not hallucinated config

### P3c: Add EvidenceGroundingChecker

Goal: reject ungrounded paths, values, numbers, and unsupported capabilities before
normalization.

Status as of 2026-05-09:

- added `core/agent/evidence_grounding.py`
- added `EvidenceGroundingContext` and `check_candidate_update_grounding()`
- interpreter v2 prompt validation now delegates path/evidence/number grounding to
  the checker
- covered invented numbers, unitless user numbers, stable context preservation,
  capability KB enum grounding, capability KB numeric rejection, unsupported KB
  grounding rejection, internal paths, and missing evidence spans
- not yet connected to candidate patch normalization or session mutation

Tasks:

- add `core/agent/evidence_grounding.py`
- check evidence source type and text span
- reject internal/private paths
- reject unsupported/deprecated KB as grounding
- reject ungrounded numeric values

Tests:

- LLM-invented number is rejected
- user number without unit asks clarification unless default unit policy exists
- existing stable value can be preserved
- capability KB can ground enum values, not arbitrary numbers

### P4: Candidate Patch Normalizer

Goal: centralize path-level update rules.

Status as of 2026-05-09:

- started P4a typed patch envelope
- added `core/agent/candidate_patch.py`
- interpreter v2 payload can now normalize into `CandidatePatchEnvelope`
- envelope can convert into existing `CandidateUpdate` without applying session
- confirmation reasons are tagged for low confidence, explicit overwrite, and
  delete/remove operations
- guarded runtime actions are preserved as separate guarded action requests
- added a read-only confirmation preview bridge that reuses the existing
  confirmation policy without mutating session state
- not yet connected to session mutation

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

Status as of 2026-05-10:

- added public `ConfirmationPolicyResult`
- added public `evaluate_confirmation_requirements()`
- added public `ConfirmationReason` constants for stable pending-confirmation
  wire values
- added public confirmation payload builder with stable `required`, `status`,
  `count`, `items`, and `available_responses` fields
- added public pending confirmation helpers for confirm-candidate construction,
  pending merge/path checks, unset checks, and pending item construction
- candidate patch preview now depends on the public API instead of private
  `_extract_*` helpers
- session manager confirmation evaluation and pending-confirmation flow now use
  the public API wrappers
- confirmation and candidate-patch tests no longer depend on confirmation
  private helper functions
- existing private helpers remain for lower-level compatibility tests and as
  implementation details
- not yet migrated confirmation into a staged patch store

Tasks:

- keep confirmation policy centralized in `core/orchestrator/confirmation_policy.py`
  until a real staged-patch boundary is introduced; do not create a duplicate
  `core/agent/confirmation_policy.py`
- define confirmation reasons as enum/string constants
- return user-visible confirmation payload
- support approve/reject/keep-original paths

Tests:

- low confidence mutation pauses
- overwrite pauses
- delete pauses
- approval applies exact staged patch
- rejection leaves state unchanged

### P5.1: StagedPatchStore + InterruptResumeController

Goal: treat confirmation as a structured paused workflow.

Tasks:

- add `core/agent/staged_patch.py`
- add `core/agent/interrupt_resume.py`
- create `confirmation_id` and `patch_hash`
- approval applies the exact staged patch
- rejection leaves session unchanged
- stale confirmation id is rejected
- edit creates a new interpretation turn

Tests:

- approval applies exact staged patch
- rejection leaves state unchanged
- stale confirmation is rejected
- mutated session invalidates older staged patch
- patch hash mismatch cannot apply

### P5.5: IdempotencyReplayPolicy

Goal: prevent repeated side effects during retry/replay.

Tasks:

- add `core/agent/idempotency.py`
- define replay behavior for interpretation, validation, session apply, Geant4 run,
  viewer launch, file write, and batch run
- introduce runtime `action_id` for non-replayable side effects

Tests:

- duplicate action id returns existing status/result
- retry after validation does not re-run Geant4
- viewer launch is never replayed automatically
- read-only summary can be repeated safely

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
- mutate-and-run in one turn must stage mutation and guard runtime action
- unsupported CT scanner must not become fake geometry
- stale confirmation must not mutate state
- domain explanation must not authorize unsupported scoring

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
- invalid JSON rate
- schema reject rate
- unsupported hallucination rate
- confirmation over-trigger and under-trigger rate
- grounding failure rate

Rules:

- never edit `.local.json` for model tests
- use `--model-override`
- never commit API keys
- full live run is opt-in

### P7.5: ModelRoutingPolicy

Goal: use stronger models only when the workflow needs them.

Suggested routing:

- cheap model for low-risk interpretation
- stronger model for ambiguous geometry/source/scoring
- escalation after validation or grounding failure
- fallback marked explicitly and never counted as success

### P8: Session Manager Decomposition

Goal: reduce maintenance risk after boundaries are proven.

Extract in this order:

1. `intent_router.py`
2. `workflow_graph.py`
3. `turn_trace.py`
4. `context_pack.py`
5. `evidence_grounding.py`
6. `candidate_patch.py`
7. `confirmation_policy.py`
8. `staged_patch.py`
9. `session_apply.py`
10. `idempotency.py`

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
- Knowledge snippets directly authorize unsupported config mutation.
- Deprecated knowledge grounds a candidate patch.
- Confirmation approval applies a different patch hash.
- Runtime action repeats after retry/replay.
- Stale confirmation can mutate state.
- Result answer uses an unavailable metric.
- Domain explanation KB is treated as capability support.

## Near-Term Execution Recommendation

P1, P2, P2.5, and the first P3a slice are complete enough to move forward.

Current checkpoint:

- `NluTurnTrace` exists and is returned by `process_turn`.
- `IntentDecision` and `WorkflowGraphSpec` are used by guard evaluators.
- `ContextPack` exists as read-only trace context.
- Scenario evaluators can check workflow nodes, runtime blocking, and context
  supported/unsupported capabilities.

Updated next steps:

1. Finish P3a by adding context-pack shape tests to guard/scenario evaluators and
   documenting the current capability KB as project support, not Geant4 support.
2. Add P3a.5 composite intent policy before starting P3b.
3. Start P3b only behind a feature flag after composite mutation/runtime behavior
   is explicitly specified.
4. Delay P3c implementation until we have at least one path-level interpretation
   frame to check.

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
