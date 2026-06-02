# Geant4 Agent v3 Next Implementation Plan

Date: 2026-05-25

This plan combines three review lenses:

- `ARC-7`: architecture review should be multi-perspective, evidence-based,
  severity-ranked, and migration-aware.
- `agent-eval-harness`: agent quality should be measured with repeatable
  run/grade/compare/calibrate pipelines, not only manual smoke tests.
- `mcp-builder`: tools should be discoverable, schema-driven, safe by default,
  and evaluated through realistic tool-use tasks.

## Goal

Make v3 an intelligent Geant4 research agent, not a keyword router and not a
form-filling pipeline.

The v3 agent should:

- understand the user's latest turn from context
- propose structured actions
- validate every proposed action before state changes or runtime execution
- preserve evidence and session memory
- expose Geant4 tools through clean contracts
- prove behavior through repeatable evals

## Non-Goals

- Do not delete legacy/v2 code in this pass.
- Do not move large directories yet.
- Do not make sweep or optimization auto-run.
- Do not replace deterministic safety checks with LLM judgment.
- Do not treat Python dictionaries as bad by themselves; the problem is
  keyword/dictionary routing as the decision policy.

## Phase 0: Baseline And Documentation Hygiene

Status: partially done.

Completed:

- Installed `ARC-7`.
- Installed agent-eval-harness components:
  - `trial-runner`
  - `trial-adapters`
  - `compare-trials`
- Identified local `mcp-builder` skill at repository root.
- Replaced stale architecture overview with v3-first `ARCHITECTURE.md`.
- Added `docs/architecture/README.md`.
- Removed obsolete HTML drafts that were already replaced by Markdown notes.

Remaining:

- Restart Codex later so newly installed skills are auto-discovered.
- Decide whether the two remaining v3 HTML snapshots should be converted to
  Markdown or deleted after content migration.
- Commit documentation cleanup separately from code changes.
- Rotate any API key that was exposed in the IDE or chat context.

Acceptance:

- `docs/architecture/README.md` clearly identifies current versus historical
  docs.
- No new primary architecture document is HTML-only.

## Phase 1: Turn Understanding Contract

Status: first implementation slice done.

Completed:

- Added `core/agent_v3/turn_understanding.py`.
- Added `V3TurnUnderstanding` and `V3RequestedChange`.
- Added explicit confirmation-event normalization.
- Service now stores the latest turn understanding in session metadata.
- Free-text confirmation/cancel remains as a compatibility fallback.
- Regex-derived modification fallbacks now pass through patch validation before
  they can affect payload rebuilds.
- Added `LLMTurnUnderstandingProvider` as the v3-first understanding path. It
  reads only `V3ContextPack` plus the latest user turn, then emits structured
  dialogue act, referenced state, requested changes, confirmation, risk intent,
  ambiguities, confidence, and reason.
- Service now runs LLM turn understanding before patch validation when an LLM
  config is supplied, so parameter edits can flow through
  `requested_changes -> patch validation -> payload rebuild` instead of relying
  on regex/dictionary routing.
- Low-confidence or invalid LLM understanding is contained as an uncertain
  understanding or deterministic fallback; it does not mutate state directly.

Remaining:

- Add richer ambiguity handling.

Add `core/agent_v3/turn_understanding.py`.

Create:

- `V3TurnUnderstanding`
- `V3RequestedChange`
- `V3ConstraintMention`
- `V3ConfirmationSignal`
- `LLMTurnUnderstandingProvider`
- deterministic schema validator

The understanding object should capture:

- user goal
- dialogue act
- referenced state
- requested changes
- constraints
- confirmation or rejection
- ambiguity
- risk intent
- confidence and reason

Do not route the main path through a tiny fixed intent enum.

Implementation notes:

- LLM reads only `V3ContextPack` plus the latest user turn.
- LLM returns JSON only.
- Invalid or low-confidence output becomes `ASK_USER`, not a guessed action.
- Existing keyword helpers become fallback only.

Tests:

- Chinese and English turns produce valid understanding.
- "解释刚才结果" references runtime result.
- "把能量改成 2 MeV" produces requested change, not direct mutation.
- malformed LLM output asks a clarification question.
- default v3 path does not call `intent_classifier.py`.

## Phase 2: Explicit UI Events

Status: second implementation slice done.

Completed:

- Pending-action confirm/cancel buttons now send `confirmation_event`.
- Backend prefers structured confirmation events over text matching.
- Frontend still displays the text turn for user readability.
- Pending actions now carry stable `action_id` values.
- UI confirm/cancel buttons prefer `pending_action.action_id`.
- Backend ignores explicit confirmation events whose `action_id` does not match
  the current pending action.

Remaining:

- Extend suggestion buttons with explicit action metadata for non-confirm
  actions.

Replace text-label guessing for UI buttons.

Add request metadata:

```json
{
  "confirmation_event": {
    "action_id": "pending-action-id",
    "decision": "confirm"
  }
}
```

UI suggestion buttons should send:

- `prefill` for editable text suggestions
- explicit event metadata for confirm/cancel/run actions
- action id when tied to a pending action

Backend behavior:

- Prefer `confirmation_event` over text matching.
- Free text confirmation remains supported through `TurnUnderstanding`.
- Confirmation can set `run_confirmed` only inside backend gate code.

Tests:

- confirm button runs the pending action after preflight.
- cancel button clears pending action.
- arbitrary LLM output cannot set `run_confirmed=True`.
- UI static tests prove default path uses `/api/v3/agent/*`.

## Phase 3: Patch-Based State Mutation

Status: first implementation slice done.

Completed:

- Added `core/agent_v3/patches.py`.
- Added `V3StatePatch` and `V3PatchValidationResult`.
- `config_overrides` now pass through patch validation before payload rebuild.
- LLM `requested_changes` and safe `parameters` are normalized through the
  same patch validator instead of bypassing service-side checks.
- Service saves `last_state_patch` in session metadata for auditability.
- Unsupported internal fields such as `run_confirmed` are rejected by the patch
  validator instead of being applied as ordinary config changes.
- Added `V3PatchApplyResult` and `apply_patches_to_state`.
- Valid payload-changing patches now invalidate stale payload, preflight,
  runtime, and commit-gate observations before the next proposal is built.
- Patch application clears stale pending actions and suggestions, removes stale
  draft artifacts, and saves `last_state_patch_apply` for auditability.
- Patch validation is now atomic: if one requested field is invalid or
  internal-only, the whole patch is rejected instead of partially mutating
  runtime configuration.
- Invalid patches now route to an `ASK_USER` clarification response, and stale
  pending run actions are cleared so the UI cannot confirm an old action after
  a failed modification attempt.
- LLM-extracted parameters in an initial design-only turn are recorded for
  audit but do not automatically imply `accept_defaults` or payload generation.

Remaining:

- Apply validated patches directly to a richer draft model when v3 has one;
  today `config_overrides` remains the compatibility bridge into payload build.
- Connect a live `TurnUnderstanding` provider directly to patches.

Add `core/agent_v3/patches.py`.

Create:

- `V3StatePatch`
- `V3PatchOperation`
- `validate_patch`
- `apply_patch_to_draft`
- stale-observation invalidation rules

Parameter edits should flow as:

```text
user turn
-> TurnUnderstanding.requested_changes
-> V3StatePatch
-> validation
-> payload rebuild
-> pending runtime action
```

This replaces regex-first config override inference as the main path.

Rules:

- Patch paths must be whitelisted by schema, not arbitrary strings.
- Unit-bearing values must be normalized before application.
- Changing source/material/geometry/events invalidates stale payload,
  preflight, and runtime observations.
- Invalid values ask a question instead of mutating state.

Tests:

- energy, particle, material, geometry thickness, and event count edits become
  patches.
- invalid material asks for clarification.
- runtime result is not reused after payload-changing patches.
- patch evidence is saved in state trace.

## Phase 4: Proposal Critic

Status: first implementation slice done.

Completed:

- Added `core/agent_v3/proposal_critic.py`.
- Wired the critic into the default v3 `AgentController`.
- Runtime execution is blocked unless the state already has a successful
  payload draft and a successful preflight observation.
- LLM-proposed runtime actions cannot skip directly to pending confirmation
  when payload/preflight evidence is missing.
- v3 state summary now distinguishes payload readiness from runtime readiness:
  a payload alone is `payload_ready`, while `runtime_ready` requires either a
  successful preflight or an active confirmation-gated pending action.
- v3 state summary now exposes `has_preflight`, `preflight_status`, and
  `runtime_ready_reason` so UI/debug panels can explain why a run can or cannot
  proceed.
- Failed or non-evaluable preflight now maps to `runtime_preflight_blocked`
  with `fix_runtime_or_modify_payload` as the next action instead of looking
  like a runnable payload.
- Proposal critic now reads `ToolRegistry` when called by the controller.
- Unknown tools, internal-only tool arguments, and schema-invalid tool
  arguments are blocked before tool invocation.
- Tool risk mismatches are recorded as review observations before commit gate;
  the controller still gates using the registered tool risk.
- Context-fact grounding now blocks payload-builder proposals whose `design`
  does not match the latest session design, and runtime proposals whose
  `payload_builder_observation` or `recommended_config` does not match the
  latest session payload.

Remaining:

- Broaden context grounding beyond design/payload/runtime payload objects where
  new tools introduce additional context-derived arguments.
- Add explicit checks for any new internal-only flags beyond the current deny
  list.

Add `core/agent_v3/proposal_critic.py`.

The critic sits between reasoner and controller action execution.

It checks:

- proposed tool exists
- tool risk level matches registry
- tool arguments match schema
- proposal references only context-visible facts
- runtime action has preflight
- risky action has confirmation
- internal-only flags are not set by LLM
- suggested sweep/optimization is not executed in the same turn

Tests:

- hallucinated tool is rejected.
- hallucinated material/particle is blocked or clarified.
- runtime execution cannot skip preflight.
- `allow_in_memory=True` is accepted only from explicit user/runtime config.

## Phase 5: MCP Tool Contract Hardening

Status: first enforcement slice done.

Completed:

- Extended v3 `ToolSpec` metadata with output schema,
  `confirmation_required`, and `idempotency_hint`.
- Added input/output contract metadata to the default Geant4 v3 tools.
- Marked runtime execution as confirmation-required at the tool contract level.
- `AgentController` now gates actions using the registered tool risk, not the
  risk claimed by a proposal.
- `ToolRegistry.invoke()` now validates tool arguments against the registered
  input schema before calling handlers.
- Invalid tool inputs return machine-readable `schema_errors` and
  `input_schema_validation_failed` instead of entering tool logic.
- Proposal-critic schema failures now include repair suggestions that the
  dialogue layer can show to the user.
- Proposal-critic grounding failures now include repair suggestions for
  rebuilding from the latest session design or payload.
- LLM design observations are post-grounded against explicit user facts such as
  `G4_*` material IDs, source particle, and source energy, so a design tool
  cannot silently replace an explicit `G4_WATER` target with a default lead
  shielding material.
- LLM turn-understanding prompts now use `V3ContextPack` only and have tests
  proving raw metadata, full runtime payloads, and raw observation secrets are
  not copied into the prompt.
- Result-analysis temporary context construction no longer copies full session
  metadata or prior raw observations when synthesizing a context pack from a
  supplied runtime result.

Remaining:

- Broaden schema support where needed beyond the current lightweight object
  validator.
- Add richer output summary schemas per tool instead of the shared observation
  envelope.

Apply `mcp-builder` guidance to `mcp/geant4` and v3 tool registry.

Changes:

- Extend `ToolSpec` with:
  - input schema
  - output summary schema
  - risk annotation
  - idempotency guidance
  - confirmation requirement
- Make tool descriptions action-oriented and discoverable.
- Return actionable errors with next-step suggestions.
- Keep agent decision logic out of `mcp/geant4`.

Target Geant4 tools:

- capability inspection
- design draft
- payload build
- runtime preflight
- runtime execution
- result summarization
- future sweep plan draft

Tests:

- every registered tool has schema, description, and risk level.
- runtime tool cannot be called without gate.
- errors include machine-readable reason and human next step.
- MCP adapter tests verify structured outputs.

## Phase 5.5: Dialogue Experience And Response Intelligence

Status: first implementation slice done.

Completed:

- v3 dialogue suggestions are now structured objects with user-facing text and
  executable prefill text.
- Blocked actions now explain the recovery path instead of pointing users only
  to raw trace.
- Proposal-critic blocks for missing payload/preflight produce concrete next
  actions.
- UI suggestion handling preserves structured confirmation events when present.
- Pending confirm/cancel fallback binding keeps sending `confirmation_event`
  instead of falling back to text-only guessing.
- Added a non-blocking response-quality evaluator for clarity, grounding,
  next-step usefulness, clickable suggestions, and raw internal trace leakage.
- `/api/v3/agent/turn` now returns `dialogue_quality`, and the session stores
  the latest quality report for eval/debug use.
- `waiting_user` responses that already have a design draft now show the design
  summary, assumptions, and explicit stop condition instead of a generic
  confirmation question.
- Dialogue labels are locale-aware for design summaries, so English responses
  no longer mix Chinese geometry/source/observable labels.
- Invalid patch clarification keeps the patch error visible even when an older
  design draft exists in state.
- v3 dialogue responses now include structured `answer_parts` in addition to
  the human `display_message`: summary, evidence, next-step suggestions, and
  dialogue act. This gives the UI and eval harness a stable response shape
  without exposing raw trace or metadata.
- `/api/v3/agent/turn` mirrors `answer_parts` at the top level and inside
  `answer`, while preserving the existing `display_message` contract.
- The web UI renders structured answer parts for agent messages, showing
  evidence and next steps as lightweight scan-friendly rows instead of relying
  only on a long paragraph.
- Response-quality checks now verify that v3 responses include structured
  answer parts, not just free text.
- Added optional `V3ResponseNaturalizer`, which rewrites only the human
  `display_message` from `V3ContextPack`, `answer_parts`, evidence, and state
  summary. It never receives raw metadata or full observations.
- Naturalized responses are accepted only after deterministic grounding checks:
  internal marker leaks, material conflicts, particle conflicts, and source
  energy conflicts fall back to the original deterministic message.
- `/api/v3/agent/turn` can enable naturalization with
  `llm_naturalize_enabled`; default behavior remains deterministic.
- `tools/run_v3_live_llm_dialogue.py` now reports dialogue eval metrics:
  turn count, ok turn count, dialogue-quality pass count/min score/warnings,
  and naturalization enabled/reported/ok/fallback counts with fallback reasons.
- The live dialogue tool accepts `--naturalize` to opt into response
  naturalization while keeping default eval runs deterministic.
- Added `docs/eval/v3_dialogue_casebank.json` as the first fixed v3 dialogue
  casebank for design, payload, confirmation, cancellation, and read-only
  result-question behavior.
- Added `tools/evaluate_v3_dialogue_casebank.py`, which runs the fixed casebank
  through `V3AgentTurnService`, emits per-case trajectories, dialogue-quality
  metrics, naturalization metrics, and optional saved eval records.
- The same tool has `--adapter` mode: it reads one JSON object from stdin and
  writes exactly one JSON object to stdout with `status`, `message`,
  `trajectory`, and safe `metadata`, giving the future agent-eval-harness a
  command-only adapter target.

Remaining:

- Expand `v3_dialogue_casebank.json` from smoke coverage to 20+ cases covering
  stale context resistance, multilingual edits, sweep suggestion safety, and
  runtime result grounding.
- Decide whether the fixed v3 casebank should stay JSON array format or move to
  JSONL before integration with the external eval harness runner.
- Add richer UI rendering for runtime facts and status once the current
  `answer_parts` contract has settled.
- Add multilingual copy tests for Chinese and English user flows.

Acceptance:

- Every v3 response should answer: what happened, what evidence it used, and
  what the user can do next.
- Suggested next actions should be clickable without losing session context.
- Safety blocks should feel helpful, not like unexplained errors.

## Phase 6: Eval Harness For Agent Behavior

Apply `agent-eval-harness` as a repeatable evaluation layer.

Create a v3 eval suite with:

- task JSONL corpus
- adapter executable for `/api/v3/agent/turn`
- grader for behavior and evidence
- compare step for baseline vs current branch
- calibration step for grader quality

Task slices:

- design-only
- clarify missing physics/material/source
- accept defaults
- modify payload after design
- run confirmation
- runtime result explanation
- stale context resistance
- sweep suggestion safety
- multilingual turns
- malformed/ambiguous user input

Metrics:

- pass rate by slice
- confirmation safety
- runtime evidence grounding
- no stale-result reuse
- no legacy endpoint use
- suggested action quality
- latency and token/cost metadata when available

Acceptance:

- one command can run a stable v3 eval suite.
- compare output can block regressions before merge.
- calibration cases catch grader false positives.

## Phase 7: Architecture Review Gate

Use ARC-7-style review before merging major v3 changes.

Review dimensions:

- architecture and contracts
- safety and secret handling
- user value and scope
- simplification opportunities
- performance and cost
- failure modes

Output should be a short review report under `docs/reports/architecture/` or a
PR comment, with severity-ranked findings.

Acceptance:

- every major v3 milestone has an architecture review note.
- findings are actionable and tied to files or contracts.
- uncertainty is labeled instead of asserted.

## Recommended Implementation Order

1. `TurnUnderstanding` contract and tests.
2. Explicit UI confirmation events.
3. Patch-based state mutation.
4. Proposal critic.
5. Tool registry schema hardening.
6. Eval harness adapter and first 20 tasks.
7. Sweep plan object and deferred execution.
8. ARC-7-style review report before merge.

## First Coding Slice

The smallest useful implementation slice is:

```text
TurnUnderstanding
-> explicit confirmation_event
-> no default v3 intent_classifier usage
-> tests for confirm/cancel/modify/result-question
```

This slice directly reduces the current keyword-routing risk without touching
the runtime adapter or moving directories.

## Merge Criteria

Ready to merge when:

- all existing v3 tests pass
- new turn-understanding and confirmation-event tests pass
- repeatable eval harness tasks cover each critical slice once the contracts are
  stable
- no API key or local config file is tracked
- documentation explains the new flow in `ARCHITECTURE.md`
