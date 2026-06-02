# Geant4 Agent v3 Intelligence Design

Date: 2026-05-25

This note defines the next v3 design step after the session/context/API
consolidation pass. The goal is to make the agent feel like an intelligent
simulation collaborator instead of a keyword router.

## Plain Goal

The agent should understand what the user is trying to do, keep track of the
current experiment, ask for missing information only when needed, and explain
results from the actual runtime evidence.

The agent should not rely on hard-coded keyword tables such as:

- "if text contains this word, choose this workflow"
- "if text matches this phrase, force this parameter"
- "if result question contains this token, use this canned explanation"

Python dictionaries are still allowed for JSON-like contracts, tool schemas,
runtime payloads, and API responses. The restriction is about agent reasoning,
not normal data representation.

## Current State

The v3 main path is now:

```text
UI
-> /api/v3/agent/turn
-> V3AgentTurnService
-> V3SessionStore
-> V3ContextPack
-> AgentController
-> Reasoner
-> ToolRegistry
-> Geant4 runtime tools
-> V3ContextPack / V3StateSummary
-> UI
```

This path is usable and tested. It already avoids feeding raw runtime blobs to
the LLM by using `V3ContextPack`.

The remaining weakness is that some turn handling is still too mechanical:

- confirmation and cancellation detection use fixed phrases
- some parameter changes are inferred with regex-heavy helpers
- sweep requests are detected with keyword checks
- `BasicGeant4Reasoner` is still a linear fallback workflow
- `intent_classifier.py` still exposes a small finite intent list

These are acceptable as temporary guardrails, but they should not be the main
intelligence layer.

## Design Principle

Use this rule for v3:

```text
LLM understands intent and proposes a structured action.
Deterministic code validates, gates, and applies the action.
```

That means the LLM may say:

```json
{
  "user_goal": "compare shielding performance at several beam energies",
  "requested_change": {
    "field": "source.energy_mev",
    "values": [1, 2, 5]
  },
  "proposed_action": "draft_sweep_plan",
  "needs_confirmation": true
}
```

But deterministic code decides whether that action is allowed, whether the
field exists, whether the units are valid, whether runtime execution needs
confirmation, and whether the session state can be changed.

## Proposed Architecture

### 1. Turn Understanding Layer

Add a v3-native `TurnUnderstanding` contract.

It should capture what the user means without forcing the text into a tiny
intent enum.

Suggested fields:

- `dialogue_act`: ask, answer, confirm, reject, revise, request_run, inspect_result
- `user_goal`: short natural-language goal
- `referenced_state`: design, payload, runtime_result, pending_action, none
- `requested_changes`: list of structured changes
- `constraints`: materials, geometry, source, energy, events, observables
- `confirmation`: confirmed, rejected, unclear, not_applicable
- `risk_intent`: read_only, draft_only, run_requested, external_side_effect
- `ambiguities`: questions the agent should ask before proceeding
- `confidence`: numeric confidence plus a short reason

This replaces the current idea of routing the turn through a fixed dictionary
of intents.

### 2. Proposal Planner

The reasoner should produce one or more `V3ActionProposal` objects from:

- `V3ContextPack`
- the tool registry's available capabilities
- the new `TurnUnderstanding`
- previous observations and unresolved questions

The planner should not decide by keyword. It should choose actions by asking:

- What does the user want now?
- What evidence do we already have?
- What is missing?
- Which tool can safely produce the missing evidence?
- Is this action read-only, draft-only, state-changing, or runtime execution?

### 3. Validation And Grounding

Every LLM proposal must pass deterministic review before it mutates state or
runs Geant4.

Checks should include:

- requested tool exists in `ToolRegistry`
- arguments match the tool schema
- referenced design/payload/runtime facts exist in context
- unit-bearing values are normalized and valid
- state changes are represented as patches
- runtime execution always goes through preflight and confirmation

This keeps the agent intelligent without making it unsafe.

### 4. Patch-Based State Mutation

Move parameter edits to a patch pipeline.

Instead of directly modifying state from text, the reasoner should produce:

```json
{
  "patches": [
    {
      "path": "source.energy_mev",
      "op": "replace",
      "value": 2.0,
      "evidence": "User asked to change beam energy to 2 MeV"
    }
  ]
}
```

The patch validator then checks whether the path exists, the value is valid,
and the change invalidates old payload/runtime observations.

This replaces ad hoc parameter inference as the primary mechanism.

### 5. Confirmation As A First-Class Event

UI buttons should send explicit metadata, not rely on the text label.

For example:

```json
{
  "confirmation_event": {
    "action_id": "...",
    "decision": "confirm"
  }
}
```

Free-text confirmations can still be understood by the LLM layer, but the
default UI path should not need keyword matching for "确认运行".

### 6. Result Reasoning

Result analysis should remain evidence-first.

The LLM receives only:

- current user question
- `latest_runtime_facts`
- payload summary
- known assumptions
- allowed next-step tools

It should not receive stale examples like Pb/gamma unless they are the actual
runtime facts. If it suggests a sweep or optimization, that suggestion becomes
a next action button and does not execute in the same turn.

### 7. Tool Catalog As Capability Surface

The tool registry should become the agent's menu of possible actions.

Each tool should expose:

- name
- risk level
- short purpose
- input schema
- output summary shape
- confirmation requirement

The reasoner should select tools from this catalog. This is different from a
dictionary router: the catalog describes what the world can do, while the
reasoner decides which capability fits the user's goal.

## Migration Plan

### Phase A: Introduce Smart Turn Understanding

Add:

- `core/agent_v3/turn_understanding.py`
- `V3TurnUnderstanding`
- `LLMTurnUnderstandingProvider`
- deterministic validator for the understanding object

Keep `intent_classifier.py` only as compatibility or remove it from the v3
default path.

Tests:

- multilingual user turns produce structured understanding
- no finite intent table is required for normal routing
- malformed LLM output falls back to a safe ask-user response

### Phase B: Replace Keyword Confirmation In UI Path

Add explicit confirmation metadata from UI suggestion buttons.

Backend should prefer:

```text
payload.confirmation_event
```

over text matching.

Free text can still be interpreted by the LLM, but button clicks should be
unambiguous.

Tests:

- clicking confirm runs the pending action
- clicking cancel clears pending action
- changing config clears stale payload/runtime observations
- no UI path depends on legacy `/api/agent/state`

### Phase C: Patch-Based Parameter Changes

Add:

- `core/agent_v3/patches.py`
- `V3StatePatch`
- patch validator
- stale-observation invalidation rules

Replace `_infer_config_overrides_from_text` as the main path. Keep numeric
regex helpers only for unit normalization after semantic extraction.

Tests:

- "change energy to 2 MeV" becomes a patch
- "use water instead" becomes a patch
- invalid material asks a question instead of mutating state
- old runtime result is not reused after a payload-changing patch

### Phase D: Proposal Critic

Add a deterministic critic between reasoner and controller action.

The critic should reject or ask for clarification when:

- the proposal references facts not in context
- the proposed tool is not registered
- runtime execution skips preflight
- confirmation is missing for risky action
- the LLM tries to set internal-only flags like `run_confirmed`

Tests:

- LLM cannot directly set `run_confirmed=True`
- LLM cannot request in-memory runtime unless user explicitly enabled it
- hallucinated material/particle is blocked or clarified

### Phase E: Agentic Sweep Plan

Represent sweep as a plan, not immediate execution.

Flow:

```text
result question
-> agent suggests sweep plan
-> user accepts
-> payloads are drafted
-> preflight
-> confirmation
-> runtime executes one planned step or approved batch
-> trend analysis from real observations
```

Tests:

- sweep suggestion never executes in the same turn
- accepted sweep still requires confirmation
- trend answer uses only collected runtime facts

## What To Keep

Keep these parts as the v3 foundation:

- `V3SessionStore`
- `V3ContextPack`
- `/api/v3/agent/state`
- `V3ActionProposal`
- `ToolRegistry`
- commit gate and runtime confirmation
- evidence-first result analysis

## What To De-Emphasize

These should become fallback or compatibility code:

- `intent_classifier.py`
- keyword-heavy confirmation detection
- keyword sweep detection
- regex-first parameter override inference
- `BasicGeant4Reasoner` as the main policy

They can stay for tests and degraded mode, but not as the normal product path.

## Acceptance Criteria

The next implementation pass is done when:

- default v3 turn handling uses `TurnUnderstanding`
- UI confirmation buttons send explicit metadata
- parameter modifications are patches before becoming payload changes
- result follow-up can suggest but not auto-run sweeps
- tests prove runtime execution cannot be triggered by LLM output alone
- no default v3 path calls the legacy intent classifier

## Mental Model

The next v3 agent should behave like this:

```text
Read the task card.
Understand the user's latest move.
Choose a safe next action from known tools.
Show what it intends to do.
Ask before anything risky.
Use real runtime evidence when explaining.
Remember what changed.
```

That is the difference between an intelligent agent and a dictionary router.
