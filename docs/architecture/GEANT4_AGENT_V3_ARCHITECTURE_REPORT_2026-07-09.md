# Geant4 Agent v3 Architecture Report - 2026-07-09

## Executive Summary

The project has now moved from a mixed v2/v3 experiment into a v3-first product
shape. The local `main` branch has been fast-forward merged to the reviewed
v3 beta product head, commit `34c5c06`, then this documentation report was
committed on top.

In user terms, the system now behaves like this:

1. The user describes a Geant4 simulation goal in the web UI.
2. The v3 agent turns that into a design draft.
3. The user can accept, revise, or ask follow-up questions across turns.
4. The agent builds a runnable payload only when the conversation reaches that
   stage.
5. Geant4 execution requires preflight plus explicit confirmation.
6. Runtime results are recorded as evidence and used for explanation.

The most important architectural improvement is that v3 now keeps a compact,
explicit context pack instead of letting scattered dictionaries and old prompt
examples leak into every prompt. This reduces prompt pollution and makes the
multi-turn workflow easier to evaluate.

## Current Merge State

- Source branch: `beta`
- Target branch: local `main`
- Merge type: fast-forward
- v3 product merge head: `34c5c06 Harden v3 merge readiness and calibrate evals`
- Current local `main` head after documentation: this report commit,
  `Document v3 main architecture after merge`
- Relationship before merge: `main` had no extra commits beyond beta's base;
  `beta` was 16 commits ahead of local `main`
- Relationship after merge: local `main` contains the reviewed v3 line

No remote push was performed in this round.

## Verification Snapshot

These checks were run on local `main` after the documentation update:

```text
pytest -q
973 passed, 3 skipped, 103 subtests passed
```

```text
python -m eval.v3.run_suite --tasks eval/v3/tasks/behavior_safety.jsonl --json
ok=true
tasks=9/9
trials=14/14
average_trial_score=1.0
```

```text
python -m eval.v3.calibrate --json
ok=true
cases=4/4
false_positive=0
false_negative=0
```

```text
python -m eval.v3.run_suite --tasks eval/v3/tasks/agent_intelligence_live.jsonl --live-llm ...
ok=true
tasks=1/1
trials=1/1
```

What this proves:

- The normal Python test suite is green.
- Runtime authorization and confirmation gates are behaving as expected.
- The deterministic grader catches known bad cases and accepts known good cases.
- The live DeepSeek path can produce a design-only answer without falling back
  to broad text parsing, without generating a payload, and without running
  Geant4.

What this does not yet prove:

- Live LLM quality across many user styles and languages.
- Long multi-session behavior under real UI usage.
- Full production readiness for autonomous optimization or sweep execution.

## Architecture In Plain Language

Think of the project as four cooperating parts.

### 1. The Conversation Layer

Location:

```text
ui/web/
core/agent_v3/service.py
core/agent_v3/session.py
core/agent_v3/context.py
```

The browser does not try to be smart by itself. It sends user turns to v3 API
endpoints. The service loads the session, summarizes the current state, runs
one agent turn, saves the updated session, and returns an answer plus UI hints.

The session now remembers:

- current goal
- latest design
- latest payload
- latest runtime facts
- open questions
- assumptions
- pending action
- suggested next actions

This makes multi-turn dialogue usable because the next turn does not have to
reconstruct the whole situation from raw text.

### 2. The Reasoning Layer

Location:

```text
core/agent_v3/reasoners.py
core/agent_v3/turn_understanding.py
core/agent_v3/dialogue_composer.py
core/agent_v3/response_quality.py
core/agent_v3/response_naturalizer.py
```

This layer decides what the user is trying to do and how the system should
reply. The current direction is deliberately not dictionary-first:

- Use structured LLM understanding when available.
- Use explicit UI metadata for buttons and suggestions.
- Keep text fallbacks small and visible.
- Record the source of turn understanding so evals can detect degradation.

The agent can ask questions, draft designs, revise existing designs, explain
state, or move toward runtime. It should not silently treat every phrase as a
hardcoded command.

### 3. The Tool And Runtime Layer

Location:

```text
core/agent_v3/tool_registry.py
core/agent_v3/runtime_policy.py
core/agent_v3/pending_action.py
core/agent_v3/tools/
mcp/geant4/
runtime/
```

The agent does not directly run arbitrary code. It proposes tool calls. The
controller checks risk, runs allowed tools, and records observations.

Runtime execution has a gate:

```text
payload built -> preflight ok -> pending action created -> user confirms -> run
```

The important safety rule is simple: a model answer alone cannot authorize a
Geant4 run. Confirmation must come from a matching explicit action or a
deterministic confirmation path that matches the current pending action.

### 4. The Evaluation Layer

Location:

```text
eval/v3/
tools/evaluate_v3_*.py
tests/
docs/eval/
```

The current primary v3 eval path is `eval/v3/`:

- `adapters/v3_turn_adapter.py`: runs one typed trial through the v3 service.
- `graders/behavior_grader.py`: checks the trial without calling the agent.
- `run_suite.py`: runs task banks and grades them.
- `compare.py`: compares baseline and candidate grade lists.
- `calibrate.py`: checks that the grader catches known bad behavior.

The older `tools/evaluate_v3_*` scripts remain useful, but should be treated as
compatibility and migration helpers, not the future center of the eval system.

## Directory Responsibilities

### Product Mainline

```text
core/agent_v3/
```

This is the agent brain and workflow owner. New v3 behavior belongs here unless
it is clearly UI-only, runtime-adapter-only, or shared domain knowledge.

```text
ui/web/
```

This is the default product UI. It should call v3 endpoints directly and render
v3-native state, trace, pending actions, and suggestions.

```text
mcp/geant4/
```

This is the Geant4 boundary. It should know how to adapt payloads, discover
runtime capability, and call the runtime. It should not decide dialogue policy.

### Domain And Support Layers

```text
builder/geometry/
knowledge/
```

These provide deterministic geometry and validated reference knowledge.

```text
nlu/
```

This provides LLM clients and historical NLU pieces. v3 can use provider
support from here, but should not inherit old intent-routing control flow.

### Compatibility Layers

```text
core/agent/
core/orchestrator/
planner/
ui/desktop/
legacy/
```

These remain for old tests, old workflows, and pure reusable helpers. They
should not receive new default UI behavior.

## Current Public Interfaces

v3 product endpoints:

```text
POST /api/v3/agent/turn
POST /api/v3/agent/state
POST /api/v3/agent/reset
```

Legacy endpoints:

```text
/api/agent/*
strict APIs
legacy APIs
```

The legacy endpoints are compatibility surfaces. New UI behavior should use
the v3 endpoints.

## Main Data Flow

```text
User message
  -> V3TurnInput
  -> V3SessionStore loads state
  -> V3ContextPack summarizes state
  -> Reasoner proposes next action
  -> Controller validates risk
  -> ToolRegistry calls draft/runtime tools
  -> Observations are recorded
  -> V3StateSummary is returned to UI
  -> V3SessionStore saves state
```

The context pack is intentionally smaller than the full session. It is safe to
include in prompts because it avoids raw observation dumps, provider config,
credentials, and unrelated historical examples.

## What Is Working

- v3 is the default web UI path.
- v3 session files are wrapped, versioned, and recoverable from older state.
- Missing or unsafe session IDs no longer collapse into a shared default.
- Corrupt or mismatched session files are quarantined instead of being silently
  trusted.
- Runtime execution requires preflight and confirmation.
- Explicit UI confirmation now requires matching action identity.
- Open questions, assumptions, active plan, pending action, and suggestions are
  represented in v3 state.
- LLM suggestions for sweep or optimization are suggestions first; they do not
  execute automatically.
- Result explanation is grounded on latest runtime facts.
- Safety eval and calibration are repeatable local gates.

## Main Risks And Gaps

### Live LLM Coverage Is Too Small

The current live DeepSeek task proves the path works, but it is only one task.
It should be expanded before live LLM behavior is treated as a hard merge gate.

Recommended next cases:

- user revises energy after a runtime result
- user asks a result follow-up that must ignore stale goal text
- Chinese multi-turn design revision
- user asks for sweep/optimization but should only receive a suggestion
- user clicks a structured UI suggestion

### Some v3 Files Are Still Large

The reasoner, config builder, Geant4 tool file, and service are large. This was
acceptable while stabilizing the product path. The next split should be
function-driven, not cosmetic:

- comprehension and turn understanding
- design drafting
- payload building
- runtime execution
- result explanation
- UI response projection

### Planner Dependency Should Shrink

`core/agent_v3/result_explainer.py` still imports a pure formatting helper from
`planner/runtime_result.py`. That helper should move into v3 before any planner
compatibility cleanup.

### UI Needs More Real Interaction Review

The UI now routes through v3, but the next quality jump should be about user
experience:

- pending-confirmation follow-up should answer user questions without losing
  the pending action
- suggestion buttons should stay visually centered and send structured metadata
- result cards should distinguish design facts, payload facts, runtime facts,
  and model interpretation

## Recommended Next Development Direction

### Step 1: Make The UI Feel Like A Helpful Lab Assistant

Focus on user-visible functionality:

- better replies during pending confirmation
- clearer result explanation sections
- better button and suggestion behavior
- state panel based only on `/api/v3/agent/state`

Acceptance:

- user can ask "what will this run do?" while a pending action exists
- the pending action remains available
- UI suggestions are centered and carry structured prefill/action metadata

### Step 2: Expand Non-Dictionary Intelligence Eval

Add a small live LLM bank, not a huge benchmark:

- 5 to 10 realistic tasks
- English and Chinese
- no mocked model output
- each task checks whether the agent used LLM/structured understanding, not
  broad keyword fallback

Acceptance:

- task reports show `turn_understanding.source`
- fallback use is visible
- failures explain which behavior regressed

### Step 3: Move Result Explanation Fully Into v3

Result explanation is central to user trust. It should be owned by v3.

Acceptance:

- no v3 result-explanation import from `planner/`
- latest runtime facts remain authoritative
- tests cover stale-context and prompt-pollution cases

### Step 4: Prepare Controlled Sweep And Optimization

Do not auto-execute sweeps yet. First make them explicit workflows:

```text
suggest sweep -> user accepts -> build sweep payload -> preflight -> confirm -> run
```

Acceptance:

- sweep suggestion never runs in the same turn
- sweep execution requires explicit user intent and confirmation
- eval covers the full gate

## Merge Recommendation After This Round

The local `main` branch now contains the reviewed v3 mainline. Based on the
checks run in this round, the merged state is acceptable for continued mainline
development.

Do not push blindly if the remote branch policy matters. Local `main` is ahead
of `origin/main`, so a push should be deliberate and may need coordination with
the remote branch protection model.

## Update Log

- Merged local `beta` into local `main` by fast-forward.
- Added this documentation commit on top of the v3 product merge.
- Ran full Python tests and current v3 eval gates.
- Confirmed live DeepSeek advisory eval passes on the design-only task.
- Updated root README to describe the v3-first product path.
- Added this architecture report as the current post-merge reference.
