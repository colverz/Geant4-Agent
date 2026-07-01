# Geant4 Agent Architecture

Current status: v3 is the main product path. v2, strict, and legacy modules are
kept only as compatibility surfaces or reusable asset libraries.

Current product strategy: functionality first, with safety and state invariants
as guardrails. New work should prioritize user-visible workflows such as
natural configuration changes, useful result explanation, UI suggestions, and
multi-turn continuity. Robustness work remains important when it protects those
flows or prevents runtime/confirmation regressions, but it is no longer the
default primary goal for every round.

## Main Flow

```text
User
-> ui/web
-> /api/v3/agent/turn
-> V3AgentTurnService
-> V3SessionStore
-> V3ContextPack
-> AgentController
-> Reasoner
-> ToolRegistry
-> mcp/geant4 runtime adapter
-> runtime observation
-> V3ContextPack + V3StateSummary
-> UI
```

In plain terms:

- The UI sends user turns to one v3 endpoint.
- The service loads the session, builds a compact context summary, and saves
  the updated state after every turn.
- The controller asks the reasoner what to do next, checks risk, calls tools,
  and records observations.
- The Geant4 adapter builds payloads, runs preflight checks, runs the local
  runtime only after confirmation, and returns evidence.
- The UI renders the answer, trace, pending action, and suggested next steps.

## Directory Responsibilities

- `core/agent_v3/`: v3 agent contracts, state, context, session, controller,
  reasoner, pending action lifecycle, dialogue composition, and Geant4 tool
  wiring.
- `mcp/geant4/`: runtime adapter boundary, payload conversion, discovery, and
  MCP-facing server code. It should not contain agent decision logic.
- `ui/web/`: browser UI and v3 API wrappers. Default product behavior should
  use `/api/v3/agent/*`.
- `core/agent/`, `core/orchestrator/`, `planner/`: legacy or reusable asset
  layers. v3 may reuse pure helpers, but must not inherit their session control
  flow.
- `nlu/`: LLM provider adapters and compatibility NLU assets. v3 reasoning
  should use structured contracts, not the old intent pipeline.
- `builder/geometry/` and `knowledge/`: deterministic geometry and domain
  reference assets.
- `docs/archive/`, `docs/reports/`, `legacy/`: historical outputs and frozen
  compatibility material.

## v3 Design Rules

1. LLM output may propose actions, but deterministic code validates and applies
   them.
2. Runtime execution must pass preflight and user confirmation.
3. Result explanation must use `latest_runtime_facts` as the source of truth.
4. UI actions should send explicit metadata, not depend on matching button
   text.
5. Keyword and regex helpers are allowed only as fallbacks, not as the main
   agent policy. User-facing parameter changes should prefer LLM/structured
   turn understanding or system-generated structured metadata; fallback parsing
   must be visible in eval metrics.
6. Tool schemas and risk levels are part of the architecture, not decoration.
7. Runtime authorization should flow through `V3PendingAction` and
   `V3ExecutionAuthorization`; legacy `metadata["pending_action"]` remains only
   as compatible storage during migration.

## Eval Harnesses

- `tools/evaluate_v3_safety_invariants.py`: guards confirmation, runtime
  authorization, and backend-invariant safety behavior.
- `tools/evaluate_v3_dialogue_casebank.py`: guards user-visible dialogue,
  suggestions, and multi-turn continuity.
- `tools/evaluate_v3_agent_intelligence.py`: guards intelligent operation. It
  records `turn_understanding.source`, state patches, suggestions, and pending
  actions so LLM/structured understanding does not silently degrade into broad
  keyword fallback.

## Active Architecture Docs

- `docs/architecture/README.md`: document map.
- `docs/architecture/GEANT4_AGENT_REBUILD_PROGRESS_2026-05-21.md`: running
  implementation log for v3.
- `docs/architecture/GEANT4_AGENT_V3_INTELLIGENCE_DESIGN_2026-05-25.md`:
  next-stage design for non-keyword agent intelligence.
- `docs/architecture/GEANT4_AGENT_V3_EVAL_AND_UPGRADE_PLAYBOOK_2026-06-04.md`:
  operating manual and acceptance gates for future v3 development.
- `docs/architecture/V3_PHASE_REVIEW_2026-06-06.md`: latest phase review and
  next-priority route check.
- `docs/architecture/V3_UPDATE_LOG.md`: plain-language update log for each v3
  mainline implementation round.
- `docs/architecture/reuse_archive/`: v2 asset reuse and legacy candidate
  index.

## Compatibility Boundary

Legacy endpoints and strict APIs are intentionally not deleted yet. They exist
for old tests, comparison, and migration safety. New work should not add product
behavior there unless it is explicitly a compatibility fix.
