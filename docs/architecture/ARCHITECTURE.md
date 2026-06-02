# Geant4 Agent Architecture

Current status: v3 is the main product path. v2, strict, and legacy modules are
kept only as compatibility surfaces or reusable asset libraries.

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
  reasoner, dialogue composition, and Geant4 tool wiring.
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
   agent policy.
6. Tool schemas and risk levels are part of the architecture, not decoration.

## Active Architecture Docs

- `docs/architecture/README.md`: document map.
- `docs/architecture/GEANT4_AGENT_REBUILD_PROGRESS_2026-05-21.md`: running
  implementation log for v3.
- `docs/architecture/GEANT4_AGENT_V3_INTELLIGENCE_DESIGN_2026-05-25.md`:
  next-stage design for non-keyword agent intelligence.
- `docs/architecture/reuse_archive/`: v2 asset reuse and legacy candidate
  index.

## Compatibility Boundary

Legacy endpoints and strict APIs are intentionally not deleted yet. They exist
for old tests, comparison, and migration safety. New work should not add product
behavior there unless it is explicitly a compatibility fix.
