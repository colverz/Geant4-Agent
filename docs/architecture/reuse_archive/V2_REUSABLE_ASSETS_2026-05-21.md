# v2 Reusable Assets for v3

Date: 2026-05-21

This file records assets that should be kept for the v3 rebuild. The goal is to preserve domain value while replacing the v2 control flow.

## Direct Reuse

| Path | Why keep | v3 use |
| --- | --- | --- |
| `core/agent/candidate_patch.py` | Candidate changes already match validate-before-apply thinking. | Basis for `V3Patch`. |
| `core/agent/staged_patch.py` | Supports staged application of state changes. | Used by apply/confirmation nodes. |
| `core/agent/evidence_grounding.py` | Guards LLM output against unsupported changes. | `ground_patch` node. |
| `core/agent/action_safety.py` | Captures safety checks for risky actions. | `runtime_guard` and confirmation policy. |
| `core/agent/context_pack.py` | Useful context construction primitives. | `build_context` node. |
| `core/agent/turn_trace.py` | Useful trace semantics for observability. | v3 trace events. |
| `core/agent/workflow_graph.py` | Existing graph workflow idea. | Reference for v3 workflow, with new contracts. |
| `core/agent/interrupt_resume.py` | Needed for multi-turn clarification and confirmation. | `confirm_or_interrupt` node. |
| `core/agent/idempotency.py` | Needed for robust repeated calls. | `apply_patch` and runtime actions. |
| `core/agent/llm_candidate_contract.py` | Existing LLM structured output contract. | Reference for v3 interpretation and patch contracts. |
| `core/agent/design_advisor.py` | Provides scenario-aware design suggestions. | Early v3 design intelligence node. |
| `core/agent/design_acceptance.py` | Converts accepted suggestions into patch-like updates. | Accept-design node or patch builder. |
| `core/agent/state_summary.py` | Gives user-facing and API-facing state clarity. | v3 answer state summary. |
| `core/agent/simulation_design.py` | Domain design helpers. | `design_candidate` node. |
| `core/agent/simulation_design_llm.py` | LLM-assisted design surface. | LLM design adapter, after contract review. |
| `core/agent/agent_plan.py` | Planning representation. | v3 plan/status reporting. |
| `core/agent/result_critic.py` | Result critique and quality checks. | `answer_user` result interpretation. |
| `core/simulation/spec.py` | Simulation domain spec. | Core v3 simulation data model. |
| `core/simulation/bridge.py` | Bridges simulation spec to runtime payload. | `simulation_adapter.py`. |
| `core/simulation/results.py` | Reads or normalizes simulation outputs. | `V3RuntimeObservation`. |
| `core/simulation/smoke_report.py` | Useful for smoke test reporting. | v3 smoke report output. |
| `mcp/geant4/tools.py` | Geant4 tool definitions. | Runtime execution layer. |
| `mcp/geant4/adapter.py` | Adapter to local Geant4 runtime behavior. | v3 runtime adapter. |
| `mcp/geant4/server.py` | MCP server integration. | Keep for tool exposure. |
| `mcp/geant4/runtime_payload.py` | Runtime payload contract. | v3 run payload generation. |
| `mcp/geant4/local_wrapper.py` | Local process wrapper. | v3 local runtime path. |
| `tools/run_minimal_agent_smoke.py` | Existing minimal natural-language smoke path. | Baseline for `run_v3_agent_smoke.py`. |
| `tools/local_geant4_smoke.py` | Local runtime smoke. | Runtime readiness check. |
| `tools/run_industrial_runtime_stage.py` | Industrial benchmark runner. | v3 runtime acceptance. |
| `docs/eval/*` | Casebanks, benchmark docs, golden metrics. | v3 regression suite. |

## Adapter Reuse

| Path | Reusable part | v3 action |
| --- | --- | --- |
| `core/orchestrator/types.py` | Shared domain-ish types. | Extract clean types or wrap. |
| `core/orchestrator/path_ops.py` | Path/state utilities. | Reuse pure helpers only. |
| `core/orchestrator/confirmation_policy.py` | Confirmation rules. | Adapt to v3 confirmation node. |
| `core/orchestrator/turn_transaction.py` | Transaction pattern. | Adapt to v3 apply node. |
| `core/orchestrator/arbiter.py` | Conflict arbitration. | Convert into explicit patch conflict checks. |
| `core/orchestrator/semantic_sync.py` | Semantic synchronization rules. | Keep only deterministic sync rules. |
| `core/orchestrator/derived_sync.py` | Derived-state synchronization. | Use after patch validation. |
| `core/dialogue/*` | Some wording and dialogue strategy. | Use for answer policy only. |
| `planner/runtime_intent.py` | Runtime intent classification. | Runtime guard input. |
| `planner/runtime_result.py` | Runtime result expression. | Answer/runtime observation adapter. |
| `nlu/llm_support/*` | Provider config and calls. | `llm_provider_adapter.py`. |
| `nlu/llm/*` | Some semantic frame ideas. | Fallback/reference, not v3 authority. |
| `nlu/runtime_semantic.py` | Runtime semantic heuristics. | Deterministic fallback checks. |
| `nlu/runtime_components/*` | Component-level runtime parsing. | Fallback/reference. |
| `core/config/prompt_profiles.py` | Prompt profile assets. | Context/prompt profile adapter. |
| `knowledge/*` | Domain knowledge snippets. | `build_context` retrieval/filtering. |

## Guardrails

- Do not import v2 session control flow into v3.
- Do not let LLM text mutate session state directly.
- Reuse functions and contracts, not accidental orchestration.
- Any reused runtime result must come from actual runtime observation.
