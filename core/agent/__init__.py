from __future__ import annotations

from .candidate_patch import (
    CandidatePatchConfirmationPreview,
    CandidatePatchEnvelope,
    GuardedActionRequest,
    PatchEvidence,
    PatchOperation,
    envelope_to_candidate_update,
    normalize_interpreter_v2_payload,
    preview_candidate_patch_confirmation,
)
from .composite_intent import CompositeIntent, detect_composite_intent
from .context_pack import ContextPack, KnowledgeSnippet, build_context_pack
from .design_advisor import DESIGN_ADVICE_SCHEMA_VERSION, DesignAdvice, DesignOption, build_design_advice
from .design_acceptance import DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION, DesignAcceptancePatch, build_design_acceptance_patch
from .evidence_grounding import EvidenceGroundingContext, EvidenceGroundingResult, check_candidate_update_grounding
from .intent_router import IntentDecision, route_user_turn
from .llm_candidate_contract import (
    LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION,
    LLM_CANDIDATE_ROLE,
    LlmCandidateContract,
    build_llm_candidate_contract,
    build_workflow_llm_candidate_report,
)
from .simulation_design import (
    ALLOWED_NEXT_ACTIONS,
    SIMULATION_DESIGN_SCHEMA_VERSION,
    SimulationDesignCandidate,
    build_simulation_design_candidate,
    build_simulation_design_reference_pack,
    check_simulation_design_capability,
)
from .agent_plan import (
    AGENT_PLAN_SCHEMA_VERSION,
    AGENT_STATE_SCHEMA_VERSION,
    AgentPlan,
    AgentState,
    build_agent_plan,
    build_agent_state,
)
from .result_critic import CRITIC_REPORT_SCHEMA_VERSION, CriticReport, build_critic_report
from .idempotency import (
    IdempotencyActionClass,
    IdempotencyDecision,
    IdempotencyDecisionResult,
    IdempotencyRecord,
    IdempotencyReplayPolicy,
    build_action_id,
    classify_idempotent_action,
)
from .interrupt_resume import InterruptResumeController, InterruptResumeResult, InterruptResumeStatus
from .staged_patch import StagedPatch, StagedPatchStatus, StagedPatchStore, build_staged_patch_reference
from .state_summary import AGENT_STATE_SUMMARY_SCHEMA_VERSION, AgentStateSummary, build_agent_state_summary
from .staged_patch_bridge import (
    StagedPatchCompatibilityPreview,
    apply_confirmation_candidate,
    preview_staged_patch_compatibility,
    staged_patch_to_confirmation_candidate,
)
from .turn_trace import NluTurnTrace
from .workflow_graph import WorkflowNode, WorkflowTerminalState, graph_path_for_intent

__all__ = [
    "ContextPack",
    "CandidatePatchEnvelope",
    "CandidatePatchConfirmationPreview",
    "CompositeIntent",
    "DESIGN_ADVICE_SCHEMA_VERSION",
    "DESIGN_ACCEPTANCE_PATCH_SCHEMA_VERSION",
    "DesignAdvice",
    "DesignAcceptancePatch",
    "DesignOption",
    "EvidenceGroundingContext",
    "EvidenceGroundingResult",
    "GuardedActionRequest",
    "IntentDecision",
    "IdempotencyActionClass",
    "IdempotencyDecision",
    "IdempotencyDecisionResult",
    "IdempotencyRecord",
    "IdempotencyReplayPolicy",
    "InterruptResumeController",
    "InterruptResumeResult",
    "InterruptResumeStatus",
    "LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION",
    "LLM_CANDIDATE_ROLE",
    "LlmCandidateContract",
    "ALLOWED_NEXT_ACTIONS",
    "AGENT_PLAN_SCHEMA_VERSION",
    "AGENT_STATE_SCHEMA_VERSION",
    "AGENT_STATE_SUMMARY_SCHEMA_VERSION",
    "CRITIC_REPORT_SCHEMA_VERSION",
    "AgentPlan",
    "AgentState",
    "AgentStateSummary",
    "SIMULATION_DESIGN_SCHEMA_VERSION",
    "SimulationDesignCandidate",
    "CriticReport",
    "KnowledgeSnippet",
    "NluTurnTrace",
    "PatchEvidence",
    "PatchOperation",
    "StagedPatch",
    "StagedPatchCompatibilityPreview",
    "StagedPatchStatus",
    "StagedPatchStore",
    "WorkflowNode",
    "WorkflowTerminalState",
    "build_context_pack",
    "build_design_advice",
    "build_design_acceptance_patch",
    "build_llm_candidate_contract",
    "build_workflow_llm_candidate_report",
    "build_simulation_design_candidate",
    "build_agent_plan",
    "build_agent_state",
    "build_agent_state_summary",
    "build_critic_report",
    "build_simulation_design_reference_pack",
    "build_llm_simulation_design_candidate",
    "build_action_id",
    "build_staged_patch_reference",
    "check_candidate_update_grounding",
    "check_simulation_design_capability",
    "SIMULATION_DESIGN_LLM_PROMPT_PROFILE_ID",
    "classify_idempotent_action",
    "detect_composite_intent",
    "envelope_to_candidate_update",
    "graph_path_for_intent",
    "normalize_interpreter_v2_payload",
    "preview_candidate_patch_confirmation",
    "preview_staged_patch_compatibility",
    "route_user_turn",
    "apply_confirmation_candidate",
    "staged_patch_to_confirmation_candidate",
]


def __getattr__(name: str):
    if name == "build_llm_simulation_design_candidate":
        from .simulation_design_llm import build_llm_simulation_design_candidate

        return build_llm_simulation_design_candidate
    if name == "SIMULATION_DESIGN_LLM_PROMPT_PROFILE_ID":
        from .simulation_design_llm import PROMPT_PROFILE_ID

        return PROMPT_PROFILE_ID
    raise AttributeError(name)
