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
from .evidence_grounding import EvidenceGroundingContext, EvidenceGroundingResult, check_candidate_update_grounding
from .intent_router import IntentDecision, route_user_turn
from .interrupt_resume import InterruptResumeController, InterruptResumeResult, InterruptResumeStatus
from .staged_patch import StagedPatch, StagedPatchStatus, StagedPatchStore, build_staged_patch_reference
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
    "EvidenceGroundingContext",
    "EvidenceGroundingResult",
    "GuardedActionRequest",
    "IntentDecision",
    "InterruptResumeController",
    "InterruptResumeResult",
    "InterruptResumeStatus",
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
    "build_staged_patch_reference",
    "check_candidate_update_grounding",
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
