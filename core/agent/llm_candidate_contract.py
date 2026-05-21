from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION = "geant4_agent_llm_candidate_contract.v1"
LLM_CANDIDATE_ROLE = "candidate_config_only"


@dataclass(frozen=True)
class LlmCandidateContract:
    """Audit boundary for LLM-produced configuration candidates.

    The LLM is allowed to propose a candidate and explain its assumptions. It is
    not the source of final runtime truth; contract alignment and runtime output
    decide what can be executed and reported.
    """

    user_goal: str
    raw_config: dict[str, Any]
    aligned_config: dict[str, Any]
    reference_pack_ids: tuple[str, ...] = ()
    choice_zones: tuple[dict[str, str], ...] = ()
    assumptions: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    requires_confirmation: tuple[str, ...] = ()
    physics_rationale: tuple[str, ...] = ()
    alignment_report: dict[str, Any] = field(default_factory=dict)

    def to_report(self, *, include_configs: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {
            "schema_version": LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION,
            "role": LLM_CANDIDATE_ROLE,
            "user_goal": self.user_goal,
            "reference_pack_ids": list(self.reference_pack_ids),
            "choice_zones": [dict(item) for item in self.choice_zones],
            "assumptions": list(self.assumptions),
            "uncertainties": list(self.uncertainties),
            "requires_confirmation": list(self.requires_confirmation),
            "physics_rationale": list(self.physics_rationale),
            "alignment": _alignment_summary(self.alignment_report),
        }
        if include_configs:
            report["raw_config"] = self.raw_config
            report["aligned_config"] = self.aligned_config
        return report


def build_llm_candidate_contract(
    *,
    user_goal: str,
    raw_config: dict[str, Any],
    aligned_config: dict[str, Any],
    reference_pack_ids: list[str] | tuple[str, ...] = (),
    choice_zones: list[dict[str, str]] | tuple[dict[str, str], ...] = (),
    alignment_report: dict[str, Any] | None = None,
    is_complete: bool = True,
    fallback_reason: str | None = None,
    runtime_contract_present: bool = False,
) -> LlmCandidateContract:
    alignment = alignment_report if isinstance(alignment_report, dict) else {}
    assumptions = _assumptions_from_choice_zones(choice_zones)
    uncertainties = _uncertainties(
        is_complete=is_complete,
        fallback_reason=fallback_reason,
        runtime_contract_present=runtime_contract_present,
    )
    confirmations = _confirmation_reasons(alignment)
    rationale = _physics_rationale(alignment)
    return LlmCandidateContract(
        user_goal=user_goal,
        raw_config=dict(raw_config),
        aligned_config=dict(aligned_config),
        reference_pack_ids=tuple(str(item) for item in reference_pack_ids),
        choice_zones=tuple(dict(item) for item in choice_zones),
        assumptions=tuple(assumptions),
        uncertainties=tuple(uncertainties),
        requires_confirmation=tuple(confirmations),
        physics_rationale=tuple(rationale),
        alignment_report=alignment,
    )


def _assumptions_from_choice_zones(choice_zones: list[dict[str, str]] | tuple[dict[str, str], ...]) -> list[str]:
    assumptions: list[str] = []
    for item in choice_zones:
        field_name = str(item.get("field") or "").strip()
        policy = str(item.get("policy") or "").strip()
        if field_name and policy:
            assumptions.append(f"{field_name}: {policy}")
    return assumptions


def _uncertainties(*, is_complete: bool, fallback_reason: str | None, runtime_contract_present: bool) -> list[str]:
    uncertainties: list[str] = []
    if not is_complete:
        uncertainties.append("LLM candidate is incomplete and must not be treated as executable without resolver checks.")
    if fallback_reason:
        uncertainties.append(f"LLM fallback path was used: {fallback_reason}.")
    if not runtime_contract_present:
        uncertainties.append("No runtime contract was available; candidate requires stronger downstream validation.")
    return uncertainties


def _confirmation_reasons(alignment: dict[str, Any]) -> list[str]:
    risk_count = int(alignment.get("risk_correction_count") or 0)
    if risk_count <= 0:
        return []
    paths = [
        str(item.get("path"))
        for item in alignment.get("correction_details", [])
        if isinstance(item, dict)
        and item.get("severity") == "override"
        and str(item.get("category") or "") in {"material_role", "geometry", "source", "detector", "scoring_role"}
    ]
    if paths:
        return [f"Runtime contract overrode high-risk LLM candidate paths: {', '.join(paths)}."]
    return [f"Runtime contract applied {risk_count} high-risk correction(s) to the LLM candidate."]


def _physics_rationale(alignment: dict[str, Any]) -> list[str]:
    categories = alignment.get("correction_categories") if isinstance(alignment.get("correction_categories"), dict) else {}
    rationale: list[str] = []
    if categories.get("material_role"):
        rationale.append("Material role binding was checked so detector material does not overwrite target/root material.")
    if categories.get("scoring_role"):
        rationale.append("Scoring roles were checked against existing target/detector volumes.")
    if categories.get("runtime_default"):
        rationale.append("Runtime defaults were completed deterministically rather than inferred as physics facts.")
    return rationale


def _alignment_summary(alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "applied": bool(alignment.get("applied")),
        "correction_count": int(alignment.get("correction_count") or 0),
        "completion_count": int(alignment.get("completion_count") or 0),
        "override_count": int(alignment.get("override_count") or 0),
        "risk_correction_count": int(alignment.get("risk_correction_count") or 0),
        "corrected_paths": list(alignment.get("corrected_paths") or []),
        "correction_categories": dict(alignment.get("correction_categories") or {}),
    }


def build_workflow_llm_candidate_report(
    *,
    user_goal: str,
    llm_used: bool,
    fallback_reason: str | None,
    prompt_profile_id: str | None,
    inference_backend: str,
    candidate_patch_paths: list[str],
    applied_paths: list[str],
    rejected_paths: list[str],
    pending_confirmation_paths: list[str],
) -> dict[str, Any]:
    """Build a main-workflow candidate report without pretending it is runtime truth."""

    proposed = list(dict.fromkeys(str(path) for path in candidate_patch_paths if str(path)))
    applied = list(dict.fromkeys(str(path) for path in applied_paths if str(path)))
    rejected = list(dict.fromkeys(str(path) for path in rejected_paths if str(path)))
    pending = list(dict.fromkeys(str(path) for path in pending_confirmation_paths if str(path)))
    uncertainties: list[str] = []
    if not llm_used:
        uncertainties.append("LLM was not used for this turn; candidate paths came from deterministic fallback components.")
    if fallback_reason:
        uncertainties.append(f"Fallback reason: {fallback_reason}.")
    if rejected:
        uncertainties.append("Some candidate paths were rejected by guard, arbitration, or validation.")
    return {
        "schema_version": LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION,
        "role": LLM_CANDIDATE_ROLE,
        "source": "process_turn",
        "user_goal": user_goal,
        "llm_used": bool(llm_used),
        "fallback_reason": fallback_reason,
        "prompt_profile_id": prompt_profile_id,
        "inference_backend": inference_backend,
        "candidate_boundary": {
            "policy": "Candidate paths are proposals. Session config changes only after confirmation policy, arbitration, validation, and commit.",
            "proposed_paths": proposed,
            "applied_paths": applied,
            "rejected_paths": rejected,
            "pending_confirmation_paths": pending,
        },
        "resolution": {
            "proposed_path_count": len(proposed),
            "applied_path_count": len(applied),
            "rejected_path_count": len(rejected),
            "pending_confirmation_count": len(pending),
            "applied_to_session": bool(applied),
            "confirmation_required": bool(pending),
        },
        "assumptions": [
            "Runtime execution and result metrics are not produced by this candidate report.",
            "Final facts must come from validated session config and Geant4 runtime output.",
        ],
        "uncertainties": uncertainties,
        "requires_confirmation": [f"User confirmation required for {path}." for path in pending],
    }


__all__ = [
    "LLM_CANDIDATE_CONTRACT_SCHEMA_VERSION",
    "LLM_CANDIDATE_ROLE",
    "LlmCandidateContract",
    "build_llm_candidate_contract",
    "build_workflow_llm_candidate_report",
]
