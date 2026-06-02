from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


V3_RESPONSE_QUALITY_SCHEMA_VERSION = "geant4_agent_v3_response_quality.v1"

_ACTS_EXPECTING_NEXT_STEP = {
    "action_cancelled",
    "action_needs_confirmation",
    "blocked",
    "design_presented",
    "needs_user_input",
    "payload_draft_presented",
    "runtime_observed",
    "runtime_result_answered",
}

_ACTS_EXPECTING_EVIDENCE = {
    "action_cancelled",
    "action_needs_confirmation",
    "blocked",
    "design_presented",
    "payload_draft_presented",
    "runtime_observed",
    "runtime_result_answered",
}

_RAW_INTERNAL_MARKERS = (
    "raw trace",
    "V3TraceEvent",
    "turn.metadata",
    "state.metadata",
    "pending_action",
    "commit_gate",
    "proposal_critic",
)


@dataclass(slots=True)
class V3ResponseQualityReport:
    ok: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_RESPONSE_QUALITY_SCHEMA_VERSION,
            "ok": self.ok,
            "score": round(self.score, 3),
            "checks": dict(self.checks),
            "warnings": list(self.warnings),
            "strengths": list(self.strengths),
        }


def evaluate_v3_response_quality(response: dict[str, Any]) -> dict[str, Any]:
    report = _evaluate(response)
    return report.to_dict()


def _evaluate(response: dict[str, Any]) -> V3ResponseQualityReport:
    dialogue = _dict(response.get("dialogue"))
    display = str(response.get("display_message") or dialogue.get("display_message") or "")
    act = str(response.get("dialogue_act") or dialogue.get("dialogue_act") or response.get("terminated_reason") or "")
    evidence = _list(response.get("evidence_used") or dialogue.get("evidence_used"))
    suggestions = _list(dialogue.get("next_suggestions"))
    answer_parts = _list(response.get("answer_parts") or dialogue.get("answer_parts"))
    observations = _list(response.get("observations"))

    checks = {
        "has_display_message": bool(display.strip()) and len(display.strip()) >= 12,
        "has_structured_answer_parts": _has_structured_answer_parts(answer_parts),
        "has_next_step": _has_next_step(act, suggestions, response),
        "has_expected_evidence": _has_expected_evidence(act, evidence, observations),
        "no_raw_internal_dump": not _contains_raw_internal_dump(display),
        "blocked_is_actionable": _blocked_is_actionable(act, display, suggestions),
        "suggestions_are_clickable": _suggestions_are_clickable(suggestions),
    }
    warnings = _warnings_for_checks(checks)
    strengths = _strengths_for_checks(checks)
    score = sum(1 for ok in checks.values() if ok) / max(1, len(checks))
    return V3ResponseQualityReport(ok=score >= 0.8 and not _hard_failure(checks), score=score, checks=checks, warnings=warnings, strengths=strengths)


def _has_next_step(act: str, suggestions: list[Any], response: dict[str, Any]) -> bool:
    if act not in _ACTS_EXPECTING_NEXT_STEP:
        return True
    if suggestions:
        return True
    answer = _dict(response.get("answer"))
    return bool(_list(answer.get("next_options")))


def _has_expected_evidence(act: str, evidence: list[Any], observations: list[Any]) -> bool:
    if act not in _ACTS_EXPECTING_EVIDENCE:
        return True
    if act == "runtime_result_answered" and not evidence and not observations:
        return False
    return bool(evidence or observations)


def _contains_raw_internal_dump(display: str) -> bool:
    lowered = display.lower()
    return any(marker.lower() in lowered for marker in _RAW_INTERNAL_MARKERS)


def _blocked_is_actionable(act: str, display: str, suggestions: list[Any]) -> bool:
    if act != "blocked":
        return True
    lowered = display.lower()
    has_recovery_words = any(token in lowered for token in ("next step", "payload", "preflight", "adjust", "modify", "下一步", "配置", "检查"))
    return bool(suggestions) and has_recovery_words


def _suggestions_are_clickable(suggestions: list[Any]) -> bool:
    if not suggestions:
        return True
    for item in suggestions:
        if isinstance(item, str):
            if item.strip():
                continue
            return False
        if not isinstance(item, dict):
            return False
        text = str(item.get("text") or item.get("label") or item.get("title") or "").strip()
        prefill = str(item.get("prefill") or item.get("text") or item.get("label") or item.get("title") or "").strip()
        if not text or not prefill:
            return False
    return True


def _has_structured_answer_parts(parts: list[Any]) -> bool:
    if not parts:
        return False
    kinds = {str(item.get("kind") or "") for item in parts if isinstance(item, dict)}
    return "summary" in kinds and ("next_step" in kinds or "evidence" in kinds)


def _warnings_for_checks(checks: dict[str, bool]) -> list[str]:
    labels = {
        "has_display_message": "response_missing_human_message",
        "has_structured_answer_parts": "response_missing_structured_answer_parts",
        "has_next_step": "response_missing_next_step",
        "has_expected_evidence": "response_missing_expected_evidence",
        "no_raw_internal_dump": "response_leaks_internal_trace_or_metadata",
        "blocked_is_actionable": "blocked_response_not_actionable",
        "suggestions_are_clickable": "suggestions_not_clickable",
    }
    return [labels[key] for key, ok in checks.items() if not ok]


def _strengths_for_checks(checks: dict[str, bool]) -> list[str]:
    labels = {
        "has_display_message": "has_human_message",
        "has_structured_answer_parts": "has_structured_answer_parts",
        "has_next_step": "has_next_step",
        "has_expected_evidence": "has_evidence",
        "no_raw_internal_dump": "hides_internal_trace",
        "blocked_is_actionable": "blocked_response_is_actionable",
        "suggestions_are_clickable": "suggestions_are_clickable",
    }
    return [labels[key] for key, ok in checks.items() if ok]


def _hard_failure(checks: dict[str, bool]) -> bool:
    return not checks.get("has_display_message", False) or not checks.get("no_raw_internal_dump", False)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


__all__ = [
    "V3_RESPONSE_QUALITY_SCHEMA_VERSION",
    "V3ResponseQualityReport",
    "evaluate_v3_response_quality",
]
