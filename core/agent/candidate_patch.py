from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.agent.action_safety import ActionSafetyClass
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp


CONFIRM_LOW_CONFIDENCE = "low_confidence"
CONFIRM_OVERWRITE = "explicit_overwrite"
CONFIRM_DELETE = "delete"


@dataclass(frozen=True)
class PatchEvidence:
    text: str
    source: str
    role: str


@dataclass(frozen=True)
class PatchOperation:
    path: str
    op: str
    value: Any
    confidence: float
    evidence: list[PatchEvidence] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardedActionRequest:
    action: str
    safety_class: ActionSafetyClass
    requested: bool
    reason: str = ""


@dataclass(frozen=True)
class CandidatePatchEnvelope:
    source: str
    intent: Intent
    operations: list[PatchOperation]
    guarded_actions: list[GuardedActionRequest] = field(default_factory=list)
    ambiguities: list[dict[str, Any]] = field(default_factory=list)
    unsupported_requests: list[dict[str, Any]] = field(default_factory=list)
    patch_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["intent"] = self.intent.value
        for item in payload.get("guarded_actions", []):
            safety = item.get("safety_class")
            item["safety_class"] = safety.value if isinstance(safety, ActionSafetyClass) else str(safety)
        return payload


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _intent_from_text(value: Any) -> Intent:
    text = str(value or "").strip().lower()
    if text == "set":
        return Intent.SET
    if text == "modify":
        return Intent.MODIFY
    if text == "remove":
        return Intent.REMOVE
    if text == "confirm":
        return Intent.CONFIRM
    if text == "reject":
        return Intent.REJECT
    if text == "question":
        return Intent.QUESTION
    return Intent.OTHER


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _coerce_evidence(items: Any) -> list[PatchEvidence]:
    if not isinstance(items, list):
        return []
    evidence: list[PatchEvidence] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        source = str(item.get("source", "") or "").strip()
        role = str(item.get("role", "") or "").strip()
        if text and source and role:
            evidence.append(PatchEvidence(text=text, source=source, role=role))
    return evidence


def _confirmation_reasons(item: dict[str, Any], confidence: float, *, min_confidence: float) -> list[str]:
    reasons: list[str] = []
    if confidence < min_confidence:
        reasons.append(CONFIRM_LOW_CONFIDENCE)
    if bool(item.get("requires_confirmation")):
        reasons.append(CONFIRM_OVERWRITE)
    if str(item.get("op", "") or "") == "remove":
        reasons.append(CONFIRM_DELETE)
    return list(dict.fromkeys(reasons))


def _coerce_guarded_actions(items: Any) -> list[GuardedActionRequest]:
    if not isinstance(items, list):
        return []
    actions: list[GuardedActionRequest] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "") or "").strip()
        if not action:
            continue
        try:
            safety_class = ActionSafetyClass(str(item.get("safety_class", "") or ActionSafetyClass.EXPENSIVE_RUNTIME.value))
        except ValueError:
            safety_class = ActionSafetyClass.EXPENSIVE_RUNTIME
        actions.append(
            GuardedActionRequest(
                action=action,
                safety_class=safety_class,
                requested=bool(item.get("requested", False)),
                reason=str(item.get("reason", "") or "").strip(),
            )
        )
    return actions


def normalize_interpreter_v2_payload(
    payload: dict[str, Any],
    *,
    source: str = "interpreter_v2",
    min_confidence: float = 0.6,
) -> CandidatePatchEnvelope:
    turn_summary = payload.get("turn_summary") if isinstance(payload.get("turn_summary"), dict) else {}
    intent = _intent_from_text(turn_summary.get("intent"))
    operations: list[PatchOperation] = []
    for item in payload.get("candidate_updates") if isinstance(payload.get("candidate_updates"), list) else []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "").strip()
        op = str(item.get("op", "") or "").strip()
        if not path or op not in {"set", "remove", "keep"}:
            continue
        confidence = _coerce_confidence(item.get("confidence"))
        reasons = _confirmation_reasons(item, confidence, min_confidence=min_confidence)
        operations.append(
            PatchOperation(
                path=path,
                op=op,
                value=item.get("value"),
                confidence=confidence,
                evidence=_coerce_evidence(item.get("evidence")),
                requires_confirmation=bool(reasons),
                confirmation_reasons=reasons,
            )
        )
    envelope = CandidatePatchEnvelope(
        source=source,
        intent=intent,
        operations=operations,
        guarded_actions=_coerce_guarded_actions(payload.get("guarded_actions")),
        ambiguities=list(payload.get("ambiguities") or []) if isinstance(payload.get("ambiguities"), list) else [],
        unsupported_requests=list(payload.get("unsupported_requests") or []) if isinstance(payload.get("unsupported_requests"), list) else [],
        patch_hash="",
    )
    return CandidatePatchEnvelope(
        source=envelope.source,
        intent=envelope.intent,
        operations=envelope.operations,
        guarded_actions=envelope.guarded_actions,
        ambiguities=envelope.ambiguities,
        unsupported_requests=envelope.unsupported_requests,
        patch_hash=_stable_hash(envelope.to_dict()),
    )


def envelope_to_candidate_update(envelope: CandidatePatchEnvelope, *, turn_id: int, producer: Producer = Producer.LLM_SEMANTIC_FRAME) -> CandidateUpdate:
    updates: list[UpdateOp] = []
    for operation in envelope.operations:
        if operation.op == "keep":
            continue
        updates.append(
            UpdateOp(
                path=operation.path,
                op="remove" if operation.op == "remove" else "set",
                value=operation.value,
                producer=producer,
                confidence=operation.confidence,
                turn_id=turn_id,
            )
        )
    confidence = min((operation.confidence for operation in envelope.operations), default=0.0)
    return CandidateUpdate(
        producer=producer,
        intent=envelope.intent,
        target_paths=sorted({operation.path for operation in envelope.operations}),
        updates=updates,
        confidence=confidence,
        rationale=f"{envelope.source}:{envelope.patch_hash}",
    )


__all__ = [
    "CONFIRM_DELETE",
    "CONFIRM_LOW_CONFIDENCE",
    "CONFIRM_OVERWRITE",
    "CandidatePatchEnvelope",
    "GuardedActionRequest",
    "PatchEvidence",
    "PatchOperation",
    "envelope_to_candidate_update",
    "normalize_interpreter_v2_payload",
]
