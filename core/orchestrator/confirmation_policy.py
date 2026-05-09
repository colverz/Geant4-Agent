from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config.field_registry import friendly_label
from core.orchestrator.path_ops import get_path
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.validation.error_codes import E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT


_EXPLICIT_TARGET_DEPENDENCIES = {
    "geometry.structure": {"geometry.chosen_skeleton", "geometry.graph_program", "geometry.root_name"},
}


@dataclass(frozen=True)
class ConfirmationPolicyResult:
    filtered_candidates: list[CandidateUpdate]
    pending: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.pending)


def evaluate_confirmation_requirements(
    state_like: Any,
    user_candidate: CandidateUpdate,
    candidates: list[CandidateUpdate],
    *,
    lang: str,
    min_confidence: float = 0.0,
    enforce_no_implicit_overwrite: bool = False,
    low_confidence_first: bool = True,
    evaluate_pending: bool = True,
) -> ConfirmationPolicyResult:
    working_candidates = list(candidates)
    rejected: list[dict[str, Any]] = []
    if enforce_no_implicit_overwrite:
        working_candidates, rejected = _enforce_no_implicit_overwrite(state_like, user_candidate, working_candidates)
    if not evaluate_pending:
        return ConfirmationPolicyResult(
            filtered_candidates=working_candidates,
            pending=[],
            rejected=rejected,
        )
    if low_confidence_first:
        working_candidates, low_confidence_pending = _extract_low_confidence_updates(
            state_like,
            working_candidates,
            min_confidence=min_confidence,
            lang=lang,
        )
        working_candidates, overwrite_pending = _extract_pending_overwrites(
            state_like,
            user_candidate,
            working_candidates,
            lang=lang,
        )
        pending = low_confidence_pending + overwrite_pending
    else:
        working_candidates, overwrite_pending = _extract_pending_overwrites(
            state_like,
            user_candidate,
            working_candidates,
            lang=lang,
        )
        working_candidates, low_confidence_pending = _extract_low_confidence_updates(
            state_like,
            working_candidates,
            min_confidence=min_confidence,
            lang=lang,
        )
        pending = overwrite_pending + low_confidence_pending
    return ConfirmationPolicyResult(
        filtered_candidates=working_candidates,
        pending=pending,
        rejected=rejected,
    )


def _path_explicitly_requested(user_candidate: CandidateUpdate, path: str) -> bool:
    expanded_targets = {str(target) for target in user_candidate.target_paths if isinstance(target, str) and target}
    for target in list(expanded_targets):
        expanded_targets.update(_EXPLICIT_TARGET_DEPENDENCIES.get(target, set()))
    for target in expanded_targets:
        if path == target:
            return True
        if path.startswith(target + "."):
            return True
    return False


def _is_unset_for_overwrite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _enforce_no_implicit_overwrite(
    state_like: Any,
    user_candidate: CandidateUpdate,
    candidates: list[CandidateUpdate],
) -> tuple[list[CandidateUpdate], list[dict]]:
    policy_rejected: list[dict] = []
    allow_overwrite = user_candidate.intent in {Intent.SET, Intent.MODIFY}
    filtered_candidates: list[CandidateUpdate] = []
    for candidate in candidates:
        kept = []
        for upd in candidate.updates:
            old = get_path(state_like.config, upd.path)
            if _is_unset_for_overwrite(old) or old == upd.value:
                kept.append(upd)
                continue
            explicitly_requested = _path_explicitly_requested(user_candidate, upd.path)
            if allow_overwrite and explicitly_requested:
                kept.append(upd)
                continue
            policy_rejected.append(
                {
                    "path": upd.path,
                    "producer": candidate.producer.value,
                    "reason_code": E_OVERWRITE_WITHOUT_EXPLICIT_USER_INTENT,
                    "detail": "overwrite blocked: user did not explicitly request this field update",
                }
            )
        if not kept:
            continue
        if len(kept) == len(candidate.updates):
            filtered_candidates.append(candidate)
            continue
        filtered_candidates.append(
            CandidateUpdate(
                producer=candidate.producer,
                intent=candidate.intent,
                target_paths=sorted({u.path for u in kept}),
                updates=kept,
                confidence=candidate.confidence,
                rationale=f"{candidate.rationale}_overwrite_guarded",
            )
        )
    return filtered_candidates, policy_rejected


def _extract_pending_overwrites(
    state_like: Any,
    user_candidate: CandidateUpdate,
    candidates: list[CandidateUpdate],
    *,
    lang: str,
) -> tuple[list[CandidateUpdate], list[dict[str, Any]]]:
    if user_candidate.intent not in {Intent.SET, Intent.MODIFY, Intent.REMOVE}:
        return candidates, []
    pending: list[dict[str, Any]] = []
    filtered_candidates: list[CandidateUpdate] = []
    for candidate in candidates:
        kept: list[UpdateOp] = []
        for upd in candidate.updates:
            old = get_path(state_like.config, upd.path)
            if upd.op == "remove" and not _is_unset_for_overwrite(old):
                pending.append(
                    _pending_item_from_update(
                        upd,
                        draft=state_like,
                        lang=lang,
                        producer=candidate.producer.value,
                        reason="remove",
                    )
                )
                continue
            if _is_unset_for_overwrite(old) or old == upd.value:
                kept.append(upd)
                continue
            if _path_explicitly_requested(user_candidate, upd.path):
                pending.append(
                    _pending_item_from_update(
                        upd,
                        draft=state_like,
                        lang=lang,
                        producer=candidate.producer.value,
                        reason="overwrite",
                    )
                )
                continue
            kept.append(upd)
        if not kept:
            continue
        if len(kept) == len(candidate.updates):
            filtered_candidates.append(candidate)
            continue
        filtered_candidates.append(
            CandidateUpdate(
                producer=candidate.producer,
                intent=candidate.intent,
                target_paths=sorted({u.path for u in kept}),
                updates=kept,
                confidence=candidate.confidence,
                rationale=f"{candidate.rationale}_overwrite_staged",
            )
        )
    return filtered_candidates, pending


def _effective_update_confidence(candidate: CandidateUpdate, update: UpdateOp) -> float:
    try:
        candidate_conf = float(candidate.confidence)
    except (TypeError, ValueError):
        candidate_conf = 0.0
    try:
        update_conf = float(update.confidence)
    except (TypeError, ValueError):
        update_conf = 0.0
    return max(0.0, min(1.0, min(candidate_conf, update_conf)))


def _extract_low_confidence_updates(
    state_like: Any,
    candidates: list[CandidateUpdate],
    *,
    min_confidence: float,
    lang: str,
) -> tuple[list[CandidateUpdate], list[dict[str, Any]]]:
    threshold = max(0.0, min(1.0, float(min_confidence)))
    if threshold <= 0.0:
        return candidates, []
    pending: list[dict[str, Any]] = []
    filtered_candidates: list[CandidateUpdate] = []
    for candidate in candidates:
        if candidate.producer != Producer.LLM_SEMANTIC_FRAME:
            filtered_candidates.append(candidate)
            continue
        kept: list[UpdateOp] = []
        for update in candidate.updates:
            effective_confidence = _effective_update_confidence(candidate, update)
            if effective_confidence < threshold:
                pending.append(
                    _pending_item_from_update(
                        update,
                        draft=state_like,
                        lang=lang,
                        producer=candidate.producer.value,
                        reason="low_confidence",
                        confidence=effective_confidence,
                    )
                )
                continue
            kept.append(update)
        if not kept:
            continue
        if len(kept) == len(candidate.updates):
            filtered_candidates.append(candidate)
            continue
        filtered_candidates.append(
            CandidateUpdate(
                producer=candidate.producer,
                intent=candidate.intent,
                target_paths=sorted({u.path for u in kept}),
                updates=kept,
                confidence=candidate.confidence,
                rationale=f"{candidate.rationale}_low_confidence_staged",
            )
        )
    return filtered_candidates, pending


def _candidate_from_pending_overwrite(items: list[dict[str, Any]], *, turn_id: int) -> CandidateUpdate:
    updates = [
        UpdateOp(
            path=str(item["path"]),
            op="remove" if str(item.get("op") or "").strip() == "remove" else "set",
            value=item.get("new"),
            producer=Producer.USER_EXPLICIT,
            confidence=1.0,
            turn_id=turn_id,
        )
        for item in items
    ]
    return CandidateUpdate(
        producer=Producer.USER_EXPLICIT,
        intent=Intent.MODIFY,
        target_paths=sorted({str(item["path"]) for item in items}),
        updates=updates,
        confidence=1.0,
        rationale="confirmed_pending_overwrite",
    )


def _merge_pending_overwrites(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in existing:
        path = str(item.get("path", "")).strip()
        if path:
            merged[path] = dict(item)
    for item in additions:
        path = str(item.get("path", "")).strip()
        if path:
            merged[path] = dict(item)
    return list(merged.values())


def _has_pending_overwrite_path(items: list[dict[str, Any]], path: str) -> bool:
    target = str(path).strip()
    if not target:
        return False
    for item in items:
        if str(item.get("path", "")).strip() == target:
            return True
    return False


def _pending_item_from_update(
    update: UpdateOp,
    *,
    draft: Any,
    lang: str,
    producer: str,
    reason: str = "overwrite",
    confidence: float | None = None,
) -> dict[str, Any]:
    return {
        "path": update.path,
        "op": update.op,
        "field": friendly_label(update.path, lang),
        "old": get_path(draft.config, update.path),
        "new": update.value,
        "producer": producer,
        "reason": reason,
        **({"confidence": confidence} if confidence is not None else {}),
    }
