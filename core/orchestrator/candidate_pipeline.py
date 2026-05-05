from __future__ import annotations

from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp


def dedupe_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        item = str(path or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def candidate_from_updates(
    *,
    intent: Intent,
    updates: list[UpdateOp],
    target_paths: list[str],
    confidence: float,
    rationale: str,
) -> CandidateUpdate | None:
    if not updates:
        return None
    return CandidateUpdate(
        producer=updates[0].producer,
        intent=intent,
        target_paths=list(target_paths),
        updates=list(updates),
        confidence=float(confidence),
        rationale=rationale,
    )


def candidate_structure(candidate: CandidateUpdate | None) -> str | None:
    if candidate is None:
        return None
    for update in candidate.updates:
        if update.path == "geometry.structure" and isinstance(update.value, str):
            return update.value
    return None


def strip_geometry_updates(candidate: CandidateUpdate | None) -> CandidateUpdate | None:
    if candidate is None or not candidate.updates:
        return candidate
    filtered = [update for update in candidate.updates if not update.path.startswith("geometry.")]
    if len(filtered) == len(candidate.updates):
        return candidate
    return CandidateUpdate(
        producer=candidate.producer,
        intent=candidate.intent,
        target_paths=[path for path in candidate.target_paths if not str(path).startswith("geometry.")],
        updates=filtered,
        confidence=candidate.confidence,
        rationale=f"{candidate.rationale}_geometry_stripped",
    )


def strip_source_updates(candidate: CandidateUpdate | None) -> CandidateUpdate | None:
    if candidate is None or not candidate.updates:
        return candidate
    filtered = [update for update in candidate.updates if not update.path.startswith("source.")]
    if len(filtered) == len(candidate.updates):
        return candidate
    return CandidateUpdate(
        producer=candidate.producer,
        intent=candidate.intent,
        target_paths=[path for path in candidate.target_paths if not str(path).startswith("source.")],
        updates=filtered,
        confidence=candidate.confidence,
        rationale=f"{candidate.rationale}_source_stripped",
    )


def candidate_has_update_prefix(candidate: CandidateUpdate | None, prefix: str) -> bool:
    if candidate is None:
        return False
    return any(str(update.path).startswith(prefix) for update in candidate.updates)


def retag_candidate(candidate: CandidateUpdate | None, *, producer: Producer, confidence: float) -> CandidateUpdate | None:
    if candidate is None:
        return None
    updates = [
        UpdateOp(
            path=update.path,
            op=update.op,
            value=update.value,
            producer=producer,
            confidence=confidence,
            turn_id=update.turn_id,
        )
        for update in candidate.updates
    ]
    return CandidateUpdate(
        producer=producer,
        intent=candidate.intent,
        target_paths=list(candidate.target_paths),
        updates=updates,
        confidence=confidence,
        rationale=f"{candidate.rationale}_interpreter",
    )


def augment_geometry_targets(
    user_candidate: CandidateUpdate | None,
    extracted_candidate: CandidateUpdate | None,
) -> CandidateUpdate | None:
    if user_candidate is None or extracted_candidate is None:
        return user_candidate
    existing_targets = [
        str(path)
        for path in user_candidate.target_paths
        if isinstance(path, str) and path
    ]
    if existing_targets and not any(path == "geometry" or path.startswith("geometry.") for path in existing_targets):
        return user_candidate
    geometry_targets = [update.path for update in extracted_candidate.updates if update.path.startswith("geometry.")]
    if not geometry_targets:
        return user_candidate
    merged_targets = dedupe_paths(list(user_candidate.target_paths) + geometry_targets)
    if merged_targets == user_candidate.target_paths:
        return user_candidate
    return CandidateUpdate(
        producer=user_candidate.producer,
        intent=user_candidate.intent,
        target_paths=merged_targets,
        updates=list(user_candidate.updates),
        confidence=user_candidate.confidence,
        rationale=f"{user_candidate.rationale}_graph_targets_augmented",
    )


def augment_user_targets(user_candidate: CandidateUpdate, extra_paths: list[str]) -> CandidateUpdate:
    merged_targets = dedupe_paths(list(user_candidate.target_paths) + [str(path) for path in extra_paths if str(path)])
    if merged_targets == list(user_candidate.target_paths):
        return user_candidate
    return CandidateUpdate(
        producer=user_candidate.producer,
        intent=user_candidate.intent,
        target_paths=merged_targets,
        updates=list(user_candidate.updates),
        confidence=user_candidate.confidence,
        rationale=f"{user_candidate.rationale}_targets_augmented",
    )
