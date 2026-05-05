from __future__ import annotations

from core.geometry.family_catalog import GEOMETRY_SLOT_TARGET_TO_PATHS
from core.orchestrator.types import CandidateUpdate, UpdateOp

_DEPENDENT_TARGETS = {
    "geometry.structure": {"geometry.chosen_skeleton", "geometry.graph_program", "geometry.root_name"},
    "geometry.graph_program": {"geometry.structure", "geometry.chosen_skeleton", "geometry.root_name"},
    "geometry.chosen_skeleton": {"geometry.structure", "geometry.graph_program", "geometry.root_name"},
}

_TARGET_ALIAS_PATHS = {
    **GEOMETRY_SLOT_TARGET_TO_PATHS,
    "materials.primary": {"materials.selected_materials", "materials.volume_material_map"},
    "source.kind": {"source.type"},
    "source.particle": {"source.particle"},
    "source.energy_mev": {"source.energy"},
    "source.position_mm": {"source.position"},
    "source.direction_vec": {"source.direction"},
    "source.spot_radius_mm": {"source.spot_radius_mm"},
    "source.spot_profile": {"source.spot_profile"},
    "source.spot_sigma_mm": {"source.spot_sigma_mm"},
    "source.divergence_half_angle_deg": {"source.divergence_half_angle_deg"},
    "source.divergence_profile": {"source.divergence_profile"},
    "source.divergence_sigma_deg": {"source.divergence_sigma_deg"},
    "detector.enabled": {"simulation.detector.enabled"},
    "detector.name": {"simulation.detector.name"},
    "detector.material": {"simulation.detector.material"},
    "detector.position_mm": {"simulation.detector.position"},
    "detector.size_triplet_mm": {"simulation.detector.size_triplet_mm"},
    "scoring.target_edep": {"scoring.target_edep"},
    "scoring.detector_crossings": {"scoring.detector_crossings"},
    "scoring.plane_crossings": {"scoring.plane_crossings"},
    "scoring.plane_name": {"scoring.plane.name"},
    "scoring.plane_z_mm": {"scoring.plane.z_mm"},
    "physics.explicit_list": {"physics.physics_list"},
    "physics.recommendation_intent": {"physics.physics_list"},
    "output.format": {"output.format", "output.path"},
    "output.path": {"output.path"},
}


def _expand_explicit_targets(target_paths: list[str]) -> list[str]:
    expanded = {str(path) for path in target_paths if isinstance(path, str) and path}
    for target in list(expanded):
        expanded.update(_TARGET_ALIAS_PATHS.get(target, set()))
        expanded.update(_DEPENDENT_TARGETS.get(target, set()))
    return sorted(expanded)


def _matches_explicit_target(path: str, target_paths: list[str]) -> bool:
    for target in target_paths:
        if not isinstance(target, str) or not target:
            continue
        if "." not in target:
            if path == target or path.startswith(target + "."):
                return True
            continue
        if path == target or path.startswith(target + "."):
            return True
    return False


def filter_candidate_by_target_scopes(candidate: CandidateUpdate, target_paths: list[str]) -> CandidateUpdate:
    scopes = {str(p).split(".", 1)[0] for p in target_paths if "." in str(p)}
    if not scopes:
        return candidate
    filtered = [u for u in candidate.updates if u.path.split(".", 1)[0] in scopes]
    if len(filtered) == len(candidate.updates):
        return candidate
    return CandidateUpdate(
        producer=candidate.producer,
        intent=candidate.intent,
        target_paths=sorted({u.path for u in filtered}),
        updates=filtered,
        confidence=candidate.confidence,
        rationale=f"{candidate.rationale}_scope_filtered",
    )


def filter_candidate_by_explicit_targets(candidate: CandidateUpdate, target_paths: list[str]) -> CandidateUpdate:
    explicit_targets = _expand_explicit_targets(target_paths)
    if not explicit_targets or not candidate.updates:
        return candidate
    filtered = [u for u in candidate.updates if _matches_explicit_target(u.path, explicit_targets)]
    if len(filtered) == len(candidate.updates):
        return candidate
    return CandidateUpdate(
        producer=candidate.producer,
        intent=candidate.intent,
        target_paths=sorted(set(explicit_targets) | {u.path for u in filtered}),
        updates=filtered,
        confidence=candidate.confidence,
        rationale=f"{candidate.rationale}_path_filtered",
    )


def drop_updates_shadowed_by_anchor(candidate: CandidateUpdate, anchor: CandidateUpdate | None) -> CandidateUpdate:
    if anchor is None or not anchor.updates or not candidate.updates:
        return candidate
    anchored_paths = {upd.path for upd in anchor.updates}
    filtered = [upd for upd in candidate.updates if upd.path not in anchored_paths]
    if len(filtered) == len(candidate.updates):
        return candidate
    return CandidateUpdate(
        producer=candidate.producer,
        intent=candidate.intent,
        target_paths=sorted({u.path for u in filtered}),
        updates=filtered,
        confidence=candidate.confidence,
        rationale=f"{candidate.rationale}_anchor_shadow_filtered",
    )


def partition_candidate_by_pending_paths(
    candidate: CandidateUpdate,
    pending_paths: list[str],
    *,
    replace_target_paths: list[str] | None = None,
) -> tuple[CandidateUpdate, list[UpdateOp], list[UpdateOp]]:
    reserved_paths = [str(path) for path in pending_paths if isinstance(path, str) and path]
    if not reserved_paths or not candidate.updates:
        return candidate, [], []
    replace_targets = [str(path) for path in (replace_target_paths or []) if isinstance(path, str) and path]
    kept: list[UpdateOp] = []
    blocked: list[UpdateOp] = []
    replacements: list[UpdateOp] = []
    for update in candidate.updates:
        if not _matches_explicit_target(update.path, reserved_paths):
            kept.append(update)
            continue
        if replace_targets and _matches_explicit_target(update.path, replace_targets):
            replacements.append(update)
            continue
        blocked.append(update)
    if len(kept) == len(candidate.updates):
        return candidate, blocked, replacements
    return (
        CandidateUpdate(
            producer=candidate.producer,
            intent=candidate.intent,
            target_paths=sorted({u.path for u in kept}),
            updates=kept,
            confidence=candidate.confidence,
            rationale=f"{candidate.rationale}_pending_filtered",
        ),
        blocked,
        replacements,
    )
