from __future__ import annotations

from core.geometry.family_catalog import GEOMETRY_SLOT_TARGET_TO_PATHS
from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from core.pipelines import (
    PIPELINE_LEGACY,
    PIPELINE_V2,
    build_legacy_geometry_updates,
    build_legacy_source_updates,
    build_v2_geometry_updates,
    build_v2_source_updates,
    select_pipelines,
)
from core.slots.slot_frame import SlotFrame

_SLOT_TARGET_TO_PATHS = {
    **GEOMETRY_SLOT_TARGET_TO_PATHS,
    "materials.primary": {"materials.selected_materials", "materials.volume_material_map"},
    "source.kind": {"source.type"},
    "source.particle": {"source.particle"},
    "source.energy_mev": {"source.energy"},
    "source.position_mm": {"source.position"},
    "source.direction_vec": {"source.direction"},
    "physics.explicit_list": {"physics.physics_list"},
    "physics.recommendation_intent": {"physics.physics_list"},
    "output.format": {"output.format", "output.path"},
    "output.path": {"output.path"},
}

_MATERIAL_CANONICAL_OVERRIDES = {
    "g4_csi": "G4_CESIUM_IODIDE",
    "csi": "G4_CESIUM_IODIDE",
    "cesium iodide": "G4_CESIUM_IODIDE",
    "caesium iodide": "G4_CESIUM_IODIDE",
    "cesium iodide crystal": "G4_CESIUM_IODIDE",
}


def _canonical_material_name(value: str) -> str:
    text = str(value or "").strip()
    return _MATERIAL_CANONICAL_OVERRIDES.get(text.lower(), text)


def _build_geometry_updates(frame: SlotFrame, *, turn_id: int, geometry_mode: str) -> tuple[list[UpdateOp], list[str]]:
    if geometry_mode == PIPELINE_V2:
        updates, targets, _ = build_v2_geometry_updates(frame, turn_id=turn_id)
        return updates, targets
    return build_legacy_geometry_updates(frame, turn_id=turn_id)


def _build_source_updates(frame: SlotFrame, *, turn_id: int, source_mode: str) -> tuple[list[UpdateOp], list[str]]:
    if source_mode == PIPELINE_V2:
        updates, targets, _ = build_v2_source_updates(frame, turn_id=turn_id)
        return updates, targets
    return build_legacy_source_updates(frame, turn_id=turn_id)


def slot_frame_to_candidates(
    frame: SlotFrame,
    *,
    turn_id: int,
    geometry_mode: str | None = None,
    source_mode: str | None = None,
) -> tuple[CandidateUpdate | None, CandidateUpdate]:
    selection = select_pipelines(geometry=geometry_mode, source=source_mode)

    updates: list[UpdateOp] = []
    target_paths: list[str] = []

    geometry_updates, geometry_targets = _build_geometry_updates(frame, turn_id=turn_id, geometry_mode=selection.geometry)
    if geometry_updates:
        updates.extend(geometry_updates)
        target_paths.extend(geometry_targets)

    source_updates, source_targets = _build_source_updates(frame, turn_id=turn_id, source_mode=selection.source)
    if source_updates:
        updates.extend(source_updates)
        target_paths.extend(source_targets)

    if frame.materials.primary:
        updates.append(
            UpdateOp(
                path="materials.selected_materials",
                op="set",
                value=[_canonical_material_name(frame.materials.primary)],
                producer=Producer.SLOT_MAPPER,
                confidence=frame.confidence or 0.8,
                turn_id=turn_id,
            )
        )
        target_paths.extend(["materials.selected_materials", "materials.volume_material_map"])

    if frame.physics.explicit_list:
        updates.append(
            UpdateOp(
                path="physics.physics_list",
                op="set",
                value=frame.physics.explicit_list,
                producer=Producer.SLOT_MAPPER,
                confidence=frame.confidence or 0.8,
                turn_id=turn_id,
            )
        )
        target_paths.append("physics.physics_list")

    if frame.output.format:
        updates.append(
            UpdateOp(
                path="output.format",
                op="set",
                value=frame.output.format,
                producer=Producer.SLOT_MAPPER,
                confidence=frame.confidence or 0.8,
                turn_id=turn_id,
            )
        )
        target_paths.extend(["output.format", "output.path"])

    if frame.output.path:
        updates.append(
            UpdateOp(
                path="output.path",
                op="set",
                value=frame.output.path,
                producer=Producer.SLOT_MAPPER,
                confidence=frame.confidence or 0.8,
                turn_id=turn_id,
            )
        )
        target_paths.append("output.path")

    dedup_updates: dict[str, UpdateOp] = {}
    for update in updates:
        dedup_updates[update.path] = update
    updates = list(dedup_updates.values())
    for slot_target in frame.target_slots:
        target_paths.extend(_SLOT_TARGET_TO_PATHS.get(slot_target, set()))
    target_paths = sorted(set(target_paths) | {u.path for u in updates})

    content_candidate = None
    if updates:
        content_candidate = CandidateUpdate(
            producer=Producer.SLOT_MAPPER,
            intent=frame.intent,
            target_paths=list(target_paths),
            updates=updates,
            confidence=frame.confidence or 0.8,
            rationale=f"slot_frame_mapper[{selection.geometry}|{selection.source}]",
        )

    user_candidate = CandidateUpdate(
        producer=Producer.USER_EXPLICIT,
        intent=frame.intent if frame.intent in {Intent.SET, Intent.MODIFY, Intent.REMOVE, Intent.CONFIRM, Intent.REJECT, Intent.QUESTION} else Intent.OTHER,
        target_paths=list(target_paths),
        updates=[],
        confidence=frame.confidence or 0.8,
        rationale="slot_frame_user_targets",
    )
    return content_candidate, user_candidate
