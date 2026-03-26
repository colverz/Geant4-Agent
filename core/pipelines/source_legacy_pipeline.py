from __future__ import annotations

from core.orchestrator.types import Producer, UpdateOp
from core.slots.slot_frame import SlotFrame


def _vector3(value: list[float]) -> dict[str, object]:
    return {"type": "vector", "value": [float(value[0]), float(value[1]), float(value[2])]}


def build_legacy_source_updates(frame: SlotFrame, *, turn_id: int) -> tuple[list[UpdateOp], list[str]]:
    updates: list[UpdateOp] = []
    target_paths: list[str] = []

    if frame.source.kind:
        updates.append(
            UpdateOp(path="source.type", op="set", value=frame.source.kind, producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append("source.type")
    if frame.source.particle:
        updates.append(
            UpdateOp(path="source.particle", op="set", value=frame.source.particle, producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append("source.particle")
    if frame.source.energy_mev is not None:
        updates.append(
            UpdateOp(path="source.energy", op="set", value=float(frame.source.energy_mev), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append("source.energy")
    if frame.source.position_mm:
        updates.append(
            UpdateOp(path="source.position", op="set", value=_vector3(frame.source.position_mm), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append("source.position")
    if frame.source.direction_vec:
        updates.append(
            UpdateOp(path="source.direction", op="set", value=_vector3(frame.source.direction_vec), producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append("source.direction")
    return updates, target_paths
