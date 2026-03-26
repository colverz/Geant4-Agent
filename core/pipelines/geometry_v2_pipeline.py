from __future__ import annotations

from core.geometry.adapters.config_fragment import geometry_spec_to_config_fragment
from core.geometry.compiler import compile_geometry_spec_from_slot_frame
from core.orchestrator.types import Producer, UpdateOp
from core.slots.slot_frame import SlotFrame


def build_v2_geometry_updates(frame: SlotFrame, *, turn_id: int) -> tuple[list[UpdateOp], list[str], dict[str, object]]:
    result = compile_geometry_spec_from_slot_frame(frame)
    if not result.ok or result.spec is None or result.spec.finalization_status != "ready":
        return [], [], {
            "compile_ok": False,
            "finalization_status": result.spec.finalization_status if result.spec else "missing",
            "missing_fields": list(result.missing_fields),
            "errors": list(result.errors),
        }

    fragment = geometry_spec_to_config_fragment(result.spec).get("geometry", {})
    updates: list[UpdateOp] = []
    target_paths: list[str] = []
    structure = fragment.get("structure")
    if structure:
        updates.append(
            UpdateOp(
                path="geometry.structure",
                op="set",
                value=structure,
                producer=Producer.SLOT_MAPPER,
                confidence=frame.confidence or 0.8,
                turn_id=turn_id,
            )
        )
        target_paths.extend(["geometry.structure", "geometry.root_name"])

    params = fragment.get("params", {}) if isinstance(fragment.get("params"), dict) else {}
    for key, value in params.items():
        path = f"geometry.params.{key}"
        updates.append(
            UpdateOp(path=path, op="set", value=value, producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append(path)

    return updates, target_paths, {
        "compile_ok": True,
        "structure": result.spec.structure,
        "finalization_status": result.spec.finalization_status,
    }
