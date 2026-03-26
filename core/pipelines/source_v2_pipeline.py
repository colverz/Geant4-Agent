from __future__ import annotations

from core.orchestrator.types import Producer, UpdateOp
from core.slots.slot_frame import SlotFrame
from core.source.adapters.config_fragment import source_spec_to_config_fragment
from core.source.compiler import compile_source_spec_from_slot_frame


def build_v2_source_updates(frame: SlotFrame, *, turn_id: int) -> tuple[list[UpdateOp], list[str], dict[str, object]]:
    result = compile_source_spec_from_slot_frame(frame)
    if not result.ok or result.spec is None or result.spec.finalization_status != "ready":
        return [], [], {
            "compile_ok": False,
            "finalization_status": result.spec.finalization_status if result.spec else "missing",
            "missing_fields": list(result.missing_fields),
            "errors": list(result.errors),
        }
    fragment = source_spec_to_config_fragment(result.spec).get("source", {})
    updates: list[UpdateOp] = []
    target_paths: list[str] = []
    for key, value in fragment.items():
        path = f"source.{key}"
        updates.append(
            UpdateOp(path=path, op="set", value=value, producer=Producer.SLOT_MAPPER, confidence=frame.confidence or 0.8, turn_id=turn_id)
        )
        target_paths.append(path)
    return updates, target_paths, {
        "compile_ok": True,
        "source_type": result.spec.source_type,
        "finalization_status": result.spec.finalization_status,
    }
