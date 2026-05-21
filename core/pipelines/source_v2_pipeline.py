from __future__ import annotations

from copy import deepcopy

from core.orchestrator.path_ops import set_path
from core.orchestrator.types import Producer, UpdateOp
from core.slots.slot_frame import SlotFrame
from core.source.adapters.config_fragment import source_spec_to_config_fragment
from core.source.compiler import compile_source_spec_from_config, compile_source_spec_from_slot_frame


def build_v2_source_updates(frame: SlotFrame, *, turn_id: int) -> tuple[list[UpdateOp], list[str], dict[str, object]]:
    result = compile_source_spec_from_slot_frame(frame)
    if not result.ok or result.spec is None or result.spec.finalization_status != "ready":
        return [], [], {
            "compile_ok": False,
            "finalization_status": result.spec.finalization_status if result.spec else "missing",
            "missing_fields": list(result.missing_fields),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "runtime_ready": False,
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
        "runtime_ready": result.spec.runtime_ready,
    }


def build_v2_source_updates_from_config(
    config: dict[str, object],
    *,
    turn_id: int,
    confidence: float = 0.8,
) -> tuple[list[UpdateOp], list[str], dict[str, object]]:
    result = compile_source_spec_from_config(config)
    if not result.ok or result.spec is None or result.spec.finalization_status != "ready":
        return [], [], {
            "compile_ok": False,
            "finalization_status": result.spec.finalization_status if result.spec else "missing",
            "missing_fields": list(result.missing_fields),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "runtime_ready": False,
        }
    fragment = source_spec_to_config_fragment(result.spec).get("source", {})
    updates: list[UpdateOp] = []
    target_paths: list[str] = []
    for key, value in fragment.items():
        path = f"source.{key}"
        updates.append(
            UpdateOp(path=path, op="set", value=value, producer=Producer.RUNTIME_SEMANTIC, confidence=confidence, turn_id=turn_id)
        )
        target_paths.append(path)
    return updates, target_paths, {
        "compile_ok": True,
        "source_type": result.spec.source_type,
        "finalization_status": result.spec.finalization_status,
        "runtime_ready": result.spec.runtime_ready,
    }


def build_v2_source_updates_from_candidate(
    base_config: dict[str, object],
    updates: list[UpdateOp],
    *,
    turn_id: int,
    confidence: float = 0.8,
) -> tuple[list[UpdateOp], list[str], dict[str, object]]:
    temp_config = deepcopy(base_config)
    for update in updates:
        if update.op != "set":
            continue
        set_path(temp_config, update.path, update.value)
    return build_v2_source_updates_from_config(temp_config, turn_id=turn_id, confidence=confidence)
