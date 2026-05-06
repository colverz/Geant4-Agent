from __future__ import annotations

from core.runtime.types import ActionSafetyClass


def is_read_only(safety_class: ActionSafetyClass | str) -> bool:
    return ActionSafetyClass(safety_class) == ActionSafetyClass.READ_ONLY


def is_runtime_side_effect(safety_class: ActionSafetyClass | str) -> bool:
    return ActionSafetyClass(safety_class) in {
        ActionSafetyClass.EXPENSIVE_RUNTIME,
        ActionSafetyClass.COSTLY_RUNTIME,
        ActionSafetyClass.EXTERNAL_EFFECT,
    }


def requires_confirmation(safety_class: ActionSafetyClass | str) -> bool:
    return ActionSafetyClass(safety_class) in {
        ActionSafetyClass.STATE_OVERWRITE,
        ActionSafetyClass.DESTRUCTIVE_MUTATION,
        ActionSafetyClass.AMBIGUOUS,
    }


__all__ = ["ActionSafetyClass", "is_read_only", "is_runtime_side_effect", "requires_confirmation"]
