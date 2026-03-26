from .slot_frame import SlotFrame
from .slot_validator import validate_slot_frame


def slot_frame_to_candidates(*args, **kwargs):
    from .slot_mapper import slot_frame_to_candidates as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "SlotFrame",
    "slot_frame_to_candidates",
    "validate_slot_frame",
]
