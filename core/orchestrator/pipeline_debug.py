from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


@dataclass(frozen=True)
class V2PipelineMetaView:
    missing_fields: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, value: object) -> V2PipelineMetaView:
        if not isinstance(value, dict):
            return cls()
        return cls(
            missing_fields=_strings(value.get("missing_fields")),
            errors=_strings(value.get("errors")),
            warnings=_strings(value.get("warnings")),
        )


@dataclass(frozen=True)
class V2PipelineDebugView:
    geometry: V2PipelineMetaView = V2PipelineMetaView()
    source: V2PipelineMetaView = V2PipelineMetaView()
    spatial: V2PipelineMetaView = V2PipelineMetaView()
    spatial_source: V2PipelineMetaView = V2PipelineMetaView()

    @classmethod
    def from_slot_debug(cls, slot_debug: dict[str, Any]) -> V2PipelineDebugView:
        spatial_meta = slot_debug.get("spatial_v2")
        spatial_source_meta = spatial_meta.get("source_meta") if isinstance(spatial_meta, dict) else None
        return cls(
            geometry=V2PipelineMetaView.from_raw(slot_debug.get("geometry_v2")),
            source=V2PipelineMetaView.from_raw(slot_debug.get("source_v2")),
            spatial=V2PipelineMetaView.from_raw(spatial_meta),
            spatial_source=V2PipelineMetaView.from_raw(spatial_source_meta),
        )
