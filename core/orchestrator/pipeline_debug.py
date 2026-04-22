from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_V2_MISSING_FIELD_PATHS = {
    ("geometry", "size_triplet_mm"): "geometry.params.module_x",
    ("geometry", "radius_mm"): "geometry.params.child_rmax",
    ("geometry", "half_length_mm"): "geometry.params.child_hz",
    ("geometry", "radius1_mm"): "geometry.params.rmax1",
    ("geometry", "radius2_mm"): "geometry.params.rmax2",
    ("geometry", "x1_mm"): "geometry.params.x1",
    ("geometry", "x2_mm"): "geometry.params.x2",
    ("geometry", "y1_mm"): "geometry.params.y1",
    ("geometry", "y2_mm"): "geometry.params.y2",
    ("geometry", "z_mm"): "geometry.params.module_z",
    ("source", "particle"): "source.particle",
    ("source", "energy_mev"): "source.energy",
    ("source", "position_mm"): "source.position",
    ("source", "direction_vec"): "source.direction",
    ("source", "source_type"): "source.type",
}


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def v2_missing_field_to_path(domain: str, field: str) -> str:
    return _V2_MISSING_FIELD_PATHS.get((domain, field), "")


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

    def missing_paths(self, domain: str) -> list[str]:
        paths: list[str] = []
        for field in self.missing_fields:
            path = v2_missing_field_to_path(domain, field)
            if path:
                paths.append(path)
        return paths


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

    def compile_missing_paths(self) -> list[str]:
        paths: list[str] = []
        if "missing_geometry_structure" in self.geometry.errors:
            paths.append("geometry.structure")
        paths.extend(self.geometry.missing_paths("geometry"))
        paths.extend(self.source.missing_paths("source"))
        paths.extend(self.spatial_source.missing_paths("source"))
        return _dedupe(paths)

    def spatial_review_missing_paths(self) -> list[str]:
        severe_warnings = {"source_inside_target", "source_on_target_face"}
        if severe_warnings & set(self.spatial.warnings):
            return ["source.position"]
        return []

    def prioritize_compile_questions(self, asked_fields: list[str], missing_fields: list[str]) -> list[str]:
        prioritized: list[str] = []
        for field in self.spatial_source.missing_fields:
            path = v2_missing_field_to_path("source", field)
            if path and path in missing_fields and path not in prioritized:
                prioritized.append(path)
        for field in self.source.missing_fields:
            path = v2_missing_field_to_path("source", field)
            if path and path in missing_fields and path not in prioritized:
                prioritized.append(path)
        for field in self.geometry.missing_fields:
            path = v2_missing_field_to_path("geometry", field)
            if path and path in missing_fields and path not in prioritized:
                prioritized.append(path)
        for path in asked_fields:
            if path not in prioritized:
                prioritized.append(path)
        return prioritized[: max(1, len(asked_fields))]

    def prioritize_spatial_questions(self, asked_fields: list[str], missing_fields: list[str]) -> list[str]:
        if not self.spatial.warnings:
            return asked_fields
        spatial_required = self.spatial_review_missing_paths()
        if not spatial_required:
            return asked_fields
        prioritized = [path for path in spatial_required if path in missing_fields]
        for path in asked_fields:
            if path not in prioritized:
                prioritized.append(path)
            if len(prioritized) >= max(2, len(asked_fields)):
                break
        return prioritized[: max(1, len(asked_fields))]


def compile_v2_missing_paths(slot_debug: dict[str, Any]) -> list[str]:
    return V2PipelineDebugView.from_slot_debug(slot_debug).compile_missing_paths()


def merge_v2_missing_paths(
    base_missing_paths: list[str],
    slot_debug: dict[str, Any],
    *,
    include_spatial: bool = True,
) -> list[str]:
    debug_view = V2PipelineDebugView.from_slot_debug(slot_debug)
    merged = _dedupe(list(base_missing_paths) + debug_view.compile_missing_paths())
    if include_spatial:
        merged = _dedupe(merged + debug_view.spatial_review_missing_paths())
    return merged


def prioritize_v2_compile_questions(
    asked_fields: list[str],
    missing_fields: list[str],
    slot_debug: dict[str, Any],
) -> list[str]:
    return V2PipelineDebugView.from_slot_debug(slot_debug).prioritize_compile_questions(asked_fields, missing_fields)


def prioritize_spatial_questions(
    asked_fields: list[str],
    missing_fields: list[str],
    slot_debug: dict[str, Any],
) -> list[str]:
    return V2PipelineDebugView.from_slot_debug(slot_debug).prioritize_spatial_questions(asked_fields, missing_fields)


def merge_v2_meta(existing: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing, dict):
        return dict(incoming)
    if existing.get("compile_ok"):
        return dict(existing)
    if incoming.get("compile_ok"):
        return dict(incoming)
    merged = dict(existing)
    merged["missing_fields"] = _dedupe(list(existing.get("missing_fields", []) or []) + list(incoming.get("missing_fields", []) or []))
    merged["errors"] = _dedupe(list(existing.get("errors", []) or []) + list(incoming.get("errors", []) or []))
    merged["warnings"] = _dedupe(list(existing.get("warnings", []) or []) + list(incoming.get("warnings", []) or []))
    merged["runtime_ready"] = bool(existing.get("runtime_ready") or incoming.get("runtime_ready"))
    merged["finalization_status"] = str(existing.get("finalization_status") or incoming.get("finalization_status") or "missing")
    return merged
