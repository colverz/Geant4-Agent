from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeometryEvidence:
    source: str
    field: str
    value: Any
    confidence: float = 1.0
    detail: str = ""


@dataclass(frozen=True)
class GeometryFieldResolution:
    field: str
    value: Any
    status: str
    evidence_sources: tuple[str, ...] = ()
    note: str = ""


@dataclass
class GeometryIntent:
    structure: str | None = None
    kind: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    evidence: list[GeometryEvidence] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    field_resolutions: dict[str, GeometryFieldResolution] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometrySpec:
    structure: str
    params: dict[str, float | list[float]]
    allowed_paths: frozenset[str] = field(default_factory=frozenset)
    required_paths: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 1.0
    finalization_status: str = "ready"
    field_resolutions: dict[str, GeometryFieldResolution] = field(default_factory=dict)
    provenance_summary: dict[str, int] = field(default_factory=dict)
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def runtime_ready(self) -> bool:
        return self.finalization_status == "ready"
