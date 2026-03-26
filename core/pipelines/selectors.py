from __future__ import annotations

import os
from dataclasses import dataclass


PIPELINE_LEGACY = "legacy"
PIPELINE_V2 = "v2"
_VALID = {PIPELINE_LEGACY, PIPELINE_V2}


@dataclass(frozen=True)
class PipelineSelection:
    geometry: str = PIPELINE_LEGACY
    source: str = PIPELINE_LEGACY


def _resolve(value: str | None, *, default: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in _VALID else default


def select_pipelines(
    *,
    geometry: str | None = None,
    source: str | None = None,
) -> PipelineSelection:
    return PipelineSelection(
        geometry=_resolve(geometry or os.getenv("GEOMETRY_PIPELINE"), default=PIPELINE_LEGACY),
        source=_resolve(source or os.getenv("SOURCE_PIPELINE"), default=PIPELINE_LEGACY),
    )
