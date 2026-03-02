from __future__ import annotations

from typing import Sequence

from core.config.path_registry import field_matches_pattern


LEGACY_PHASE_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("geometry_core", ("geometry.structure",)),
    ("geometry_params", ("geometry.params.",)),
    ("materials", ("materials.",)),
    ("source_core", ("source.particle", "source.type")),
    ("source_kinematics", ("source.energy", "source.position", "source.direction")),
    ("physics", ("physics.physics_list",)),
    ("output", ("output.",)),
)


_PHASE_TITLES = {
    "geometry": {"en": "Geometry", "zh": "\u51e0\u4f55"},
    "materials": {"en": "Materials", "zh": "\u6750\u6599"},
    "source": {"en": "Source", "zh": "\u6e90\u9879"},
    "physics": {"en": "Physics", "zh": "\u7269\u7406\u8fc7\u7a0b"},
    "output": {"en": "Output", "zh": "\u8f93\u51fa"},
    "finalize": {"en": "Finalize", "zh": "\u6536\u5c3e"},
    "geometry_core": {"en": "Geometry Structure", "zh": "\u51e0\u4f55\u7ed3\u6784"},
    "geometry_params": {"en": "Geometry Parameters", "zh": "\u51e0\u4f55\u53c2\u6570"},
    "source_core": {"en": "Source Core", "zh": "\u6e90\u9879\u57fa\u7840"},
    "source_kinematics": {"en": "Source Kinematics", "zh": "\u6e90\u9879\u8fd0\u52a8\u5b66"},
    "complete": {"en": "Complete", "zh": "\u5df2\u5b8c\u6210"},
}


def _lang_key(lang: str) -> str:
    return "zh" if lang == "zh" else "en"


def phase_title(phase: str, lang: str) -> str:
    key = _lang_key(lang)
    return _PHASE_TITLES.get(phase, {}).get(key, phase)


def select_phase_fields(missing_fields: Sequence[str]) -> tuple[str, list[str]]:
    for phase, patterns in LEGACY_PHASE_ORDER:
        selected: list[str] = []
        for item in missing_fields:
            if any(field_matches_pattern(item, pattern) for pattern in patterns):
                selected.append(item)
        if selected:
            return phase, selected
    return "complete", []
