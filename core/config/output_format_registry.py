from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


_DATA_PATH = Path("knowledge/data/output_formats.json")
_FALLBACK_OFFICIAL = ("csv", "hdf5", "root", "xml")
_FALLBACK_PROJECT = ("json",)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, object]:
    if not _DATA_PATH.exists():
        return {
            "official_items": list(_FALLBACK_OFFICIAL),
            "project_extensions": list(_FALLBACK_PROJECT),
            "items": sorted(set(_FALLBACK_OFFICIAL + _FALLBACK_PROJECT)),
            "aliases": {"h5": "hdf5"},
        }
    data = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    official = [str(x).strip().lower() for x in data.get("official_items", []) if str(x).strip()]
    project = [str(x).strip().lower() for x in data.get("project_extensions", []) if str(x).strip()]
    items = [str(x).strip().lower() for x in data.get("items", []) if str(x).strip()]
    if not official:
        official = list(_FALLBACK_OFFICIAL)
    if not project:
        project = [x for x in items if x not in official] or list(_FALLBACK_PROJECT)
    if not items:
        items = sorted(set(official + project))
    aliases = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in dict(data.get("aliases", {})).items()
        if str(k).strip() and str(v).strip()
    }
    return {
        "official_items": official,
        "project_extensions": project,
        "items": items,
        "aliases": aliases,
    }


def official_output_formats() -> tuple[str, ...]:
    return tuple(_catalog()["official_items"])


def project_output_extensions() -> tuple[str, ...]:
    return tuple(_catalog()["project_extensions"])


def accepted_output_formats() -> tuple[str, ...]:
    return tuple(_catalog()["items"])


def canonical_output_format(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = dict(_catalog()["aliases"])
    text = aliases.get(text, text)
    if text in accepted_output_formats():
        return text
    return None

