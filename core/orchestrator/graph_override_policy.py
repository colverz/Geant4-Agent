from __future__ import annotations

import re


GRAPH_STRUCTURES = {"ring", "grid", "nest", "stack", "shell", "boolean"}


def has_explicit_graph_cue(text: str, structure: str | None) -> bool:
    low = str(text or "").lower()
    key = str(structure or "").strip().lower()
    if key == "boolean":
        if re.search(
            r"\b(?:box|cube|cuboid|cylinder|sphere|solid|target)\b.{0,80}\bminus\b.{0,80}\b(?:box|cube|cuboid|cylinder|sphere|solid|target)\b",
            low,
        ):
            return True
        return bool(
            re.search(
                r"\b(?:boolean|union|intersect|intersection|subtract|subtraction|difference|cut out|hole)\b",
                low,
            )
        )
    if key == "ring":
        return any(token in low for token in ("ring", "annulus", "circular array", "\u73af", "\u5706\u73af"))
    if key == "grid":
        return any(token in low for token in ("grid", "array", "matrix", "\u9635\u5217", "\u7f51\u683c"))
    if key == "nest":
        return any(token in low for token in ("inside", "nested", "embedded", "within", "\u5185\u5d4c", "\u5185\u90e8"))
    if key == "stack":
        return any(token in low for token in ("stack", "layer", "layered", "\u5806\u53e0", "\u5c42"))
    if key == "shell":
        return any(token in low for token in ("shell", "coating", "layer around", "\u58f3", "\u5305\u5c42"))
    return False


def should_prefer_extracted_graph(
    *,
    text: str,
    extracted_structure: str | None,
    slot_structure: str | None,
    slot_geometry_ready: bool,
) -> bool:
    return (
        extracted_structure in GRAPH_STRUCTURES
        and slot_structure not in GRAPH_STRUCTURES
        and (not slot_geometry_ready or has_explicit_graph_cue(text, extracted_structure))
    )
