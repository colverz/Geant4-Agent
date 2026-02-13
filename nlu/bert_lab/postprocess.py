from __future__ import annotations

import re
from typing import Dict, List, Tuple

MM_WORDS = ("mm", "\u6beb\u7c73")
CM_WORDS = ("cm", "\u5398\u7c73")
M_WORDS = ("\u7c73",)


def _parse_value_with_unit(text: str) -> float | None:
    text_l = text.strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", text_l)
    if not m:
        return None
    value = float(m.group(1))

    if any(w in text_l for w in MM_WORDS):
        return value
    if any(w in text_l for w in CM_WORDS):
        return value * 10.0
    if any(w in text_l for w in M_WORDS):
        return value * 1000.0
    if re.search(r"\d(?:\.\d+)?\s*m\b", text_l) and ("mm" not in text_l and "cm" not in text_l):
        return value * 1000.0
    return value


def _first_match(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    return _parse_value_with_unit(m.group(1))


def _module_triplet(text: str) -> Tuple[float, float, float] | None:
    unit = r"(?:mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)"
    m = re.search(
        rf"(\d*\.?\d+\s*{unit})\s*(?:x|X|\*)\s*"
        rf"(\d*\.?\d+\s*{unit})\s*(?:x|X|\*)\s*"
        rf"(\d*\.?\d+\s*{unit})",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    x = _parse_value_with_unit(m.group(1))
    y = _parse_value_with_unit(m.group(2))
    z = _parse_value_with_unit(m.group(3))
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _cube_edge(text: str) -> float | None:
    unit = r"(?:mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)"
    patterns = [
        rf"(?:edge|side)\s*[:=]?\s*([\d\.]+\s*{unit})",
        rf"\u8fb9\u957f\s*[:=]?\s*([\d\.]+\s*{unit})",
        rf"([\d\.]+\s*{unit})\s*(?:cube|box)",
        rf"([\d\.]+\s*{unit})\s*\u7acb\u65b9\u4f53",
    ]
    for pattern in patterns:
        val = _first_match(pattern, text)
        if val is not None:
            return val
    return None


def _fill_by_patterns(text: str, out: Dict[str, float], notes: List[str]) -> None:
    unit = r"(?:mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)?"
    for key, pattern in [
        ("module_x", rf"(?:module[_\s-]*x)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("module_y", rf"(?:module[_\s-]*y)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("module_z", rf"(?:module[_\s-]*z)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("parent_x", rf"(?:parent[_\s-]*x)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("parent_y", rf"(?:parent[_\s-]*y)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("parent_z", rf"(?:parent[_\s-]*z)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("child_rmax", rf"(?:child[_\s-]*rmax|rmax)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("child_hz", rf"(?:child[_\s-]*hz|hz)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("stack_x", rf"(?:stack[_\s-]*x)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("stack_y", rf"(?:stack[_\s-]*y)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("t1", rf"\bt1\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("t2", rf"\bt2\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("t3", rf"\bt3\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("stack_clearance", rf"(?:stack[_\s-]*clearance)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("nest_clearance", rf"(?:nest[_\s-]*clearance)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("inner_r", rf"(?:inner[_\s-]*r)\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("th1", rf"\bth1\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("th2", rf"\bth2\s*[:=]\s*([\d\.]+\s*{unit})"),
        ("th3", rf"\bth3\s*[:=]\s*([\d\.]+\s*{unit})"),
    ]:
        if key in out:
            continue
        val = _first_match(pattern, text)
        if val is None:
            continue
        out[key] = float(val)
        notes.append(f"filled {key} from key/value")

    if "n" not in out:
        m = re.search(r"\b(\d+)\s+modules?\b", text, flags=re.IGNORECASE)
        if m:
            out["n"] = int(m.group(1))
            notes.append("filled n from modules count")


def merge_params(text: str, params: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    out = dict(params)
    notes: List[str] = []
    _fill_by_patterns(text, out, notes)

    triplet = _module_triplet(text)
    if triplet and not ("module_x" in out and "module_y" in out and "module_z" in out):
        out.setdefault("module_x", triplet[0])
        out.setdefault("module_y", triplet[1])
        out.setdefault("module_z", triplet[2])
        notes.append("filled module_x/module_y/module_z from triplet")
    elif not ("module_x" in out and "module_y" in out and "module_z" in out):
        edge = _cube_edge(text)
        if edge is not None:
            out.setdefault("module_x", edge)
            out.setdefault("module_y", edge)
            out.setdefault("module_z", edge)
            notes.append("filled module_x/module_y/module_z from cube edge")

    for key, pattern in [
        ("radius", r"radius\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("radius", r"\bR\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("clearance", r"clearance\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("clearance", r"gap\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("pitch_x", r"pitch[_\s-]*x\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("pitch_y", r"pitch[_\s-]*y\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("nx", r"\bnx\s*[:=]?\s*(\d+)"),
        ("ny", r"\bny\s*[:=]?\s*(\d+)"),
        ("n", r"\bcount\s*[:=]?\s*(\d+)"),
        ("n", r"\bn\s*[:=]?\s*(\d+)"),
    ]:
        if key in out:
            continue
        val = _first_match(pattern, text)
        if val is None:
            continue
        out[key] = int(round(val)) if key in {"nx", "ny", "n"} else float(val)
        notes.append(f"filled {key} from text")

    for k, v in list(out.items()):
        if k in {"nx", "ny", "n"}:
            out[k] = int(round(v))
        else:
            out[k] = float(v)
        if out[k] < 0:
            out[k] = abs(out[k])
            notes.append(f"clamped {k} to abs")

    return out, notes
