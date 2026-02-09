from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _parse_value_with_unit(text: str) -> float | None:
    text = text.strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", text)
    if not m:
        return None
    value = float(m.group(1))
    if "cm" in text:
        value *= 10.0
    return value


def _first_match(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    return _parse_value_with_unit(m.group(1))


def _module_triplet(text: str) -> Tuple[float, float, float] | None:
    m = re.search(
        r"(\d*\.?\d+)\s*(?:x|X)\s*(\d*\.?\d+)\s*(?:x|X)\s*(\d*\.?\d+)\s*(mm|cm)?",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    x = float(m.group(1))
    y = float(m.group(2))
    z = float(m.group(3))
    unit = m.group(4)
    if unit and unit.lower() == "cm":
        x *= 10.0
        y *= 10.0
        z *= 10.0
    return x, y, z


def merge_params(text: str, params: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    out = dict(params)
    notes: List[str] = []

    triplet = _module_triplet(text)
    if triplet and not ("module_x" in out and "module_y" in out and "module_z" in out):
        out.setdefault("module_x", triplet[0])
        out.setdefault("module_y", triplet[1])
        out.setdefault("module_z", triplet[2])
        notes.append("filled module_x/module_y/module_z from triplet")

    for key, pattern in [
        ("radius", r"radius\s*[:=]?\s*([\d\.]+\s*(?:mm|cm)?)"),
        ("radius", r"\bR\s*[:=]?\s*([\d\.]+\s*(?:mm|cm)?)"),
        ("clearance", r"clearance\s*[:=]?\s*([\d\.]+\s*(?:mm|cm)?)"),
        ("clearance", r"gap\s*[:=]?\s*([\d\.]+\s*(?:mm|cm)?)"),
        ("pitch_x", r"pitch[_\s-]*x\s*[:=]?\s*([\d\.]+\s*(?:mm|cm)?)"),
        ("pitch_y", r"pitch[_\s-]*y\s*[:=]?\s*([\d\.]+\s*(?:mm|cm)?)"),
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

    # basic cleanup
    for k, v in list(out.items()):
        if k in {"nx", "ny", "n"}:
            out[k] = int(round(v))
        else:
            out[k] = float(v)
        if out[k] < 0:
            out[k] = abs(out[k])
            notes.append(f"clamped {k} to abs")

    return out, notes
