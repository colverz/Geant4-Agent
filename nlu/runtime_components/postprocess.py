from __future__ import annotations

import re
from typing import Dict, List, Tuple

MM_WORDS = ("mm", "\u6beb\u7c73")
CM_WORDS = ("cm", "\u5398\u7c73")
M_WORDS = ("\u7c73",)


INT_KEYS = {"nx", "ny", "n"}
CANONICAL_NUMERIC_KEYS = {
    "module_x",
    "module_y",
    "module_z",
    "nx",
    "ny",
    "pitch_x",
    "pitch_y",
    "n",
    "radius",
    "clearance",
    "parent_x",
    "parent_y",
    "parent_z",
    "child_x",
    "child_y",
    "child_z",
    "child_rmax",
    "child_hz",
    "rmax1",
    "rmax2",
    "x1",
    "x2",
    "y1",
    "y2",
    "z1",
    "z2",
    "z3",
    "r1",
    "r2",
    "r3",
    "tilt_x",
    "tilt_y",
    "bool_a_x",
    "bool_a_y",
    "bool_a_z",
    "bool_b_x",
    "bool_b_y",
    "bool_b_z",
    "stack_x",
    "stack_y",
    "t1",
    "t2",
    "t3",
    "stack_clearance",
    "nest_clearance",
    "inner_r",
    "th1",
    "th2",
    "th3",
    "hz",
    "tx",
    "ty",
    "tz",
    "rx",
    "ry",
    "rz",
}
DIRECT_PARAM_ALIASES = {
    "num_elements": "n",
    "element_count": "n",
    "elements": "n",
    "count": "n",
    "ring_count": "n",
    "module_count": "n",
    "element_radius": "radius",
    "ring_radius": "radius",
    "module_radius": "radius",
    "element_clearance": "clearance",
    "module_clearance": "clearance",
    "gap": "clearance",
    "spacing": "clearance",
    "dim_x": "module_x",
    "dim_y": "module_y",
    "dim_z": "module_z",
    "dimension_x": "module_x",
    "dimension_y": "module_y",
    "dimension_z": "module_z",
    "size_x": "module_x",
    "size_y": "module_y",
    "size_z": "module_z",
}
TRIPLET_PARAM_ALIASES = {
    "element_size",
    "module_size",
    "size",
    "dimensions",
    "box_size",
    "target_size",
    "module_dimensions",
    "element_dimensions",
}

BOOLEAN_TOKENS = (
    "boolean",
    "union",
    "subtraction",
    "intersection",
    "subtract",
    "intersect",
    "minus",
    "difference",
    "cut out",
    "cutout",
    "hole",
    "减去",
    "差集",
    "并",
    "并集",
    "合并",
    "挖空",
    "开孔",
    "打孔",
)

GRID_TOKENS = ("grid", "array", "matrix", "阵列", "二维阵列", "探测板", "网格", "排布")
NEST_TOKENS = (
    "nest",
    "nested",
    "inside",
    "contains",
    "inner",
    "outer",
    "parent",
    "child",
    "嵌套",
    "内嵌",
    "外盒",
    "盒子里",
    "外盒体",
    "内盒体",
    "包住",
    "包裹",
)
SHELL_TOKENS = ("shell", "concentric", "coaxial", "壳", "同心", "屏蔽层", "多层壳", "外壳")


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


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
    sep = r"(?:x|X|\*|by|×)"
    m = re.search(
        rf"(\d*\.?\d+\s*{unit})\s*{sep}\s*"
        rf"(\d*\.?\d+\s*{unit})\s*{sep}\s*"
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
        m2 = re.search(
            rf"(\d*\.?\d+)\s*{sep}\s*(\d*\.?\d+)\s*{sep}\s*(\d*\.?\d+)\s*({unit})",
            text,
            flags=re.IGNORECASE,
        )
        if not m2:
            return None
        suffix = m2.group(4)
        x = _parse_value_with_unit(f"{m2.group(1)} {suffix}")
        y = _parse_value_with_unit(f"{m2.group(2)} {suffix}")
        z = _parse_value_with_unit(f"{m2.group(3)} {suffix}")
        if x is None or y is None or z is None:
            return None
    return x, y, z


def _all_triplet_matches(text: str) -> List[Tuple[int, int, Tuple[float, float, float]]]:
    unit = r"(?:mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)"
    sep = r"(?:x|X|\*|by|×)"
    pattern = re.compile(
        rf"(\d*\.?\d+\s*{unit})\s*{sep}\s*"
        rf"(\d*\.?\d+\s*{unit})\s*{sep}\s*"
        rf"(\d*\.?\d+\s*{unit})",
        flags=re.IGNORECASE,
    )
    out: List[Tuple[int, int, Tuple[float, float, float]]] = []
    for m in pattern.finditer(text):
        x = _parse_value_with_unit(m.group(1))
        y = _parse_value_with_unit(m.group(2))
        z = _parse_value_with_unit(m.group(3))
        if x is None or y is None or z is None:
            continue
        out.append((m.start(), m.end(), (x, y, z)))
    compact = re.compile(
        rf"(\d*\.?\d+)\s*{sep}\s*(\d*\.?\d+)\s*{sep}\s*(\d*\.?\d+)\s*({unit})",
        flags=re.IGNORECASE,
    )
    for m in compact.finditer(text):
        suffix = m.group(4)
        x = _parse_value_with_unit(f"{m.group(1)} {suffix}")
        y = _parse_value_with_unit(f"{m.group(2)} {suffix}")
        z = _parse_value_with_unit(f"{m.group(3)} {suffix}")
        if x is None or y is None or z is None:
            continue
        out.append((m.start(), m.end(), (x, y, z)))
    return out


def _all_triplets(text: str) -> List[Tuple[float, float, float]]:
    return [item[2] for item in _all_triplet_matches(text)]


def _ordered_boolean_triplets(text: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]] | None:
    matches = _all_triplet_matches(text)
    if len(matches) < 2:
        return None
    low = text.lower()
    from_index = low.find(" from ")
    if ("subtract" in low or "减去" in low) and from_index >= 0:
        before_from = [triplet for start, end, triplet in matches if end <= from_index]
        after_from = [triplet for start, end, triplet in matches if start >= from_index]
        if before_from and after_from:
            return after_from[0], before_from[0]
    if "hole" in low or "cut out" in low or "cutout" in low or "挖空" in low or "开孔" in low or "打孔" in low:
        return matches[0][2], matches[1][2]
    if "minus" in low or "difference" in low or "差集" in low:
        return matches[0][2], matches[1][2]
    return matches[0][2], matches[1][2]


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


def _apply_numeric_value(out: Dict[str, float], key: str, value: float, notes: List[str], reason: str) -> None:
    if key in out:
        return
    out[key] = int(round(value)) if key in INT_KEYS else float(value)
    notes.append(f"filled {key} from {reason}")


def _canonicalize_input_params(params: Dict[str, float], notes: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in params.items():
        key = _normalize_key(str(k))
        target = DIRECT_PARAM_ALIASES.get(key, key)
        if target in CANONICAL_NUMERIC_KEYS:
            try:
                value = float(v)
            except (TypeError, ValueError):
                continue
            _apply_numeric_value(out, target, value, notes, f"input alias {k}")
            continue

        if key in TRIPLET_PARAM_ALIASES and isinstance(v, str):
            triplet = _module_triplet(v)
            if triplet:
                _apply_numeric_value(out, "module_x", triplet[0], notes, f"input alias {k}")
                _apply_numeric_value(out, "module_y", triplet[1], notes, f"input alias {k}")
                _apply_numeric_value(out, "module_z", triplet[2], notes, f"input alias {k}")
    return out


def _apply_alias_key_values(text: str, out: Dict[str, float], notes: List[str]) -> None:
    clauses = re.split(r"[;\n]", text)
    for clause in clauses:
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_\- ]*)\s*[:=]\s*(.+?)\s*$", clause)
        if not m:
            continue
        raw_key = m.group(1)
        raw_value = m.group(2).strip()
        key = _normalize_key(raw_key)

        if key in TRIPLET_PARAM_ALIASES:
            triplet = _module_triplet(raw_value)
            if triplet:
                _apply_numeric_value(out, "module_x", triplet[0], notes, f"alias key {raw_key}")
                _apply_numeric_value(out, "module_y", triplet[1], notes, f"alias key {raw_key}")
                _apply_numeric_value(out, "module_z", triplet[2], notes, f"alias key {raw_key}")
            continue

        target = DIRECT_PARAM_ALIASES.get(key, key)
        if target not in CANONICAL_NUMERIC_KEYS:
            continue
        value = _parse_value_with_unit(raw_value)
        if value is None:
            continue
        _apply_numeric_value(out, target, value, notes, f"alias key {raw_key}")


def _is_boolean_context(text: str) -> bool:
    low = text.lower()
    return any(token in low for token in BOOLEAN_TOKENS)


def _is_grid_context(text: str) -> bool:
    low = text.lower()
    return any(token in low for token in GRID_TOKENS)


def _is_nest_context(text: str) -> bool:
    low = text.lower()
    return any(token in low for token in NEST_TOKENS)


def _is_shell_context(text: str) -> bool:
    low = text.lower()
    return any(token in low for token in SHELL_TOKENS)


def _fill_by_patterns(text: str, out: Dict[str, float], notes: List[str]) -> None:
    unit = r"(?:mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)?"
    num = r"[+\-]?\d*\.?\d+"
    for key, pattern in [
        ("module_x", rf"(?:module[_\s-]*x)\s*[:=]?\s*({num}\s*{unit})"),
        ("module_y", rf"(?:module[_\s-]*y)\s*[:=]?\s*({num}\s*{unit})"),
        ("module_z", rf"(?:module[_\s-]*z)\s*[:=]?\s*({num}\s*{unit})"),
        ("parent_x", rf"(?:parent[_\s-]*x)\s*[:=]?\s*({num}\s*{unit})"),
        ("parent_y", rf"(?:parent[_\s-]*y)\s*[:=]?\s*({num}\s*{unit})"),
        ("parent_z", rf"(?:parent[_\s-]*z)\s*[:=]?\s*({num}\s*{unit})"),
        ("child_x", rf"(?:child[_\s-]*x)\s*[:=]?\s*({num}\s*{unit})"),
        ("child_y", rf"(?:child[_\s-]*y)\s*[:=]?\s*({num}\s*{unit})"),
        ("child_z", rf"(?:child[_\s-]*z)\s*[:=]?\s*({num}\s*{unit})"),
        ("child_rmax", rf"(?:child[_\s-]*rmax|rmax)\s*[:=]?\s*({num}\s*{unit})"),
        ("child_hz", rf"(?:child[_\s-]*hz|hz)\s*[:=]?\s*({num}\s*{unit})"),
        ("rmax1", rf"\brmax1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("rmax2", rf"\brmax2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("x1", rf"\bx1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("x2", rf"\bx2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("y1", rf"\by1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("y2", rf"\by2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("z1", rf"\bz1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("z2", rf"\bz2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("z3", rf"\bz3\b\s*[:=]?\s*({num}\s*{unit})"),
        ("r1", rf"\br1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("r2", rf"\br2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("r3", rf"\br3\b\s*[:=]?\s*({num}\s*{unit})"),
        ("tilt_x", rf"(?:tilt[_\s-]*x)\s*[:=]?\s*({num})"),
        ("tilt_y", rf"(?:tilt[_\s-]*y)\s*[:=]?\s*({num})"),
        ("bool_a_x", rf"(?:bool[_\s-]*a[_\s-]*x)\s*[:=]?\s*({num}\s*{unit})"),
        ("bool_a_y", rf"(?:bool[_\s-]*a[_\s-]*y)\s*[:=]?\s*({num}\s*{unit})"),
        ("bool_a_z", rf"(?:bool[_\s-]*a[_\s-]*z)\s*[:=]?\s*({num}\s*{unit})"),
        ("bool_b_x", rf"(?:bool[_\s-]*b[_\s-]*x)\s*[:=]?\s*({num}\s*{unit})"),
        ("bool_b_y", rf"(?:bool[_\s-]*b[_\s-]*y)\s*[:=]?\s*({num}\s*{unit})"),
        ("bool_b_z", rf"(?:bool[_\s-]*b[_\s-]*z)\s*[:=]?\s*({num}\s*{unit})"),
        ("stack_x", rf"(?:stack[_\s-]*x)\s*[:=]?\s*({num}\s*{unit})"),
        ("stack_y", rf"(?:stack[_\s-]*y)\s*[:=]?\s*({num}\s*{unit})"),
        ("t1", rf"\bt1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("t2", rf"\bt2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("t3", rf"\bt3\b\s*[:=]?\s*({num}\s*{unit})"),
        ("stack_clearance", rf"(?:stack[_\s-]*clearance)\s*[:=]?\s*({num}\s*{unit})"),
        ("nest_clearance", rf"(?:nest[_\s-]*clearance)\s*[:=]?\s*({num}\s*{unit})"),
        ("inner_r", rf"(?:inner[_\s-]*r)\s*[:=]?\s*({num}\s*{unit})"),
        ("th1", rf"\bth1\b\s*[:=]?\s*({num}\s*{unit})"),
        ("th2", rf"\bth2\b\s*[:=]?\s*({num}\s*{unit})"),
        ("th3", rf"\bth3\b\s*[:=]?\s*({num}\s*{unit})"),
        ("tx", rf"\btx\b\s*[:=]?\s*({num}\s*{unit})"),
        ("ty", rf"\bty\b\s*[:=]?\s*({num}\s*{unit})"),
        ("tz", rf"\btz\b\s*[:=]?\s*({num}\s*{unit})"),
        ("rx", rf"\brx\b\s*[:=]?\s*({num})"),
        ("ry", rf"\bry\b\s*[:=]?\s*({num})"),
        ("rz", rf"\brz\b\s*[:=]?\s*({num})"),
    ]:
        if key in out:
            continue
        val = _first_match(pattern, text)
        if val is None:
            continue
        out[key] = float(val)
        notes.append(f"filled {key} from key/value")

    if "n" not in out:
        patterns = [
            r"\b(\d+)\s+modules?\b",
            r"(\d+)\s*\u4e2a?\s*(?:\u6a21\u5757|\u63a2\u6d4b\u5668|\u5355\u5143)",
            r"(\d+)\s*(?:\u6a21\u5757|\u63a2\u6d4b\u5668|\u5355\u5143).{0,8}?(?:\u6210\u73af|\u56f4\u6210\u4e00\u5708|\u73af\u7ed5\u6210\u73af|\u73af\u7ed5)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                out["n"] = int(m.group(1))
                notes.append("filled n from modules count")
                break


def _fill_stack_params(text: str, out: Dict[str, float], notes: List[str]) -> None:
    text_l = text.lower()
    stack_context = any(
        token in text_l
        for token in (
            "stack",
            "layer",
            "layers",
            "along z",
            "stacked",
            "堆叠",
            "沿 z 方向",
            "沿z方向",
            "层厚",
            "层间",
        )
    )
    if not stack_context:
        return

    unit = r"(?:mm|cm|m|\u6beb\u7c73|\u5398\u7c73|\u7c73)?"
    num = r"[+\-]?\d*\.?\d+"
    sep = r"(?:x|X|\*|by)"

    if "stack_x" not in out or "stack_y" not in out:
        footprint_patterns = [
            rf"(?:footprint|stack(?:ed)?\s*layers?[^,;]*?footprint)\s*[:=]?\s*({num}\s*{unit})\s*{sep}\s*({num}\s*{unit})",
            rf"(?:\u5e95\u9762|占地|层面尺寸)\s*[:=]?\s*({num}\s*{unit})\s*{sep}\s*({num}\s*{unit})",
        ]
        for pattern in footprint_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            x_val = _parse_value_with_unit(m.group(1))
            y_val = _parse_value_with_unit(m.group(2))
            if x_val is None or y_val is None:
                continue
            if "stack_x" not in out:
                out["stack_x"] = float(x_val)
                notes.append("filled stack_x from stack footprint")
            if "stack_y" not in out:
                out["stack_y"] = float(y_val)
                notes.append("filled stack_y from stack footprint")
            break

    if "stack_x" not in out:
        val = _first_match(rf"(?:^|[\s,;(])x\s*[:=]?\s*({num}\s*{unit})", text)
        if val is not None:
            out["stack_x"] = float(val)
            notes.append("filled stack_x from stack x")
    if "stack_y" not in out:
        val = _first_match(rf"(?:^|[\s,;(])y\s*[:=]?\s*({num}\s*{unit})", text)
        if val is not None:
            out["stack_y"] = float(val)
            notes.append("filled stack_y from stack y")

    if "stack_clearance" not in out:
        val = _first_match(rf"(?:stack[_\s-]*)?clearance\s*[:=]?\s*({num}\s*{unit})", text)
        if val is not None:
            out["stack_clearance"] = float(val)
            notes.append("filled stack_clearance from stack clearance")

    container_clearance = _first_match(
        rf"(?:container|outer(?:\s*box)?|nest(?:ed)?)\s*clearance\s*[:=]?\s*({num}\s*{unit})",
        text,
    )
    if container_clearance is not None:
        out["nest_clearance"] = float(container_clearance)
        notes.append("filled nest_clearance from container clearance")
    elif "nest_clearance" not in out and "stack_clearance" in out:
        out["nest_clearance"] = float(out["stack_clearance"])
        notes.append("filled nest_clearance from stack_clearance")

    if all(k in out for k in ("t1", "t2", "t3")):
        pass
    else:
        thickness_patterns = [
            rf"thickness(?:es)?\s*[:=]?\s*({num}\s*{unit})\s*(?:and|,|/|;)\s*({num}\s*{unit})\s*(?:and|,|/|;)\s*({num}\s*{unit})",
            rf"layers?\s*[:=]?\s*({num}\s*{unit})\s*(?:and|,|/|;)\s*({num}\s*{unit})\s*(?:and|,|/|;)\s*({num}\s*{unit})",
            rf"thickness(?:es)?\s*[:=]?\s*({num})[\s,;/]+({num})[\s,;/]+({num})\s*({unit})",
            rf"layers?\s*[:=]?\s*({num})[\s,;/]+({num})[\s,;/]+({num})\s*({unit})",
            rf"(?:层厚|厚度)\s*[:：=]?\s*({num}\s*{unit})\s*(?:和|及|,|/|;)\s*({num}\s*{unit})\s*(?:和|及|,|/|;)\s*({num}\s*{unit})",
            rf"(?:三层厚度|三层)\s*[:：=]?\s*({num})[\s,;/]+({num})[\s,;/]+({num})\s*({unit})",
        ]
        for pattern in thickness_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            if len(m.groups()) == 3:
                values = (
                    _parse_value_with_unit(m.group(1)),
                    _parse_value_with_unit(m.group(2)),
                    _parse_value_with_unit(m.group(3)),
                )
            else:
                suffix = m.group(4).strip() or "mm"
                values = (
                    _parse_value_with_unit(f"{m.group(1)} {suffix}"),
                    _parse_value_with_unit(f"{m.group(2)} {suffix}"),
                    _parse_value_with_unit(f"{m.group(3)} {suffix}"),
                )
            if any(v is None for v in values):
                continue
            if "t1" not in out:
                out["t1"] = float(values[0])
                notes.append("filled t1 from stack thicknesses")
            if "t2" not in out:
                out["t2"] = float(values[1])
                notes.append("filled t2 from stack thicknesses")
            if "t3" not in out:
                out["t3"] = float(values[2])
                notes.append("filled t3 from stack thicknesses")
            break

    if not all(k in out for k in ("parent_x", "parent_y", "parent_z")):
        outer_box_patterns = [
            rf"(?:outer\s*box|container|parent)\s*[:=]?\s*({num}\s*{unit})\s*{sep}\s*({num}\s*{unit})\s*{sep}\s*({num}\s*{unit})",
            rf"(?:\u5916\u76d2|容器)\s*[:=]?\s*({num}\s*{unit})\s*{sep}\s*({num}\s*{unit})\s*{sep}\s*({num}\s*{unit})",
        ]
        for pattern in outer_box_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            values = [_parse_value_with_unit(m.group(idx)) for idx in range(1, 4)]
            if any(v is None for v in values):
                continue
            if "parent_x" not in out:
                out["parent_x"] = float(values[0])
                notes.append("filled parent_x from outer box")
            if "parent_y" not in out:
                out["parent_y"] = float(values[1])
                notes.append("filled parent_y from outer box")
            if "parent_z" not in out:
                out["parent_z"] = float(values[2])
                notes.append("filled parent_z from outer box")
            break


def _fill_grid_params(text: str, out: Dict[str, float], notes: List[str]) -> None:
    text_l = text.lower()
    has_pitch_hint = "pitch" in text_l or "pitch_x" in text_l or "pitch_y" in text_l
    if not _is_grid_context(text) and not has_pitch_hint:
        return
    if "nx" not in out or "ny" not in out:
        patterns = [
            r"\b(\d+)\s*(?:x|X|by|×)\s*(\d+)\b(?:[^;\n]{0,24})\b(?:grid|array|matrix|modules?)\b",
            r"(?:grid|array|matrix)\s*(?:of)?\s*(\d+)\s*(?:x|X|by|×)\s*(\d+)",
            r"\b(\d+)\s*(?:x|X|by|×)\s*(\d+)\s+modules?\b",
            r"(\d+)\s*(?:x|X|by|×)\s*(\d+)\s*(?:网格|阵列|二维阵列)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if "nx" not in out:
                out["nx"] = int(match.group(1))
                notes.append("filled nx from grid count")
            if "ny" not in out:
                out["ny"] = int(match.group(2))
                notes.append("filled ny from grid count")
            break
        if "nx" not in out or "ny" not in out:
            loose_match = re.search(r"\b(\d+)\s*(?:x|X|by|×)\s*(\d+)\b", text, flags=re.IGNORECASE)
            if loose_match and has_pitch_hint:
                if "nx" not in out:
                    out["nx"] = int(loose_match.group(1))
                    notes.append("filled nx from loose grid count")
                if "ny" not in out:
                    out["ny"] = int(loose_match.group(2))
                    notes.append("filled ny from loose grid count")

    if "pitch_x" not in out or "pitch_y" not in out:
        unit = r"(?:mm|cm|m|毫米|厘米|米)"
        match = re.search(
            rf"pitch\s*[:=]?\s*([+\-]?\d*\.?\d+\s*{unit})\s*(?:x|X|by|×)\s*([+\-]?\d*\.?\d+\s*{unit})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            pitch_x = _parse_value_with_unit(match.group(1))
            pitch_y = _parse_value_with_unit(match.group(2))
            if pitch_x is not None and "pitch_x" not in out:
                out["pitch_x"] = float(pitch_x)
                notes.append("filled pitch_x from shared pitch")
            if pitch_y is not None and "pitch_y" not in out:
                out["pitch_y"] = float(pitch_y)
                notes.append("filled pitch_y from shared pitch")
        else:
            alt_match = re.search(
                rf"pitch(?:es)?\s*[:=]?\s*([+\-]?\d*\.?\d+\s*{unit})\s*(?:and|,|/|;)\s*([+\-]?\d*\.?\d+\s*{unit})",
                text,
                flags=re.IGNORECASE,
            )
            if alt_match:
                pitch_x = _parse_value_with_unit(alt_match.group(1))
                pitch_y = _parse_value_with_unit(alt_match.group(2))
                if pitch_x is not None and "pitch_x" not in out:
                    out["pitch_x"] = float(pitch_x)
                    notes.append("filled pitch_x from paired pitch values")
                if pitch_y is not None and "pitch_y" not in out:
                    out["pitch_y"] = float(pitch_y)
                    notes.append("filled pitch_y from paired pitch values")
            else:
                axis_match = re.search(
                    rf"(?:x\s*方向间距|pitch_x)\s*[:=]?\s*([+\-]?\d*\.?\d+\s*{unit}).*?(?:y\s*方向间距|pitch_y)\s*[:=]?\s*([+\-]?\d*\.?\d+\s*{unit})",
                    text,
                    flags=re.IGNORECASE,
                )
                if axis_match:
                    pitch_x = _parse_value_with_unit(axis_match.group(1))
                    pitch_y = _parse_value_with_unit(axis_match.group(2))
                    if pitch_x is not None and "pitch_x" not in out:
                        out["pitch_x"] = float(pitch_x)
                        notes.append("filled pitch_x from axis pitch values")
                    if pitch_y is not None and "pitch_y" not in out:
                        out["pitch_y"] = float(pitch_y)
                        notes.append("filled pitch_y from axis pitch values")


def _fill_nest_shell_params(text: str, out: Dict[str, float], notes: List[str]) -> None:
    unit = r"(?:mm|cm|m|毫米|厘米|米)?"
    num = r"[+\-]?\d*\.?\d+"
    nest_context = _is_nest_context(text)
    shell_context = _is_shell_context(text)

    if nest_context:
        triplets = _all_triplets(text)
        if len(triplets) >= 2:
            parent_triplet, child_triplet = triplets[0], triplets[1]
            for key, value in zip(("parent_x", "parent_y", "parent_z"), parent_triplet):
                if key not in out:
                    out[key] = float(value)
                    notes.append(f"filled {key} from nested parent triplet")
            for key, value in zip(("child_x", "child_y", "child_z"), child_triplet):
                if key not in out:
                    out[key] = float(value)
                    notes.append(f"filled {key} from nested child triplet")

    if nest_context and all(k in out for k in ("module_x", "module_y", "module_z")):
        if "parent_x" not in out:
            out["parent_x"] = float(out["module_x"])
            notes.append("filled parent_x from outer box size")
        if "parent_y" not in out:
            out["parent_y"] = float(out["module_y"])
            notes.append("filled parent_y from outer box size")
        if "parent_z" not in out:
            out["parent_z"] = float(out["module_z"])
            notes.append("filled parent_z from outer box size")

    if "child_rmax" not in out:
        patterns = [
            rf"(?:nested\s*core|core|child|cylinder)[^\n]*?(?:radius)\s*[:=]?\s*({num}\s*{unit})",
            rf"(?:内核|子体|子圆柱)[^\n]*?(?:半径)\s*[:=]?\s*({num}\s*{unit})",
        ]
        for pattern in patterns:
            value = _first_match(pattern, text)
            if value is not None:
                out["child_rmax"] = float(value)
                notes.append("filled child_rmax from nested radius")
                break

    if "child_hz" not in out:
        patterns = [
            rf"(?:nested\s*core|core|child|cylinder)[^\n]*?(?:half[_\s-]*length|half[_\s-]*len|half[_\s-]*l)\s*[:=]?\s*({num}\s*{unit})",
            rf"(?:内核|子体|子圆柱)[^\n]*?(?:半长|半长度)\s*[:=]?\s*({num}\s*{unit})",
        ]
        for pattern in patterns:
            value = _first_match(pattern, text)
            if value is not None:
                out["child_hz"] = float(value)
                notes.append("filled child_hz from nested half length")
                break

    if shell_context and "inner_r" not in out:
        value = _first_match(
            rf"(?:inner[_\s-]*radius|inner[_\s-]*r|inner radius|内半径|内径)\s*[:=]?\s*({num}\s*{unit})",
            text,
        )
        if value is not None:
            out["inner_r"] = float(value)
            notes.append("filled inner_r from shell inner radius")

    if shell_context and "hz" not in out:
        value = _first_match(
            rf"(?:half[_\s-]*length|half[_\s-]*len|half[_\s-]*l|half length|半长|半长度)\s*[:=]?\s*({num}\s*{unit})",
            text,
        )
        if value is not None:
            out["hz"] = float(value)
            notes.append("filled hz from shell half length")

    if shell_context and not any(k in out for k in ("th1", "th2", "th3")):
        thickness_patterns = [
            rf"thickness(?:es)?\s*[:=]?\s*({num})\s*{unit}\s*(?:and|,|/|;)\s*({num})(?:\s*{unit})?(?:\s*(?:and|,|/|;)\s*({num})(?:\s*{unit})?)?",
            rf"厚度\s*[:=]?\s*({num})\s*{unit}\s*(?:和|,|/|;)\s*({num})(?:\s*{unit})?(?:\s*(?:和|,|/|;)\s*({num})(?:\s*{unit})?)?",
        ]
        for pattern in thickness_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            values = [match.group(1), match.group(2), match.group(3)]
            for idx, raw in enumerate(values, start=1):
                if not raw:
                    continue
                key = f"th{idx}"
                if key in out:
                    continue
                parsed = _parse_value_with_unit(f"{raw} mm")
                if parsed is None:
                    continue
                out[key] = float(parsed)
                notes.append(f"filled {key} from shell thicknesses")
            break


def merge_params(text: str, params: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    notes: List[str] = []
    out = _canonicalize_input_params(params, notes)
    _apply_alias_key_values(text, out, notes)
    _fill_by_patterns(text, out, notes)
    _fill_grid_params(text, out, notes)
    _fill_stack_params(text, out, notes)
    _fill_nest_shell_params(text, out, notes)

    boolean_context = _is_boolean_context(text)
    if not boolean_context:
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
        _fill_nest_shell_params(text, out, notes)

    if boolean_context or ("bool_a_x" in out or "bool_b_x" in out):
        ordered = _ordered_boolean_triplets(text)
        if ordered is not None:
            a, b = ordered
            if "bool_a_x" not in out:
                out["bool_a_x"] = a[0]
                out["bool_a_y"] = a[1]
                out["bool_a_z"] = a[2]
                notes.append("filled bool_a_* from boolean triplet order")
            if "bool_b_x" not in out:
                out["bool_b_x"] = b[0]
                out["bool_b_y"] = b[1]
                out["bool_b_z"] = b[2]
                notes.append("filled bool_b_* from boolean triplet order")
    elif not any(
        key in out
        for key in ("nx", "ny", "n", "radius", "parent_x", "parent_y", "parent_z", "stack_x", "stack_y", "inner_r")
    ):
        ordered = _ordered_boolean_triplets(text)
        low = text.lower()
        dual_box_phrases = low.count("set geometry to box") >= 2 or low.count("box with size") >= 2
        if ordered is not None and dual_box_phrases:
            a, b = ordered
            out["bool_a_x"] = a[0]
            out["bool_a_y"] = a[1]
            out["bool_a_z"] = a[2]
            out["bool_b_x"] = b[0]
            out["bool_b_y"] = b[1]
            out["bool_b_z"] = b[2]
            notes.append("filled bool_a_*/bool_b_* from dual-box heuristic")

    for key, pattern in [
        ("radius", r"(?:\bradius\b|半径)\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("radius", r"\bR(?!\d)\b\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("clearance", r"(?:clearance|间隙|间隔|嵌套间隙)\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
        ("clearance", r"(?:gap|spacing)\s*[:=]?\s*([\d\.]+\s*(?:mm|cm|m)?)"),
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
        if k in INT_KEYS:
            out[k] = int(round(v))
        else:
            out[k] = float(v)

    return out, notes
