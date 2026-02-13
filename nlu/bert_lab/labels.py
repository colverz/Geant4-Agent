from __future__ import annotations

PARAM_KEYS = [
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
    "child_rmax",
    "child_hz",
    "inner_r",
    "th1",
    "th2",
    "th3",
    "hz",
    "stack_x",
    "stack_y",
    "t1",
    "t2",
    "t3",
    "stack_clearance",
    "nest_clearance",
]

ENTITY_KEYS = [
    "material",
    "particle",
    "physics_list",
    "source_type",
    "output_format",
]

STRUCTURE_LABELS = [
    "nest",
    "grid",
    "ring",
    "stack",
    "shell",
    "single_box",
    "single_tubs",
    "unknown",
]

TOKEN_LABELS = ["O"] + [f"B-{k}" for k in (PARAM_KEYS + ENTITY_KEYS)] + [
    f"I-{k}" for k in (PARAM_KEYS + ENTITY_KEYS)
]
