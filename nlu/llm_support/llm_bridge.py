from __future__ import annotations

from typing import Dict, Iterable, List
import json

from core.config.prompt_profiles import PromptTask, build_prompt


PARAM_DESCRIPTIONS: Dict[str, str] = {
    "module_x": "module size in x (mm)",
    "module_y": "module size in y (mm)",
    "module_z": "module size in z (mm)",
    "nx": "number of modules along x (integer)",
    "ny": "number of modules along y (integer)",
    "pitch_x": "grid pitch in x (mm)",
    "pitch_y": "grid pitch in y (mm)",
    "n": "number of modules on ring (integer)",
    "radius": "ring radius (mm)",
    "clearance": "clearance/gap (mm)",
    "parent_x": "parent box size x (mm)",
    "parent_y": "parent box size y (mm)",
    "parent_z": "parent box size z (mm)",
    "child_rmax": "child cylinder outer radius (mm)",
    "child_hz": "child cylinder half-length (mm)",
    "rmax1": "cone max radius at -z side (mm)",
    "rmax2": "cone max radius at +z side (mm)",
    "x1": "trd half-length in x at -z side (mm)",
    "x2": "trd half-length in x at +z side (mm)",
    "y1": "trd half-length in y at -z side (mm)",
    "y2": "trd half-length in y at +z side (mm)",
    "z1": "polycone z plane 1 (mm)",
    "z2": "polycone z plane 2 (mm)",
    "z3": "polycone z plane 3 (mm)",
    "r1": "polycone radius at z1 (mm)",
    "r2": "polycone radius at z2 (mm)",
    "r3": "polycone radius at z3 (mm)",
    "tilt_x": "cuttubs cut tilt in x (deg proxy)",
    "tilt_y": "cuttubs cut tilt in y (deg proxy)",
    "bool_a_x": "boolean left box x (mm)",
    "bool_a_y": "boolean left box y (mm)",
    "bool_a_z": "boolean left box z (mm)",
    "bool_b_x": "boolean right box x (mm)",
    "bool_b_y": "boolean right box y (mm)",
    "bool_b_z": "boolean right box z (mm)",
    "inner_r": "inner radius (mm)",
    "th1": "shell thickness 1 (mm)",
    "th2": "shell thickness 2 (mm)",
    "th3": "shell thickness 3 (mm)",
    "hz": "shell half-length (mm)",
    "stack_x": "stack footprint x (mm)",
    "stack_y": "stack footprint y (mm)",
    "t1": "layer thickness 1 (mm)",
    "t2": "layer thickness 2 (mm)",
    "t3": "layer thickness 3 (mm)",
    "stack_clearance": "clearance between stacked layers (mm)",
    "nest_clearance": "clearance for nesting (mm)",
    "tx": "translation x (mm)",
    "ty": "translation y (mm)",
    "tz": "translation z (mm)",
    "rx": "rotation around x (deg)",
    "ry": "rotation around y (deg)",
    "rz": "rotation around z (deg)",
}


def describe_params(keys: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for key in keys:
        desc = PARAM_DESCRIPTIONS.get(key, key)
        lines.append(f"- {key}: {desc}")
    return lines


def build_missing_params_schema(structure: str, missing: Iterable[str]) -> Dict[str, object]:
    keys = list(missing)
    properties: Dict[str, object] = {}
    required: List[str] = []
    for key in keys:
        typ = "integer" if key in {"nx", "ny", "n"} else "number"
        properties[key] = {"type": typ, "description": PARAM_DESCRIPTIONS.get(key, key)}
        required.append(key)
    return {
        "title": f"{structure} missing parameters",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_missing_params_prompt(structure: str, missing: Iterable[str], fmt: str = "text") -> str:
    missing_list = list(missing)
    if not missing_list:
        return ""
    if fmt == "json_schema":
        schema = build_missing_params_schema(structure, missing_list)
        return "Return a JSON object that satisfies this JSON schema:\n" + json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        )
    header = (
        "Some required geometry parameters are missing. "
        "Please provide the following values (numbers, units in mm if not specified):"
    )
    lines = "\n".join(describe_params(missing_list))
    return f"[structure={structure}]\n{header}\n{lines}\nReturn a JSON object with these keys."


def build_normalization_prompt(user_text: str, context_summary: str = "") -> str:
    return build_prompt(
        PromptTask.NORMALIZE_USER_TURN,
        "en",
        {
            "user_text": user_text,
            "context_summary": context_summary,
        },
    ).prompt
