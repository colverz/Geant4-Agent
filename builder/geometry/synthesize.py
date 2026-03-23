from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Tuple

from .dsl import graph_to_dict
from .feasibility import check_feasibility
from .library import SKELETONS


STRUCTURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "nest": ("nest_box_box", "nest_box_tubs", "stack_in_box", "shell_nested"),
    "grid": ("grid_modules",),
    "ring": ("ring_modules",),
    "stack": ("stack_in_box",),
    "shell": ("shell_nested",),
    "box_tubs": ("nest_box_tubs",),
    "box_box": ("nest_box_box",),
    "box": ("single_box",),
    "cube": ("single_box",),
    "single_box": ("single_box",),
    "tubs": ("single_tubs",),
    "cylinder": ("single_tubs",),
    "single_tubs": ("single_tubs",),
    "sphere": ("single_sphere",),
    "single_sphere": ("single_sphere",),
    "orb": ("single_orb",),
    "single_orb": ("single_orb",),
    "cons": ("single_cons",),
    "single_cons": ("single_cons",),
    "trd": ("single_trd",),
    "single_trd": ("single_trd",),
    "polycone": ("single_polycone",),
    "single_polycone": ("single_polycone",),
    "cuttubs": ("single_cuttubs",),
    "single_cuttubs": ("single_cuttubs",),
    "boolean": ("boolean_union_boxes",),
    "boolean_union": ("boolean_union_boxes",),
    "boolean_subtraction": ("boolean_subtraction_boxes",),
    "boolean_intersection": ("boolean_intersection_boxes",),
    "tilted_box": ("tilted_box_in_parent",),
    "tilted_box_in_parent": ("tilted_box_in_parent",),
}


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _select_skeleton(structure: str):
    if structure in STRUCTURE_ALIASES:
        names = STRUCTURE_ALIASES[structure]
        for sk in SKELETONS:
            if sk.name == names[0]:
                return sk
    for sk in SKELETONS:
        if sk.name == structure:
            return sk
    raise ValueError(f"Unknown structure: {structure}")


def _fill_params(base: Dict[str, float], defaults: Dict[str, float]) -> Dict[str, float]:
    out = dict(defaults)
    for k, v in base.items():
        out[k] = v
    return out


def _missing_params(sk, user_params: Dict[str, float]) -> List[str]:
    return [k for k in sk.param_keys if k not in user_params]


def _apply_autofix(sk_name: str, p: Dict[str, float]) -> Dict[str, float]:
    out = dict(p)
    # basic clamps
    for key in ("clearance", "stack_clearance", "nest_clearance"):
        if key in out and out[key] < 0:
            out[key] = 0.0

    if sk_name == "grid_modules":
        needed_x = out["module_x"] + 2.0 * out["clearance"]
        needed_y = out["module_y"] + 2.0 * out["clearance"]
        if out["pitch_x"] < needed_x:
            out["pitch_x"] = needed_x
        if out["pitch_y"] < needed_y:
            out["pitch_y"] = needed_y
    elif sk_name == "ring_modules":
        n = max(int(out["n"]), 1)
        needed = max(out["module_x"], out["module_y"]) + 2.0 * out["clearance"]
        denom = 2.0 * math.sin(math.pi / n)
        if denom > 0:
            min_radius = needed / denom
            if out["radius"] < min_radius:
                out["radius"] = min_radius
    elif sk_name == "nest_box_box":
        needed_x = out["child_x"] + 2.0 * out["clearance"]
        needed_y = out["child_y"] + 2.0 * out["clearance"]
        needed_z = out["child_z"] + 2.0 * out["clearance"]
        if out["parent_x"] < needed_x:
            out["parent_x"] = needed_x
        if out["parent_y"] < needed_y:
            out["parent_y"] = needed_y
        if out["parent_z"] < needed_z:
            out["parent_z"] = needed_z
    elif sk_name == "nest_box_tubs":
        needed_xy = 2.0 * out["child_rmax"] + 2.0 * out["clearance"]
        needed_z = 2.0 * out["child_hz"] + 2.0 * out["clearance"]
        if out["parent_x"] < needed_xy:
            out["parent_x"] = needed_xy
        if out["parent_y"] < needed_xy:
            out["parent_y"] = needed_xy
        if out["parent_z"] < needed_z:
            out["parent_z"] = needed_z
    elif sk_name == "stack_in_box":
        stack_z = out["t1"] + out["t2"] + out["t3"] + 2.0 * out["stack_clearance"]
        needed_x = out["stack_x"] + 2.0 * out["nest_clearance"]
        needed_y = out["stack_y"] + 2.0 * out["nest_clearance"]
        needed_z = stack_z + 2.0 * out["nest_clearance"]
        if out["parent_x"] < needed_x:
            out["parent_x"] = needed_x
        if out["parent_y"] < needed_y:
            out["parent_y"] = needed_y
        if out["parent_z"] < needed_z:
            out["parent_z"] = needed_z
    elif sk_name == "shell_nested":
        thickness_total = sum(
            float(out[key])
            for key in ("th1", "th2", "th3")
            if key in out and out[key] is not None
        )
        shell_rmax = out["inner_r"] + thickness_total
        clearance = float(out.get("clearance", 0.0))
        child_rmax = float(out.get("child_rmax", max(out["inner_r"] - clearance, 0.1)))
        child_hz = float(out.get("child_hz", max(out["hz"] - clearance, 0.1)))
        needed_rmax = child_rmax + clearance
        if shell_rmax < needed_rmax:
            out["th3"] = float(out.get("th3", 0.0)) + (needed_rmax - shell_rmax)
        needed_hz = child_hz + clearance
        if out["hz"] < needed_hz:
            out["hz"] = needed_hz
    return out


def synthesize_from_params(
    structure: str,
    user_params: Dict[str, float],
    seed: int,
    apply_autofix: bool = False,
) -> Dict[str, object]:
    if not structure:
        raise ValueError("Missing 'structure'")
    if not isinstance(user_params, dict):
        raise ValueError("'params' must be an object")

    rng = random.Random(seed)
    sk = _select_skeleton(structure)
    missing = _missing_params(sk, user_params)
    default_params = sk.param_sampler(rng)
    params = _fill_params(user_params, default_params)
    if sk.name == "shell_nested":
        for optional_key in ("th3", "child_rmax", "child_hz", "clearance"):
            if optional_key not in user_params:
                params.pop(optional_key, None)
    if apply_autofix:
        params = _apply_autofix(sk.name, params)

    graph = sk.build_fn(params)
    report = check_feasibility(graph)

    return {
        "structure": structure,
        "skeleton": sk.name,
        "missing_params": missing,
        "needs_user_input": len(missing) > 0,
        "params_filled": params,
        "feasible": report.ok,
        "errors": [e.code.value for e in report.errors],
        "warnings": [w.code.value for w in report.warnings],
        "suggestions": [s.message for s in report.suggestions],
        "dsl": graph_to_dict(graph),
    }


def run_synthesize(input_json: str, outdir: str, seed: int, apply_autofix: bool) -> None:
    _ensure_outdir(outdir)
    with open(input_json, "r") as f:
        payload = json.load(f)

    structure = payload.get("structure", "")
    user_params = payload.get("params", {})

    out = synthesize_from_params(structure, user_params, seed, apply_autofix)

    out_path = os.path.join(outdir, "synthesis_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

