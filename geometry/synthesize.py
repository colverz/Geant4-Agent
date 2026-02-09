from __future__ import annotations

import json
import os
import random
from typing import Dict, Tuple

from .dsl import graph_to_dict
from .feasibility import check_feasibility
from .library import SKELETONS


STRUCTURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "nest": ("nest_box_tubs", "stack_in_box", "shell_nested"),
    "grid": ("grid_modules",),
    "ring": ("ring_modules",),
    "stack": ("stack_in_box",),
    "shell": ("shell_nested",),
    "box_tubs": ("nest_box_tubs",),
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


def synthesize_from_params(structure: str, user_params: Dict[str, float], seed: int) -> Dict[str, object]:
    if not structure:
        raise ValueError("Missing 'structure'")
    if not isinstance(user_params, dict):
        raise ValueError("'params' must be an object")

    rng = random.Random(seed)
    sk = _select_skeleton(structure)
    default_params = sk.param_sampler(rng)
    params = _fill_params(user_params, default_params)

    graph = sk.build_fn(params)
    report = check_feasibility(graph)

    return {
        "structure": structure,
        "skeleton": sk.name,
        "params_filled": params,
        "feasible": report.ok,
        "errors": [e.code.value for e in report.errors],
        "warnings": [w.code.value for w in report.warnings],
        "dsl": graph_to_dict(graph),
    }


def run_synthesize(input_json: str, outdir: str, seed: int) -> None:
    _ensure_outdir(outdir)
    with open(input_json, "r") as f:
        payload = json.load(f)

    structure = payload.get("structure", "")
    user_params = payload.get("params", {})

    out = synthesize_from_params(structure, user_params, seed)

    out_path = os.path.join(outdir, "synthesis_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
