from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List

from builder.geometry.library import (
    sample_grid_modules,
    sample_nest_box_tubs,
    sample_ring_modules,
    sample_shell_nested,
    sample_stack_in_box,
)

STRUCTURES = ["nest", "grid", "ring", "stack", "shell"]
DISTRACTORS = {
    "ring": ["array", "grid", "matrix", "nx", "ny", "pitch"],
    "grid": ["ring", "circular", "radius", "arc"],
    "nest": ["stack", "layer", "pitch"],
    "stack": ["nest", "inside", "contain"],
    "shell": ["ring", "grid", "pitch", "nx"],
}


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}"


def _render_case(structure: str, p: Dict[str, float], rng: random.Random) -> str:
    # Hard cases intentionally include distractor words from other structures.
    if structure == "ring":
        lead = (
            f"Use circular placement for detector modules: n={int(round(p['n']))}, "
            f"radius={_fmt(p['radius'])} mm, module={_fmt(p['module_x'])}x{_fmt(p['module_y'])}x{_fmt(p['module_z'])} mm, "
            f"clearance={_fmt(p['clearance'])} mm."
        )
    elif structure == "grid":
        lead = (
            f"Use rectangular array: nx={int(round(p['nx']))}, ny={int(round(p['ny']))}, "
            f"pitch_x={_fmt(p['pitch_x'])} mm, pitch_y={_fmt(p['pitch_y'])} mm, "
            f"module={_fmt(p['module_x'])}x{_fmt(p['module_y'])}x{_fmt(p['module_z'])} mm, clearance={_fmt(p['clearance'])} mm."
        )
    elif structure == "nest":
        lead = (
            f"Place child cylinder inside parent box: parent={_fmt(p['parent_x'])}x{_fmt(p['parent_y'])}x{_fmt(p['parent_z'])} mm, "
            f"child_rmax={_fmt(p['child_rmax'])} mm, child_hz={_fmt(p['child_hz'])} mm, clearance={_fmt(p['clearance'])} mm."
        )
    elif structure == "stack":
        lead = (
            f"Stack layers along Z: stack_xy={_fmt(p['stack_x'])}x{_fmt(p['stack_y'])} mm, "
            f"thicknesses=[{_fmt(p['t1'])},{_fmt(p['t2'])},{_fmt(p['t3'])}] mm, stack_clearance={_fmt(p['stack_clearance'])} mm, "
            f"container={_fmt(p['parent_x'])}x{_fmt(p['parent_y'])}x{_fmt(p['parent_z'])} mm, nest_clearance={_fmt(p['nest_clearance'])} mm."
        )
    else:
        lead = (
            f"Concentric shells: inner_r={_fmt(p['inner_r'])} mm, "
            f"thicknesses=[{_fmt(p['th1'])},{_fmt(p['th2'])},{_fmt(p['th3'])}] mm, hz={_fmt(p['hz'])} mm, "
            f"child_rmax={_fmt(p['child_rmax'])} mm, child_hz={_fmt(p['child_hz'])} mm, clearance={_fmt(p['clearance'])} mm."
        )

    distractors = DISTRACTORS[structure]
    picked = rng.sample(distractors, k=min(3, len(distractors)))
    tail = (
        " Note: terms like "
        + ", ".join(picked)
        + " may appear in previous designs; keep this request in the current structure."
    )
    return lead + tail


def _sample_params(structure: str, rng: random.Random) -> Dict[str, float]:
    if structure == "ring":
        return sample_ring_modules(rng)
    if structure == "grid":
        return sample_grid_modules(rng)
    if structure == "nest":
        return sample_nest_box_tubs(rng)
    if structure == "stack":
        return sample_stack_in_box(rng)
    return sample_shell_nested(rng)


def generate(n: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for i in range(n):
        structure = STRUCTURES[i % len(STRUCTURES)]
        params = _sample_params(structure, rng)
        text = _render_case(structure, params, rng)
        rows.append({"text": text, "structure": structure, "params": params})
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard structure classification samples")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: Dict[str, int] = {}
    for r in rows:
        s = str(r["structure"])
        counts[s] = counts.get(s, 0) + 1
    print(json.dumps({"n": len(rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
