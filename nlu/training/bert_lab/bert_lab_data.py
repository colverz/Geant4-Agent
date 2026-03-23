from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Tuple

from builder.geometry.library import (
    sample_grid_modules,
    sample_nest_box_tubs,
    sample_ring_modules,
    sample_shell_nested,
    sample_stack_in_box,
)

UNITS = ["mm", "cm"]
FILLER = [
    "please",
    "roughly",
    "approximately",
    "centered",
    "aligned",
    "as a prototype",
    "for testing",
]


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}"


def _maybe_unit(rng: random.Random, x: float) -> str:
    if rng.random() < 0.5:
        unit = rng.choice(UNITS)
        if unit == "cm":
            return f"{_fmt_num(x / 10.0)} {unit}"
        return f"{_fmt_num(x)} {unit}"
    return _fmt_num(x)


def _shuffle_phrases(rng: random.Random, parts: List[str], noise_level: str) -> str:
    if noise_level == "none":
        return ", ".join(parts)
    rng.shuffle(parts)
    sep = rng.choice([", ", "; ", " / "])
    return sep.join(parts)


def _add_noise(
    rng: random.Random,
    text: str,
    spans: List[Dict[str, object]] | None,
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]] | None]:
    prefix = ""
    suffix = ""
    if noise_level != "none" and rng.random() < 0.4:
        prefix = rng.choice(FILLER) + " "
    if noise_level == "full" and rng.random() < 0.3:
        suffix = " (" + rng.choice(FILLER) + ")"
    # Avoid word replacements when spans are requested; they would shift offsets.
    if spans is None and noise_level == "full":
        if rng.random() < 0.2:
            text = text.replace("module", "unit")
        if rng.random() < 0.2:
            text = text.replace("clearance", "gap")
        if rng.random() < 0.2:
            text = text.replace("radius", "R")
    if spans is None:
        return prefix + text + suffix, None
    out = []
    for sp in spans:
        out.append(
            {
                "key": sp["key"],
                "start": sp["start"] + len(prefix),
                "end": sp["end"] + len(prefix),
            }
        )
    return prefix + text + suffix, out


def _part_with_span(key: str, value: str, template: str) -> Tuple[str, Dict[str, object]]:
    part = template.format(value=value)
    start = part.index(value)
    end = start + len(value)
    return part, {"key": key, "start": start, "end": end}


def _assemble_parts(
    rng: random.Random,
    parts: List[Tuple[str, List[Dict[str, object]]]],
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]]]:
    if noise_level != "none":
        rng.shuffle(parts)
        sep = rng.choice([", ", "; ", " / "])
    else:
        sep = ", "
    text = ""
    spans: List[Dict[str, object]] = []
    for i, (part_text, part_spans) in enumerate(parts):
        if i > 0:
            text += sep
        offset = len(text)
        text += part_text
        for sp in part_spans:
            spans.append(
                {
                    "key": sp["key"],
                    "start": sp["start"] + offset,
                    "end": sp["end"] + offset,
                }
            )
    return text, spans


def _text_ring(
    rng: random.Random,
    p: Dict[str, float],
    with_spans: bool,
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]] | None]:
    variants = [
        "Arrange modules on a circle",
        "Modules placed on a circular path",
        "Circular arrangement of modules",
    ]
    prefix = rng.choice(variants)
    if not with_spans:
        parts = [
            f"module size {_maybe_unit(rng, p['module_x'])} by {_maybe_unit(rng, p['module_y'])} by {_maybe_unit(rng, p['module_z'])}",
            f"count {_fmt_num(p['n'])}",
            f"radius {_maybe_unit(rng, p['radius'])}",
            f"clearance {_maybe_unit(rng, p['clearance'])}",
        ]
        return _add_noise(rng, prefix + ": " + _shuffle_phrases(rng, parts, noise_level) + ".", None, noise_level)
    parts_with_spans = [
        _part_with_span("module_x", _maybe_unit(rng, p["module_x"]), "module x {value}"),
        _part_with_span("module_y", _maybe_unit(rng, p["module_y"]), "module y {value}"),
        _part_with_span("module_z", _maybe_unit(rng, p["module_z"]), "module z {value}"),
        _part_with_span("n", _fmt_num(p["n"]), "count {value}"),
        _part_with_span("radius", _maybe_unit(rng, p["radius"]), "radius {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts(rng, [(t, [sp]) for t, sp in parts_with_spans], noise_level)
    text = prefix + ": " + text + "."
    spans = [{"key": sp["key"], "start": sp["start"] + len(prefix) + 2, "end": sp["end"] + len(prefix) + 2} for sp in spans]
    return _add_noise(rng, text, spans, noise_level)


def _text_grid(
    rng: random.Random,
    p: Dict[str, float],
    with_spans: bool,
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]] | None]:
    variants = [
        "Arrange a rectangular array",
        "Place a rectangular array of modules",
        "Modules arranged in rows and columns",
    ]
    prefix = rng.choice(variants)
    if not with_spans:
        parts = [
            f"module {_maybe_unit(rng, p['module_x'])} x {_maybe_unit(rng, p['module_y'])} x {_maybe_unit(rng, p['module_z'])}",
            f"nx {_fmt_num(p['nx'])}",
            f"ny {_fmt_num(p['ny'])}",
            f"pitch_x {_maybe_unit(rng, p['pitch_x'])}",
            f"pitch_y {_maybe_unit(rng, p['pitch_y'])}",
            f"clearance {_maybe_unit(rng, p['clearance'])}",
        ]
        return _add_noise(rng, prefix + ": " + _shuffle_phrases(rng, parts, noise_level) + ".", None, noise_level)
    parts_with_spans = [
        _part_with_span("module_x", _maybe_unit(rng, p["module_x"]), "module x {value}"),
        _part_with_span("module_y", _maybe_unit(rng, p["module_y"]), "module y {value}"),
        _part_with_span("module_z", _maybe_unit(rng, p["module_z"]), "module z {value}"),
        _part_with_span("nx", _fmt_num(p["nx"]), "nx {value}"),
        _part_with_span("ny", _fmt_num(p["ny"]), "ny {value}"),
        _part_with_span("pitch_x", _maybe_unit(rng, p["pitch_x"]), "pitch_x {value}"),
        _part_with_span("pitch_y", _maybe_unit(rng, p["pitch_y"]), "pitch_y {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts(rng, [(t, [sp]) for t, sp in parts_with_spans], noise_level)
    text = prefix + ": " + text + "."
    spans = [{"key": sp["key"], "start": sp["start"] + len(prefix) + 2, "end": sp["end"] + len(prefix) + 2} for sp in spans]
    return _add_noise(rng, text, spans, noise_level)


def _text_nest(
    rng: random.Random,
    p: Dict[str, float],
    with_spans: bool,
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]] | None]:
    variants = [
        "Place a cylinder inside a box",
        "A box contains a cylindrical child",
        "Cylinder centered inside a box",
    ]
    prefix = rng.choice(variants)
    if not with_spans:
        parts = [
            f"parent box {_maybe_unit(rng, p['parent_x'])} by {_maybe_unit(rng, p['parent_y'])} by {_maybe_unit(rng, p['parent_z'])}",
            f"child tub rmax {_maybe_unit(rng, p['child_rmax'])}",
            f"child hz {_maybe_unit(rng, p['child_hz'])}",
            f"clearance {_maybe_unit(rng, p['clearance'])}",
        ]
        return _add_noise(rng, prefix + ": " + _shuffle_phrases(rng, parts, noise_level) + ".", None, noise_level)
    parts_with_spans = [
        _part_with_span("parent_x", _maybe_unit(rng, p["parent_x"]), "parent x {value}"),
        _part_with_span("parent_y", _maybe_unit(rng, p["parent_y"]), "parent y {value}"),
        _part_with_span("parent_z", _maybe_unit(rng, p["parent_z"]), "parent z {value}"),
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "child rmax {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "child hz {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts(rng, [(t, [sp]) for t, sp in parts_with_spans], noise_level)
    text = prefix + ": " + text + "."
    spans = [{"key": sp["key"], "start": sp["start"] + len(prefix) + 2, "end": sp["end"] + len(prefix) + 2} for sp in spans]
    return _add_noise(rng, text, spans, noise_level)


def _text_stack(
    rng: random.Random,
    p: Dict[str, float],
    with_spans: bool,
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]] | None]:
    variants = [
        "Stack three layers along Z",
        "Three slabs stacked along Z",
        "Layers stacked in the Z direction",
    ]
    prefix = rng.choice(variants)
    if not with_spans:
        parts = [
            f"footprint {_maybe_unit(rng, p['stack_x'])} by {_maybe_unit(rng, p['stack_y'])}",
            f"thicknesses {_maybe_unit(rng, p['t1'])}, {_maybe_unit(rng, p['t2'])}, {_maybe_unit(rng, p['t3'])}",
            f"stack clearance {_maybe_unit(rng, p['stack_clearance'])}",
            f"container box {_maybe_unit(rng, p['parent_x'])} by {_maybe_unit(rng, p['parent_y'])} by {_maybe_unit(rng, p['parent_z'])}",
            f"nest clearance {_maybe_unit(rng, p['nest_clearance'])}",
        ]
        return _add_noise(rng, prefix + ": " + _shuffle_phrases(rng, parts, noise_level) + ".", None, noise_level)
    parts_with_spans = [
        _part_with_span("stack_x", _maybe_unit(rng, p["stack_x"]), "stack x {value}"),
        _part_with_span("stack_y", _maybe_unit(rng, p["stack_y"]), "stack y {value}"),
        _part_with_span("t1", _maybe_unit(rng, p["t1"]), "t1 {value}"),
        _part_with_span("t2", _maybe_unit(rng, p["t2"]), "t2 {value}"),
        _part_with_span("t3", _maybe_unit(rng, p["t3"]), "t3 {value}"),
        _part_with_span("stack_clearance", _maybe_unit(rng, p["stack_clearance"]), "stack clearance {value}"),
        _part_with_span("parent_x", _maybe_unit(rng, p["parent_x"]), "parent x {value}"),
        _part_with_span("parent_y", _maybe_unit(rng, p["parent_y"]), "parent y {value}"),
        _part_with_span("parent_z", _maybe_unit(rng, p["parent_z"]), "parent z {value}"),
        _part_with_span("nest_clearance", _maybe_unit(rng, p["nest_clearance"]), "nest clearance {value}"),
    ]
    text, spans = _assemble_parts(rng, [(t, [sp]) for t, sp in parts_with_spans], noise_level)
    text = prefix + ": " + text + "."
    spans = [{"key": sp["key"], "start": sp["start"] + len(prefix) + 2, "end": sp["end"] + len(prefix) + 2} for sp in spans]
    return _add_noise(rng, text, spans, noise_level)


def _text_shell(
    rng: random.Random,
    p: Dict[str, float],
    with_spans: bool,
    noise_level: str,
) -> Tuple[str, List[Dict[str, object]] | None]:
    variants = [
        "Concentric tubular shells",
        "Coaxial tubular shells",
        "Nested cylindrical shell",
    ]
    prefix = rng.choice(variants)
    if not with_spans:
        parts = [
            f"inner radius {_maybe_unit(rng, p['inner_r'])}",
            f"thicknesses {_maybe_unit(rng, p['th1'])}, {_maybe_unit(rng, p['th2'])}, {_maybe_unit(rng, p['th3'])}",
            f"half-length {_maybe_unit(rng, p['hz'])}",
            f"child tub rmax {_maybe_unit(rng, p['child_rmax'])}",
            f"child hz {_maybe_unit(rng, p['child_hz'])}",
            f"clearance {_maybe_unit(rng, p['clearance'])}",
        ]
        return _add_noise(rng, prefix + ": " + _shuffle_phrases(rng, parts, noise_level) + ".", None, noise_level)
    parts_with_spans = [
        _part_with_span("inner_r", _maybe_unit(rng, p["inner_r"]), "inner radius {value}"),
        _part_with_span("th1", _maybe_unit(rng, p["th1"]), "th1 {value}"),
        _part_with_span("th2", _maybe_unit(rng, p["th2"]), "th2 {value}"),
        _part_with_span("th3", _maybe_unit(rng, p["th3"]), "th3 {value}"),
        _part_with_span("hz", _maybe_unit(rng, p["hz"]), "hz {value}"),
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "child rmax {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "child hz {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts(rng, [(t, [sp]) for t, sp in parts_with_spans], noise_level)
    text = prefix + ": " + text + "."
    spans = [{"key": sp["key"], "start": sp["start"] + len(prefix) + 2, "end": sp["end"] + len(prefix) + 2} for sp in spans]
    return _add_noise(rng, text, spans, noise_level)


def _text_unknown(rng: random.Random, noise_level: str) -> str:
    variants = [
        "Set up a prototype with some modules; exact arrangement can be decided later",
        "Arrange several components with standard clearances; details are flexible",
        "Prepare a module layout with typical spacing; choose a reasonable pattern",
    ]
    prefix = rng.choice(variants)
    parts = [
        f"module size {_maybe_unit(rng, rng.uniform(3.0, 12.0))} by {_maybe_unit(rng, rng.uniform(3.0, 12.0))} by {_maybe_unit(rng, rng.uniform(1.0, 5.0))}",
        f"clearance {_maybe_unit(rng, rng.uniform(0.1, 1.0))}",
    ]
    text = prefix + ": " + _shuffle_phrases(rng, parts, noise_level) + "."
    out, _ = _add_noise(rng, text, None, noise_level)
    return out


def generate_samples(
    n: int,
    seed: int,
    with_spans: bool,
    noise_level: str,
    unknown_rate: float,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    samples: List[Dict[str, object]] = []
    generators = [
        ("ring", sample_ring_modules, _text_ring),
        ("grid", sample_grid_modules, _text_grid),
        ("nest", sample_nest_box_tubs, _text_nest),
        ("stack", sample_stack_in_box, _text_stack),
        ("shell", sample_shell_nested, _text_shell),
    ]

    for i in range(n):
        if (not with_spans) and rng.random() < unknown_rate:
            text = _text_unknown(rng, noise_level)
            samples.append({"text": text, "structure": "unknown", "params": {}})
            continue
        structure, sampler, renderer = rng.choice(generators)
        params = sampler(rng)
        text, spans = renderer(rng, params, with_spans, noise_level)
        sample = {"text": text, "structure": structure, "params": params}
        if spans is not None:
            sample["spans"] = spans
        samples.append(sample)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic text+label samples for BERT lab")
    parser.add_argument("--out", default="nlu/bert_lab/data/bert_lab_samples.jsonl")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--with_spans", action="store_true", help="Include spans for token tagging")
    parser.add_argument(
        "--noise_level",
        default="none",
        choices=["none", "light", "full"],
        help="Text noise level (default: none)",
    )
    parser.add_argument(
        "--unknown_rate",
        type=float,
        default=0.15,
        help="Fraction of unknown-structure samples (classification only)",
    )
    args = parser.parse_args()

    unknown_rate = 0.0 if args.with_spans else max(0.0, min(1.0, args.unknown_rate))
    samples = generate_samples(args.n, args.seed, args.with_spans, args.noise_level, unknown_rate)
    with open(args.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()


