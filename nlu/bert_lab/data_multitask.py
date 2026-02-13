from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from builder.geometry.library import (
    sample_grid_modules,
    sample_nest_box_tubs,
    sample_ring_modules,
    sample_shell_nested,
    sample_stack_in_box,
    sample_single_box,
    sample_single_tubs,
)


UNITS = ["mm", "cm"]

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


def _part_with_span(key: str, value: str, template: str) -> Tuple[str, Dict[str, object]]:
    part = template.format(value=value)
    start = part.index(value)
    end = start + len(value)
    return part, {"key": key, "start": start, "end": end}


def _assemble_parts(parts: List[Tuple[str, List[Dict[str, object]]]], sep: str) -> Tuple[str, List[Dict[str, object]]]:
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


def _load_knowledge() -> Dict[str, List[str]]:
    root = Path(__file__).resolve().parents[2]
    data = root / "knowledge" / "data"
    materials = json.loads((data / "materials_geant4_nist.json").read_text(encoding="utf-8")).get("materials", [])
    physics_lists = json.loads((data / "physics_lists.json").read_text(encoding="utf-8")).get("items", [])
    particles = json.loads((data / "particles.json").read_text(encoding="utf-8")).get("items", [])
    output_formats = json.loads((data / "output_formats.json").read_text(encoding="utf-8")).get("items", [])
    source_types = json.loads((data / "source_constraints.json").read_text(encoding="utf-8")).get("types", [])
    if not source_types:
        source_types = ["point", "beam", "plane", "isotropic"]
    return {
        "materials": materials,
        "physics_lists": physics_lists,
        "particles": particles,
        "output_formats": output_formats,
        "source_types": source_types,
    }


def _pick_subset(rng: random.Random, items: List[str], max_items: int) -> List[str]:
    clean = [x for x in items if x]
    if not clean:
        return []
    rng.shuffle(clean)
    return clean[:max_items]


def _text_single_box(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("module_x", _maybe_unit(rng, p["module_x"]), "box x {value}"),
        _part_with_span("module_y", _maybe_unit(rng, p["module_y"]), "box y {value}"),
        _part_with_span("module_z", _maybe_unit(rng, p["module_z"]), "box z {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Single box: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 12, "end": sp["end"] + 12} for sp in spans
    ]


def _text_single_tubs(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "rmax {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "hz {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Single cylinder: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 16, "end": sp["end"] + 16} for sp in spans
    ]


def _text_ring(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("module_x", _maybe_unit(rng, p["module_x"]), "module x {value}"),
        _part_with_span("module_y", _maybe_unit(rng, p["module_y"]), "module y {value}"),
        _part_with_span("module_z", _maybe_unit(rng, p["module_z"]), "module z {value}"),
        _part_with_span("n", _fmt_num(p["n"]), "count {value}"),
        _part_with_span("radius", _maybe_unit(rng, p["radius"]), "radius {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Ring layout: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 13, "end": sp["end"] + 13} for sp in spans
    ]


def _text_grid(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("module_x", _maybe_unit(rng, p["module_x"]), "module x {value}"),
        _part_with_span("module_y", _maybe_unit(rng, p["module_y"]), "module y {value}"),
        _part_with_span("module_z", _maybe_unit(rng, p["module_z"]), "module z {value}"),
        _part_with_span("nx", _fmt_num(p["nx"]), "nx {value}"),
        _part_with_span("ny", _fmt_num(p["ny"]), "ny {value}"),
        _part_with_span("pitch_x", _maybe_unit(rng, p["pitch_x"]), "pitch_x {value}"),
        _part_with_span("pitch_y", _maybe_unit(rng, p["pitch_y"]), "pitch_y {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Grid layout: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 13, "end": sp["end"] + 13} for sp in spans
    ]


def _text_nest(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("parent_x", _maybe_unit(rng, p["parent_x"]), "parent x {value}"),
        _part_with_span("parent_y", _maybe_unit(rng, p["parent_y"]), "parent y {value}"),
        _part_with_span("parent_z", _maybe_unit(rng, p["parent_z"]), "parent z {value}"),
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "child rmax {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "child hz {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Nest layout: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 13, "end": sp["end"] + 13} for sp in spans
    ]


def _text_stack(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
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
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Stack layout: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 14, "end": sp["end"] + 14} for sp in spans
    ]


def _text_shell(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("inner_r", _maybe_unit(rng, p["inner_r"]), "inner radius {value}"),
        _part_with_span("th1", _maybe_unit(rng, p["th1"]), "th1 {value}"),
        _part_with_span("th2", _maybe_unit(rng, p["th2"]), "th2 {value}"),
        _part_with_span("th3", _maybe_unit(rng, p["th3"]), "th3 {value}"),
        _part_with_span("hz", _maybe_unit(rng, p["hz"]), "hz {value}"),
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "child rmax {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "child hz {value}"),
        _part_with_span("clearance", _maybe_unit(rng, p["clearance"]), "clearance {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Shell layout: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 14, "end": sp["end"] + 14} for sp in spans
    ]


def _entity_parts(
    rng: random.Random,
    materials: List[str],
    particles: List[str],
    physics_lists: List[str],
    source_types: List[str],
    output_formats: List[str],
) -> List[Tuple[str, List[Dict[str, object]]]]:
    material = rng.choice(materials) if materials else "G4_WATER"
    particle = rng.choice(particles) if particles else "gamma"
    physics_list = rng.choice(physics_lists) if physics_lists else "FTFP_BERT"
    source_type = rng.choice(source_types) if source_types else "point"
    output_fmt = rng.choice(output_formats) if output_formats else "root"

    parts = [
        _part_with_span("material", material, "material {value}"),
        _part_with_span("particle", particle, "particle {value}"),
        _part_with_span("physics_list", physics_list, "physics list {value}"),
        _part_with_span("source_type", source_type, "source type {value}"),
        _part_with_span("output_format", output_fmt, "output {value}"),
    ]
    return [(t, [sp]) for t, sp in parts]


def generate_samples(n: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    knowledge = _load_knowledge()

    materials = _pick_subset(rng, knowledge["materials"], 30)
    particles = _pick_subset(rng, knowledge["particles"], 20)
    physics_lists = _pick_subset(rng, knowledge["physics_lists"], 20)
    source_types = _pick_subset(rng, knowledge["source_types"], 6)
    output_formats = _pick_subset(rng, knowledge["output_formats"], 6)

    generators = [
        ("ring", sample_ring_modules, _text_ring),
        ("grid", sample_grid_modules, _text_grid),
        ("nest", sample_nest_box_tubs, _text_nest),
        ("stack", sample_stack_in_box, _text_stack),
        ("shell", sample_shell_nested, _text_shell),
        ("single_box", sample_single_box, _text_single_box),
        ("single_tubs", sample_single_tubs, _text_single_tubs),
    ]

    samples: List[Dict[str, object]] = []
    for _ in range(n):
        structure, sampler, renderer = rng.choice(generators)
        params = sampler(rng)
        geom_text, geom_spans = renderer(rng, params)
        ent_parts = _entity_parts(rng, materials, particles, physics_lists, source_types, output_formats)
        ent_text, ent_spans = _assemble_parts(ent_parts, "; ")

        full_text = geom_text + " | " + ent_text
        spans = [
            {"key": sp["key"], "start": sp["start"], "end": sp["end"]} for sp in geom_spans
        ]
        offset = len(geom_text) + 3
        for sp in ent_spans:
            spans.append({"key": sp["key"], "start": sp["start"] + offset, "end": sp["end"] + offset})

        samples.append(
            {
                "text": full_text,
                "structure": structure,
                "params": params,
                "spans": spans,
                "entities": {
                    "material": ent_parts[0][0].split()[-1],
                    "particle": ent_parts[1][0].split()[-1],
                    "physics_list": ent_parts[2][0].split()[-1],
                    "source_type": ent_parts[3][0].split()[-1],
                    "output_format": ent_parts[4][0].split()[-1],
                },
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-task NLU samples (geometry + entities)")
    parser.add_argument("--out", default="nlu/bert_lab/data/bert_lab_multitask_samples.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    samples = generate_samples(args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
