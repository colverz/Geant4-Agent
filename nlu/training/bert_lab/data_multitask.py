from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from builder.geometry.library import (
    sample_boolean_boxes,
    sample_single_cuttubs,
    sample_single_elltube,
    sample_single_ellipsoid,
    sample_grid_modules,
    sample_nest_box_tubs,
    sample_single_para,
    sample_single_polycone,
    sample_single_polyhedra,
    sample_ring_modules,
    sample_shell_nested,
    sample_stack_in_box,
    sample_single_box,
    sample_single_cons,
    sample_single_orb,
    sample_single_sphere,
    sample_single_torus,
    sample_single_trap,
    sample_single_trd,
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
    "trap_x1",
    "trap_x2",
    "trap_x3",
    "trap_x4",
    "trap_y1",
    "trap_y2",
    "trap_z",
    "para_x",
    "para_y",
    "para_z",
    "para_alpha",
    "para_theta",
    "para_phi",
    "torus_rtor",
    "torus_rmax",
    "ellipsoid_ax",
    "ellipsoid_by",
    "ellipsoid_cz",
    "elltube_ax",
    "elltube_by",
    "elltube_hz",
    "polyhedra_nsides",
    "tilt_x",
    "tilt_y",
    "bool_a_x",
    "bool_a_y",
    "bool_a_z",
    "bool_b_x",
    "bool_b_y",
    "bool_b_z",
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


def _text_single_sphere(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "rmax {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Single sphere: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 15, "end": sp["end"] + 15} for sp in spans
    ]


def _text_single_orb(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "rmax {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Single orb: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 12, "end": sp["end"] + 12} for sp in spans
    ]


def _text_single_cons(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("rmax1", _maybe_unit(rng, p["rmax1"]), "rmax1 {value}"),
        _part_with_span("rmax2", _maybe_unit(rng, p["rmax2"]), "rmax2 {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "hz {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Single frustum: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 16, "end": sp["end"] + 16} for sp in spans
    ]


def _text_single_trd(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    parts = [
        _part_with_span("x1", _maybe_unit(rng, p["x1"]), "x1 {value}"),
        _part_with_span("x2", _maybe_unit(rng, p["x2"]), "x2 {value}"),
        _part_with_span("y1", _maybe_unit(rng, p["y1"]), "y1 {value}"),
        _part_with_span("y2", _maybe_unit(rng, p["y2"]), "y2 {value}"),
        _part_with_span("module_z", _maybe_unit(rng, p["module_z"]), "z {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return "Single trd: " + text + ".", [
        {"key": sp["key"], "start": sp["start"] + 12, "end": sp["end"] + 12} for sp in spans
    ]


def _text_single_polycone(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single polycone: "
    parts = [
        _part_with_span("z1", _maybe_unit(rng, p["z1"]), "z1 {value}"),
        _part_with_span("z2", _maybe_unit(rng, p["z2"]), "z2 {value}"),
        _part_with_span("z3", _maybe_unit(rng, p["z3"]), "z3 {value}"),
        _part_with_span("r1", _maybe_unit(rng, p["r1"]), "r1 {value}"),
        _part_with_span("r2", _maybe_unit(rng, p["r2"]), "r2 {value}"),
        _part_with_span("r3", _maybe_unit(rng, p["r3"]), "r3 {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_cuttubs(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single cuttubs: "
    parts = [
        _part_with_span("child_rmax", _maybe_unit(rng, p["child_rmax"]), "rmax {value}"),
        _part_with_span("child_hz", _maybe_unit(rng, p["child_hz"]), "hz {value}"),
        _part_with_span("tilt_x", _fmt_num(p["tilt_x"]), "tilt_x {value}"),
        _part_with_span("tilt_y", _fmt_num(p["tilt_y"]), "tilt_y {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_trap(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single trap: "
    parts = [
        _part_with_span("trap_x1", _maybe_unit(rng, p["trap_x1"]), "x1 {value}"),
        _part_with_span("trap_x2", _maybe_unit(rng, p["trap_x2"]), "x2 {value}"),
        _part_with_span("trap_x3", _maybe_unit(rng, p["trap_x3"]), "x3 {value}"),
        _part_with_span("trap_x4", _maybe_unit(rng, p["trap_x4"]), "x4 {value}"),
        _part_with_span("trap_y1", _maybe_unit(rng, p["trap_y1"]), "y1 {value}"),
        _part_with_span("trap_y2", _maybe_unit(rng, p["trap_y2"]), "y2 {value}"),
        _part_with_span("trap_z", _maybe_unit(rng, p["trap_z"]), "z {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_para(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single para: "
    parts = [
        _part_with_span("para_x", _maybe_unit(rng, p["para_x"]), "x {value}"),
        _part_with_span("para_y", _maybe_unit(rng, p["para_y"]), "y {value}"),
        _part_with_span("para_z", _maybe_unit(rng, p["para_z"]), "z {value}"),
        _part_with_span("para_alpha", _fmt_num(p["para_alpha"]), "alpha {value}"),
        _part_with_span("para_theta", _fmt_num(p["para_theta"]), "theta {value}"),
        _part_with_span("para_phi", _fmt_num(p["para_phi"]), "phi {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_torus(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single torus: "
    parts = [
        _part_with_span("torus_rtor", _maybe_unit(rng, p["torus_rtor"]), "major radius {value}"),
        _part_with_span("torus_rmax", _maybe_unit(rng, p["torus_rmax"]), "tube radius {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_ellipsoid(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single ellipsoid: "
    parts = [
        _part_with_span("ellipsoid_ax", _maybe_unit(rng, p["ellipsoid_ax"]), "ax {value}"),
        _part_with_span("ellipsoid_by", _maybe_unit(rng, p["ellipsoid_by"]), "by {value}"),
        _part_with_span("ellipsoid_cz", _maybe_unit(rng, p["ellipsoid_cz"]), "cz {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_elltube(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single elliptical tube: "
    parts = [
        _part_with_span("elltube_ax", _maybe_unit(rng, p["elltube_ax"]), "ax {value}"),
        _part_with_span("elltube_by", _maybe_unit(rng, p["elltube_by"]), "by {value}"),
        _part_with_span("elltube_hz", _maybe_unit(rng, p["elltube_hz"]), "hz {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_single_polyhedra(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Single polyhedra: "
    parts = [
        _part_with_span("polyhedra_nsides", str(int(p["polyhedra_nsides"])), "nsides {value}"),
        _part_with_span("z1", _maybe_unit(rng, p["z1"]), "z1 {value}"),
        _part_with_span("z2", _maybe_unit(rng, p["z2"]), "z2 {value}"),
        _part_with_span("z3", _maybe_unit(rng, p["z3"]), "z3 {value}"),
        _part_with_span("r1", _maybe_unit(rng, p["r1"]), "r1 {value}"),
        _part_with_span("r2", _maybe_unit(rng, p["r2"]), "r2 {value}"),
        _part_with_span("r3", _maybe_unit(rng, p["r3"]), "r3 {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
    ]


def _text_boolean(rng: random.Random, p: Dict[str, float]) -> Tuple[str, List[Dict[str, object]]]:
    prefix = "Boolean solid (union placeholder): "
    parts = [
        _part_with_span("bool_a_x", _maybe_unit(rng, p["bool_a_x"]), "a.x {value}"),
        _part_with_span("bool_a_y", _maybe_unit(rng, p["bool_a_y"]), "a.y {value}"),
        _part_with_span("bool_a_z", _maybe_unit(rng, p["bool_a_z"]), "a.z {value}"),
        _part_with_span("bool_b_x", _maybe_unit(rng, p["bool_b_x"]), "b.x {value}"),
        _part_with_span("bool_b_y", _maybe_unit(rng, p["bool_b_y"]), "b.y {value}"),
        _part_with_span("bool_b_z", _maybe_unit(rng, p["bool_b_z"]), "b.z {value}"),
    ]
    text, spans = _assemble_parts([(t, [sp]) for t, sp in parts], ", ")
    return prefix + text + ".", [
        {"key": sp["key"], "start": sp["start"] + len(prefix), "end": sp["end"] + len(prefix)} for sp in spans
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
        ("single_sphere", sample_single_sphere, _text_single_sphere),
        ("single_orb", sample_single_orb, _text_single_orb),
        ("single_cons", sample_single_cons, _text_single_cons),
        ("single_trd", sample_single_trd, _text_single_trd),
        ("single_polycone", sample_single_polycone, _text_single_polycone),
        ("single_cuttubs", sample_single_cuttubs, _text_single_cuttubs),
        ("single_trap", sample_single_trap, _text_single_trap),
        ("single_para", sample_single_para, _text_single_para),
        ("single_torus", sample_single_torus, _text_single_torus),
        ("single_ellipsoid", sample_single_ellipsoid, _text_single_ellipsoid),
        ("single_elltube", sample_single_elltube, _text_single_elltube),
        ("single_polyhedra", sample_single_polyhedra, _text_single_polyhedra),
        ("boolean", sample_boolean_boxes, _text_boolean),
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
