from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .dsl import (
    Box,
    Graph,
    GridXY,
    Nest,
    Ring,
    ShellTubsFromThicknesses,
    StackZ,
    Tubs,
)


@dataclass(frozen=True)
class Skeleton:
    name: str
    build_fn: Callable[[Dict[str, float]], Graph]
    param_sampler: Callable[[random.Random], Dict[str, float]]
    param_keys: Tuple[str, ...]


def _sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _sample_int(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


# ===== Skeletons =====

def build_nest_box_tubs(p: Dict[str, float]) -> Graph:
    nodes = {
        "parent": Box(id="parent", x=p["parent_x"], y=p["parent_y"], z=p["parent_z"]),
        "child": Tubs(id="child", rmax=p["child_rmax"], hz=p["child_hz"]),
        "nest": Nest(id="nest", parent="parent", child="child", clearance=p["clearance"]),
    }
    return Graph(nodes=nodes, root="nest")


def sample_nest_box_tubs(rng: random.Random) -> Dict[str, float]:
    return {
        "parent_x": _sample_uniform(rng, 20.0, 120.0),
        "parent_y": _sample_uniform(rng, 20.0, 120.0),
        "parent_z": _sample_uniform(rng, 20.0, 120.0),
        "child_rmax": _sample_uniform(rng, 2.0, 40.0),
        "child_hz": _sample_uniform(rng, 2.0, 40.0),
        "clearance": _sample_uniform(rng, 0.0, 5.0),
    }


def build_grid_modules(p: Dict[str, float]) -> Graph:
    nodes = {
        "module": Box(id="module", x=p["module_x"], y=p["module_y"], z=p["module_z"]),
        "grid": GridXY(
            id="grid",
            module="module",
            nx=int(p["nx"]),
            ny=int(p["ny"]),
            pitch_x=p["pitch_x"],
            pitch_y=p["pitch_y"],
            clearance=p["clearance"],
        ),
    }
    return Graph(nodes=nodes, root="grid")


def sample_grid_modules(rng: random.Random) -> Dict[str, float]:
    module_x = _sample_uniform(rng, 2.0, 20.0)
    module_y = _sample_uniform(rng, 2.0, 20.0)
    return {
        "module_x": module_x,
        "module_y": module_y,
        "module_z": _sample_uniform(rng, 1.0, 10.0),
        "nx": float(_sample_int(rng, 1, 6)),
        "ny": float(_sample_int(rng, 1, 6)),
        "pitch_x": _sample_uniform(rng, module_x * 0.6, module_x * 1.6),
        "pitch_y": _sample_uniform(rng, module_y * 0.6, module_y * 1.6),
        "clearance": _sample_uniform(rng, 0.0, 2.0),
    }


def build_ring_modules(p: Dict[str, float]) -> Graph:
    nodes = {
        "module": Box(id="module", x=p["module_x"], y=p["module_y"], z=p["module_z"]),
        "ring": Ring(
            id="ring",
            module="module",
            n=int(p["n"]),
            radius=p["radius"],
            clearance=p["clearance"],
        ),
    }
    return Graph(nodes=nodes, root="ring")


def sample_ring_modules(rng: random.Random) -> Dict[str, float]:
    module_x = _sample_uniform(rng, 2.0, 20.0)
    module_y = _sample_uniform(rng, 2.0, 20.0)
    return {
        "module_x": module_x,
        "module_y": module_y,
        "module_z": _sample_uniform(rng, 1.0, 10.0),
        "n": float(_sample_int(rng, 3, 16)),
        "radius": _sample_uniform(rng, 10.0, 80.0),
        "clearance": _sample_uniform(rng, 0.0, 2.0),
    }


def build_stack_in_box(p: Dict[str, float]) -> Graph:
    stack = StackZ(
        id="stack",
        x=p["stack_x"],
        y=p["stack_y"],
        thicknesses=(p["t1"], p["t2"], p["t3"]),
        clearance=p["stack_clearance"],
    )
    parent = Box(id="parent", x=p["parent_x"], y=p["parent_y"], z=p["parent_z"])
    nest = Nest(id="nest", parent="parent", child="stack", clearance=p["nest_clearance"])
    nodes = {"stack": stack, "parent": parent, "nest": nest}
    return Graph(nodes=nodes, root="nest")


def sample_stack_in_box(rng: random.Random) -> Dict[str, float]:
    return {
        "stack_x": _sample_uniform(rng, 5.0, 40.0),
        "stack_y": _sample_uniform(rng, 5.0, 40.0),
        "t1": _sample_uniform(rng, 0.5, 10.0),
        "t2": _sample_uniform(rng, 0.5, 10.0),
        "t3": _sample_uniform(rng, 0.5, 10.0),
        "stack_clearance": _sample_uniform(rng, 0.0, 2.0),
        "parent_x": _sample_uniform(rng, 10.0, 80.0),
        "parent_y": _sample_uniform(rng, 10.0, 80.0),
        "parent_z": _sample_uniform(rng, 10.0, 80.0),
        "nest_clearance": _sample_uniform(rng, 0.0, 2.0),
    }


def build_shell_nested(p: Dict[str, float]) -> Graph:
    shell = ShellTubsFromThicknesses(
        id="shell",
        inner_r=p["inner_r"],
        thicknesses=(p["th1"], p["th2"], p["th3"]),
        hz=p["hz"],
    )
    child = Tubs(id="child", rmax=p["child_rmax"], hz=p["child_hz"])
    nest = Nest(id="nest", parent="shell", child="child", clearance=p["clearance"])
    nodes = {"shell": shell, "child": child, "nest": nest}
    return Graph(nodes=nodes, root="nest")


def sample_shell_nested(rng: random.Random) -> Dict[str, float]:
    return {
        "inner_r": _sample_uniform(rng, 1.0, 20.0),
        "th1": _sample_uniform(rng, 0.5, 5.0),
        "th2": _sample_uniform(rng, 0.5, 5.0),
        "th3": _sample_uniform(rng, 0.5, 5.0),
        "hz": _sample_uniform(rng, 2.0, 40.0),
        "child_rmax": _sample_uniform(rng, 1.0, 30.0),
        "child_hz": _sample_uniform(rng, 1.0, 30.0),
        "clearance": _sample_uniform(rng, 0.0, 2.0),
    }


SKELETONS: List[Skeleton] = [
    Skeleton(
        name="nest_box_tubs",
        build_fn=build_nest_box_tubs,
        param_sampler=sample_nest_box_tubs,
        param_keys=("parent_x", "parent_y", "parent_z", "child_rmax", "child_hz", "clearance"),
    ),
    Skeleton(
        name="grid_modules",
        build_fn=build_grid_modules,
        param_sampler=sample_grid_modules,
        param_keys=("module_x", "module_y", "module_z", "nx", "ny", "pitch_x", "pitch_y", "clearance"),
    ),
    Skeleton(
        name="ring_modules",
        build_fn=build_ring_modules,
        param_sampler=sample_ring_modules,
        param_keys=("module_x", "module_y", "module_z", "n", "radius", "clearance"),
    ),
    Skeleton(
        name="stack_in_box",
        build_fn=build_stack_in_box,
        param_sampler=sample_stack_in_box,
        param_keys=(
            "stack_x",
            "stack_y",
            "t1",
            "t2",
            "t3",
            "stack_clearance",
            "parent_x",
            "parent_y",
            "parent_z",
            "nest_clearance",
        ),
    ),
    Skeleton(
        name="shell_nested",
        build_fn=build_shell_nested,
        param_sampler=sample_shell_nested,
        param_keys=("inner_r", "th1", "th2", "th3", "hz", "child_rmax", "child_hz", "clearance"),
    ),
]


PARAM_SIGNATURE_KEYS: Tuple[str, ...] = (
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
)


def sample_param_signature(rng: random.Random) -> Dict[str, float]:
    # A generic sampling space that can feed multiple skeletons.
    return {
        "module_x": _sample_uniform(rng, 2.0, 20.0),
        "module_y": _sample_uniform(rng, 2.0, 20.0),
        "module_z": _sample_uniform(rng, 1.0, 10.0),
        "nx": float(_sample_int(rng, 1, 6)),
        "ny": float(_sample_int(rng, 1, 6)),
        "pitch_x": _sample_uniform(rng, 1.0, 30.0),
        "pitch_y": _sample_uniform(rng, 1.0, 30.0),
        "n": float(_sample_int(rng, 3, 16)),
        "radius": _sample_uniform(rng, 10.0, 80.0),
        "clearance": _sample_uniform(rng, 0.0, 2.0),
        "parent_x": _sample_uniform(rng, 20.0, 120.0),
        "parent_y": _sample_uniform(rng, 20.0, 120.0),
        "parent_z": _sample_uniform(rng, 20.0, 120.0),
        "child_rmax": _sample_uniform(rng, 2.0, 40.0),
        "child_hz": _sample_uniform(rng, 2.0, 40.0),
        "inner_r": _sample_uniform(rng, 1.0, 20.0),
        "th1": _sample_uniform(rng, 0.5, 5.0),
        "th2": _sample_uniform(rng, 0.5, 5.0),
        "th3": _sample_uniform(rng, 0.5, 5.0),
        "hz": _sample_uniform(rng, 2.0, 40.0),
        "stack_x": _sample_uniform(rng, 5.0, 40.0),
        "stack_y": _sample_uniform(rng, 5.0, 40.0),
        "t1": _sample_uniform(rng, 0.5, 10.0),
        "t2": _sample_uniform(rng, 0.5, 10.0),
        "t3": _sample_uniform(rng, 0.5, 10.0),
        "stack_clearance": _sample_uniform(rng, 0.0, 2.0),
        "nest_clearance": _sample_uniform(rng, 0.0, 2.0),
    }
