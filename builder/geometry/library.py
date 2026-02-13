from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .dsl import (
    Box,
    Cons,
    Graph,
    GridXY,
    Nest,
    Ring,
    ShellTubsFromThicknesses,
    Sphere,
    StackZ,
    Transform,
    Trd,
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


def build_single_box(p: Dict[str, float]) -> Graph:
    nodes = {
        "box": Box(id="box", x=p["module_x"], y=p["module_y"], z=p["module_z"]),
    }
    return Graph(nodes=nodes, root="box")


def sample_single_box(rng: random.Random) -> Dict[str, float]:
    return {
        "module_x": _sample_uniform(rng, 5.0, 200.0),
        "module_y": _sample_uniform(rng, 5.0, 200.0),
        "module_z": _sample_uniform(rng, 5.0, 200.0),
    }


def build_single_tubs(p: Dict[str, float]) -> Graph:
    nodes = {
        "tubs": Tubs(id="tubs", rmax=p["child_rmax"], hz=p["child_hz"]),
    }
    return Graph(nodes=nodes, root="tubs")


def sample_single_tubs(rng: random.Random) -> Dict[str, float]:
    return {
        "child_rmax": _sample_uniform(rng, 2.0, 100.0),
        "child_hz": _sample_uniform(rng, 2.0, 100.0),
    }


def build_single_sphere(p: Dict[str, float]) -> Graph:
    nodes = {"sphere": Sphere(id="sphere", rmax=p["child_rmax"])}
    return Graph(nodes=nodes, root="sphere")


def sample_single_sphere(rng: random.Random) -> Dict[str, float]:
    return {"child_rmax": _sample_uniform(rng, 2.0, 120.0)}


def build_single_cons(p: Dict[str, float]) -> Graph:
    nodes = {"cons": Cons(id="cons", rmax1=p["rmax1"], rmax2=p["rmax2"], hz=p["child_hz"])}
    return Graph(nodes=nodes, root="cons")


def sample_single_cons(rng: random.Random) -> Dict[str, float]:
    return {
        "rmax1": _sample_uniform(rng, 2.0, 80.0),
        "rmax2": _sample_uniform(rng, 2.0, 80.0),
        "child_hz": _sample_uniform(rng, 2.0, 100.0),
    }


def build_single_trd(p: Dict[str, float]) -> Graph:
    nodes = {
        "trd": Trd(
            id="trd",
            x1=p["x1"],
            x2=p["x2"],
            y1=p["y1"],
            y2=p["y2"],
            z=p["module_z"],
        )
    }
    return Graph(nodes=nodes, root="trd")


def sample_single_trd(rng: random.Random) -> Dict[str, float]:
    return {
        "x1": _sample_uniform(rng, 2.0, 80.0),
        "x2": _sample_uniform(rng, 2.0, 80.0),
        "y1": _sample_uniform(rng, 2.0, 80.0),
        "y2": _sample_uniform(rng, 2.0, 80.0),
        "module_z": _sample_uniform(rng, 2.0, 120.0),
    }


def build_tilted_box_in_parent(p: Dict[str, float]) -> Graph:
    nodes = {
        "parent": Box(id="parent", x=p["parent_x"], y=p["parent_y"], z=p["parent_z"]),
        "child": Box(id="child", x=p["module_x"], y=p["module_y"], z=p["module_z"]),
        "placed": Transform(
            id="placed",
            target="child",
            tx=p.get("tx", 0.0),
            ty=p.get("ty", 0.0),
            tz=p.get("tz", 0.0),
            rx=p.get("rx", 0.0),
            ry=p.get("ry", 0.0),
            rz=p.get("rz", 30.0),
        ),
        "nest": Nest(id="nest", parent="parent", child="placed", clearance=p["clearance"]),
    }
    return Graph(nodes=nodes, root="nest")


def sample_tilted_box_in_parent(rng: random.Random) -> Dict[str, float]:
    return {
        "parent_x": _sample_uniform(rng, 20.0, 120.0),
        "parent_y": _sample_uniform(rng, 20.0, 120.0),
        "parent_z": _sample_uniform(rng, 20.0, 120.0),
        "module_x": _sample_uniform(rng, 5.0, 40.0),
        "module_y": _sample_uniform(rng, 5.0, 40.0),
        "module_z": _sample_uniform(rng, 5.0, 40.0),
        "clearance": _sample_uniform(rng, 0.0, 3.0),
        "rx": _sample_uniform(rng, 0.0, 30.0),
        "ry": _sample_uniform(rng, 0.0, 30.0),
        "rz": _sample_uniform(rng, 0.0, 45.0),
        "tx": _sample_uniform(rng, -5.0, 5.0),
        "ty": _sample_uniform(rng, -5.0, 5.0),
        "tz": _sample_uniform(rng, -5.0, 5.0),
    }


SKELETONS: List[Skeleton] = []


def register_skeleton(skeleton: Skeleton) -> None:
    SKELETONS.append(skeleton)


register_skeleton(
    Skeleton(
        name="nest_box_tubs",
        build_fn=build_nest_box_tubs,
        param_sampler=sample_nest_box_tubs,
        param_keys=("parent_x", "parent_y", "parent_z", "child_rmax", "child_hz", "clearance"),
    )
)
register_skeleton(
    Skeleton(
        name="grid_modules",
        build_fn=build_grid_modules,
        param_sampler=sample_grid_modules,
        param_keys=("module_x", "module_y", "module_z", "nx", "ny", "pitch_x", "pitch_y", "clearance"),
    )
)
register_skeleton(
    Skeleton(
        name="ring_modules",
        build_fn=build_ring_modules,
        param_sampler=sample_ring_modules,
        param_keys=("module_x", "module_y", "module_z", "n", "radius", "clearance"),
    )
)
register_skeleton(
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
    )
)
register_skeleton(
    Skeleton(
        name="shell_nested",
        build_fn=build_shell_nested,
        param_sampler=sample_shell_nested,
        param_keys=("inner_r", "th1", "th2", "th3", "hz", "child_rmax", "child_hz", "clearance"),
    )
)
register_skeleton(
    Skeleton(
        name="single_box",
        build_fn=build_single_box,
        param_sampler=sample_single_box,
        param_keys=("module_x", "module_y", "module_z"),
    )
)
register_skeleton(
    Skeleton(
        name="single_tubs",
        build_fn=build_single_tubs,
        param_sampler=sample_single_tubs,
        param_keys=("child_rmax", "child_hz"),
    )
)
register_skeleton(
    Skeleton(
        name="single_sphere",
        build_fn=build_single_sphere,
        param_sampler=sample_single_sphere,
        param_keys=("child_rmax",),
    )
)
register_skeleton(
    Skeleton(
        name="single_cons",
        build_fn=build_single_cons,
        param_sampler=sample_single_cons,
        param_keys=("rmax1", "rmax2", "child_hz"),
    )
)
register_skeleton(
    Skeleton(
        name="single_trd",
        build_fn=build_single_trd,
        param_sampler=sample_single_trd,
        param_keys=("x1", "x2", "y1", "y2", "module_z"),
    )
)
register_skeleton(
    Skeleton(
        name="tilted_box_in_parent",
        build_fn=build_tilted_box_in_parent,
        param_sampler=sample_tilted_box_in_parent,
        param_keys=(
            "parent_x",
            "parent_y",
            "parent_z",
            "module_x",
            "module_y",
            "module_z",
            "clearance",
            "rx",
            "ry",
            "rz",
            "tx",
            "ty",
            "tz",
        ),
    )
)


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
    "rmax1",
    "rmax2",
    "x1",
    "x2",
    "y1",
    "y2",
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
    "tx",
    "ty",
    "tz",
    "rx",
    "ry",
    "rz",
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
        "rmax1": _sample_uniform(rng, 2.0, 40.0),
        "rmax2": _sample_uniform(rng, 2.0, 40.0),
        "x1": _sample_uniform(rng, 2.0, 40.0),
        "x2": _sample_uniform(rng, 2.0, 40.0),
        "y1": _sample_uniform(rng, 2.0, 40.0),
        "y2": _sample_uniform(rng, 2.0, 40.0),
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
        "tx": _sample_uniform(rng, -5.0, 5.0),
        "ty": _sample_uniform(rng, -5.0, 5.0),
        "tz": _sample_uniform(rng, -5.0, 5.0),
        "rx": _sample_uniform(rng, 0.0, 30.0),
        "ry": _sample_uniform(rng, 0.0, 30.0),
        "rz": _sample_uniform(rng, 0.0, 45.0),
    }

