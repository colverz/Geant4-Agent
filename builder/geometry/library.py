from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .dsl import (
    BooleanBinary,
    Box,
    Cons,
    CutTubs,
    EllipticalTube,
    Ellipsoid,
    Graph,
    GridXY,
    Nest,
    Para,
    Polycone,
    Polyhedra,
    Ring,
    ShellTubsFromThicknesses,
    Orb,
    Sphere,
    StackZ,
    Torus,
    Trap,
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

def build_nest_box_box(p: Dict[str, float]) -> Graph:
    nodes = {
        "parent": Box(id="parent", x=p["parent_x"], y=p["parent_y"], z=p["parent_z"]),
        "child": Box(id="child", x=p["child_x"], y=p["child_y"], z=p["child_z"]),
        "nest": Nest(id="nest", parent="parent", child="child", clearance=p["clearance"]),
    }
    return Graph(nodes=nodes, root="nest")


def sample_nest_box_box(rng: random.Random) -> Dict[str, float]:
    return {
        "parent_x": _sample_uniform(rng, 20.0, 120.0),
        "parent_y": _sample_uniform(rng, 20.0, 120.0),
        "parent_z": _sample_uniform(rng, 20.0, 120.0),
        "child_x": _sample_uniform(rng, 5.0, 60.0),
        "child_y": _sample_uniform(rng, 5.0, 60.0),
        "child_z": _sample_uniform(rng, 5.0, 60.0),
        "clearance": _sample_uniform(rng, 0.0, 5.0),
    }


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
    stack_clearance = p.get("stack_clearance", 0.0)
    nest_clearance = p.get("nest_clearance", 0.0)
    stack = StackZ(
        id="stack",
        x=p["stack_x"],
        y=p["stack_y"],
        thicknesses=(p["t1"], p["t2"], p["t3"]),
        clearance=stack_clearance,
    )
    stack_z = p["t1"] + p["t2"] + p["t3"] + 2.0 * stack_clearance
    parent_x = p.get("parent_x", p["stack_x"] + 2.0 * nest_clearance)
    parent_y = p.get("parent_y", p["stack_y"] + 2.0 * nest_clearance)
    parent_z = p.get("parent_z", stack_z + 2.0 * nest_clearance)
    parent = Box(id="parent", x=parent_x, y=parent_y, z=parent_z)
    nest = Nest(id="nest", parent="parent", child="stack", clearance=nest_clearance)
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
    thicknesses = tuple(
        p[key]
        for key in ("th1", "th2", "th3")
        if key in p and p[key] is not None
    )
    if not thicknesses:
        raise KeyError("shell_nested requires at least one shell thickness")
    shell = ShellTubsFromThicknesses(
        id="shell",
        inner_r=p["inner_r"],
        thicknesses=thicknesses,
        hz=p["hz"],
    )
    clearance = p.get("clearance", 0.0)
    child_rmax = p.get("child_rmax", max(p["inner_r"] - clearance, 0.1))
    child_hz = p.get("child_hz", max(p["hz"] - clearance, 0.1))
    child = Tubs(id="child", rmax=child_rmax, hz=child_hz)
    nest = Nest(id="nest", parent="shell", child="child", clearance=clearance)
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


def build_single_orb(p: Dict[str, float]) -> Graph:
    nodes = {"orb": Orb(id="orb", rmax=p["child_rmax"])}
    return Graph(nodes=nodes, root="orb")


def sample_single_orb(rng: random.Random) -> Dict[str, float]:
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


def build_single_polycone(p: Dict[str, float]) -> Graph:
    nodes = {
        "polycone": Polycone(
            id="polycone",
            z_planes=(p["z1"], p["z2"], p["z3"]),
            rmax=(p["r1"], p["r2"], p["r3"]),
        )
    }
    return Graph(nodes=nodes, root="polycone")


def sample_single_polycone(rng: random.Random) -> Dict[str, float]:
    hz = _sample_uniform(rng, 5.0, 120.0)
    return {
        "z1": -hz,
        "z2": 0.0,
        "z3": hz,
        "r1": _sample_uniform(rng, 2.0, 80.0),
        "r2": _sample_uniform(rng, 2.0, 80.0),
        "r3": _sample_uniform(rng, 2.0, 80.0),
    }


def build_single_cuttubs(p: Dict[str, float]) -> Graph:
    nodes = {
        "cuttubs": CutTubs(
            id="cuttubs",
            rmax=p["child_rmax"],
            hz=p["child_hz"],
            tilt_x=p["tilt_x"],
            tilt_y=p["tilt_y"],
        )
    }
    return Graph(nodes=nodes, root="cuttubs")


def sample_single_cuttubs(rng: random.Random) -> Dict[str, float]:
    return {
        "child_rmax": _sample_uniform(rng, 2.0, 90.0),
        "child_hz": _sample_uniform(rng, 2.0, 120.0),
        "tilt_x": _sample_uniform(rng, 0.0, 15.0),
        "tilt_y": _sample_uniform(rng, 0.0, 15.0),
    }


def build_single_trap(p: Dict[str, float]) -> Graph:
    nodes = {
        "trap": Trap(
            id="trap",
            x1=p["trap_x1"],
            x2=p["trap_x2"],
            x3=p["trap_x3"],
            x4=p["trap_x4"],
            y1=p["trap_y1"],
            y2=p["trap_y2"],
            z=p["trap_z"],
        )
    }
    return Graph(nodes=nodes, root="trap")


def sample_single_trap(rng: random.Random) -> Dict[str, float]:
    return {
        "trap_x1": _sample_uniform(rng, 2.0, 80.0),
        "trap_x2": _sample_uniform(rng, 2.0, 80.0),
        "trap_x3": _sample_uniform(rng, 2.0, 80.0),
        "trap_x4": _sample_uniform(rng, 2.0, 80.0),
        "trap_y1": _sample_uniform(rng, 2.0, 80.0),
        "trap_y2": _sample_uniform(rng, 2.0, 80.0),
        "trap_z": _sample_uniform(rng, 2.0, 120.0),
    }


def build_single_para(p: Dict[str, float]) -> Graph:
    nodes = {
        "para": Para(
            id="para",
            x=p["para_x"],
            y=p["para_y"],
            z=p["para_z"],
            alpha=p["para_alpha"],
            theta=p["para_theta"],
            phi=p["para_phi"],
        )
    }
    return Graph(nodes=nodes, root="para")


def sample_single_para(rng: random.Random) -> Dict[str, float]:
    return {
        "para_x": _sample_uniform(rng, 2.0, 100.0),
        "para_y": _sample_uniform(rng, 2.0, 100.0),
        "para_z": _sample_uniform(rng, 2.0, 120.0),
        "para_alpha": _sample_uniform(rng, 0.0, 20.0),
        "para_theta": _sample_uniform(rng, 0.0, 20.0),
        "para_phi": _sample_uniform(rng, 0.0, 20.0),
    }


def build_single_torus(p: Dict[str, float]) -> Graph:
    nodes = {"torus": Torus(id="torus", rtor=p["torus_rtor"], rmax=p["torus_rmax"])}
    return Graph(nodes=nodes, root="torus")


def sample_single_torus(rng: random.Random) -> Dict[str, float]:
    major = _sample_uniform(rng, 10.0, 120.0)
    minor = _sample_uniform(rng, 2.0, max(3.0, major * 0.4))
    return {
        "torus_rtor": major,
        "torus_rmax": minor,
    }


def build_single_ellipsoid(p: Dict[str, float]) -> Graph:
    nodes = {
        "ellipsoid": Ellipsoid(
            id="ellipsoid",
            ax=p["ellipsoid_ax"],
            by=p["ellipsoid_by"],
            cz=p["ellipsoid_cz"],
        )
    }
    return Graph(nodes=nodes, root="ellipsoid")


def build_single_elltube(p: Dict[str, float]) -> Graph:
    nodes = {
        "elltube": EllipticalTube(
            id="elltube",
            ax=p["elltube_ax"],
            by=p["elltube_by"],
            hz=p["elltube_hz"],
        )
    }
    return Graph(nodes=nodes, root="elltube")


def sample_single_elltube(rng: random.Random) -> Dict[str, float]:
    return {
        "elltube_ax": _sample_uniform(rng, 2.0, 80.0),
        "elltube_by": _sample_uniform(rng, 2.0, 80.0),
        "elltube_hz": _sample_uniform(rng, 2.0, 120.0),
    }


def build_single_polyhedra(p: Dict[str, float]) -> Graph:
    z_planes = tuple(sorted((p["z1"], p["z2"], p["z3"])))
    nodes = {
        "polyhedra": Polyhedra(
            id="polyhedra",
            nsides=int(p["polyhedra_nsides"]),
            z_planes=z_planes,
            rmax=(p["r1"], p["r2"], p["r3"]),
        )
    }
    return Graph(nodes=nodes, root="polyhedra")


def sample_single_polyhedra(rng: random.Random) -> Dict[str, float]:
    z1 = _sample_uniform(rng, -80.0, -10.0)
    z2 = _sample_uniform(rng, -5.0, 5.0)
    z3 = _sample_uniform(rng, 10.0, 80.0)
    z1, z2, z3 = sorted((z1, z2, z3))
    return {
        "polyhedra_nsides": float(_sample_int(rng, 3, 12)),
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "r1": _sample_uniform(rng, 2.0, 40.0),
        "r2": _sample_uniform(rng, 2.0, 40.0),
        "r3": _sample_uniform(rng, 2.0, 40.0),
    }


def sample_single_ellipsoid(rng: random.Random) -> Dict[str, float]:
    return {
        "ellipsoid_ax": _sample_uniform(rng, 2.0, 100.0),
        "ellipsoid_by": _sample_uniform(rng, 2.0, 100.0),
        "ellipsoid_cz": _sample_uniform(rng, 2.0, 120.0),
    }


def build_boolean_union_boxes(p: Dict[str, float]) -> Graph:
    nodes = {
        "a": Box(id="a", x=p["bool_a_x"], y=p["bool_a_y"], z=p["bool_a_z"]),
        "b": Box(id="b", x=p["bool_b_x"], y=p["bool_b_y"], z=p["bool_b_z"]),
        "bool": BooleanBinary(id="bool", op="union", left="a", right="b"),
    }
    return Graph(nodes=nodes, root="bool")


def build_boolean_subtraction_boxes(p: Dict[str, float]) -> Graph:
    nodes = {
        "a": Box(id="a", x=p["bool_a_x"], y=p["bool_a_y"], z=p["bool_a_z"]),
        "b": Box(id="b", x=p["bool_b_x"], y=p["bool_b_y"], z=p["bool_b_z"]),
        "bool": BooleanBinary(id="bool", op="subtraction", left="a", right="b"),
    }
    return Graph(nodes=nodes, root="bool")


def build_boolean_intersection_boxes(p: Dict[str, float]) -> Graph:
    nodes = {
        "a": Box(id="a", x=p["bool_a_x"], y=p["bool_a_y"], z=p["bool_a_z"]),
        "b": Box(id="b", x=p["bool_b_x"], y=p["bool_b_y"], z=p["bool_b_z"]),
        "bool": BooleanBinary(id="bool", op="intersection", left="a", right="b"),
    }
    return Graph(nodes=nodes, root="bool")


def sample_boolean_boxes(rng: random.Random) -> Dict[str, float]:
    return {
        "bool_a_x": _sample_uniform(rng, 5.0, 80.0),
        "bool_a_y": _sample_uniform(rng, 5.0, 80.0),
        "bool_a_z": _sample_uniform(rng, 5.0, 80.0),
        "bool_b_x": _sample_uniform(rng, 5.0, 80.0),
        "bool_b_y": _sample_uniform(rng, 5.0, 80.0),
        "bool_b_z": _sample_uniform(rng, 5.0, 80.0),
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
        name="nest_box_box",
        build_fn=build_nest_box_box,
        param_sampler=sample_nest_box_box,
        param_keys=("parent_x", "parent_y", "parent_z", "child_x", "child_y", "child_z", "clearance"),
    )
)
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
        ),
    )
)
register_skeleton(
    Skeleton(
        name="shell_nested",
        build_fn=build_shell_nested,
        param_sampler=sample_shell_nested,
        param_keys=("inner_r", "th1", "th2", "hz"),
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
        name="single_orb",
        build_fn=build_single_orb,
        param_sampler=sample_single_orb,
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
register_skeleton(
    Skeleton(
        name="single_polycone",
        build_fn=build_single_polycone,
        param_sampler=sample_single_polycone,
        param_keys=("z1", "z2", "z3", "r1", "r2", "r3"),
    )
)
register_skeleton(
    Skeleton(
        name="single_cuttubs",
        build_fn=build_single_cuttubs,
        param_sampler=sample_single_cuttubs,
        param_keys=("child_rmax", "child_hz", "tilt_x", "tilt_y"),
    )
)
register_skeleton(
    Skeleton(
        name="single_trap",
        build_fn=build_single_trap,
        param_sampler=sample_single_trap,
        param_keys=("trap_x1", "trap_x2", "trap_x3", "trap_x4", "trap_y1", "trap_y2", "trap_z"),
    )
)
register_skeleton(
    Skeleton(
        name="single_para",
        build_fn=build_single_para,
        param_sampler=sample_single_para,
        param_keys=("para_x", "para_y", "para_z", "para_alpha", "para_theta", "para_phi"),
    )
)
register_skeleton(
    Skeleton(
        name="single_torus",
        build_fn=build_single_torus,
        param_sampler=sample_single_torus,
        param_keys=("torus_rtor", "torus_rmax"),
    )
)
register_skeleton(
    Skeleton(
        name="single_ellipsoid",
        build_fn=build_single_ellipsoid,
        param_sampler=sample_single_ellipsoid,
        param_keys=("ellipsoid_ax", "ellipsoid_by", "ellipsoid_cz"),
    )
)
register_skeleton(
    Skeleton(
        name="single_elltube",
        build_fn=build_single_elltube,
        param_sampler=sample_single_elltube,
        param_keys=("elltube_ax", "elltube_by", "elltube_hz"),
    )
)
register_skeleton(
    Skeleton(
        name="single_polyhedra",
        build_fn=build_single_polyhedra,
        param_sampler=sample_single_polyhedra,
        param_keys=("polyhedra_nsides", "z1", "z2", "z3", "r1", "r2", "r3"),
    )
)
register_skeleton(
    Skeleton(
        name="boolean_union_boxes",
        build_fn=build_boolean_union_boxes,
        param_sampler=sample_boolean_boxes,
        param_keys=("bool_a_x", "bool_a_y", "bool_a_z", "bool_b_x", "bool_b_y", "bool_b_z"),
    )
)
register_skeleton(
    Skeleton(
        name="boolean_subtraction_boxes",
        build_fn=build_boolean_subtraction_boxes,
        param_sampler=sample_boolean_boxes,
        param_keys=("bool_a_x", "bool_a_y", "bool_a_z", "bool_b_x", "bool_b_y", "bool_b_z"),
    )
)
register_skeleton(
    Skeleton(
        name="boolean_intersection_boxes",
        build_fn=build_boolean_intersection_boxes,
        param_sampler=sample_boolean_boxes,
        param_keys=("bool_a_x", "bool_a_y", "bool_a_z", "bool_b_x", "bool_b_y", "bool_b_z"),
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
        "z1": _sample_uniform(rng, -60.0, -5.0),
        "z2": 0.0,
        "z3": _sample_uniform(rng, 5.0, 60.0),
        "r1": _sample_uniform(rng, 2.0, 40.0),
        "r2": _sample_uniform(rng, 2.0, 40.0),
        "r3": _sample_uniform(rng, 2.0, 40.0),
        "torus_rtor": _sample_uniform(rng, 10.0, 120.0),
        "torus_rmax": _sample_uniform(rng, 2.0, 20.0),
        "ellipsoid_ax": _sample_uniform(rng, 2.0, 100.0),
        "ellipsoid_by": _sample_uniform(rng, 2.0, 100.0),
        "ellipsoid_cz": _sample_uniform(rng, 2.0, 120.0),
        "elltube_ax": _sample_uniform(rng, 2.0, 80.0),
        "elltube_by": _sample_uniform(rng, 2.0, 80.0),
        "elltube_hz": _sample_uniform(rng, 2.0, 120.0),
        "polyhedra_nsides": float(_sample_int(rng, 3, 12)),
        "tilt_x": _sample_uniform(rng, 0.0, 15.0),
        "tilt_y": _sample_uniform(rng, 0.0, 15.0),
        "bool_a_x": _sample_uniform(rng, 5.0, 60.0),
        "bool_a_y": _sample_uniform(rng, 5.0, 60.0),
        "bool_a_z": _sample_uniform(rng, 5.0, 60.0),
        "bool_b_x": _sample_uniform(rng, 5.0, 60.0),
        "bool_b_y": _sample_uniform(rng, 5.0, 60.0),
        "bool_b_z": _sample_uniform(rng, 5.0, 60.0),
    }

