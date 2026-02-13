from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ===== DSL Node Definitions =====

@dataclass(frozen=True)
class Part:
    id: str


@dataclass(frozen=True)
class Box(Part):
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Tubs(Part):
    rmax: float
    hz: float


@dataclass(frozen=True)
class Sphere(Part):
    rmax: float


@dataclass(frozen=True)
class Cons(Part):
    rmax1: float
    rmax2: float
    hz: float


@dataclass(frozen=True)
class Trd(Part):
    x1: float
    x2: float
    y1: float
    y2: float
    z: float


@dataclass(frozen=True)
class ShellTubsFromThicknesses(Part):
    inner_r: float
    thicknesses: Tuple[float, ...]
    hz: float


@dataclass(frozen=True)
class Operator:
    id: str


@dataclass(frozen=True)
class Nest(Operator):
    parent: str
    child: str
    clearance: float


@dataclass(frozen=True)
class StackZ(Operator):
    x: float
    y: float
    thicknesses: Tuple[float, ...]
    clearance: float


@dataclass(frozen=True)
class GridXY(Operator):
    module: str
    nx: int
    ny: int
    pitch_x: float
    pitch_y: float
    clearance: float


@dataclass(frozen=True)
class Ring(Operator):
    module: str
    n: int
    radius: float
    clearance: float


@dataclass(frozen=True)
class Transform(Operator):
    target: str
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


Node = Union[Box, Tubs, Sphere, Cons, Trd, ShellTubsFromThicknesses, Nest, StackZ, GridXY, Ring, Transform]


@dataclass
class Graph:
    nodes: Dict[str, Node]
    root: str
    constraints: List[Constraint] = field(default_factory=list)


# ===== Parsing / Serialization =====

_TYPE_MAP = {
    "Box": Box,
    "Tubs": Tubs,
    "Sphere": Sphere,
    "Cons": Cons,
    "Trd": Trd,
    "ShellTubsFromThicknesses": ShellTubsFromThicknesses,
    "Nest": Nest,
    "StackZ": StackZ,
    "GridXY": GridXY,
    "Ring": Ring,
    "Transform": Transform,
}


def _as_tuple_floats(value: Any) -> Tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    raise ValueError("Expected list/tuple of floats")


def parse_graph(data: Dict[str, Any]) -> Graph:
    nodes: Dict[str, Node] = {}
    for n in data.get("nodes", []):
        n_type = n.get("type")
        cls = _TYPE_MAP.get(n_type)
        if cls is None:
            raise ValueError(f"Unknown node type: {n_type}")
        nid = n.get("id")
        if not nid:
            raise ValueError("Node missing id")
        if cls is Box:
            node = Box(id=nid, x=float(n["x"]), y=float(n["y"]), z=float(n["z"]))
        elif cls is Tubs:
            node = Tubs(id=nid, rmax=float(n["rmax"]), hz=float(n["hz"]))
        elif cls is Sphere:
            node = Sphere(id=nid, rmax=float(n["rmax"]))
        elif cls is Cons:
            node = Cons(id=nid, rmax1=float(n["rmax1"]), rmax2=float(n["rmax2"]), hz=float(n["hz"]))
        elif cls is Trd:
            node = Trd(
                id=nid,
                x1=float(n["x1"]),
                x2=float(n["x2"]),
                y1=float(n["y1"]),
                y2=float(n["y2"]),
                z=float(n["z"]),
            )
        elif cls is ShellTubsFromThicknesses:
            node = ShellTubsFromThicknesses(
                id=nid,
                inner_r=float(n["inner_r"]),
                thicknesses=_as_tuple_floats(n["thicknesses"]),
                hz=float(n["hz"]),
            )
        elif cls is Nest:
            node = Nest(id=nid, parent=n["parent"], child=n["child"], clearance=float(n["clearance"]))
        elif cls is StackZ:
            node = StackZ(
                id=nid,
                x=float(n["x"]),
                y=float(n["y"]),
                thicknesses=_as_tuple_floats(n["thicknesses"]),
                clearance=float(n["clearance"]),
            )
        elif cls is GridXY:
            node = GridXY(
                id=nid,
                module=n["module"],
                nx=int(n["nx"]),
                ny=int(n["ny"]),
                pitch_x=float(n["pitch_x"]),
                pitch_y=float(n["pitch_y"]),
                clearance=float(n["clearance"]),
            )
        elif cls is Ring:
            node = Ring(
                id=nid,
                module=n["module"],
                n=int(n["n"]),
                radius=float(n["radius"]),
                clearance=float(n["clearance"]),
            )
        elif cls is Transform:
            node = Transform(
                id=nid,
                target=n["target"],
                tx=float(n.get("tx", 0.0)),
                ty=float(n.get("ty", 0.0)),
                tz=float(n.get("tz", 0.0)),
                rx=float(n.get("rx", 0.0)),
                ry=float(n.get("ry", 0.0)),
                rz=float(n.get("rz", 0.0)),
            )
        else:
            raise ValueError(f"Unhandled node type: {n_type}")
        nodes[nid] = node

    constraints: List[Constraint] = []
    for c in data.get("constraints", []):
        cid = c.get("id", "")
        kind = c.get("kind", "")
        params = dict(c.get("params", {}))
        constraints.append(Constraint(id=cid, kind=kind, params=params))

    root = data.get("root")
    if not root:
        raise ValueError("Graph missing root")

    return Graph(nodes=nodes, root=root, constraints=constraints)


def parse_graph_json(text: str) -> Graph:
    return parse_graph(json.loads(text))


def graph_to_dict(graph: Graph) -> Dict[str, Any]:
    out_nodes: List[Dict[str, Any]] = []
    for node in graph.nodes.values():
        if isinstance(node, Box):
            out_nodes.append({"id": node.id, "type": "Box", "x": node.x, "y": node.y, "z": node.z})
        elif isinstance(node, Tubs):
            out_nodes.append({"id": node.id, "type": "Tubs", "rmax": node.rmax, "hz": node.hz})
        elif isinstance(node, Sphere):
            out_nodes.append({"id": node.id, "type": "Sphere", "rmax": node.rmax})
        elif isinstance(node, Cons):
            out_nodes.append({"id": node.id, "type": "Cons", "rmax1": node.rmax1, "rmax2": node.rmax2, "hz": node.hz})
        elif isinstance(node, Trd):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "Trd",
                    "x1": node.x1,
                    "x2": node.x2,
                    "y1": node.y1,
                    "y2": node.y2,
                    "z": node.z,
                }
            )
        elif isinstance(node, ShellTubsFromThicknesses):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "ShellTubsFromThicknesses",
                    "inner_r": node.inner_r,
                    "thicknesses": list(node.thicknesses),
                    "hz": node.hz,
                }
            )
        elif isinstance(node, Nest):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "Nest",
                    "parent": node.parent,
                    "child": node.child,
                    "clearance": node.clearance,
                }
            )
        elif isinstance(node, StackZ):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "StackZ",
                    "x": node.x,
                    "y": node.y,
                    "thicknesses": list(node.thicknesses),
                    "clearance": node.clearance,
                }
            )
        elif isinstance(node, GridXY):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "GridXY",
                    "module": node.module,
                    "nx": node.nx,
                    "ny": node.ny,
                    "pitch_x": node.pitch_x,
                    "pitch_y": node.pitch_y,
                    "clearance": node.clearance,
                }
            )
        elif isinstance(node, Ring):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "Ring",
                    "module": node.module,
                    "n": node.n,
                    "radius": node.radius,
                    "clearance": node.clearance,
                }
            )
        elif isinstance(node, Transform):
            out_nodes.append(
                {
                    "id": node.id,
                    "type": "Transform",
                    "target": node.target,
                    "tx": node.tx,
                    "ty": node.ty,
                    "tz": node.tz,
                    "rx": node.rx,
                    "ry": node.ry,
                    "rz": node.rz,
                }
            )
        else:
            raise ValueError(f"Unhandled node type in serialization: {type(node)}")

    return {
        "nodes": out_nodes,
        "root": graph.root,
        "constraints": [
            {"id": c.id, "kind": c.kind, "params": dict(c.params)} for c in graph.constraints
        ],
    }


def graph_to_json(graph: Graph, indent: int = 2) -> str:
    return json.dumps(graph_to_dict(graph), indent=indent)

