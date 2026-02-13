from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .dsl import (
    Box,
    Cons,
    Transform,
    Graph,
    GridXY,
    Nest,
    Ring,
    ShellTubsFromThicknesses,
    Sphere,
    StackZ,
    Trd,
    Tubs,
)
from .geom import (
    AABB,
    aabb_apply_transform,
    aabb_from_box,
    aabb_from_cons,
    aabb_from_sphere,
    aabb_from_trd,
    aabb_from_tubs,
    aabb_ring,
    aabb_stackz,
    aabb_union_xy,
)


class ErrorCode(str, Enum):
    E_SOLID_PARAM = "E_SOLID_PARAM"
    E_CONTAINMENT = "E_CONTAINMENT"
    E_PARAM_INFEASIBLE = "E_PARAM_INFEASIBLE"
    E_OVERLAP_RISK = "E_OVERLAP_RISK"
    E_DANGLING_REF = "E_DANGLING_REF"
    E_UNKNOWN = "E_UNKNOWN"


@dataclass
class Issue:
    code: ErrorCode
    node_id: str
    message: str


@dataclass
class Report:
    ok: bool
    errors: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)
    suggestions: List[Issue] = field(default_factory=list)
    aabb_by_node: Dict[str, AABB] = field(default_factory=dict)


class FeasibilityChecker:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.report = Report(ok=True)

    def check(self) -> Report:
        self._eval_node(self.graph.root)
        self.report.ok = len(self.report.errors) == 0
        return self.report

    def _error(self, code: ErrorCode, node_id: str, message: str) -> None:
        self.report.errors.append(Issue(code=code, node_id=node_id, message=message))

    def _warn(self, code: ErrorCode, node_id: str, message: str) -> None:
        self.report.warnings.append(Issue(code=code, node_id=node_id, message=message))

    def _suggest(self, node_id: str, message: str) -> None:
        self.report.suggestions.append(Issue(code=ErrorCode.E_PARAM_INFEASIBLE, node_id=node_id, message=message))

    def _get_node(self, node_id: str):
        node = self.graph.nodes.get(node_id)
        if node is None:
            self._error(ErrorCode.E_DANGLING_REF, node_id, "Dangling reference")
        return node

    def _eval_node(self, node_id: str) -> Optional[AABB]:
        if node_id in self.report.aabb_by_node:
            return self.report.aabb_by_node[node_id]

        node = self._get_node(node_id)
        if node is None:
            return None

        aabb: Optional[AABB] = None

        if isinstance(node, Box):
            if node.x <= 0 or node.y <= 0 or node.z <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Box dimensions must be > 0")
                self._suggest(node.id, "Set x,y,z to positive values")
            aabb = aabb_from_box(node.x, node.y, node.z)

        elif isinstance(node, Tubs):
            if node.rmax <= 0 or node.hz <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Tubs rmax and hz must be > 0")
                self._suggest(node.id, "Set rmax>0 and hz>0")
            aabb = aabb_from_tubs(node.rmax, node.hz)

        elif isinstance(node, Sphere):
            if node.rmax <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Sphere rmax must be > 0")
                self._suggest(node.id, "Set rmax>0")
            aabb = aabb_from_sphere(node.rmax)

        elif isinstance(node, Cons):
            if node.rmax1 <= 0 or node.rmax2 <= 0 or node.hz <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Cons rmax1,rmax2,hz must be > 0")
                self._suggest(node.id, "Set rmax1>0, rmax2>0, hz>0")
            aabb = aabb_from_cons(node.rmax1, node.rmax2, node.hz)

        elif isinstance(node, Trd):
            if node.x1 <= 0 or node.x2 <= 0 or node.y1 <= 0 or node.y2 <= 0 or node.z <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Trd x1,x2,y1,y2,z must be > 0")
                self._suggest(node.id, "Set x1,x2,y1,y2,z > 0")
            aabb = aabb_from_trd(node.x1, node.x2, node.y1, node.y2, node.z)

        elif isinstance(node, ShellTubsFromThicknesses):
            if node.inner_r < 0 or node.hz <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "ShellTubs inner_r >= 0 and hz > 0")
                self._suggest(node.id, "Set inner_r>=0 and hz>0")
            if any(t <= 0 for t in node.thicknesses):
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "All thicknesses must be > 0")
                self._suggest(node.id, "Set all thicknesses > 0")
            rmax = node.inner_r + sum(node.thicknesses)
            if rmax <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "ShellTubs derived rmax must be > 0")
                self._suggest(node.id, "Increase inner_r or thicknesses so rmax>0")
            aabb = aabb_from_tubs(rmax, node.hz)

        elif isinstance(node, Nest):
            parent_aabb = self._eval_node(node.parent)
            child_aabb = self._eval_node(node.child)
            if node.clearance < 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Clearance must be >= 0")
                self._suggest(node.id, "Set clearance >= 0")
            if parent_aabb and child_aabb:
                if not parent_aabb.contains(child_aabb, node.clearance):
                    self._error(ErrorCode.E_CONTAINMENT, node.id, "Child not contained in parent (AABB)")
                    needed_x = child_aabb.x + 2.0 * node.clearance
                    needed_y = child_aabb.y + 2.0 * node.clearance
                    needed_z = child_aabb.z + 2.0 * node.clearance
                    self._suggest(
                        node.id,
                        f"Increase parent size to >= ({needed_x:.3f}, {needed_y:.3f}, {needed_z:.3f}) "
                        "or reduce child/clearance",
                    )
                aabb = parent_aabb

        elif isinstance(node, StackZ):
            if node.x <= 0 or node.y <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "StackZ x,y must be > 0")
                self._suggest(node.id, "Set stack x,y > 0")
            if node.clearance < 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "StackZ clearance must be >= 0")
                self._suggest(node.id, "Set stack clearance >= 0")
            if any(t <= 0 for t in node.thicknesses):
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "StackZ thicknesses must be > 0")
                self._suggest(node.id, "Set all thicknesses > 0")
            aabb = aabb_stackz(node.x, node.y, node.thicknesses, node.clearance)

        elif isinstance(node, GridXY):
            module_aabb = self._eval_node(node.module)
            if node.nx <= 0 or node.ny <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "GridXY nx, ny must be > 0")
                self._suggest(node.id, "Set nx, ny to positive integers")
            if node.pitch_x <= 0 or node.pitch_y <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "GridXY pitch_x, pitch_y must be > 0")
                self._suggest(node.id, "Set pitch_x, pitch_y > 0")
            if node.clearance < 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "GridXY clearance must be >= 0")
                self._suggest(node.id, "Set clearance >= 0")
            if module_aabb:
                needed_x = module_aabb.x + 2.0 * node.clearance
                needed_y = module_aabb.y + 2.0 * node.clearance
                if node.pitch_x < needed_x or node.pitch_y < needed_y:
                    self._error(
                        ErrorCode.E_PARAM_INFEASIBLE,
                        node.id,
                        "GridXY pitch too small for module + clearance",
                    )
                    self._suggest(
                        node.id,
                        f"Set pitch_x >= {needed_x:.3f} and pitch_y >= {needed_y:.3f}",
                    )
                elif node.pitch_x < module_aabb.x or node.pitch_y < module_aabb.y:
                    self._warn(ErrorCode.E_OVERLAP_RISK, node.id, "Pitch smaller than module envelope")
                aabb = aabb_union_xy(module_aabb, node.nx, node.ny, node.pitch_x, node.pitch_y)

        elif isinstance(node, Ring):
            module_aabb = self._eval_node(node.module)
            if node.n <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Ring n must be > 0")
                self._suggest(node.id, "Set n to a positive integer")
            if node.radius <= 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Ring radius must be > 0")
                self._suggest(node.id, "Set radius > 0")
            if node.clearance < 0:
                self._error(ErrorCode.E_SOLID_PARAM, node.id, "Ring clearance must be >= 0")
                self._suggest(node.id, "Set clearance >= 0")
            if module_aabb and node.n > 0:
                needed = max(module_aabb.x, module_aabb.y) + 2.0 * node.clearance
                chord = 2.0 * node.radius * math.sin(math.pi / max(node.n, 1))
                if chord < needed:
                    self._error(
                        ErrorCode.E_PARAM_INFEASIBLE,
                        node.id,
                        "Ring spacing too small for module + clearance",
                    )
                    min_radius = needed / (2.0 * math.sin(math.pi / max(node.n, 1)))
                    self._suggest(node.id, f"Increase radius to >= {min_radius:.3f} or reduce module/clearance")
                aabb = aabb_ring(module_aabb, node.radius)
        elif isinstance(node, Transform):
            child = self._eval_node(node.target)
            if child:
                if abs(node.rx) > 1e-9 or abs(node.ry) > 1e-9 or abs(node.rz) > 1e-9:
                    self._warn(
                        ErrorCode.E_OVERLAP_RISK,
                        node.id,
                        "Rotation handled by conservative AABB envelope",
                    )
                aabb = aabb_apply_transform(child, node.rx, node.ry, node.rz)
        else:
            self._error(ErrorCode.E_UNKNOWN, node_id, "Unknown node type")

        if aabb is not None:
            self.report.aabb_by_node[node_id] = aabb
        return aabb


def check_feasibility(graph: Graph) -> Report:
    return FeasibilityChecker(graph).check()

