from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AABB:
    x: float
    y: float
    z: float

    @property
    def half(self) -> Tuple[float, float, float]:
        return (self.x * 0.5, self.y * 0.5, self.z * 0.5)

    def contains(self, other: "AABB", clearance: float = 0.0) -> bool:
        return (
            other.x + 2.0 * clearance <= self.x
            and other.y + 2.0 * clearance <= self.y
            and other.z + 2.0 * clearance <= self.z
        )


def aabb_from_box(x: float, y: float, z: float) -> AABB:
    return AABB(x=x, y=y, z=z)


def aabb_from_tubs(rmax: float, hz: float) -> AABB:
    d = 2.0 * rmax
    return AABB(x=d, y=d, z=2.0 * hz)


def aabb_from_sphere(rmax: float) -> AABB:
    d = 2.0 * rmax
    return AABB(x=d, y=d, z=d)


def aabb_from_cons(rmax1: float, rmax2: float, hz: float) -> AABB:
    rmax = max(rmax1, rmax2)
    d = 2.0 * rmax
    return AABB(x=d, y=d, z=2.0 * hz)


def aabb_from_trd(x1: float, x2: float, y1: float, y2: float, z: float) -> AABB:
    return AABB(x=max(x1, x2), y=max(y1, y2), z=z)


def aabb_apply_transform(base: AABB, rx: float, ry: float, rz: float) -> AABB:
    # Conservative rotation envelope; translation does not change size.
    if abs(rx) < 1e-9 and abs(ry) < 1e-9 and abs(rz) < 1e-9:
        return base
    d = (base.x ** 2 + base.y ** 2 + base.z ** 2) ** 0.5
    return AABB(x=d, y=d, z=d)


def aabb_union_xy(module: AABB, nx: int, ny: int, pitch_x: float, pitch_y: float) -> AABB:
    span_x = (nx - 1) * pitch_x + module.x
    span_y = (ny - 1) * pitch_y + module.y
    return AABB(x=span_x, y=span_y, z=module.z)


def aabb_ring(module: AABB, radius: float) -> AABB:
    span = 2.0 * (radius + 0.5 * max(module.x, module.y))
    return AABB(x=span, y=span, z=module.z)


def aabb_stackz(x: float, y: float, thicknesses: Tuple[float, ...], clearance: float) -> AABB:
    if len(thicknesses) == 0:
        return AABB(x=0.0, y=0.0, z=0.0)
    span_z = sum(thicknesses) + max(0, len(thicknesses) - 1) * clearance
    return AABB(x=x, y=y, z=span_z)

