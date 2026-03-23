from __future__ import annotations

from typing import Iterable


_GROUPS_BY_SKELETON: dict[str, dict[str, tuple[str, ...]]] = {
    "ring_modules": {
        "geometry.ask.ring.module_size": ("module_x", "module_y", "module_z"),
        "geometry.ask.ring.count": ("n",),
        "geometry.ask.ring.radius": ("radius",),
        "geometry.ask.ring.clearance": ("clearance",),
    },
    "grid_modules": {
        "geometry.ask.grid.module_size": ("module_x", "module_y", "module_z"),
        "geometry.ask.grid.count_x": ("nx",),
        "geometry.ask.grid.count_y": ("ny",),
        "geometry.ask.grid.pitch_x": ("pitch_x",),
        "geometry.ask.grid.pitch_y": ("pitch_y",),
        "geometry.ask.grid.clearance": ("clearance",),
    },
    "nest_box_box": {
        "geometry.ask.nest.parent_size": ("parent_x", "parent_y", "parent_z"),
        "geometry.ask.nest.child_size": ("child_x", "child_y", "child_z"),
        "geometry.ask.nest.clearance": ("clearance",),
    },
    "nest_box_tubs": {
        "geometry.ask.nest.parent_size": ("parent_x", "parent_y", "parent_z"),
        "geometry.ask.nest.child_radius": ("child_rmax",),
        "geometry.ask.nest.child_half_length": ("child_hz",),
        "geometry.ask.nest.clearance": ("clearance",),
    },
    "stack_in_box": {
        "geometry.ask.stack.footprint": ("stack_x", "stack_y"),
        "geometry.ask.stack.thicknesses": ("t1", "t2", "t3"),
        "geometry.ask.stack.layer_clearance": ("stack_clearance",),
    },
    "shell_nested": {
        "geometry.ask.shell.inner_radius": ("inner_r",),
        "geometry.ask.shell.thicknesses": ("th1", "th2", "th3"),
        "geometry.ask.shell.half_length": ("hz",),
    },
    "boolean_union_boxes": {
        "geometry.ask.boolean.solid_a_size": ("bool_a_x", "bool_a_y", "bool_a_z"),
        "geometry.ask.boolean.solid_b_size": ("bool_b_x", "bool_b_y", "bool_b_z"),
    },
    "boolean_subtraction_boxes": {
        "geometry.ask.boolean.solid_a_size": ("bool_a_x", "bool_a_y", "bool_a_z"),
        "geometry.ask.boolean.solid_b_size": ("bool_b_x", "bool_b_y", "bool_b_z"),
    },
    "boolean_intersection_boxes": {
        "geometry.ask.boolean.solid_a_size": ("bool_a_x", "bool_a_y", "bool_a_z"),
        "geometry.ask.boolean.solid_b_size": ("bool_b_x", "bool_b_y", "bool_b_z"),
    },
}


def graph_dialogue_missing_paths(chosen_skeleton: str | None, missing_params: Iterable[str]) -> list[str]:
    skeleton = str(chosen_skeleton or "").strip()
    missing = {str(item).strip() for item in missing_params if str(item).strip()}
    if not skeleton or not missing:
        return []
    groups = _GROUPS_BY_SKELETON.get(skeleton, {})
    if not groups:
        return [f"geometry.params.{name}" for name in sorted(missing)]

    out: list[str] = []
    covered: set[str] = set()
    for dialogue_path, param_names in groups.items():
        if any(name in missing for name in param_names):
            out.append(dialogue_path)
            covered.update(name for name in param_names if name in missing)
    for name in sorted(missing - covered):
        out.append(f"geometry.params.{name}")
    return out
