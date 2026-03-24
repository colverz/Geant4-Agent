from __future__ import annotations

from typing import Any

from core.domain.geometry_family import get_geometry_family
from core.orchestrator.path_ops import get_path
from core.orchestrator.types import ValidationReport
from core.validation.error_codes import E_NAME_BINDING, E_RANGE_INVALID, E_REQUIRED_MISSING, E_TYPE_INVALID
from core.validation.geometry_registry import prune_out_of_scope_params
from core.validation.minimal_schema import get_minimal_required_paths

_POSITIVE_NUMERIC_PATHS = {
    "geometry.params.module_x",
    "geometry.params.module_y",
    "geometry.params.module_z",
    "geometry.params.child_rmax",
    "geometry.params.child_hz",
    "geometry.params.rmax1",
    "geometry.params.rmax2",
    "geometry.params.x1",
    "geometry.params.x2",
    "geometry.params.y1",
    "geometry.params.y2",
    "geometry.params.r1",
    "geometry.params.r2",
    "geometry.params.r3",
    "geometry.params.trap_x1",
    "geometry.params.trap_x2",
    "geometry.params.trap_x3",
    "geometry.params.trap_x4",
    "geometry.params.trap_y1",
    "geometry.params.trap_y2",
    "geometry.params.trap_z",
    "geometry.params.para_x",
    "geometry.params.para_y",
    "geometry.params.para_z",
    "geometry.params.torus_rtor",
    "geometry.params.torus_rmax",
    "geometry.params.ellipsoid_ax",
    "geometry.params.ellipsoid_by",
    "geometry.params.ellipsoid_cz",
    "geometry.params.elltube_ax",
    "geometry.params.elltube_by",
    "geometry.params.elltube_hz",
    "geometry.params.polyhedra_nsides",
    "geometry.params.bool_a_x",
    "geometry.params.bool_a_y",
    "geometry.params.bool_a_z",
    "geometry.params.bool_b_x",
    "geometry.params.bool_b_y",
    "geometry.params.bool_b_z",
}

_FINITE_NUMERIC_PATHS = {
    "geometry.params.z1",
    "geometry.params.z2",
    "geometry.params.z3",
    "geometry.params.tilt_x",
    "geometry.params.tilt_y",
    "geometry.params.para_alpha",
    "geometry.params.para_theta",
    "geometry.params.para_phi",
    "geometry.params.tx",
    "geometry.params.ty",
    "geometry.params.tz",
    "geometry.params.rx",
    "geometry.params.ry",
    "geometry.params.rz",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (dict, list)) and len(value) == 0:
        return True
    return False


def _report(ok: bool, errors: list[dict], missing: list[str], warnings: list[dict] | None = None) -> ValidationReport:
    return ValidationReport(
        ok=ok,
        errors=errors,
        warnings=warnings or [],
        missing_required_paths=missing,
        first_error=errors[0] if errors else None,
        candidate_rejection_reason_codes=sorted({e["code"] for e in errors}),
    )


def validate_layer_a_params(config: dict, registry: dict | None = None) -> ValidationReport:
    errors: list[dict] = []
    missing: list[str] = []

    for path in sorted(_POSITIVE_NUMERIC_PATHS | _FINITE_NUMERIC_PATHS):
        val = get_path(config, path)
        if _is_missing(val):
            continue
        if not isinstance(val, (int, float)):
            errors.append({"code": E_TYPE_INVALID, "path": path, "message": "must be number"})
            continue
        if path in _POSITIVE_NUMERIC_PATHS and float(val) <= 0:
            errors.append({"code": E_RANGE_INVALID, "path": path, "message": "must be > 0"})

    energy = get_path(config, "source.energy")
    if not _is_missing(energy):
        if not isinstance(energy, (int, float)):
            errors.append({"code": E_TYPE_INVALID, "path": "source.energy", "message": "must be number"})
        elif float(energy) <= 0:
            errors.append({"code": E_RANGE_INVALID, "path": "source.energy", "message": "must be > 0"})

    source_type = get_path(config, "source.type")
    if _is_missing(source_type):
        missing.append("source.type")
    elif str(source_type) not in {"point", "beam", "plane", "isotropic"}:
        errors.append({"code": E_TYPE_INVALID, "path": "source.type", "message": "invalid enum"})

    ok = len(errors) == 0
    return _report(ok, errors, missing)


def validate_layer_b_consistency(config: dict, registry: dict | None = None) -> ValidationReport:
    errors: list[dict] = []
    missing: list[str] = []
    warnings: list[dict] = []

    _, scope_errors = prune_out_of_scope_params(config, family_registry=registry)
    warnings.extend(scope_errors)

    structure = str(get_path(config, "geometry.structure", "") or "")
    family = get_geometry_family(structure)

    for path in family.get("required_paths", set()):
        if _is_missing(get_path(config, path)):
            missing.append(path)
            errors.append({"code": E_REQUIRED_MISSING, "path": path, "message": "required by structure family"})

    # name-binding: map key must match geometry root name when available
    vmap = get_path(config, "materials.volume_material_map", {})
    if isinstance(vmap, dict) and vmap:
        geo_root = str(get_path(config, "geometry.root_name", "") or "").strip()
        if not geo_root:
            graph_program = get_path(config, "geometry.graph_program", {})
            if isinstance(graph_program, dict):
                geo_root = str(graph_program.get("root", "") or "").strip()
        if not geo_root:
            # Default root aliases for current prototype
            geo_root = "box" if structure == "single_box" else "target"
        if geo_root not in vmap:
            errors.append(
                {
                    "code": E_NAME_BINDING,
                    "path": "materials.volume_material_map",
                    "message": f"volume map key must include geometry root '{geo_root}'",
                }
            )

    ok = len(errors) == 0
    return _report(ok, errors, missing, warnings)


def validate_layer_c_completeness(config: dict, minimal_schema: dict | None = None) -> ValidationReport:
    errors: list[dict] = []
    missing: list[str] = []

    required_paths = get_minimal_required_paths(config)
    for path in required_paths:
        if _is_missing(get_path(config, path)):
            missing.append(path)
            errors.append({"code": E_REQUIRED_MISSING, "path": path, "message": "minimal schema required"})

    ok = len(errors) == 0
    return _report(ok, errors, missing)


def merge_reports(*reports: ValidationReport) -> ValidationReport:
    errors: list[dict] = []
    warnings: list[dict] = []
    missing: list[str] = []
    for rep in reports:
        errors.extend(rep.errors)
        warnings.extend(rep.warnings)
        missing.extend(rep.missing_required_paths)
    dedup_missing: list[str] = []
    seen = set()
    for item in missing:
        if item not in seen:
            seen.add(item)
            dedup_missing.append(item)
    ok = len(errors) == 0
    return _report(ok, errors, dedup_missing, warnings)


def validate_all(config: dict, registry: dict | None = None, minimal_schema: dict | None = None) -> ValidationReport:
    a = validate_layer_a_params(config, registry=registry)
    b = validate_layer_b_consistency(config, registry=registry)
    c = validate_layer_c_completeness(config, minimal_schema=minimal_schema)
    return merge_reports(a, b, c)
