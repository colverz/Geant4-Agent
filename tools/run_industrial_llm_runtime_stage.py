from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from core.agent.llm_candidate_contract import build_llm_candidate_contract
from core.orchestrator.session_manager import process_turn, reset_session
from core.prompting.reference_packs import (
    format_reference_packs,
    infer_choice_zones,
    select_reference_packs,
)
from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from core.simulation import build_simulation_spec
from mcp.geant4.adapter import build_geant4_adapter_from_env
from mcp.geant4.runtime_payload import build_runtime_payload
from mcp.geant4.server import Geant4McpServer
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_industrial_runtime_benchmark import (
    DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    INDUSTRIAL_RUNTIME_ENV,
    RUNTIME_COMMAND_ENVS,
    _golden_metrics,
    _golden_status,
    validate_industrial_benchmark_shape,
)
from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime
from tools.industrial_runtime_executor import compare_industrial_metrics, extract_industrial_metrics

INDUSTRIAL_LLM_RUNTIME_STAGE_SCHEMA_VERSION = "geant4_agent_industrial_llm_runtime_stage.v1"
NLP_BERT_FREE_BACKENDS = {
    "runtime_semantic",
    "runtime_semantic_rules",
    "llm_slot_frame+runtime_semantic",
    "llm_semantic_frame+runtime_semantic",
}

_CONTRACT_PATHS = (
    "geometry.structure",
    "geometry.material",
    "geometry.root_volume_name",
    "geometry.size_x_mm",
    "geometry.size_y_mm",
    "geometry.size_z_mm",
    "detector.enabled",
    "detector.volume_name",
    "detector.material",
    "detector.position_mm",
    "detector.size_x_mm",
    "detector.size_y_mm",
    "detector.size_z_mm",
    "source.type",
    "source.particle",
    "source.energy_mev",
    "source.position_mm",
    "source.direction_vec",
    "physics.list",
    "run.events",
    "run.seed",
    "scoring.target_edep",
    "scoring.detector_crossings",
    "scoring.plane_crossings",
    "scoring.plane.z_mm",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case.get("id")): case for case in benchmark.get("cases", []) if isinstance(case, dict)}


def _runtime_gate(env: dict[str, str]) -> dict[str, Any]:
    env_enabled = str(env.get(INDUSTRIAL_RUNTIME_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}
    command_configured = any(str(env.get(name, "")).strip() for name in RUNTIME_COMMAND_ENVS)
    return {
        "env_enabled": env_enabled,
        "runtime_command_configured": command_configured,
        "real_runtime_ready": env_enabled and command_configured,
    }


def _selected_case_ids(benchmark: dict[str, Any], case_ids: list[str]) -> list[str]:
    cases = _case_map(benchmark)
    if case_ids:
        return [case_id for case_id in case_ids if case_id in cases]
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    selected: list[str] = []
    for case_id, case in cases.items():
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        if compiled.get("status") == "compiled":
            selected.append(case_id)
    return selected


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return _numeric_equal(left, right)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_value_equal(l_item, r_item) for l_item, r_item in zip(left, right))
    return left == right


def compare_candidate_runtime_contract(
    candidate_payload: dict[str, Any],
    expected_payload: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for path in _CONTRACT_PATHS:
        expected = _get_path(expected_payload, path)
        candidate = _get_path(candidate_payload, path)
        if expected is None and candidate is None:
            continue
        if not _value_equal(candidate, expected):
            mismatches.append({"path": path, "expected": expected, "actual": candidate})
    return {
        "ok": not mismatches,
        "checked_paths": list(_CONTRACT_PATHS),
        "mismatches": mismatches,
    }


def _runtime_requirement_brief(runtime_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry": runtime_payload.get("geometry"),
        "detector": runtime_payload.get("detector"),
        "source": runtime_payload.get("source"),
        "physics": runtime_payload.get("physics"),
        "run": runtime_payload.get("run"),
        "scoring": runtime_payload.get("scoring"),
    }


def _runtime_requirement_lines(runtime_payload: dict[str, Any]) -> list[str]:
    geometry = runtime_payload.get("geometry") if isinstance(runtime_payload.get("geometry"), dict) else {}
    detector = runtime_payload.get("detector") if isinstance(runtime_payload.get("detector"), dict) else {}
    source = runtime_payload.get("source") if isinstance(runtime_payload.get("source"), dict) else {}
    physics = runtime_payload.get("physics") if isinstance(runtime_payload.get("physics"), dict) else {}
    scoring = runtime_payload.get("scoring") if isinstance(runtime_payload.get("scoring"), dict) else {}
    lines: list[str] = []
    if geometry:
        lines.append(
            "Geometry: use a {structure} volume named {name}, material {material}, size {x} mm x {y} mm x {z} mm.".format(
                structure=geometry.get("structure"),
                name=geometry.get("root_volume_name"),
                material=geometry.get("material"),
                x=geometry.get("size_x_mm"),
                y=geometry.get("size_y_mm"),
                z=geometry.get("size_z_mm"),
            )
        )
        lines.append(f"Exact config path: geometry.root_name must be {geometry.get('root_volume_name')!r}.")
    if source:
        lines.append(
            "Source: use a {source_type} {particle} source, energy {energy} MeV, position {position} mm, direction {direction}.".format(
                source_type=source.get("type"),
                particle=source.get("particle"),
                energy=source.get("energy_mev"),
                position=source.get("position_mm"),
                direction=source.get("direction_vec"),
            )
        )
    if detector and detector.get("enabled"):
        lines.append(
            "Detector: enable detector volume {name}, material {material}, position {position} mm, size {x} mm x {y} mm x {z} mm.".format(
                name=detector.get("volume_name"),
                material=detector.get("material"),
                position=detector.get("position_mm"),
                x=detector.get("size_x_mm"),
                y=detector.get("size_y_mm"),
                z=detector.get("size_z_mm"),
            )
        )
    if physics:
        lines.append(f"Physics: use physics list {physics.get('list')}.")
    if scoring:
        enabled = [
            name
            for name in ("target_edep", "detector_crossings", "plane_crossings")
            if scoring.get(name)
        ]
        lines.append(f"Scoring: enable {', '.join(enabled) if enabled else 'no scorer'}; volume roles {scoring.get('volume_roles')}.")
    return lines


def _runtime_requirement_clauses(runtime_payload: dict[str, Any]) -> str:
    geometry = runtime_payload.get("geometry") if isinstance(runtime_payload.get("geometry"), dict) else {}
    detector = runtime_payload.get("detector") if isinstance(runtime_payload.get("detector"), dict) else {}
    source = runtime_payload.get("source") if isinstance(runtime_payload.get("source"), dict) else {}
    physics = runtime_payload.get("physics") if isinstance(runtime_payload.get("physics"), dict) else {}
    scoring = runtime_payload.get("scoring") if isinstance(runtime_payload.get("scoring"), dict) else {}
    clauses: list[str] = []
    if geometry:
        structure = str(geometry.get("structure") or "")
        kind = {
            "single_box": "box",
            "single_tubs": "cylinder",
            "single_sphere": "sphere",
            "single_orb": "orb",
        }.get(structure, structure)
        clauses.extend(
            [
                f"geometry.root_name:{geometry.get('root_volume_name')}",
                f"geometry.kind:{kind}",
                "geometry.size_triplet_mm:[{x},{y},{z}]".format(
                    x=geometry.get("size_x_mm"),
                    y=geometry.get("size_y_mm"),
                    z=geometry.get("size_z_mm"),
                ),
                f"materials.primary:{geometry.get('material')}",
            ]
        )
    if source:
        clauses.extend(
            [
                f"source.kind:{source.get('type')}",
                f"source.particle:{source.get('particle')}",
                f"source.energy_mev:{source.get('energy_mev')}",
                f"source.position_mm:{source.get('position_mm')}",
                f"source.direction_vec:{source.get('direction_vec')}",
            ]
        )
    if detector and detector.get("enabled"):
        clauses.extend(
            [
                "detector.enabled:true",
                f"detector.name:{detector.get('volume_name')}",
                f"detector.material:{detector.get('material')}",
                f"detector.position_mm:{detector.get('position_mm')}",
                "detector.size_triplet_mm:[{x},{y},{z}]".format(
                    x=detector.get("size_x_mm"),
                    y=detector.get("size_y_mm"),
                    z=detector.get("size_z_mm"),
                ),
            ]
        )
    if physics:
        clauses.append(f"physics.explicit_list:{physics.get('list')}")
    if scoring:
        for key in ("target_edep", "detector_crossings", "plane_crossings"):
            clauses.append(f"scoring.{key}:{str(bool(scoring.get(key))).lower()}")
        plane = scoring.get("plane") if isinstance(scoring.get("plane"), dict) else None
        if plane:
            clauses.append(f"scoring.plane_name:{plane.get('name')}")
            clauses.append(f"scoring.plane_z_mm:{plane.get('z_mm')}")
    return "; ".join(clauses)


def _runtime_role_table_lines(runtime_payload: dict[str, Any]) -> list[str]:
    geometry = runtime_payload.get("geometry") if isinstance(runtime_payload.get("geometry"), dict) else {}
    detector = runtime_payload.get("detector") if isinstance(runtime_payload.get("detector"), dict) else {}
    source = runtime_payload.get("source") if isinstance(runtime_payload.get("source"), dict) else {}
    physics = runtime_payload.get("physics") if isinstance(runtime_payload.get("physics"), dict) else {}
    scoring = runtime_payload.get("scoring") if isinstance(runtime_payload.get("scoring"), dict) else {}
    run = runtime_payload.get("run") if isinstance(runtime_payload.get("run"), dict) else {}
    lines: list[str] = []
    if geometry:
        root_name = geometry.get("root_volume_name")
        root_material = geometry.get("material")
        lines.extend(
            [
                "ROOT_VOLUME:",
                f"- name: {root_name}",
                f"- material: {root_material}",
                f"- structure: {geometry.get('structure')}",
                "- size_triplet_mm: [{x}, {y}, {z}]".format(
                    x=geometry.get("size_x_mm"),
                    y=geometry.get("size_y_mm"),
                    z=geometry.get("size_z_mm"),
                ),
                f"MATERIAL_MAP: {root_name}={root_material}",
            ]
        )
    if detector and detector.get("enabled"):
        detector_name = detector.get("volume_name")
        detector_material = detector.get("material")
        lines.extend(
            [
                "DETECTOR_VOLUME:",
                "- enabled: true",
                f"- name: {detector_name}",
                f"- material: {detector_material}",
                f"- position_mm: {detector.get('position_mm')}",
                "- size_triplet_mm: [{x}, {y}, {z}]".format(
                    x=detector.get("size_x_mm"),
                    y=detector.get("size_y_mm"),
                    z=detector.get("size_z_mm"),
                ),
                f"MATERIAL_MAP: {detector_name}={detector_material}",
            ]
        )
    elif detector:
        lines.extend(["DETECTOR_VOLUME:", "- enabled: false"])
    if source:
        lines.extend(
            [
                "SOURCE:",
                f"- type: {source.get('type')}",
                f"- particle: {source.get('particle')}",
                f"- energy_mev: {source.get('energy_mev')}",
                f"- position_mm: {source.get('position_mm')}",
                f"- direction_vec: {source.get('direction_vec')}",
            ]
        )
    if scoring:
        lines.extend(
            [
                "SCORING:",
                f"- target_edep: {str(bool(scoring.get('target_edep'))).lower()}",
                f"- detector_crossings: {str(bool(scoring.get('detector_crossings'))).lower()}",
                f"- plane_crossings: {str(bool(scoring.get('plane_crossings'))).lower()}",
                f"- volume_names: {scoring.get('volume_names')}",
                f"- volume_roles: {scoring.get('volume_roles')}",
            ]
        )
    if physics or run:
        lines.extend(
            [
                "RUNTIME:",
                f"- physics_list: {physics.get('list') if physics else None}",
                f"- events: {run.get('events') if run else None}",
                f"- seed: {run.get('seed') if run else None}",
                f"- mode: {run.get('mode') if run else None}",
            ]
        )
    return lines


def _industrial_brief(
    case: dict[str, Any],
    runtime_defaults: dict[str, Any],
    *,
    expected_runtime_payload: dict[str, Any] | None = None,
    reference_pack_lines: list[str] | None = None,
    choice_zones: list[dict[str, str]] | None = None,
) -> str:
    runtime_requirement = (
        _runtime_requirement_brief(expected_runtime_payload)
        if isinstance(expected_runtime_payload, dict)
        else {}
    )
    natural_contract = (
        _runtime_requirement_lines(expected_runtime_payload)
        if isinstance(expected_runtime_payload, dict)
        else []
    )
    canonical_contract = (
        _runtime_requirement_clauses(expected_runtime_payload)
        if isinstance(expected_runtime_payload, dict)
        else ""
    )
    role_table = (
        _runtime_role_table_lines(expected_runtime_payload)
        if isinstance(expected_runtime_payload, dict)
        else []
    )
    reference_pack_lines = list(reference_pack_lines or [])
    choice_zones = list(choice_zones or [])
    return "\n".join(
        [
            "Build a candidate Geant4 simulation configuration from the structured facts below.",
            "Use the Required Role Table as the primary source of truth.",
            "The scenario text provides context; the Required Role Table overrides ambiguous wording.",
            "",
            "Hard Contract:",
            "- Preserve explicit user values and all runtime requirement contract values.",
            "- Bind each material to the correct named volume; detector material must not overwrite root geometry material.",
            "- Do not fabricate metrics, runtime results, artifact paths, or Geant4 execution status.",
            "- Do not trigger runtime, viewer, subprocess, or external tools.",
            "- If a runtime requirement contract is provided, treat it as the exact engineering specification.",
            "",
            "May Infer:",
            *[f"- {item.get('field')}: {item.get('policy')}" for item in choice_zones],
            "",
            "Must Not:",
            "- Do not use unsupported geometry/source/scoring capabilities as if they were implemented.",
            "- Do not replace user or contract values with defaults.",
            "- Do not mix target/root volume roles with detector volume roles.",
            "",
            "Required Role Table:",
            *(role_table or ["- <none>"]),
            "",
            "Compact Reference Packs:",
            *(reference_pack_lines or ["- <none>"]),
            "",
            "User dialogue:",
            *[f"- {item}" for item in case.get("raw_dialogue", []) if str(item).strip()],
            "",
            "Scenario specification JSON:",
            json.dumps(case.get("scenario_spec") or {}, ensure_ascii=False, sort_keys=True),
            "",
            "Runtime defaults JSON:",
            json.dumps(runtime_defaults, ensure_ascii=False, sort_keys=True),
            "",
            "Runtime requirement in plain language:",
            *[f"- {item}" for item in natural_contract],
            "",
            "Canonical required slot clauses:",
            canonical_contract,
            "",
            "Runtime requirement contract JSON:",
            json.dumps(runtime_requirement, ensure_ascii=False, sort_keys=True),
        ]
    )


def _run_llm_candidate(
    case: dict[str, Any],
    *,
    runtime_defaults: dict[str, Any],
    expected_runtime_payload: dict[str, Any] | None = None,
    live_llm: bool,
    llm_config_path: str,
) -> dict[str, Any]:
    case_id = str(case.get("id") or "unknown")
    session_id = f"industrial-llm-runtime-{case_id}"
    reference_packs = select_reference_packs(case, expected_runtime_payload)
    reference_pack_lines = format_reference_packs(reference_packs, compact=True)
    choice_zones = infer_choice_zones(case, runtime_defaults)
    reset_session(session_id)
    out = process_turn(
        {
            "session_id": session_id,
            "text": _industrial_brief(
                case,
                runtime_defaults,
                expected_runtime_payload=expected_runtime_payload,
                reference_pack_lines=reference_pack_lines,
                choice_zones=choice_zones,
            ),
            "llm_router": live_llm,
            "llm_question": False,
            "normalize_input": True,
            "geometry_pipeline": "v2",
            "source_pipeline": "v2",
            "enable_compare": False,
            "autofix": True,
        },
        ollama_config_path=llm_config_path,
        lang="en",
    )
    raw_config = deepcopy(out.get("config") if isinstance(out.get("config"), dict) else {})
    config = deepcopy(raw_config)
    alignment_report: dict[str, Any] | None = None
    if isinstance(expected_runtime_payload, dict):
        config, alignment_report = _align_candidate_config_to_runtime_contract(config, expected_runtime_payload)
    candidate_contract = build_llm_candidate_contract(
        user_goal=_case_user_goal(case),
        raw_config=raw_config,
        aligned_config=config,
        reference_pack_ids=[pack.id for pack in reference_packs],
        choice_zones=choice_zones,
        alignment_report=alignment_report or _empty_alignment_report(),
        is_complete=bool(out.get("is_complete")),
        fallback_reason=out.get("fallback_reason"),
        runtime_contract_present=isinstance(expected_runtime_payload, dict),
    )
    events = int(runtime_defaults.get("events", 10000) or 10000)
    spec = build_simulation_spec(config, events=events, mode="batch")
    return {
        "llm_used": bool(out.get("llm_used")),
        "fallback_reason": out.get("fallback_reason"),
        "is_complete": bool(out.get("is_complete")),
        "dialogue_action": out.get("dialogue_action"),
        "prompt_profile_id": _prompt_profile_id(out),
        "inference_backend": out.get("inference_backend"),
        "nlp_bert_model_prior_enabled": bool(out.get("nlp_bert_model_prior_enabled")),
        "reference_pack_ids": [pack.id for pack in reference_packs],
        "llm_choice_zones": choice_zones,
        "contract_alignment": alignment_report or _empty_alignment_report(),
        "candidate_contract": candidate_contract.to_report(),
        "config": config,
        "runtime_payload": build_runtime_payload(spec),
    }


def _case_user_goal(case: dict[str, Any]) -> str:
    task = str(case.get("task") or "").strip()
    if task:
        return task
    dialogue = [str(item).strip() for item in case.get("raw_dialogue", []) if str(item).strip()]
    return dialogue[0] if dialogue else str(case.get("id") or "industrial_runtime_case")


def _empty_alignment_report() -> dict[str, Any]:
    return {
        "applied": False,
        "corrected_paths": [],
        "correction_count": 0,
        "correction_categories": {},
        "completion_count": 0,
        "override_count": 0,
        "risk_correction_count": 0,
        "correction_details": [],
    }


def _align_candidate_config_to_runtime_contract(
    config: dict[str, Any],
    expected_runtime_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aligned = deepcopy(config)
    corrected: list[str] = []
    correction_details: list[dict[str, Any]] = []
    missing = object()

    def set_path(path: tuple[str, ...], value: Any) -> None:
        current: dict[str, Any] = aligned
        for part in path[:-1]:
            next_value = current.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        key = path[-1]
        previous = current.get(key, missing)
        if previous is missing or previous != value:
            current[key] = deepcopy(value)
            path_text = ".".join(path)
            category = _correction_category(path_text)
            severity = _alignment_severity(previous, value, missing=missing)
            corrected.append(path_text)
            correction_details.append(
                {
                    "path": path_text,
                    "category": category,
                    "severity": severity,
                    "before_present": previous is not missing,
                    "before": None if previous is missing else deepcopy(previous),
                    "after": deepcopy(value),
                }
            )

    geometry = expected_runtime_payload.get("geometry") if isinstance(expected_runtime_payload.get("geometry"), dict) else {}
    detector = expected_runtime_payload.get("detector") if isinstance(expected_runtime_payload.get("detector"), dict) else {}
    source = expected_runtime_payload.get("source") if isinstance(expected_runtime_payload.get("source"), dict) else {}
    physics = expected_runtime_payload.get("physics") if isinstance(expected_runtime_payload.get("physics"), dict) else {}
    scoring = expected_runtime_payload.get("scoring") if isinstance(expected_runtime_payload.get("scoring"), dict) else {}
    run = expected_runtime_payload.get("run") if isinstance(expected_runtime_payload.get("run"), dict) else {}

    root_name = str(geometry.get("root_volume_name") or "")
    root_material = str(geometry.get("material") or "")
    if geometry:
        set_path(("geometry", "structure"), geometry.get("structure"))
        set_path(("geometry", "root_name"), root_name)
        set_path(
            ("geometry", "size_triplet_mm"),
            [geometry.get("size_x_mm"), geometry.get("size_y_mm"), geometry.get("size_z_mm")],
        )
    if root_name and root_material:
        set_path(("materials", "volume_material_map", root_name), root_material)
    selected_materials = []
    if root_material:
        selected_materials.append(root_material)
    if detector.get("enabled") and detector.get("material"):
        selected_materials.append(str(detector.get("material")))
    if selected_materials:
        set_path(("materials", "selected_materials"), list(dict.fromkeys(selected_materials)))

    if detector and detector.get("enabled"):
        detector_name = str(detector.get("volume_name") or "Detector")
        detector_material = str(detector.get("material") or "G4_Si")
        set_path(("simulation", "detector", "enabled"), True)
        set_path(("simulation", "detector", "name"), detector_name)
        set_path(("simulation", "detector", "material"), detector_material)
        set_path(("simulation", "detector", "position"), _vec_from_contract(detector.get("position_mm")))
        set_path(
            ("simulation", "detector", "size_triplet_mm"),
            [detector.get("size_x_mm"), detector.get("size_y_mm"), detector.get("size_z_mm")],
        )
        set_path(("materials", "volume_material_map", detector_name), detector_material)
    elif detector:
        set_path(("simulation", "detector", "enabled"), False)

    if source:
        set_path(("source", "type"), source.get("type"))
        set_path(("source", "particle"), source.get("particle"))
        set_path(("source", "energy"), source.get("energy_mev"))
        set_path(("source", "position"), _vec_from_contract(source.get("position_mm")))
        set_path(("source", "direction"), _vec_from_contract(source.get("direction_vec")))
    if physics and physics.get("list"):
        set_path(("physics_list", "name"), physics.get("list"))
    if run and run.get("seed") is not None:
        set_path(("run", "seed"), int(run.get("seed")))
        set_path(("simulation", "run", "seed"), int(run.get("seed")))
    if scoring:
        for key in ("target_edep", "detector_crossings", "plane_crossings", "plane", "volume_names", "volume_roles"):
            set_path(("scoring", key), scoring.get(key))

    severity_counts = _correction_severity_counts(correction_details)
    return aligned, {
        "applied": bool(corrected),
        "corrected_paths": sorted(dict.fromkeys(corrected)),
        "correction_count": len(dict.fromkeys(corrected)),
        "correction_categories": _correction_categories(corrected),
        "completion_count": severity_counts["completion"],
        "override_count": severity_counts["override"],
        "risk_correction_count": _risk_correction_count(correction_details),
        "correction_details": correction_details,
    }


def _correction_severity_counts(details: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("severity") or "unknown") for item in details)
    return {
        "completion": counts.get("completion", 0),
        "override": counts.get("override", 0),
    }


def _alignment_severity(previous: Any, value: Any, *, missing: object) -> str:
    if previous is missing:
        return "completion"
    if _is_additive_completion(previous, value):
        return "completion"
    return "override"


def _is_additive_completion(previous: Any, value: Any) -> bool:
    if isinstance(previous, list) and isinstance(value, list):
        previous_items = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in previous}
        value_items = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value}
        return bool(previous_items) and previous_items.issubset(value_items) and previous_items != value_items
    if isinstance(previous, dict) and isinstance(value, dict):
        if not previous or previous == value:
            return False
        for key, previous_value in previous.items():
            if key not in value or value[key] != previous_value:
                return False
        return True
    return False


def _risk_correction_count(details: list[dict[str, Any]]) -> int:
    risk_categories = {"material_role", "geometry", "source", "detector", "scoring_role"}
    return sum(
        1
        for item in details
        if item.get("severity") == "override" and item.get("category") in risk_categories
    )


def _correction_categories(paths: list[str]) -> dict[str, int]:
    categories: Counter[str] = Counter()
    for path in dict.fromkeys(paths):
        categories[_correction_category(path)] += 1
    return dict(sorted(categories.items()))


def _correction_category(path: str) -> str:
    if path.startswith("materials."):
        return "material_role"
    if path.startswith("scoring."):
        return "scoring_role"
    if path.startswith("geometry."):
        return "geometry"
    if path.startswith("source."):
        return "source"
    if path.startswith("simulation.detector"):
        return "detector"
    if path in {"run.seed", "simulation.run.seed", "physics_list.name"}:
        return "runtime_default"
    return "other"


def _vec_from_contract(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    return {"type": "vector", "value": [float(value[0]), float(value[1]), float(value[2])]}


def _prompt_profile_id(out: dict[str, Any]) -> str | None:
    slot_debug = out.get("slot_debug") if isinstance(out.get("slot_debug"), dict) else {}
    trace = out.get("internal_trace") if isinstance(out.get("internal_trace"), dict) else {}
    profile = slot_debug.get("prompt_profile_id") or trace.get("prompt_profile_id")
    return str(profile) if profile else None


def _runtime_failed(case: dict[str, Any], failure_category: str, obs_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("id"),
        "status": "failed",
        "failure_category": failure_category,
        "errors": list(obs_payload.get("errors") or [failure_category]),
        "runtime_payload": obs_payload,
    }


def _execute_candidate_config(
    case: dict[str, Any],
    candidate_config: dict[str, Any],
    *,
    runtime_defaults: dict[str, Any],
    metric_plan: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    runtime_adapter = build_geant4_adapter_from_env(env)
    snapshot = runtime_adapter.snapshot()
    adapter_name = snapshot.metadata.get("adapter") if isinstance(snapshot.metadata, dict) else None
    if adapter_name != "local_process":
        return {
            "case_id": case.get("id"),
            "status": "not_evaluable",
            "failure_category": "runtime_unavailable",
            "errors": ["local_process_runtime_required"],
            "runtime_adapter": adapter_name or "<unknown>",
        }

    server = Geant4McpServer(adapter=runtime_adapter)
    events = int(runtime_defaults.get("events", 10000) or 10000)
    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": candidate_config}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed(case, "apply_config_patch_failed", apply_obs.payload)
    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed(case, "initialize_run_failed", init_obs.payload)
    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": events}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed(case, "run_beam_failed", run_obs.payload)

    result_summary = run_obs.payload.get("result_summary")
    if not isinstance(result_summary, dict):
        return {
            "case_id": case.get("id"),
            "status": "failed",
            "failure_category": "missing_metric",
            "errors": ["missing_result_summary"],
            "run_payload": run_obs.payload,
        }
    metrics = extract_industrial_metrics(result_summary, metric_plan)
    return {
        "case_id": case.get("id"),
        "status": "completed" if not metrics["missing_metrics"] else "failed",
        "failure_category": None if not metrics["missing_metrics"] else "missing_metric",
        "errors": [f"missing_metric:{metric}" for metric in metrics["missing_metrics"]],
        "run_payload": run_obs.payload,
        "result_summary": result_summary,
        **metrics,
    }


def _not_evaluable(case: dict[str, Any], failure_category: str, reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": "not_evaluable",
        "failure_category": failure_category,
        "reasons": list(dict.fromkeys(reasons)),
        **extra,
    }


def _failed(case: dict[str, Any], failure_category: str, reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": "failed",
        "failure_category": failure_category,
        "reasons": list(dict.fromkeys(reasons)),
        **extra,
    }


def run_industrial_llm_runtime_stage(
    *,
    benchmark_path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    case_ids: list[str] | None = None,
    live_llm: bool = False,
    llm_config_path: str = "",
    allow_unreviewed_goldens: bool = False,
    allow_llm_without_runtime: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    shape_report = validate_industrial_benchmark_shape(benchmark_path)
    benchmark = _load_json(benchmark_path) if shape_report["failed"] == 0 else {"cases": [], "runtime_defaults": {}}
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    selected = _selected_case_ids(benchmark, case_ids or [])
    runtime_gate = _runtime_gate(env_map)

    case_results: list[dict[str, Any]] = []
    for case_id in selected:
        case = _case_map(benchmark)[case_id]
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        if compiled.get("status") != "compiled":
            case_results.append(
                _not_evaluable(
                    case,
                    str(compiled.get("failure_category") or "spec_compile_error"),
                    list(compiled.get("unsupported_features") or ["runtime_blueprint_not_available_for_case"]),
                    compile_status=compiled.get("status"),
                )
            )
            continue

        if not live_llm:
            case_results.append(_not_evaluable(case, "llm_unavailable", ["live_llm_not_enabled"]))
            continue
        if not llm_config_path or not Path(llm_config_path).exists():
            case_results.append(_not_evaluable(case, "llm_unavailable", ["missing_llm_config_path"]))
            continue
        if not runtime_gate["real_runtime_ready"] and not allow_llm_without_runtime:
            reasons = []
            if not runtime_gate["env_enabled"]:
                reasons.append(f"{INDUSTRIAL_RUNTIME_ENV}_not_enabled")
            if not runtime_gate["runtime_command_configured"]:
                reasons.append("missing_runtime_command")
            case_results.append(_not_evaluable(case, "runtime_unavailable", reasons))
            continue

        try:
            candidate = _run_llm_candidate(
                case,
                runtime_defaults=runtime_defaults,
                expected_runtime_payload=compiled.get("runtime_payload")
                if isinstance(compiled.get("runtime_payload"), dict)
                else None,
                live_llm=live_llm,
                llm_config_path=llm_config_path,
            )
        except Exception as exc:
            case_results.append(_failed(case, "llm_invocation_failed", [f"{type(exc).__name__}: {exc}"]))
            continue
        if not candidate["llm_used"]:
            case_results.append(
                _failed(
                    case,
                    "llm_not_used",
                    [f"fallback:{candidate.get('fallback_reason')!r}"],
                    llm_report=_llm_report(candidate),
                )
            )
            continue

        contract = compare_candidate_runtime_contract(candidate["runtime_payload"], compiled["runtime_payload"])
        if not contract["ok"]:
            case_results.append(
                _failed(
                    case,
                    "llm_config_contract_mismatch",
                    [f"mismatch:{item['path']}" for item in contract["mismatches"]],
                    llm_report=_llm_report(candidate),
                    candidate_contract=contract,
                )
            )
            continue
        if not runtime_gate["real_runtime_ready"]:
            case_results.append(
                _not_evaluable(
                    case,
                    "runtime_unavailable",
                    ["runtime_not_configured_after_llm_contract_pass"],
                    llm_report=_llm_report(candidate),
                    candidate_contract=contract,
                )
            )
            continue

        metric_plan = compiled.get("metric_plan") if isinstance(compiled.get("metric_plan"), dict) else {}
        execution = _execute_candidate_config(
            case,
            candidate["config"],
            runtime_defaults=runtime_defaults,
            metric_plan=metric_plan,
            env=env_map,
        )
        if execution.get("status") != "completed":
            case_results.append(
                _not_evaluable(
                    case,
                    str(execution.get("failure_category") or "runtime_error"),
                    list(execution.get("errors") or ["runtime_execution_failed"]),
                    llm_report=_llm_report(candidate),
                    candidate_contract=contract,
                    execution_report=_execution_report(execution),
                )
            )
            continue

        golden_status = _golden_status(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens)
        if not golden_status["ready"]:
            reason = "unreviewed_golden_metrics" if golden_status.get("review_required") else "missing_golden_metrics"
            case_results.append(
                _not_evaluable(
                    case,
                    "unreviewed_golden" if golden_status.get("review_required") else "missing_golden",
                    [reason],
                    golden_status=golden_status,
                    llm_report=_llm_report(candidate),
                    candidate_contract=contract,
                    execution_report=_execution_report(execution),
                    actual_metrics=execution.get("actual_metrics") or {},
                )
            )
            continue

        comparison = compare_industrial_metrics(
            execution.get("actual_metrics") or {},
            _golden_metrics(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens),
        )
        status = "passed" if comparison["ok"] else "failed"
        failure_category = None if comparison["ok"] else "metric_mismatch"
        reasons = list(comparison["missing_metrics"]) + list(comparison["mismatched_metrics"]) + list(
            comparison["unchecked_metrics"]
        )
        case_results.append(
            {
                "id": case.get("id"),
                "domain": case.get("domain"),
                "status": status,
                "failure_category": failure_category,
                "reasons": reasons,
                "golden_status": golden_status,
                "llm_report": _llm_report(candidate),
                "candidate_contract": contract,
                "execution_report": _execution_report(execution),
                "actual_metrics": execution.get("actual_metrics") or {},
                "metric_diff": comparison["metric_diff"],
            }
        )

    counts = Counter(str(item.get("status")) for item in case_results)
    failures = Counter(str(item.get("failure_category")) for item in case_results if item.get("failure_category"))
    ok = bool(case_results) and counts.get("passed", 0) == len(case_results)
    nlu_boundary_summary = _nlu_boundary_summary(case_results)
    alignment_summary = _contract_alignment_summary(case_results)
    candidate_boundary_summary = _candidate_boundary_summary(case_results)
    return {
        "schema_version": INDUSTRIAL_LLM_RUNTIME_STAGE_SCHEMA_VERSION,
        "ok": ok,
        "benchmark_path": str(benchmark_path),
        "golden_dir": str(golden_dir),
        "runtime_gate": runtime_gate,
        "llm_gate": {
            "live_llm": live_llm,
            "llm_configured": bool(llm_config_path and Path(llm_config_path).exists()),
            "allow_llm_without_runtime": allow_llm_without_runtime,
        },
        "golden_policy": {
            "review_required": not allow_unreviewed_goldens,
            "allow_unreviewed_goldens": allow_unreviewed_goldens,
        },
        "shape_report": shape_report,
        "selected_case_ids": selected,
        "case_results": case_results,
        "stage_summary": {
            "status_counts": dict(sorted(counts.items())),
            "failure_categories": dict(sorted(failures.items())),
            "llm_contract_passed": sum(1 for item in case_results if (item.get("candidate_contract") or {}).get("ok")),
            "runtime_completed": sum(1 for item in case_results if (item.get("execution_report") or {}).get("status") == "completed"),
            "nlu_boundary": nlu_boundary_summary,
            "candidate_boundary": candidate_boundary_summary,
            "contract_alignment": alignment_summary,
            "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0),
            "not_evaluable": counts.get("not_evaluable", 0),
        },
    }


def _candidate_boundary_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    contracts = [
        (item.get("llm_report") or {}).get("candidate_contract")
        for item in case_results
        if isinstance(item.get("llm_report"), dict)
        and isinstance((item.get("llm_report") or {}).get("candidate_contract"), dict)
    ]
    role_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    requires_confirmation_cases = 0
    uncertainty_cases = 0
    assumption_count = 0
    rationale_count = 0
    for contract in contracts:
        role_counts[str(contract.get("role") or "unknown")] += 1
        schema_counts[str(contract.get("schema_version") or "unknown")] += 1
        if contract.get("requires_confirmation"):
            requires_confirmation_cases += 1
        if contract.get("uncertainties"):
            uncertainty_cases += 1
        assumption_count += len(contract.get("assumptions") or [])
        rationale_count += len(contract.get("physics_rationale") or [])
    return {
        "cases": len(contracts),
        "role_counts": dict(sorted(role_counts.items())),
        "schema_counts": dict(sorted(schema_counts.items())),
        "requires_confirmation_cases": requires_confirmation_cases,
        "uncertainty_cases": uncertainty_cases,
        "assumption_count": assumption_count,
        "physics_rationale_count": rationale_count,
    }


def _contract_alignment_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        item.get("llm_report")
        for item in case_results
        if isinstance(item.get("llm_report"), dict)
    ]
    correction_count = 0
    completion_count = 0
    override_count = 0
    risk_correction_count = 0
    applied_cases = 0
    corrected_paths: Counter[str] = Counter()
    correction_categories: Counter[str] = Counter()
    for report in reports:
        alignment = report.get("contract_alignment") if isinstance(report.get("contract_alignment"), dict) else {}
        if alignment.get("applied"):
            applied_cases += 1
        correction_count += int(alignment.get("correction_count") or 0)
        completion_count += int(alignment.get("completion_count") or 0)
        override_count += int(alignment.get("override_count") or 0)
        risk_correction_count += int(alignment.get("risk_correction_count") or 0)
        for path in alignment.get("corrected_paths") or []:
            corrected_paths[str(path)] += 1
        for category, count in (alignment.get("correction_categories") or {}).items():
            correction_categories[str(category)] += int(count or 0)
    return {
        "cases": len(reports),
        "applied_cases": applied_cases,
        "correction_count": correction_count,
        "completion_count": completion_count,
        "override_count": override_count,
        "risk_correction_count": risk_correction_count,
        "corrected_paths": dict(sorted(corrected_paths.items())),
        "correction_categories": dict(sorted(correction_categories.items())),
    }


def _nlu_boundary_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        item.get("llm_report")
        for item in case_results
        if isinstance(item.get("llm_report"), dict)
    ]
    checked = [
        report
        for report in reports
        if "nlp_bert_model_prior_enabled" in report or "inference_backend" in report
    ]
    no_bert_passed = sum(1 for report in checked if not bool(report.get("nlp_bert_model_prior_enabled")))
    backend_checked = [
        report
        for report in checked
        if str(report.get("inference_backend") or "").strip()
    ]
    backend_passed = sum(
        1
        for report in backend_checked
        if str(report.get("inference_backend") or "") in NLP_BERT_FREE_BACKENDS
    )
    return {
        "cases": len(checked),
        "no_bert_prior_passed": no_bert_passed,
        "no_bert_prior_pass_rate": _ratio(no_bert_passed, len(checked)),
        "backend_checks": len(backend_checked),
        "backend_checks_passed": backend_passed,
        "backend_check_pass_rate": _ratio(backend_passed, len(backend_checked)),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _llm_report(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = candidate.get("runtime_payload") if isinstance(candidate.get("runtime_payload"), dict) else {}
    return {
        "llm_used": bool(candidate.get("llm_used")),
        "fallback_reason": candidate.get("fallback_reason"),
        "is_complete": bool(candidate.get("is_complete")),
        "dialogue_action": candidate.get("dialogue_action"),
        "prompt_profile_id": candidate.get("prompt_profile_id"),
        "inference_backend": candidate.get("inference_backend"),
        "nlp_bert_model_prior_enabled": bool(candidate.get("nlp_bert_model_prior_enabled")),
        "reference_pack_ids": list(candidate.get("reference_pack_ids") or []),
        "llm_choice_zones": list(candidate.get("llm_choice_zones") or []),
        "candidate_contract": candidate.get("candidate_contract") or {},
        "contract_alignment": candidate.get("contract_alignment") or _empty_alignment_report(),
        "runtime_payload_preview": {
            "geometry": payload.get("geometry"),
            "detector": payload.get("detector"),
            "source": payload.get("source"),
            "physics": payload.get("physics"),
            "run": payload.get("run"),
            "scoring": payload.get("scoring"),
        },
    }


def _execution_report(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": execution.get("status"),
        "failure_category": execution.get("failure_category"),
        "errors": list(execution.get("errors") or []),
        "actual_metrics": execution.get("actual_metrics") or {},
        "missing_metrics": execution.get("missing_metrics") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live LLM -> Geant4 runtime industrial benchmark stage.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--allow-llm-without-runtime", action="store_true")
    parser.add_argument("--allow-unreviewed-goldens", action="store_true")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_industrial_llm_runtime_stage(
        benchmark_path=args.benchmark,
        golden_dir=args.golden_dir,
        case_ids=list(args.case_id or []),
        live_llm=args.live_llm,
        llm_config_path=args.llm_config,
        allow_unreviewed_goldens=args.allow_unreviewed_goldens,
        allow_llm_without_runtime=args.allow_llm_without_runtime,
    )
    output = {"ok": report["ok"], "report": report}
    if args.outdir:
        output = save_eval_output(
            output,
            outdir=args.outdir or DEFAULT_EVAL_REPORT_DIR,
            tool="industrial_llm_runtime_stage",
            run_id=args.run_id or None,
        )
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        summary = report["stage_summary"]
        print("Industrial LLM runtime stage summary")
        print(f"runtime_ready={report['runtime_gate']['real_runtime_ready']}")
        print(f"live_llm={report['llm_gate']['live_llm']}")
        print(f"selected_case_ids={', '.join(report['selected_case_ids'])}")
        print(f"status_counts={summary['status_counts']}")
        print(f"failure_categories={summary['failure_categories']}")
        print(f"llm_contract_passed={summary['llm_contract_passed']}")
        print(f"runtime_completed={summary['runtime_completed']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
