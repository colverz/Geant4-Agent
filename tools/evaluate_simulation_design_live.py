from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from core.agent.simulation_design import (
    ALLOWED_NEXT_ACTIONS,
    SIMULATION_DESIGN_SCHEMA_VERSION,
    build_simulation_design_reference_pack,
    check_simulation_design_capability,
)
from nlu.llm_support import ollama_client
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output


DEFAULT_CASEBANK = Path("docs/eval/simulation_design_live_casebank.json")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"casebank must be a list: {path}")
    return [case for case in payload if isinstance(case, dict)]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _build_prompt(case: dict[str, Any], reference_pack: dict[str, Any]) -> str:
    lang = str(case.get("lang") or "en")
    goal = str(case.get("goal") or "")
    language_rule = "Use Chinese for natural-language strings." if lang == "zh" else "Use English for natural-language strings."
    return (
        "You are designing a Geant4 simulation candidate for a bounded agent workflow.\n"
        "Return JSON only. Do not use markdown. Do not invent runtime capabilities.\n"
        "The candidate is read-only: do not claim that a simulation was run and do not create final config.\n"
        f"The output schema_version must be exactly {SIMULATION_DESIGN_SCHEMA_VERSION}. Do not copy the reference pack schema_version.\n"
        "recommended_setup must be a flat object: geometry, material, source, detector, scoring.\n"
        "geometry must be one canonical ID from: single_box, step_wedge, pipe, void, inclusion, multi_layer.\n"
        "A target slab plus a downstream detector is still geometry=single_box with detector set; do not call it multi_layer.\n"
        "source must be one canonical ID from: beam, point, isotropic.\n"
        "observables and recommended_setup.scoring must use canonical IDs only: target_edep, detector_crossing_count, detector_edep, plane_crossing_count, region_contrast, depth_bins, transmission_factor.\n"
        "transmission_factor is a derived observable. It is acceptable when detector_crossing_count or plane_crossing_count is included; do not mark transmission_factor itself unsupported.\n"
        "For detector response, include both detector_crossing_count and detector_edep when a detector volume is proposed.\n"
        "knowledge_references must use canonical IDs like materials:G4_Pb, geometry:pipe, scoring:depth_bins. Do not write prose in knowledge_references.\n"
        "If a requested capability is unsupported, put it in unsupported_capabilities and set next_action to unsupported_capability.\n"
        "Use simplifications only for unsupported-to-supported approximations that require approval, not for ordinary simple supported modeling choices.\n"
        "If an approximation is required, list it in simplifications, add user_decisions_required, and set next_action to ask_user_to_choose_approximation.\n"
        "If next_action is unsupported_capability, add at least one user_decisions_required item asking the user to choose a supported approximation or defer.\n"
        "If the setup is fully supported by current runtime capabilities, set next_action to build_candidate_config.\n"
        f"{language_rule}\n\n"
        "Allowed next_action values:\n"
        f"{_json_dumps(sorted(ALLOWED_NEXT_ACTIONS))}\n\n"
        "Output schema fields:\n"
        f"{_json_dumps(['schema_version','goal','recommended_setup','observables','assumptions','simplifications','unsupported_capabilities','user_decisions_required','knowledge_references','capability_check','next_action'])}\n\n"
        "Reference pack for choosing canonical IDs. This is not the output schema:\n"
        f"{_json_dumps(reference_pack)}\n\n"
        "User goal:\n"
        f"{goal}\n"
    )


def _checked_next_action(capability_check: dict[str, Any]) -> str:
    if capability_check.get("unsupported_capabilities"):
        return "unsupported_capability"
    if capability_check.get("requires_user_approval"):
        return "ask_user_to_choose_approximation"
    if capability_check.get("supported"):
        return "build_candidate_config"
    return "needs_more_information"


def _normalize_candidate(raw: dict[str, Any], goal: str) -> dict[str, Any]:
    setup = raw.get("recommended_setup") if isinstance(raw.get("recommended_setup"), dict) else {}
    setup = _normalize_setup(setup, raw.get("recommended_setup"))
    raw_observables = list(raw.get("observables") or [])
    raw_scoring = setup.get("scoring") if isinstance(setup.get("scoring"), list) else []
    observables = _canonical_observables(raw_observables + raw_scoring)
    unsupported = _filter_irrelevant_unsupported(
        _canonical_unsupported(raw.get("unsupported_capabilities") or []),
        goal=goal,
        observables=observables,
    )
    simplifications = [str(item) for item in raw.get("simplifications") or [] if str(item)]
    if any(item in unsupported for item in ("step_wedge_geometry", "curved_pipe_geometry")):
        for item in list(unsupported):
            if item in {"step_wedge_geometry", "curved_pipe_geometry"}:
                simplifications.append(f"Approximation required for {item}.")
                unsupported.remove(item)
    normalized = {
        "schema_version": SIMULATION_DESIGN_SCHEMA_VERSION,
        "goal": str(raw.get("goal") or goal),
        "recommended_setup": setup,
        "observables": observables,
        "assumptions": _string_list(raw.get("assumptions")),
        "simplifications": list(dict.fromkeys(simplifications)),
        "unsupported_capabilities": unsupported,
        "user_decisions_required": [str(item) for item in raw.get("user_decisions_required") or [] if str(item)],
        "knowledge_references": _canonical_references(raw.get("knowledge_references") or [], setup, observables, unsupported),
        "capability_check": raw.get("capability_check") if isinstance(raw.get("capability_check"), dict) else {},
        "next_action": str(raw.get("next_action") or "needs_more_information"),
    }
    if normalized["next_action"] not in ALLOWED_NEXT_ACTIONS:
        normalized["next_action"] = "needs_more_information"
    if normalized["unsupported_capabilities"] and not normalized["user_decisions_required"]:
        normalized["user_decisions_required"] = ["Choose a supported approximation or defer until runtime capability is added."]
    return normalized


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _filter_irrelevant_unsupported(unsupported: list[str], *, goal: str, observables: list[str]) -> list[str]:
    text = str(goal or "").lower()
    observable_set = set(observables)
    filtered: list[str] = []
    for item in unsupported:
        if item == "depth_binned_scoring" and "depth_bins" not in observable_set:
            if not any(token in text for token in ("depth", "dose", "bragg", "深度", "剂量")):
                continue
        if item == "region_contrast_scoring" and "region_contrast" not in observable_set:
            if not any(token in text for token in ("region", "contrast", "void", "inclusion", "区域", "对比", "空洞", "夹杂")):
                continue
        filtered.append(item)
    return list(dict.fromkeys(filtered))


def _canonical_from_text(value: Any, allowed: tuple[str, ...]) -> str:
    text = str(value or "").lower()
    for item in allowed:
        if item.lower() in text:
            return item
    return ""


def _normalize_setup(setup: dict[str, Any], raw_setup: Any) -> dict[str, Any]:
    text = json.dumps(raw_setup, ensure_ascii=False).lower() if raw_setup is not None else ""
    geometry_value = setup.get("geometry")
    if isinstance(geometry_value, dict):
        geometry_value = geometry_value.get("type") or geometry_value.get("shape") or geometry_value.get("description")
    geometry = _canonical_from_text(
        geometry_value or text,
        ("step_wedge", "single_box", "pipe", "void", "inclusion", "multi_layer"),
    )
    if not geometry and "single box" in text:
        geometry = "single_box"
    source_value = setup.get("source")
    if isinstance(source_value, dict):
        source_value = source_value.get("type") or source_value.get("mode") or source_value.get("description")
    source = _canonical_from_text(source_value or text, ("beam", "point", "isotropic"))
    material = _canonical_from_text(
        setup.get("material") or setup.get("materials") or text,
        ("G4_STAINLESS-STEEL", "G4_PLASTIC_SC_VINYLTOLUENE", "G4_POLYETHYLENE", "G4_WATER", "G4_Pb", "G4_Si", "G4_AIR", "G4_Cu", "G4_Al"),
    )
    scoring = _canonical_observables([setup.get("scoring"), text])
    detector = setup.get("detector")
    if detector and "detector_edep" in scoring and "detector_crossing_count" not in scoring:
        scoring.append("detector_crossing_count")
    return {
        "geometry": geometry,
        "material": material,
        "source": source,
        "detector": detector if isinstance(detector, dict) else None,
        "scoring": scoring,
    }


def _canonical_observables(values: Any) -> list[str]:
    allowed = (
        "target_edep",
        "detector_crossing_count",
        "detector_edep",
        "plane_crossing_count",
        "region_contrast",
        "depth_bins",
        "transmission_factor",
    )
    text = json.dumps(values, ensure_ascii=False).lower()
    found = [item for item in allowed if item.lower() in text]
    if "depth-binned" in text or "depth binned" in text:
        found.append("depth_bins")
    if "detector" in text and "response" in text:
        found.extend(["detector_crossing_count", "detector_edep"])
    return list(dict.fromkeys(found))


def _canonical_unsupported(values: Any) -> list[str]:
    mapping = {
        "depth_binned_scoring": ("depth_binned_scoring", "depth_bins", "depth-binned", "depth binned"),
        "region_contrast_scoring": ("region_contrast_scoring", "region_contrast"),
        "embedded_void_geometry": ("embedded_void_geometry", "void_geometry", "embedded void", "embedded_void", "空洞", "孔洞"),
        "embedded_inclusion_geometry": ("embedded_inclusion_geometry", "inclusion_geometry", "embedded inclusion", "夹杂", "内含物"),
        "multi_layer_geometry": ("multi_layer_geometry", "multi-layer", "multi layer"),
        "isotropic_source_sampling": ("isotropic_source_sampling", "isotropic"),
        "curved_pipe_geometry": ("curved_pipe_geometry", "curved pipe", "pipe_geometry"),
        "step_wedge_geometry": ("step_wedge_geometry", "step_wedge"),
        "cad_import": ("cad_import", "cad"),
    }
    text = json.dumps(values, ensure_ascii=False).lower()
    return [name for name, aliases in mapping.items() if any(alias.lower() in text for alias in aliases)]


def _canonical_references(values: Any, setup: dict[str, Any], observables: list[str], unsupported: list[str]) -> list[str]:
    text = json.dumps(values, ensure_ascii=False)
    refs: list[str] = []
    for material in ("G4_STAINLESS-STEEL", "G4_PLASTIC_SC_VINYLTOLUENE", "G4_POLYETHYLENE", "G4_WATER", "G4_Pb", "G4_Si", "G4_AIR", "G4_Cu", "G4_Al"):
        if material in text or material == setup.get("material"):
            refs.append(f"materials:{material}")
    geometry = setup.get("geometry")
    if geometry:
        refs.append(f"geometry:{geometry}")
    for observable in observables:
        if observable != "transmission_factor":
            refs.append(f"scoring:{observable}")
    for item in unsupported:
        if item == "depth_binned_scoring":
            refs.append("scoring:depth_bins")
        if item == "region_contrast_scoring":
            refs.append("scoring:region_contrast")
        if item == "embedded_void_geometry":
            refs.append("geometry:void")
    for raw in values:
        raw_text = str(raw)
        if ":" in raw_text and " " not in raw_text:
            refs.append(raw_text)
    return list(dict.fromkeys(refs))


def _check_expected(case: dict[str, Any], candidate: dict[str, Any], checked: dict[str, Any]) -> list[str]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    errors: list[str] = []
    checked_action = _checked_next_action(checked)
    if expected.get("checked_next_action") and checked_action != expected["checked_next_action"]:
        errors.append(f"checked_next_action:expected={expected['checked_next_action']}:actual={checked_action}")
    if "must_be_supported" in expected and bool(checked.get("supported")) != bool(expected["must_be_supported"]):
        errors.append(f"supported:expected={expected['must_be_supported']}:actual={checked.get('supported')}")
    if "must_require_user_decision" in expected:
        has_decision = bool(candidate.get("user_decisions_required") or checked.get("requires_user_approval"))
        if has_decision != bool(expected["must_require_user_decision"]):
            errors.append(f"user_decision_required:expected={expected['must_require_user_decision']}:actual={has_decision}")
    if expected.get("must_have_simplification") and not candidate.get("simplifications"):
        errors.append("simplifications:missing")
    observables = set(candidate.get("observables") or [])
    for item in expected.get("must_include_observables") or []:
        if item not in observables:
            errors.append(f"observables:missing={item}")
    unsupported = set(candidate.get("unsupported_capabilities") or []) | set(checked.get("unsupported_capabilities") or [])
    for item in expected.get("must_include_unsupported") or []:
        if item not in unsupported:
            errors.append(f"unsupported_capabilities:missing={item}")
    references = set(candidate.get("knowledge_references") or [])
    reference_options = set(expected.get("must_reference_any") or [])
    if reference_options and not references.intersection(reference_options):
        errors.append(f"knowledge_references:missing_any={sorted(reference_options)}")
    if candidate.get("schema_version") != SIMULATION_DESIGN_SCHEMA_VERSION:
        errors.append(f"schema_version:expected={SIMULATION_DESIGN_SCHEMA_VERSION}:actual={candidate.get('schema_version')}")
    return errors


def _process_case(case: dict[str, Any], *, live_llm: bool, llm_config: str, model_override: str | None) -> dict[str, Any]:
    goal = str(case.get("goal") or "")
    reference_pack = build_simulation_design_reference_pack(goal)
    prompt = _build_prompt(case, reference_pack)
    errors: list[str] = []
    llm_used = False
    raw_text = ""
    raw_json: dict[str, Any] | None = None
    if not live_llm:
        errors.append("live_llm_not_enabled")
    else:
        previous_override = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
        try:
            if model_override:
                os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = model_override
            response = ollama_client.chat(prompt, config_path=llm_config, temperature=0.0)
            llm_used = True
            raw_text = str(response.get("response") or "")
            raw_json = ollama_client.extract_json(raw_text)
        except Exception as exc:
            errors.append(f"llm_call_failed:{type(exc).__name__}:{exc}")
        finally:
            if model_override:
                if previous_override is None:
                    os.environ.pop("GEANT4_LLM_MODEL_OVERRIDE", None)
                else:
                    os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = previous_override
    if raw_json is None:
        errors.append("llm_json_parse_failed")
        raw_json = {}
    candidate = _normalize_candidate(raw_json, goal)
    checked = check_simulation_design_capability(candidate, reference_pack["runtime_capabilities"])
    candidate["capability_check"] = checked
    checked_action = _checked_next_action(checked)
    candidate["checked_next_action"] = checked_action
    if not errors:
        errors.extend(_check_expected(case, candidate, checked))
    status = "passed" if not errors else "failed"
    return {
        "id": case.get("id"),
        "status": status,
        "errors": errors,
        "llm_used": llm_used,
        "goal": goal,
        "prompt_profile": "simulation_design_live_v1",
        "raw_llm_response": raw_text,
        "candidate": candidate,
        "reference_pack_summary": {
            "materials": [item.get("id") for item in reference_pack.get("materials", [])],
            "scoring": [item.get("id") for item in reference_pack.get("scoring", [])],
            "geometry": [item.get("id") for item in reference_pack.get("geometry", [])],
            "runtime_capabilities": reference_pack.get("runtime_capabilities"),
        },
    }


def evaluate_simulation_design_live(
    *,
    casebank: Path = DEFAULT_CASEBANK,
    live_llm: bool = False,
    llm_config: str = "",
    model_override: str | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = _load_cases(casebank)
    if max_cases and max_cases > 0:
        cases = cases[:max_cases]
    results = [
        _process_case(case, live_llm=live_llm, llm_config=llm_config, model_override=model_override)
        for case in cases
    ]
    passed = sum(1 for result in results if result["status"] == "passed")
    failed = len(results) - passed
    supported_count = sum(1 for result in results if result["candidate"]["checked_next_action"] == "build_candidate_config")
    approximation_required_count = sum(
        1 for result in results if result["candidate"]["checked_next_action"] == "ask_user_to_choose_approximation"
    )
    unsupported_count = sum(1 for result in results if result["candidate"]["checked_next_action"] == "unsupported_capability")
    user_decision_required_count = sum(1 for result in results if result["candidate"].get("user_decisions_required"))
    return {
        "schema_version": "geant4_agent_simulation_design_live_eval.v1",
        "ok": failed == 0,
        "mode": "live_llm" if live_llm else "offline",
        "model_override": model_override,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "accuracy": passed / len(results) if results else 0.0,
        "stage_summary": {
            "simulation_design": {
                "supported_count": supported_count,
                "approximation_required_count": approximation_required_count,
                "unsupported_count": unsupported_count,
                "user_decision_required_count": user_decision_required_count,
            }
        },
        "case_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate live LLM simulation design candidates.")
    parser.add_argument("--casebank", type=Path, default=DEFAULT_CASEBANK)
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--llm-config", default=os.environ.get("GEANT4_LLM_CONFIG", ""))
    parser.add_argument("--model", default=os.environ.get("GEANT4_LLM_MODEL_OVERRIDE", ""))
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_EVAL_REPORT_DIR)
    parser.add_argument("--run-id", default="simulation-design-live")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate_simulation_design_live(
        casebank=args.casebank,
        live_llm=bool(args.live_llm),
        llm_config=args.llm_config,
        model_override=args.model or None,
        max_cases=args.max_cases or None,
    )
    payload = save_eval_output(report, outdir=args.outdir, tool="simulation_design_live", run_id=args.run_id)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"ok={report['ok']} total={report['total']} passed={report['passed']} failed={report['failed']}")
        print(f"report_path={payload['eval_record']['report_path']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
