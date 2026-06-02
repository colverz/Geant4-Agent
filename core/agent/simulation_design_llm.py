from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.agent.simulation_design import (
    ALLOWED_NEXT_ACTIONS,
    SIMULATION_DESIGN_SCHEMA_VERSION,
    build_simulation_design_reference_pack,
    check_simulation_design_capability,
)
from nlu.llm_support import ollama_client


PROMPT_PROFILE_ID = "simulation_design_live_v2_human_collab"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_simulation_design_prompt(user_goal: str, reference_pack: dict[str, Any], *, lang: str = "en") -> str:
    language_rule = "Use Chinese for natural-language strings." if str(lang).lower().startswith("zh") else "Use English for natural-language strings."
    return (
        "You are a senior Geant4 simulation designer inside a bounded agent workflow.\n"
        "Your job is to understand the user's physical goal, propose a practical simulation design, and keep the design executable by the current runtime.\n"
        "Return JSON only. Do not use markdown. Do not expose hidden chain-of-thought. Do not invent runtime capabilities.\n"
        "The candidate is read-only: do not claim that a simulation was run and do not create final config.\n"
        "Write natural-language fields like a concise senior colleague helping the user think, not like a validation report.\n"
        "Avoid bureaucratic phrasing such as 'the system', 'the user requirement', 'this module', or long nested clauses.\n"
        "Use direct, grounded language: name the physical choice, why it helps, and what remains uncertain.\n"
        "Keep assumptions to at most 3 short items. Merge ordinary distance/size defaults into one item when possible.\n"
        "Keep design_rationale to one or two short sentences. Do not repeat the same fact already present in geometry/material/source.\n"
        "Keep alternatives_considered as a JSON array of at most 3 short strings, each with a practical reason.\n"
        "For Chinese natural-language strings, use plain collaborative Chinese; avoid academic report tone and duplicated punctuation.\n"
        "Think internally in this order before writing JSON: physical goal -> target object -> environment -> source model -> detector/scoring -> runtime support -> user decisions.\n"
        "Prefer a useful runnable candidate with explicit assumptions over repeatedly asking for missing ordinary parameters.\n"
        "Use reasonable defaults only when they are standard/simple and list them in assumptions.\n"
        "Never hide a physically meaningful approximation. If an approximation changes the physical model, require user approval.\n"
        "The reference pack provides full catalogs for materials, sources, scoring, and geometry. These catalogs are the knowledge context.\n"
        "query_hints are only orientation hints from the user text. They are non-binding and must not restrict your design choices.\n"
        "Choose the setup by comparing the full catalog tags, use_cases, notes, and runtime capabilities.\n"
        "Do not behave like a keyword extractor. Make a design choice, then justify it briefly in recommended_setup.design_rationale.\n"
        "If you reject plausible alternatives, put compact reasons in recommended_setup.alternatives_considered.\n"
        f"The output schema_version must be exactly {SIMULATION_DESIGN_SCHEMA_VERSION}. Do not copy the reference pack schema_version.\n"
        "recommended_setup must include: geometry, material, source, detector, scoring.\n"
        "recommended_setup may also include: target_material, environment_material, void_material, dimensions_mm, source_particle, source_energy_mev, source_position_mm, source_direction, design_rationale, alternatives_considered.\n"
        "material is the primary target/scoring material unless there is no target and the user is explicitly modeling only an environment.\n"
        "environment_material is the surrounding medium or world-like transport medium when relevant.\n"
        "void_material is the material inside an embedded defect or cavity when relevant.\n"
        "geometry is a structured object describing the physical setup. Format:\n"
        '  {"volumes": [{"name": "descriptive", "shape": "box|sphere|tubs|cons|trd", "material": "G4_ ID",\n'
        '    "dimensions": {"size_x_mm": N, ...}  // shape-appropriate keys (see below),\n'
        '    "position_mm": [x, y, z]}],  // optional, default origin\n'
        '   "environment": {"material": "G4_Galactic for vacuum/space, G4_AIR for air"}}\n'
        "Shape-appropriate dimension keys:\n"
        "  box: size_x_mm, size_y_mm, size_z_mm\n"
        "  sphere: radius_mm\n"
        "  tubs/cylinder: radius_mm, half_length_mm (along z)\n"
        "  cons: radius1_mm, radius2_mm, half_length_mm\n"
        "For multi-volume setups (layers, embedded voids, step wedges), use multiple volumes\n"
        "with appropriate positions. Describe the physical setup, not Geant4 implementation.\n"
        "source must be one canonical ID from: beam, point, isotropic.\n"
        "observables and recommended_setup.scoring must use canonical IDs only: target_edep, detector_crossing_count, detector_edep, plane_crossing_count, region_contrast, depth_bins, transmission_factor.\n"
        "transmission_factor is a derived observable. It is acceptable when detector_crossing_count or plane_crossing_count is included; do not mark transmission_factor itself unsupported.\n"
        "For detector response, include both detector_crossing_count and detector_edep when a detector volume is proposed.\n"
        "Use G4_Galactic for vacuum, space-like, beamline vacuum, 真空, or 虚空 environments.\n"
        "Do not use G4_AIR as a vacuum substitute. G4_AIR means air or an air gap only.\n"
        "For an air-filled defect/cavity, void_material may be G4_AIR. For a vacuum cavity, void_material must be G4_Galactic.\n"
        "If the user requests a copper/lead/steel/etc. target in vacuum, keep material as the target material and set environment_material=G4_Galactic.\n"
        "If the user requests only a vacuum environment without a target, set material=G4_Galactic.\n"
        "knowledge_references must use canonical IDs like materials:G4_Pb, geometry:pipe, scoring:depth_bins. Do not write prose in knowledge_references.\n"
        "If a requested capability is unsupported, put it in unsupported_capabilities and set next_action to unsupported_capability.\n"
        "Use simplifications only for unsupported-to-supported approximations that require approval, not for ordinary simple supported modeling choices.\n"
        "Unreasonable approximations are forbidden. Examples: air as vacuum; curved pipe as a flat slab without approval; region contrast without region volumes; depth-dose without depth bins.\n"
        "If an approximation is required, list it in simplifications, add user_decisions_required, and set next_action to ask_user_to_choose_approximation.\n"
        "If next_action is unsupported_capability, add at least one user_decisions_required item asking the user to choose a supported approximation or defer.\n"
        "If the setup is fully supported by current runtime capabilities, set next_action to build_candidate_config.\n"
        f"{language_rule}\n\n"
        "Allowed next_action values:\n"
        f"{_json_dumps(sorted(ALLOWED_NEXT_ACTIONS))}\n\n"
        "Output schema fields:\n"
        f"{_json_dumps(['schema_version','goal','recommended_setup','observables','assumptions','simplifications','unsupported_capabilities','user_decisions_required','knowledge_references','capability_check','next_action'])}\n\n"
        "Reference pack for choosing canonical IDs. materials/sources/scoring/geometry are full catalogs; query_hints are not binding. This is not the output schema:\n"
        f"{_json_dumps(reference_pack)}\n\n"
        "User goal:\n"
        f"{user_goal}\n"
    )


def checked_next_action(capability_check: dict[str, Any]) -> str:
    if capability_check.get("unsupported_capabilities"):
        return "unsupported_capability"
    if capability_check.get("requires_user_approval"):
        return "ask_user_to_choose_approximation"
    if capability_check.get("supported"):
        return "build_candidate_config"
    return "needs_more_information"


def normalize_simulation_design_candidate(raw: dict[str, Any], goal: str) -> dict[str, Any]:
    setup = raw.get("recommended_setup") if isinstance(raw.get("recommended_setup"), dict) else {}
    setup = _normalize_setup(setup, raw.get("recommended_setup"), goal=goal)
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
        "assumptions": _compact_natural_language_items(raw.get("assumptions"), limit=3),
        "simplifications": _compact_natural_language_items(simplifications, limit=4),
        "unsupported_capabilities": unsupported,
        "user_decisions_required": _compact_natural_language_items(raw.get("user_decisions_required"), limit=4),
        "knowledge_references": _canonical_references(raw.get("knowledge_references") or [], setup, observables, unsupported),
        "capability_check": raw.get("capability_check") if isinstance(raw.get("capability_check"), dict) else {},
        "next_action": str(raw.get("next_action") or "needs_more_information"),
    }
    if normalized["next_action"] not in ALLOWED_NEXT_ACTIONS:
        normalized["next_action"] = "needs_more_information"
    if normalized["unsupported_capabilities"] and not normalized["user_decisions_required"]:
        normalized["user_decisions_required"] = ["Choose a supported approximation or defer until runtime capability is added."]
    return normalized


def build_llm_simulation_design_candidate(
    user_goal: str,
    *,
    config_path: str,
    lang: str = "en",
    runtime_capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_pack = build_simulation_design_reference_pack(user_goal, runtime_capabilities)
    prompt = build_simulation_design_prompt(user_goal, reference_pack, lang=lang)
    raw_text = ""
    try:
        response = ollama_client.chat(prompt, config_path=config_path, temperature=0.0)
        raw_text = str(response.get("response") or "")
        raw_json = ollama_client.extract_json(raw_text)
        if raw_json is None:
            return {
                "ok": False,
                "fallback_reason": "llm_json_parse_failed",
                "raw_response": raw_text,
                "prompt_profile_id": PROMPT_PROFILE_ID,
                "prompt_validation": {"ok": False, "errors": ["llm_json_parse_failed"]},
                "reference_pack": reference_pack,
            }
        candidate = normalize_simulation_design_candidate(raw_json, user_goal)
        checked = check_simulation_design_capability(candidate, reference_pack["runtime_capabilities"])
        candidate["capability_check"] = checked
        candidate["checked_next_action"] = checked_next_action(checked)
        if candidate.get("next_action") != candidate["checked_next_action"]:
            candidate["next_action"] = candidate["checked_next_action"]
        return {
            "ok": True,
            "candidate": candidate,
            "raw_response": raw_text,
            "prompt_profile_id": PROMPT_PROFILE_ID,
            "prompt_validation": {"ok": True, "errors": []},
            "reference_pack": reference_pack,
        }
    except Exception as exc:
        logging.warning("simulation_design_llm_fallback: %s", type(exc).__name__)
        return {
            "ok": False,
            "fallback_reason": f"llm_call_failed:{type(exc).__name__}",
            "raw_response": raw_text,
            "prompt_profile_id": PROMPT_PROFILE_ID,
            "prompt_validation": {"ok": False, "errors": [f"llm_call_failed:{type(exc).__name__}"]},
            "reference_pack": reference_pack,
        }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _compact_natural_language_items(value: Any, *, limit: int) -> list[str]:
    compacted: list[str] = []
    for item in _string_list(value):
        text = _clean_natural_language_item(item)
        if text:
            compacted.append(text)
    return list(dict.fromkeys(compacted))[:limit]


def _clean_natural_language_item(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"[。．.；;，,、\s]+$", "", text)
    text = text.replace("。。", "。").replace("，，", "，")
    return text.strip()


def _normalize_setup_text_field(key: str, value: Any) -> Any:
    if key == "alternatives_considered":
        return _compact_natural_language_items(value, limit=3)
    if key in {"design_rationale", "user_explanation"}:
        return _clean_natural_language_item(" ".join(_string_list(value)))
    return value


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


_MATERIAL_IDS = (
    "G4_STAINLESS-STEEL",
    "G4_PLASTIC_SC_VINYLTOLUENE",
    "G4_POLYETHYLENE",
    "G4_Galactic",
    "G4_WATER",
    "G4_Pb",
    "G4_Si",
    "G4_AIR",
    "G4_Cu",
    "G4_Al",
)


def _normalize_setup(setup: dict[str, Any], raw_setup: Any, *, goal: str = "") -> dict[str, Any]:
    text = json.dumps(raw_setup, ensure_ascii=False).lower() if raw_setup is not None else ""
    goal_text = str(goal or "").lower()
    geometry_value = setup.get("geometry")
    # New format: LLM outputs structured geometry with volumes list — keep as-is
    if isinstance(geometry_value, dict) and geometry_value.get("volumes"):
        geometry = geometry_value  # Pass dict through to config builder
    else:
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
    inferred_goal_material = _infer_material_from_goal(goal_text)
    material = _canonical_from_text(setup.get("material") or setup.get("materials") or text, _MATERIAL_IDS)
    target_material = _canonical_from_text(setup.get("target_material"), _MATERIAL_IDS)
    environment_material = _canonical_from_text(setup.get("environment_material"), _MATERIAL_IDS)
    void_material = _canonical_from_text(setup.get("void_material"), _MATERIAL_IDS)
    if not material:
        material = inferred_goal_material
    if not target_material and inferred_goal_material != "G4_Galactic":
        target_material = inferred_goal_material
    if _mentions_vacuum(goal_text):
        environment_material = "G4_Galactic"
        if geometry == "void":
            void_material = "G4_Galactic"
        if material == "G4_AIR" and not target_material:
            material = "G4_Galactic"
        elif material == "G4_AIR" and target_material:
            material = target_material
    if not material:
        material = "G4_Cu"
    scoring = _canonical_observables([setup.get("scoring"), text])
    detector = setup.get("detector")
    if detector and "detector_edep" in scoring and "detector_crossing_count" not in scoring:
        scoring.append("detector_crossing_count")
    normalized = {
        "geometry": geometry,
        "material": material,
        "source": source,
        "detector": detector if isinstance(detector, dict) else None,
        "scoring": scoring,
    }
    if target_material:
        normalized["target_material"] = target_material
    if environment_material:
        normalized["environment_material"] = environment_material
    if void_material:
        normalized["void_material"] = void_material
    for key in (
        "dimensions_mm",
        "source_particle",
        "source_energy_mev",
        "source_position_mm",
        "source_direction",
        "design_rationale",
        "alternatives_considered",
        "user_explanation",
    ):
        if key in setup:
            normalized[key] = _normalize_setup_text_field(key, setup[key])
    return normalized


def _mentions_vacuum(text: str) -> bool:
    return any(token in text for token in ("vacuum", "space", "beamline vacuum", "真空", "虚空", "宇宙", "太空"))


def _infer_material_from_goal(text: str) -> str:
    def has_word(*tokens: str) -> bool:
        return any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)

    if "铜" in text or has_word("copper", "cu"):
        return "G4_Cu"
    if "铅" in text or "屏蔽" in text or has_word("lead", "pb", "shield"):
        return "G4_Pb"
    if any(token in text for token in ("钢", "管", "楔")) or has_word("steel", "pipe", "wedge"):
        return "G4_STAINLESS-STEEL"
    if "铝" in text or has_word("aluminum", "aluminium", "al"):
        return "G4_Al"
    if "水" in text or "模体" in text or has_word("water", "phantom"):
        return "G4_WATER"
    if "聚乙烯" in text or "中子" in text or has_word("polyethylene", "neutron"):
        return "G4_POLYETHYLENE"
    if "闪烁体" in text or has_word("scintillator"):
        return "G4_PLASTIC_SC_VINYLTOLUENE"
    if "硅" in text or has_word("silicon"):
        return "G4_Si"
    if _mentions_vacuum(text):
        return "G4_Galactic"
    if "空气" in text or "气隙" in text or has_word("air"):
        return "G4_AIR"
    return ""


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
    for material in (
        *_MATERIAL_IDS,
    ):
        if material in text or material == setup.get("material") or material == setup.get("environment_material") or material == setup.get("void_material"):
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


__all__ = [
    "PROMPT_PROFILE_ID",
    "build_llm_simulation_design_candidate",
    "build_simulation_design_prompt",
    "checked_next_action",
    "normalize_simulation_design_candidate",
]
