from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferencePack:
    id: str
    title: str
    bullets: tuple[str, ...]


REFERENCE_PACKS: dict[str, ReferencePack] = {
    "geometry_basic": ReferencePack(
        id="geometry_basic",
        title="Geometry basics",
        bullets=(
            "The root geometry volume is the main transport medium or target volume.",
            "The root geometry material is independent from detector material.",
            "Box dimensions are expressed as size_triplet_mm in millimeters.",
            "Use stable volume names from the runtime contract when available.",
        ),
    ),
    "material_roles": ReferencePack(
        id="material_roles",
        title="Material role binding",
        bullets=(
            "selected_materials should include every material used by named volumes.",
            "volume_material_map must bind each named volume to its own material.",
            "Do not copy detector material onto the root volume unless the contract says the root volume is the detector.",
            "For an air gap plus silicon detector, root material is G4_AIR and detector material is G4_Si.",
        ),
    ),
    "source_basic": ReferencePack(
        id="source_basic",
        title="Source basics",
        bullets=(
            "A beam source has particle, energy_mev, position_mm, and direction_vec.",
            "For a beam through a target along +z, the source is normally upstream at negative z and direction is [0, 0, 1].",
            "Do not change explicitly provided particle, energy, position, or direction.",
        ),
    ),
    "detector_basic": ReferencePack(
        id="detector_basic",
        title="Detector basics",
        bullets=(
            "Detector settings belong under the detector volume, not the root volume.",
            "Detector name, material, position, and size should match the runtime contract when provided.",
            "A detector behind a target is downstream along the source direction.",
        ),
    ),
    "scoring_roles": ReferencePack(
        id="scoring_roles",
        title="Scoring roles",
        bullets=(
            "target_edep scores energy deposition in target/root role volumes.",
            "detector_crossings and detector edep use detector role volumes.",
            "volume_roles.target and volume_roles.detector must reference existing volume names.",
            "If plane_crossings is false, do not invent a scoring plane.",
        ),
    ),
    "physics_list": ReferencePack(
        id="physics_list",
        title="Physics list",
        bullets=(
            "Use the explicit physics list from the user or runtime defaults.",
            "FTFP_BERT is a Geant4 physics list; it is unrelated to the removed NLP-BERT model prior.",
            "Do not substitute a different physics list unless the user asks or the allowed defaults require it.",
        ),
    ),
    "runtime_defaults": ReferencePack(
        id="runtime_defaults",
        title="Runtime defaults",
        bullets=(
            "Use default events, seed, mode, and output format from runtime_defaults when the user leaves them unspecified.",
            "Runtime defaults are allowed inferred values, not hallucinated physics results.",
            "Never fabricate runtime metrics; metrics must come from Geant4 runtime output.",
        ),
    ),
    "unsupported_boundary": ReferencePack(
        id="unsupported_boundary",
        title="Unsupported capability boundary",
        bullets=(
            "CAD import, moving gantry, paired multi-run comparisons, and depth-binned scoring require explicit runtime support.",
            "If a requested capability is unsupported, state the capability gap instead of inventing a config.",
            "Unsupported features must not trigger Geant4 runtime automatically.",
        ),
    ),
}


def select_reference_packs(
    case: dict[str, Any],
    runtime_payload: dict[str, Any] | None = None,
    *,
    max_packs: int = 5,
) -> list[ReferencePack]:
    """Select compact prompt references for the current benchmark case.

    This intentionally selects role/contract references, not phrase dictionaries.
    """

    runtime_payload = runtime_payload if isinstance(runtime_payload, dict) else {}
    scenario = case.get("scenario_spec") if isinstance(case.get("scenario_spec"), dict) else {}
    text = " ".join(
        [
            str(case.get("domain") or ""),
            str(case.get("task") or ""),
            " ".join(str(item) for item in case.get("raw_dialogue", []) if str(item)),
            " ".join(str(value) for value in scenario.values()),
            " ".join(str(item) for item in case.get("capability_pressure", []) if str(item)),
        ]
    ).lower()
    detector = runtime_payload.get("detector") if isinstance(runtime_payload.get("detector"), dict) else {}
    scoring = runtime_payload.get("scoring") if isinstance(runtime_payload.get("scoring"), dict) else {}

    selected: list[str] = ["geometry_basic", "material_roles", "source_basic"]
    if detector.get("enabled") or "detector" in text or "silicon" in text or "scintillator" in text:
        selected.append("detector_basic")
    if scoring or "scoring" in text or "crossing" in text or "edep" in text or "response" in text:
        selected.append("scoring_roles")
    if "physics" in text or "neutron" in text or "proton" in text or runtime_payload.get("physics"):
        selected.append("physics_list")
    selected.append("runtime_defaults")
    if case.get("domain") == "unsupported_boundary" or any(
        cue in text for cue in ("cad", "gantry", "multi-run", "depth bin", "paired")
    ):
        selected.append("unsupported_boundary")

    unique = [pack_id for pack_id in dict.fromkeys(selected) if pack_id in REFERENCE_PACKS]
    return [REFERENCE_PACKS[pack_id] for pack_id in unique[:max_packs]]


def format_reference_packs(packs: list[ReferencePack], *, compact: bool = False) -> list[str]:
    lines: list[str] = []
    for pack in packs:
        if compact:
            lines.append(f"[{pack.id}] {pack.title}: {' '.join(pack.bullets[:2])}")
        else:
            lines.append(f"[{pack.id}] {pack.title}")
            lines.extend(f"- {bullet}" for bullet in pack.bullets)
    return lines


def infer_choice_zones(case: dict[str, Any], runtime_defaults: dict[str, Any]) -> list[dict[str, str]]:
    zones = [
        {
            "field": "run.seed",
            "policy": f"May use runtime default {runtime_defaults.get('seed', 1337)} when user did not specify a seed.",
        },
        {
            "field": "run.events",
            "policy": f"May use runtime default {runtime_defaults.get('events', 10000)} when user did not specify event count.",
        },
        {
            "field": "volume_names",
            "policy": "May use conventional names like Detector only when no explicit contract name is provided.",
        },
    ]
    scenario = case.get("scenario_spec") if isinstance(case.get("scenario_spec"), dict) else {}
    if "detector" in " ".join(str(value).lower() for value in scenario.values()):
        zones.append(
            {
                "field": "simulation.detector",
                "policy": "May infer a downstream detector placement only when the user says the detector is behind/downstream.",
            }
        )
    return zones
