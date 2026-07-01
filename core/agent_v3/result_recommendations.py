from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeResultFacts:
    material: str = ""
    particle: str = ""
    source_energy_mev: float | None = None
    target_thickness_mm: float | None = None
    target_edep_total_mev: float | None = None
    detector_crossing_count: float | None = None
    plane_crossing_count: float | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "RuntimeResultFacts":
        return cls(
            material=str(raw.get("material") or ""),
            particle=str(raw.get("particle") or "").lower(),
            source_energy_mev=_positive_float(raw.get("source_energy_mev")),
            target_thickness_mm=_positive_float(raw.get("target_thickness_mm")),
            target_edep_total_mev=_optional_number(raw.get("target_edep_total_mev")),
            detector_crossing_count=_optional_number(raw.get("detector_crossing_count")),
            plane_crossing_count=_optional_number(raw.get("plane_crossing_count")),
        )

    @property
    def is_photon_source(self) -> bool:
        return self.particle in ("gamma", "photon")

    @property
    def has_core_source(self) -> bool:
        return self.is_photon_source and self.source_energy_mev is not None

    @property
    def downstream_counts(self) -> list[float]:
        return [
            value
            for value in (self.detector_crossing_count, self.plane_crossing_count)
            if value is not None
        ]

    @property
    def downstream_metrics_missing(self) -> bool:
        return self.detector_crossing_count is None and self.plane_crossing_count is None

    @property
    def downstream_counts_all_zero(self) -> bool:
        counts = self.downstream_counts
        return bool(counts) and all(value <= 0 for value in counts)

    def fact_basis(self, *fields: str) -> dict[str, Any]:
        values = {
            "material": self.material,
            "particle": self.particle,
            "source_energy_mev": self.source_energy_mev,
            "target_thickness_mm": self.target_thickness_mm,
            "target_edep_total_mev": self.target_edep_total_mev,
            "detector_crossing_count": self.detector_crossing_count,
            "plane_crossing_count": self.plane_crossing_count,
        }
        selected = fields or tuple(values)
        return {field: values[field] for field in selected if field in values}


def build_runtime_result_recommendations(response: dict[str, Any], *, locale: str = "zh-CN") -> list[dict[str, Any]]:
    """Build next-action suggestions from structured runtime facts only.

    This is intentionally not a user-text keyword system. It reads the latest
    runtime facts, normalizes them into a typed object, and proposes safe
    follow-up actions that still require a later explicit user turn and the
    normal confirmation gate.
    """

    raw_facts = _runtime_facts(response)
    if not raw_facts:
        return []
    facts = RuntimeResultFacts.from_mapping(raw_facts)
    recommendations: list[dict[str, Any]] = []
    for builder in (
        _downstream_scoring_recommendation,
        _thickness_recommendation,
        _energy_sweep_recommendation,
    ):
        recommendation = builder(facts, locale=locale)
        if recommendation:
            recommendations.append(recommendation)
    return recommendations


def _downstream_scoring_recommendation(facts: RuntimeResultFacts, *, locale: str) -> dict[str, Any] | None:
    if not facts.has_core_source or facts.target_edep_total_mev is None:
        return None
    if not facts.downstream_metrics_missing:
        return None
    return {
        "text": "Add downstream scoring",
        "prefill": "add downstream detector and plane scoring and run again",
        "kind": "result_driven_scoring_addition",
        "rationale": "The result has target energy deposition but no downstream crossing metric, so transmission cannot be compared yet.",
        "fact_basis": facts.fact_basis(
            "material",
            "particle",
            "source_energy_mev",
            "target_edep_total_mev",
            "detector_crossing_count",
            "plane_crossing_count",
        ),
    }


def _thickness_recommendation(facts: RuntimeResultFacts, *, locale: str) -> dict[str, Any] | None:
    if not facts.has_core_source or facts.target_thickness_mm is None:
        return None
    if not facts.downstream_counts_all_zero:
        return None
    next_thickness = max(1.0, facts.target_thickness_mm * 0.5)
    if abs(next_thickness - facts.target_thickness_mm) < 1e-9:
        return None
    return {
        "text": "Reduce target thickness",
        "prefill": f"change target thickness to {_format_number(next_thickness)} mm and run again",
        "kind": "result_driven_thickness_change",
        "rationale": "No downstream crossings were observed; a thinner target can test whether attenuation is dominated by thickness.",
        "fact_basis": {
            **facts.fact_basis(
                "material",
                "particle",
                "source_energy_mev",
                "target_thickness_mm",
                "detector_crossing_count",
                "plane_crossing_count",
            ),
            "next_target_thickness_mm": next_thickness,
        },
    }


def _energy_sweep_recommendation(facts: RuntimeResultFacts, *, locale: str) -> dict[str, Any] | None:
    if not facts.has_core_source or not facts.material:
        return None
    if not facts.downstream_counts_all_zero:
        return None
    values = _sweep_values_around(facts.source_energy_mev)
    values_text = " ".join(_format_number(value) for value in values)
    return {
        "text": "Compare source energy",
        "prefill": f"run sweep {values_text} MeV",
        "kind": "result_driven_sweep",
        "rationale": f"{facts.material} stopped all observed downstream gamma counts at {facts.source_energy_mev:g} MeV.",
        "fact_basis": facts.fact_basis(
            "material",
            "particle",
            "source_energy_mev",
            "detector_crossing_count",
            "plane_crossing_count",
        ),
    }


def _runtime_facts(response: dict[str, Any]) -> dict[str, Any]:
    context = response.get("context") if isinstance(response.get("context"), dict) else {}
    facts = context.get("latest_runtime_facts") if isinstance(context.get("latest_runtime_facts"), dict) else {}
    if facts:
        return dict(facts)
    runtime = _runtime_data(response)
    result = _dict(runtime.get("result_summary"))
    config = _dict(result.get("configuration"))
    payload = _dict(runtime.get("runtime_payload"))
    source = _dict(payload.get("source"))
    geometry = _dict(payload.get("geometry"))
    scoring = _dict(result.get("scoring"))
    target = _dict(scoring.get("target"))
    detector = _dict(scoring.get("detector_crossing"))
    plane = _dict(scoring.get("plane_crossing"))
    return {
        "material": _first_present(config.get("material"), geometry.get("material")),
        "particle": _first_present(config.get("particle"), source.get("particle")),
        "source_energy_mev": _first_present(source.get("energy_mev"), payload.get("energy")),
        "target_thickness_mm": _target_thickness_mm(geometry),
        "target_edep_total_mev": target.get("target_edep_total_mev"),
        "detector_crossing_count": detector.get("detector_crossing_count"),
        "plane_crossing_count": plane.get("plane_crossing_count"),
    }


def _runtime_data(response: dict[str, Any]) -> dict[str, Any]:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    state = response.get("state") if isinstance(response.get("state"), dict) else {}
    state_observations = state.get("observations") if isinstance(state.get("observations"), list) else []
    for observation in reversed([*state_observations, *observations]):
        if isinstance(observation, dict) and observation.get("source") == "geant4_runtime_tool":
            data = observation.get("data")
            return data if isinstance(data, dict) else {}
    return {}


def _sweep_values_around(energy: float) -> list[float]:
    return [max(0.001, energy * 0.5), energy, energy * 2.0]


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    return f"{value:g}"


def _target_thickness_mm(geometry: dict[str, Any]) -> float | None:
    for value in (
        _dict(geometry.get("params")).get("module_z"),
        geometry.get("size_z_mm"),
        _dict(geometry.get("dimensions")).get("size_z_mm"),
        geometry.get("target_thickness_mm"),
        geometry.get("thickness_mm"),
    ):
        parsed = _positive_float(value)
        if parsed is not None:
            return parsed
    volumes = geometry.get("volumes")
    if isinstance(volumes, list):
        for volume in volumes:
            if not isinstance(volume, dict):
                continue
            size = volume.get("size_mm")
            if isinstance(size, list) and len(size) >= 3:
                parsed = _positive_float(size[2])
                if parsed is not None:
                    return parsed
            parsed = _positive_float(_dict(volume.get("dimensions")).get("size_z_mm"))
            if parsed is not None:
                return parsed
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["RuntimeResultFacts", "build_runtime_result_recommendations"]
