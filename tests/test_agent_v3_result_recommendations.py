from __future__ import annotations

from core.agent_v3.result_recommendations import build_runtime_result_recommendations


def test_gamma_zero_downstream_counts_suggests_energy_sweep_from_runtime_facts() -> None:
    response = {
        "context": {
            "latest_runtime_facts": {
                "material": "G4_Pb",
                "particle": "gamma",
                "source_energy_mev": 1.0,
                "detector_crossing_count": 0,
                "plane_crossing_count": 0,
            }
        }
    }

    suggestions = build_runtime_result_recommendations(response, locale="en-US")

    assert suggestions
    assert suggestions[0]["text"] == "Compare source energy"
    assert suggestions[0]["prefill"] == "run sweep 0.5 1 2 MeV"
    assert suggestions[0]["kind"] == "result_driven_sweep"
    assert suggestions[0]["fact_basis"]["material"] == "G4_Pb"


def test_positive_downstream_counts_do_not_suggest_energy_sweep() -> None:
    response = {
        "context": {
            "latest_runtime_facts": {
                "material": "G4_Pb",
                "particle": "gamma",
                "source_energy_mev": 1.0,
                "detector_crossing_count": 3,
                "plane_crossing_count": 0,
            }
        }
    }

    assert build_runtime_result_recommendations(response, locale="en-US") == []


def test_zero_downstream_counts_with_thickness_suggests_thinner_target_first() -> None:
    response = {
        "context": {
            "latest_runtime_facts": {
                "material": "G4_Pb",
                "particle": "gamma",
                "source_energy_mev": 1.0,
                "target_thickness_mm": 20.0,
                "detector_crossing_count": 0,
                "plane_crossing_count": 0,
            }
        }
    }

    suggestions = build_runtime_result_recommendations(response, locale="en-US")

    assert suggestions[0]["kind"] == "result_driven_thickness_change"
    assert suggestions[0]["text"] == "Reduce target thickness"
    assert suggestions[0]["prefill"] == "change target thickness to 10 mm and run again"
    assert suggestions[0]["fact_basis"]["target_thickness_mm"] == 20.0
    assert suggestions[0]["fact_basis"]["next_target_thickness_mm"] == 10.0
    assert any(item["kind"] == "result_driven_sweep" for item in suggestions)


def test_target_edep_without_downstream_counts_suggests_adding_scoring() -> None:
    response = {
        "context": {
            "latest_runtime_facts": {
                "material": "G4_Pb",
                "particle": "gamma",
                "source_energy_mev": 1.0,
                "target_thickness_mm": 10.0,
                "target_edep_total_mev": 12.5,
                "detector_crossing_count": None,
                "plane_crossing_count": None,
            }
        }
    }

    suggestions = build_runtime_result_recommendations(response, locale="en-US")

    assert suggestions[0]["kind"] == "result_driven_scoring_addition"
    assert suggestions[0]["text"] == "Add downstream scoring"
    assert suggestions[0]["prefill"] == "add downstream detector and plane scoring and run again"
    assert suggestions[0]["fact_basis"]["target_edep_total_mev"] == 12.5


def test_runtime_payload_geometry_can_supply_target_thickness() -> None:
    response = {
        "observations": [
            {
                "source": "geant4_runtime_tool",
                "status": "ok",
                "data": {
                    "runtime_payload": {
                        "source": {"particle": "gamma", "energy_mev": 1.0},
                        "geometry": {
                            "material": "G4_Pb",
                            "params": {"module_z": 12.0},
                        },
                    },
                    "result_summary": {
                        "scoring": {
                            "detector_crossing": {"detector_crossing_count": 0},
                            "plane_crossing": {"plane_crossing_count": 0},
                        }
                    },
                },
            }
        ]
    }

    suggestions = build_runtime_result_recommendations(response, locale="en-US")

    assert suggestions[0]["kind"] == "result_driven_thickness_change"
    assert suggestions[0]["prefill"] == "change target thickness to 6 mm and run again"
