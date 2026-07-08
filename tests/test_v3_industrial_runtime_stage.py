from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime
from tools.industrial_runtime_contract import (
    V3IndustrialCandidateRequirements,
    compare_v3_candidate_runtime_contract,
)
from tools.run_v3_industrial_runtime_stage import _run_v3_case


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "docs" / "eval" / "industrial_runtime_benchmark.json"


def _compiled_lead_case() -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    case = next(item for item in benchmark["cases"] if item["id"] == "shielding_lead_gamma_transmission")
    return case, compile_industrial_case_to_runtime(case, runtime_defaults=benchmark["runtime_defaults"])


def _compiled_case(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    case = next(item for item in benchmark["cases"] if item["id"] == case_id)
    return case, compile_industrial_case_to_runtime(case, runtime_defaults=benchmark["runtime_defaults"])


def _payload_response(runtime_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "terminated_reason": "final_answer",
        "observations": [
            {
                "source": "geant4_llm_design_tool",
                "status": "ok",
                "data": {"llm": {"used": True, "ok": True, "prompt_profile_id": "test"}},
            },
            {
                "source": "geant4_payload_builder_tool",
                "status": "ok",
                "data": {"runtime_payload": runtime_payload},
            }
        ],
        "state": {"metadata": {"turn_understanding": {"source": "llm"}}},
    }


def _valid_noncanonical_lead_payload(compiled: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(compiled["runtime_payload"])
    candidate["geometry"]["structure"] = "llm_structured"
    candidate["geometry"]["root_volume_name"] = "lead_shield"
    candidate["geometry"]["size_x_mm"] = 80.0
    candidate["geometry"]["size_y_mm"] = 80.0
    candidate["source"]["position_mm"] = [0.0, 0.0, -200.0]
    candidate["detector"]["position_mm"] = [0.0, 0.0, 25.0]
    candidate["detector"]["size_x_mm"] = 80.0
    candidate["detector"]["size_y_mm"] = 80.0
    candidate["scoring"]["plane_crossings"] = True
    candidate["scoring"]["plane"] = {"name": "ExitPlane", "z_mm": 15.0}
    return candidate


class FakeService:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def run_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        return self.responses[len(self.requests) - 1]


def test_semantic_contract_allows_physically_valid_design_freedom() -> None:
    _, compiled = _compiled_lead_case()
    candidate = _valid_noncanonical_lead_payload(compiled)

    report = compare_v3_candidate_runtime_contract(candidate, compiled["runtime_payload"])

    assert report["ok"] is True
    assert report["mismatches"] == []
    assert "geometry.root_volume_name" in report["allowed_variations"]


def test_semantic_contract_rejects_wrong_thickness_and_upstream_order() -> None:
    _, compiled = _compiled_lead_case()
    candidate = _valid_noncanonical_lead_payload(compiled)
    candidate["geometry"]["size_z_mm"] = 20.0
    candidate["source"]["position_mm"] = [0.0, 0.0, 0.0]

    report = compare_v3_candidate_runtime_contract(candidate, compiled["runtime_payload"])

    assert report["ok"] is False
    fields = {item["field"] for item in report["mismatches"]}
    assert "target.thickness_mm" in fields
    assert "source.upstream_position" in fields


def test_semantic_contract_allows_optional_detector_for_neutron_case() -> None:
    case, compiled = _compiled_case("shielding_polyethylene_neutron_moderation")
    candidate = deepcopy(compiled["runtime_payload"])
    candidate["detector"] = {
        "enabled": True,
        "material": "G4_Si",
        "position_mm": [0.0, 0.0, 80.0],
    }
    candidate["scoring"]["detector_crossings"] = True
    requirements = V3IndustrialCandidateRequirements.from_case(case, compiled["runtime_payload"])

    report = compare_v3_candidate_runtime_contract(
        candidate,
        compiled["runtime_payload"],
        requirements=requirements,
    )

    assert report["ok"] is True


def test_semantic_contract_accepts_detector_material_as_primary_when_material_set_is_complete() -> None:
    case, compiled = _compiled_case("detector_silicon_gamma_response")
    candidate = deepcopy(compiled["runtime_payload"])
    candidate["geometry"]["material"] = "G4_Si"
    candidate["geometry"]["volumes"] = [
        {"name": "AirGap", "material": "G4_AIR"},
        {"name": "SiliconSensor", "material": "G4_Si"},
    ]
    requirements = V3IndustrialCandidateRequirements.from_case(case, compiled["runtime_payload"])

    report = compare_v3_candidate_runtime_contract(
        candidate,
        compiled["runtime_payload"],
        requirements=requirements,
    )

    assert report["ok"] is True


def test_semantic_contract_rejects_declared_but_unmaterialized_depth_bins() -> None:
    case, compiled = _compiled_case("medical_proton_water_depth_dose")
    candidate = deepcopy(compiled["runtime_payload"])
    candidate["geometry"]["volumes"] = candidate["geometry"]["volumes"][:1]
    candidate["scoring"]["volume_roles"] = {"target": ["WaterPhantom"]}
    requirements = V3IndustrialCandidateRequirements.from_case(case, compiled["runtime_payload"])

    report = compare_v3_candidate_runtime_contract(
        candidate,
        compiled["runtime_payload"],
        requirements=requirements,
    )

    assert report["ok"] is False
    assert "scoring.depth_bins.executable" in {item["field"] for item in report["mismatches"]}


def test_contract_mismatch_never_reaches_preflight_or_confirmation(tmp_path: Path) -> None:
    case, compiled = _compiled_lead_case()
    candidate = deepcopy(compiled["runtime_payload"])
    candidate["source"]["energy_mev"] = 2.0
    service = FakeService([_payload_response(candidate)])

    result = _run_v3_case(
        service,  # type: ignore[arg-type]
        case,
        compiled,
        llm_config_path="llm.local.json",
        runtime_policy={"allow_in_memory": False, "env": {}},
        golden_dir=tmp_path,
        allow_unreviewed_goldens=False,
    )

    assert result["failure_category"] == "v3_candidate_contract_mismatch"
    assert result["runtime_attempted"] is False
    assert len(service.requests) == 1
    assert "confirmation_event" not in service.requests[0]


def test_llm_fallback_is_not_counted_as_an_llm_candidate(tmp_path: Path) -> None:
    case, compiled = _compiled_lead_case()
    response = _payload_response(deepcopy(compiled["runtime_payload"]))
    response["observations"][0]["data"]["llm"] = {
        "used": True,
        "ok": False,
        "fallback_reason": "llm_call_failed:URLError",
    }
    service = FakeService([response])

    result = _run_v3_case(
        service,  # type: ignore[arg-type]
        case,
        compiled,
        llm_config_path="llm.local.json",
        runtime_policy={"allow_in_memory": False, "env": {}},
        golden_dir=tmp_path,
        allow_unreviewed_goldens=False,
    )

    assert result["failure_category"] == "llm_unavailable"
    assert result["runtime_attempted"] is False
    assert result["llm_report"]["fallback_reason"] == "llm_call_failed:URLError"
    assert len(service.requests) == 1


def test_matching_candidate_uses_pending_action_id_for_confirmation(tmp_path: Path) -> None:
    case, compiled = _compiled_lead_case()
    service = FakeService(
        [
            _payload_response(deepcopy(compiled["runtime_payload"])),
            {
                "ok": True,
                "terminated_reason": "waiting_confirmation",
                "observations": [],
                "pending_action": {"action_id": "v3-action-exact"},
                "state": {"metadata": {"turn_understanding": {"source": "fallback"}}},
            },
            {
                "ok": True,
                "terminated_reason": "observed",
                "observations": [
                    {
                        "source": "geant4_runtime_tool",
                        "status": "ok",
                        "data": {"adapter": "local_process", "result_summary": {}},
                    }
                ],
                "state": {"metadata": {"turn_understanding": {"source": "explicit_event"}}},
            },
        ]
    )

    result = _run_v3_case(
        service,  # type: ignore[arg-type]
        case,
        compiled,
        llm_config_path="llm.local.json",
        runtime_policy={"allow_in_memory": False, "env": {}},
        golden_dir=tmp_path,
        allow_unreviewed_goldens=False,
    )

    assert result["runtime_attempted"] is True
    assert result["failure_category"] == "missing_metric"
    assert len(service.requests) == 3
    assert service.requests[1]["run"] is True
    assert service.requests[2]["confirmation_event"] == {
        "action_id": "v3-action-exact",
        "decision": "confirm",
    }
    assert all(request["allow_in_memory"] is False for request in service.requests)


def test_valid_noncanonical_candidate_runs_without_canonical_golden_comparison(tmp_path: Path) -> None:
    case, compiled = _compiled_lead_case()
    candidate = _valid_noncanonical_lead_payload(compiled)
    service = FakeService(
        [
            _payload_response(candidate),
            {
                "ok": True,
                "terminated_reason": "waiting_confirmation",
                "observations": [],
                "pending_action": {"action_id": "v3-action-semantic"},
                "state": {"metadata": {"turn_understanding": {"source": "fallback"}}},
            },
            {
                "ok": True,
                "terminated_reason": "observed",
                "observations": [
                    {
                        "source": "geant4_runtime_tool",
                        "status": "ok",
                        "data": {
                            "adapter": "local_process",
                            "result_summary": {
                                "run": {"events_completed": 10000},
                                "scoring": {
                                    "detector_crossing": {"detector_crossing_count": 4000},
                                    "roles": {"detector": {"edep_total_mev": 42.0}},
                                },
                            },
                        },
                    }
                ],
                "state": {"metadata": {"turn_understanding": {"source": "explicit_event"}}},
            },
        ]
    )

    result = _run_v3_case(
        service,  # type: ignore[arg-type]
        case,
        compiled,
        llm_config_path="llm.local.json",
        runtime_policy={"allow_in_memory": False, "env": {}},
        golden_dir=tmp_path,
        allow_unreviewed_goldens=False,
    )

    assert result["status"] == "passed"
    assert result["comparison_scope"] == "semantic_contract_and_real_runtime"
    assert result["candidate_contract"]["ok"] is True
    assert result["canonical_alignment"]["ok"] is False
    assert result["golden_comparison"]["performed"] is False
    assert result["actual_metrics"]["transmission_factor"] == 0.4
    assert service.requests[2]["confirmation_event"]["action_id"] == "v3-action-semantic"


def test_canonical_candidate_still_uses_reviewed_golden_comparison() -> None:
    case, compiled = _compiled_lead_case()
    service = FakeService(
        [
            _payload_response(deepcopy(compiled["runtime_payload"])),
            {
                "ok": True,
                "terminated_reason": "waiting_confirmation",
                "observations": [],
                "pending_action": {"action_id": "v3-action-canonical"},
                "state": {"metadata": {"turn_understanding": {"source": "fallback"}}},
            },
            {
                "ok": True,
                "terminated_reason": "observed",
                "observations": [
                    {
                        "source": "geant4_runtime_tool",
                        "status": "ok",
                        "data": {
                            "adapter": "local_process",
                            "result_summary": {
                                "run": {"events_completed": 10000},
                                "scoring": {
                                    "detector_crossing": {"detector_crossing_count": 4659},
                                    "roles": {"detector": {"edep_total_mev": 53.9888}},
                                },
                            },
                        },
                    }
                ],
                "state": {"metadata": {"turn_understanding": {"source": "explicit_event"}}},
            },
        ]
    )

    result = _run_v3_case(
        service,  # type: ignore[arg-type]
        case,
        compiled,
        llm_config_path="llm.local.json",
        runtime_policy={"allow_in_memory": False, "env": {}},
        golden_dir=ROOT / "docs" / "eval" / "golden" / "industrial_runtime",
        allow_unreviewed_goldens=False,
    )

    assert result["status"] == "passed"
    assert result["comparison_scope"] == "canonical_golden"
    assert result["canonical_alignment"]["ok"] is True
    assert result["golden_comparison"] == {"performed": True, "ok": True}


def test_paired_case_is_explicitly_deferred_without_service_call(tmp_path: Path) -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    case = next(item for item in benchmark["cases"] if item["id"] == "shielding_concrete_gamma_transmission")
    compiled = compile_industrial_case_to_runtime(case, runtime_defaults=benchmark["runtime_defaults"])
    service = FakeService([])

    result = _run_v3_case(
        service,  # type: ignore[arg-type]
        case,
        compiled,
        llm_config_path="llm.local.json",
        runtime_policy={"allow_in_memory": False, "env": {}},
        golden_dir=tmp_path,
        allow_unreviewed_goldens=False,
    )

    assert result["failure_category"] == "v3_paired_runtime_not_supported"
    assert service.requests == []
