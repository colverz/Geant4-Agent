from __future__ import annotations

from core.agent_v3.context import (
    V3_CONTEXT_SCHEMA_VERSION,
    V3_STATE_SUMMARY_SCHEMA_VERSION,
    build_v3_context_pack,
    build_v3_state_summary,
    update_v3_workflow_state,
)
from core.agent_v3.contracts import V3AgentState, V3Observation, V3ObservationStatus
from core.agent_v3.tools.geant4_tools import (
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
)


def test_context_pack_uses_safe_summaries_not_raw_runtime_payload() -> None:
    state = V3AgentState(session_id="ctx-1", goal="old lead goal")
    state.add_observation(
        V3Observation(
            source=GEANT4_RUNTIME_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "runtime_payload": {
                    "geometry": {"material": "G4_WATER", "structure": "single_box", "raw_secret": "do-not-copy"},
                    "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                    "large_raw_payload": {"nested": ["not copied"]},
                },
                "result_summary": {
                    "run": {"events_requested": 1000, "events_completed": 999},
                    "configuration": {"material": "G4_WATER", "particle": "proton", "source_type": "beam"},
                    "scoring": {
                        "target": {"target_edep_total_mev": 123.4},
                        "detector_crossing": {"detector_crossing_count": 2},
                        "plane_crossing": {"plane_crossing_count": 1},
                    },
                },
            },
        )
    )

    context = build_v3_context_pack(state, last_user_turn="explain")

    assert context["schema_version"] == V3_CONTEXT_SCHEMA_VERSION
    facts = context["latest_runtime_facts"]
    assert facts["material"] == "G4_WATER"
    assert facts["particle"] == "proton"
    assert facts["source_energy_mev"] == 100.0
    assert "runtime_payload" not in context
    assert "large_raw_payload" not in str(context)
    assert "do-not-copy" not in str(context)


def test_context_pack_summarizes_target_thickness_without_raw_geometry() -> None:
    state = V3AgentState(session_id="ctx-thickness", goal="gamma shielding")
    state.add_observation(
        V3Observation(
            source=GEANT4_RUNTIME_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "runtime_payload": {
                    "geometry": {
                        "material": "G4_Pb",
                        "structure": "single_box",
                        "params": {"module_x": 100.0, "module_y": 100.0, "module_z": 20.0},
                    },
                    "source": {"particle": "gamma", "energy_mev": 1.0, "type": "beam"},
                },
                "result_summary": {
                    "run": {"events_requested": 10, "events_completed": 10},
                    "configuration": {"material": "G4_Pb", "particle": "gamma", "source_type": "beam"},
                    "scoring": {
                        "detector_crossing": {"detector_crossing_count": 0},
                        "plane_crossing": {"plane_crossing_count": 0},
                    },
                },
            },
        )
    )

    context = build_v3_context_pack(state, last_user_turn="explain")

    facts = context["latest_runtime_facts"]
    assert facts["target_thickness_mm"] == 20.0
    assert "module_x" not in str(context)
    assert "module_y" not in str(context)


def test_workflow_state_summary_tracks_phase_plan_and_assumptions() -> None:
    state = V3AgentState(session_id="ctx-2", goal="gamma shielding")
    state.add_observation(
        V3Observation(
            source=GEANT4_DESIGN_TEMPLATE_TOOL,
            status=V3ObservationStatus.OK,
            data={"design": {"assumptions": ["detector downstream"], "recommended_setup": {"material": "G4_Pb"}}},
        )
    )
    state.add_observation(
        V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "recommended_config_assumptions": ["detector downstream", "events=1000"],
                "simulation_spec": {
                    "geometry": {"material": "G4_Pb", "structure": "single_box"},
                    "source": {"particle": "gamma", "energy_mev": 1.0, "type": "beam"},
                    "run": {"events": 1000},
                },
            },
        )
    )
    state.metadata["pending_action"] = {"kind": "run_simulation", "risk_level": "runtime_execution", "requires_confirmation": True}

    update_v3_workflow_state(state, last_user_turn="accept defaults")
    summary = build_v3_state_summary(state)

    assert summary["schema_version"] == V3_STATE_SUMMARY_SCHEMA_VERSION
    assert summary["phase"] == "await_confirmation"
    assert summary["needs_confirmation"] is True
    assert summary["runtime_ready"] is True
    assert summary["runtime_ready_reason"] == "awaiting_user_confirmation"
    assert summary["has_preflight"] is False
    assert summary["preflight_status"] == ""
    assert state.active_plan == ["draft_design", "build_payload", "await_confirmation"]
    assert state.assumptions == ["detector downstream", "events=1000"]


def test_payload_without_preflight_is_not_runtime_ready() -> None:
    state = V3AgentState(session_id="ctx-3", goal="run proton in water")
    state.add_observation(
        V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "simulation_spec": {
                    "geometry": {"material": "G4_WATER", "structure": "single_box"},
                    "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                    "run": {"events": 1000},
                }
            },
        )
    )

    update_v3_workflow_state(state, last_user_turn="accept defaults")
    summary = build_v3_state_summary(state)

    assert summary["phase"] == "payload_ready"
    assert summary["has_payload"] is True
    assert summary["runtime_ready"] is False
    assert summary["runtime_ready_reason"] == "preflight_not_run"
    assert summary["has_preflight"] is False
    assert summary["next_action"] == "confirm_run_or_modify_payload"
    assert state.active_plan == ["build_payload"]


def test_failed_preflight_is_not_runtime_ready_and_sets_fix_phase() -> None:
    state = V3AgentState(session_id="ctx-4", goal="run proton in water")
    state.add_observation(
        V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "simulation_spec": {
                    "geometry": {"material": "G4_WATER", "structure": "single_box"},
                    "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                    "run": {"events": 1000},
                }
            },
        )
    )
    state.add_observation(
        V3Observation(
            source=GEANT4_RUNTIME_PREFLIGHT_TOOL,
            status=V3ObservationStatus.NOT_EVALUABLE,
            not_evaluable_reason="local_process_runtime_required",
        )
    )

    update_v3_workflow_state(state, last_user_turn="run it")
    summary = build_v3_state_summary(state)

    assert summary["phase"] == "runtime_preflight_blocked"
    assert summary["runtime_ready"] is False
    assert summary["runtime_ready_reason"] == "local_process_runtime_required"
    assert summary["has_preflight"] is True
    assert summary["preflight_status"] == "not_evaluable"
    assert summary["next_action"] == "fix_runtime_or_modify_payload"
    assert state.active_plan == ["build_payload", "fix_preflight"]


def test_ok_preflight_marks_runtime_ready_without_pending_action() -> None:
    state = V3AgentState(session_id="ctx-5", goal="run proton in water")
    state.add_observation(
        V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "simulation_spec": {
                    "geometry": {"material": "G4_WATER", "structure": "single_box"},
                    "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                    "run": {"events": 1000},
                }
            },
        )
    )
    state.add_observation(V3Observation(source=GEANT4_RUNTIME_PREFLIGHT_TOOL, status=V3ObservationStatus.OK))

    update_v3_workflow_state(state, last_user_turn="run it")
    summary = build_v3_state_summary(state)

    assert summary["phase"] == "runtime_preflight"
    assert summary["runtime_ready"] is True
    assert summary["runtime_ready_reason"] == "preflight_ok"
    assert summary["has_preflight"] is True
    assert summary["preflight_status"] == "ok"
    assert summary["next_action"] == "confirm_run_or_fix_runtime"
    assert state.active_plan == ["build_payload", "run_preflight"]
