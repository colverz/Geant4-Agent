from __future__ import annotations

from core.agent_v3.contracts import V3AgentState, V3Observation, V3ObservationStatus
from core.agent_v3.patches import (
    apply_patches_to_state,
    build_patches_from_config_overrides,
    build_patches_from_requested_changes,
    config_overrides_from_llm_parameters,
    stale_sources_for_patches,
)


def test_build_patches_from_config_overrides_normalizes_known_fields() -> None:
    result = build_patches_from_config_overrides(
        {
            "source_energy_mev": "2.5",
            "run_events": "12",
            "target_material": "G4_WATER",
            "target_thickness_mm": "20",
            "enable_downstream_scoring": "true",
        },
        evidence="unit-test",
    )

    assert result.ok
    assert result.config_overrides == {
        "source_energy_mev": 2.5,
        "run_events": 12,
        "target_material": "G4_WATER",
        "target_thickness_mm": 20.0,
        "enable_downstream_scoring": True,
    }
    assert [patch.path for patch in result.patches] == [
        "source_energy_mev",
        "run_events",
        "target_material",
        "target_thickness_mm",
        "enable_downstream_scoring",
    ]
    assert result.patches[0].to_dict()["evidence"] == "unit-test"


def test_build_patches_rejects_unknown_or_invalid_fields() -> None:
    result = build_patches_from_config_overrides(
        {
            "run_confirmed": True,
            "source_energy_mev": -1,
            "target_material": "water",
        }
    )

    assert not result.ok
    assert result.config_overrides == {}
    assert result.patches == []
    assert "unsupported_patch_field:run_confirmed" in result.errors
    assert "invalid_patch_value:source_energy_mev:must_be_positive" in result.errors
    assert "invalid_patch_value:target_material:must_be_geant4_material_id" in result.errors


def test_build_patches_are_atomic_when_any_field_is_invalid() -> None:
    result = build_patches_from_config_overrides(
        {
            "source_energy_mev": 2.0,
            "run_confirmed": True,
        }
    )

    assert not result.ok
    assert result.config_overrides == {}
    assert result.patches == []
    assert "unsupported_patch_field:run_confirmed" in result.errors


def test_stale_sources_for_patches_marks_runtime_chain_stale() -> None:
    result = build_patches_from_config_overrides({"source_energy_mev": 2.0})

    assert stale_sources_for_patches(result.patches) == {
        "geant4_payload_builder_tool",
        "geant4_runtime_preflight_tool",
        "geant4_runtime_tool",
        "commit_gate",
    }


def test_requested_changes_use_canonical_patch_fields() -> None:
    result = build_patches_from_requested_changes(
        [
            {"field": "source.energy_mev", "value": "3.0"},
            {"field": "run.events", "value": "25"},
        ]
    )

    assert result.ok
    assert result.config_overrides == {"source_energy_mev": 3.0, "run_events": 25}
    assert [patch.path for patch in result.patches] == ["source_energy_mev", "run_events"]


def test_llm_parameters_map_to_patch_overrides() -> None:
    assert config_overrides_from_llm_parameters(
        {
            "energy_mev": 5,
            "events": 20,
            "material": "G4_Pb",
            "particle": "gamma",
        }
    ) == {
        "source_energy_mev": 5,
        "run_events": 20,
        "target_material": "G4_Pb",
    }


def test_apply_patches_invalidates_stale_runtime_chain() -> None:
    state = V3AgentState(session_id="patch-apply")
    state.observations = [
        V3Observation(source="geant4_design_template_tool", status=V3ObservationStatus.OK, message="design"),
        V3Observation(source="geant4_payload_builder_tool", status=V3ObservationStatus.OK, message="payload"),
        V3Observation(source="geant4_runtime_preflight_tool", status=V3ObservationStatus.OK, message="preflight"),
        V3Observation(source="commit_gate", status=V3ObservationStatus.BLOCKED, message="pending"),
        V3Observation(source="geant4_runtime_tool", status=V3ObservationStatus.OK, message="runtime"),
    ]
    state.artifacts = {
        "geant4_design_draft": {"kept": True},
        "geant4_runtime_payload_draft": {"stale": True},
        "preflight_geant4_runtime": {"stale": True},
    }
    state.metadata = {
        "pending_action": {"kind": "run_simulation"},
        "suggested_next_actions": [{"text": "old"}],
    }
    patch_result = build_patches_from_config_overrides({"source_energy_mev": 2.0})

    apply_result = apply_patches_to_state(state, patch_result)

    assert apply_result.ok
    assert apply_result.applied
    assert [item.source for item in state.observations] == ["geant4_design_template_tool"]
    assert "geant4_design_draft" in state.artifacts
    assert "geant4_runtime_payload_draft" not in state.artifacts
    assert "preflight_geant4_runtime" not in state.artifacts
    assert "pending_action" not in state.metadata
    assert "suggested_next_actions" not in state.metadata
    assert state.metadata["last_state_patch_apply"]["removed_artifacts"] == [
        "geant4_runtime_payload_draft",
        "preflight_geant4_runtime",
    ]


def test_apply_invalid_patch_does_not_clear_state() -> None:
    state = V3AgentState(session_id="invalid-patch-apply")
    state.observations = [
        V3Observation(source="geant4_payload_builder_tool", status=V3ObservationStatus.OK, message="payload"),
    ]
    state.artifacts = {"geant4_runtime_payload_draft": {"kept": True}}
    state.metadata = {"pending_action": {"kind": "run_simulation"}}
    patch_result = build_patches_from_config_overrides({"source_energy_mev": -1})

    apply_result = apply_patches_to_state(state, patch_result)

    assert not apply_result.ok
    assert not apply_result.applied
    assert [item.source for item in state.observations] == ["geant4_payload_builder_tool"]
    assert state.artifacts == {"geant4_runtime_payload_draft": {"kept": True}}
    assert state.metadata == {"pending_action": {"kind": "run_simulation"}}
