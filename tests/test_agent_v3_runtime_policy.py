from __future__ import annotations

from core.agent_v3.contracts import V3AgentState, V3Observation, V3ObservationStatus, V3TurnInput
from core.agent_v3.reasoners import BasicGeant4Reasoner
from core.agent_v3.runtime_policy import (
    RUNTIME_POLICY_SCHEMA_VERSION,
    V3RuntimePolicy,
    runtime_policy_from_turn,
    runtime_tool_arguments,
    set_runtime_policy,
)
from core.agent_v3.tools.geant4_tools import GEANT4_PAYLOAD_BUILDER_TOOL, GEANT4_RUNTIME_PREFLIGHT_TOOL
from core.agent_v3.tools.geant4_tools import (
    GEANT4_CAPABILITY_TOOL,
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_RUNTIME_TOOL,
)


def test_runtime_policy_from_payload_shape_syncs_legacy_fields() -> None:
    policy = V3RuntimePolicy.from_payload(
        {
            "allow_in_memory": True,
            "runtime_policy": {"backend_preference": "in_memory", "source": "test"},
        },
        runtime_env={"GEANT4_ROOT": "F:/Geant4"},
    )
    metadata: dict[str, object] = {}

    set_runtime_policy(metadata, policy)

    assert metadata["runtime_policy"]["schema_version"] == RUNTIME_POLICY_SCHEMA_VERSION
    assert metadata["runtime_policy"]["allow_in_memory"] is True
    assert metadata["runtime_policy"]["backend_preference"] == "in_memory"
    assert metadata["allow_in_memory"] is True
    assert metadata["runtime_env"] == {"GEANT4_ROOT": "F:/Geant4"}


def test_runtime_tool_arguments_prefers_structured_policy_over_legacy_flag() -> None:
    turn = V3TurnInput(
        session_id="s1",
        user_text="run",
        metadata={
            "allow_in_memory": False,
            "runtime_policy": {
                "allow_in_memory": True,
                "env": {"GEANT4_ROOT": "F:/Geant4"},
                "backend_preference": "in_memory",
            },
        },
    )

    args = runtime_tool_arguments(turn)

    assert args == {"allow_in_memory": True, "env": {"GEANT4_ROOT": "F:/Geant4"}}
    assert turn.metadata["allow_in_memory"] is True
    assert turn.metadata["runtime_backend"] == "in_memory"


def test_basic_reasoner_uses_runtime_policy_for_preflight_tool_arguments() -> None:
    state = V3AgentState(session_id="s1", goal="gamma shielding")
    state.observations.append(
        V3Observation(
            source=GEANT4_CAPABILITY_TOOL,
            status=V3ObservationStatus.OK,
            data={"runtime_capabilities": {"supported": True}},
        )
    )
    state.observations.append(
        V3Observation(
            source=GEANT4_DESIGN_TEMPLATE_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "design": {
                    "goal": "gamma shielding",
                    "recommended_setup": {"geometry": "single_box", "material": "G4_Pb", "source": "beam"},
                    "next_action": "build_candidate_config",
                }
            },
        )
    )
    state.observations.append(
        V3Observation(
            source=GEANT4_PAYLOAD_BUILDER_TOOL,
            status=V3ObservationStatus.OK,
            data={
                "recommended_config": {
                    "geometry": {"material": "G4_Pb"},
                    "source": {"particle": "gamma", "energy_mev": 1.0},
                    "run": {"events": 5},
                }
            },
        )
    )
    turn = V3TurnInput(
        session_id="s1",
        user_text="run it",
        metadata={
            "run": True,
            "events": 5,
            "runtime_policy": {"allow_in_memory": True, "env": {"GEANT4_ROOT": "F:/Geant4"}},
        },
    )

    proposal = BasicGeant4Reasoner().propose(turn, state)

    assert proposal.tool_call is not None
    assert proposal.tool_call.tool_name == GEANT4_RUNTIME_PREFLIGHT_TOOL
    assert proposal.tool_call.arguments["allow_in_memory"] is True
    assert proposal.tool_call.arguments["env"] == {"GEANT4_ROOT": "F:/Geant4"}


def test_runtime_policy_from_turn_keeps_legacy_metadata_available() -> None:
    turn = V3TurnInput(
        session_id="s1",
        user_text="run",
        metadata={"allow_in_memory": True, "runtime_env": {"GEANT4_ROOT": "F:/Geant4"}},
    )

    policy = runtime_policy_from_turn(turn)

    assert policy.allow_in_memory is True
    assert turn.metadata["runtime_policy"]["allow_in_memory"] is True
    assert turn.metadata["allow_in_memory"] is True


def test_basic_reasoner_prefers_execution_authorization_for_runtime_confirmation() -> None:
    state = _runtime_ready_state()
    turn = V3TurnInput(
        session_id="s1",
        user_text="confirm run",
        metadata={
            "run": True,
            "events": 5,
            "execution_authorization": {
                "schema_version": "geant4_agent_v3_execution_authorization.v1",
                "action_id": "v3-action-test",
                "confirmed_by": "user_text",
                "confirmed_turn_text": "confirm run",
                "authorized_tool_call_hash": "abc123",
            },
            "runtime_policy": {"allow_in_memory": True},
        },
    )

    proposal = BasicGeant4Reasoner().propose(turn, state)

    assert proposal.kind.value == "run_simulation"
    assert proposal.tool_call is not None
    assert proposal.tool_call.tool_name == GEANT4_RUNTIME_TOOL
    assert proposal.confirmed is True
    assert proposal.requires_confirmation is False
    assert proposal.arguments["authorization_source"] == "execution_authorization"


def test_basic_reasoner_requires_confirmation_without_execution_authorization() -> None:
    state = _runtime_ready_state()
    turn = V3TurnInput(
        session_id="s1",
        user_text="run it",
        metadata={"run": True, "events": 5, "runtime_policy": {"allow_in_memory": True}},
    )

    proposal = BasicGeant4Reasoner().propose(turn, state)

    assert proposal.kind.value == "run_simulation"
    assert proposal.confirmed is False


def _runtime_ready_state() -> V3AgentState:
    state = V3AgentState(session_id="s1", goal="gamma shielding")
    state.observations.extend(
        [
            V3Observation(
                source=GEANT4_CAPABILITY_TOOL,
                status=V3ObservationStatus.OK,
                data={"runtime_capabilities": {"supported": True}},
            ),
            V3Observation(
                source=GEANT4_DESIGN_TEMPLATE_TOOL,
                status=V3ObservationStatus.OK,
                data={
                    "design": {
                        "goal": "gamma shielding",
                        "recommended_setup": {"geometry": "single_box", "material": "G4_Pb", "source": "beam"},
                        "next_action": "build_candidate_config",
                    }
                },
            ),
            V3Observation(
                source=GEANT4_PAYLOAD_BUILDER_TOOL,
                status=V3ObservationStatus.OK,
                data={
                    "recommended_config": {
                        "geometry": {"material": "G4_Pb"},
                        "source": {"particle": "gamma", "energy_mev": 1.0},
                        "run": {"events": 5},
                    }
                },
            ),
            V3Observation(
                source=GEANT4_RUNTIME_PREFLIGHT_TOOL,
                status=V3ObservationStatus.OK,
                data={"config_ok": True},
            ),
        ]
    )
    return state
