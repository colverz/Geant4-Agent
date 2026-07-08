from __future__ import annotations

from core.agent_v3.contracts import V3AgentState, V3TurnInput
from core.agent_v3.pending_action import (
    EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
    PENDING_ACTION_SCHEMA_VERSION,
    V3PendingAction,
    V3PendingActionManager,
    runtime_authorization_source,
    runtime_execution_authorized,
)


def test_pending_action_manager_reads_legacy_metadata_dict() -> None:
    state = V3AgentState(
        session_id="s1",
        metadata={
            "pending_action": {
                "kind": "run_simulation",
                "tool_call": {"arguments": {"events": 7, "allow_in_memory": True}},
            }
        },
    )

    pending = V3PendingActionManager.get(state)

    assert pending is not None
    assert pending.kind == "run_simulation"
    assert pending.schema_version == PENDING_ACTION_SCHEMA_VERSION
    assert not V3PendingActionManager.confirmation_event_matches({}, pending)
    assert V3PendingActionManager.confirmation_event_matches({"action_id": pending.action_id}, pending)
    assert not V3PendingActionManager.confirmation_event_matches({"action_id": "wrong"}, pending)


def test_pending_action_manager_applies_authorization_without_losing_compatibility_flags() -> None:
    pending = V3PendingAction(
        action_id="v3-action-test",
        kind="run_simulation",
        intent="run_geant4_runtime",
        risk_level="runtime_execution",
        tool_call={
            "tool_name": "run_geant4",
            "arguments": {"events": 9, "allow_in_memory": True, "env": {"GEANT4_ROOT": "F:/Geant4"}},
        },
    )
    turn = V3TurnInput(session_id="s1", user_text="confirm run", metadata={"events": 1})

    authorization = V3PendingActionManager.apply_to_turn(
        turn,
        pending,
        confirmed_by="confirmation_event",
    )

    assert authorization is not None
    assert authorization.schema_version == EXECUTION_AUTHORIZATION_SCHEMA_VERSION
    assert authorization.action_id == "v3-action-test"
    assert turn.metadata["run"] is True
    assert turn.metadata["run_confirmed"] is True
    assert turn.metadata["accept_defaults"] is True
    assert turn.metadata["events"] == 9
    assert turn.metadata["allow_in_memory"] is True
    assert turn.metadata["runtime_env"] == {"GEANT4_ROOT": "F:/Geant4"}
    assert turn.metadata["execution_authorization"]["confirmed_by"] == "confirmation_event"
    assert runtime_execution_authorized(turn.metadata)
    assert runtime_authorization_source(turn.metadata) == "execution_authorization"


def test_pending_action_manager_stores_and_clears_compatible_dict() -> None:
    state = V3AgentState(session_id="s1")
    pending = V3PendingAction(action_id="v3-action-test", kind="run_simulation")

    V3PendingActionManager.store(state, pending)

    assert state.metadata["pending_action"]["schema_version"] == PENDING_ACTION_SCHEMA_VERSION
    assert V3PendingActionManager.get_dict(state)["action_id"] == "v3-action-test"

    V3PendingActionManager.clear(state)

    assert "pending_action" not in state.metadata


def test_legacy_run_confirmed_is_only_a_compatibility_authorization_source() -> None:
    metadata = {"run_confirmed": True}

    assert runtime_execution_authorized(metadata)
    assert runtime_authorization_source(metadata) == "legacy_run_confirmed"
