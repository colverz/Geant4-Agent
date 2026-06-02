from __future__ import annotations

from core.agent_v3.contracts import V3AgentState, V3Observation, V3ObservationStatus, V3TurnInput
from core.agent_v3.turn_understanding import (
    LLMTurnUnderstandingProvider,
    build_v3_turn_understanding,
    normalize_confirmation_event,
)


def test_confirmation_event_becomes_first_class_understanding() -> None:
    turn = V3TurnInput(
        session_id="s1",
        user_text="confirm run",
        metadata={"confirmation_event": {"action_id": "run-1", "decision": "confirm"}},
    )

    understanding = build_v3_turn_understanding(turn, V3AgentState(session_id="s1"))

    assert understanding.dialogue_act == "confirm"
    assert understanding.confirmation == "confirmed"
    assert understanding.referenced_state == "pending_action"
    assert understanding.source == "explicit_event"
    assert understanding.to_dict()["schema_version"] == "geant4_agent_v3_turn_understanding.v1"


def test_cancel_event_normalizes_to_rejected() -> None:
    event = normalize_confirmation_event({"action_id": "run-1", "decision": "cancel"})

    assert event == {"action_id": "run-1", "decision": "reject"}


def test_config_overrides_become_requested_changes() -> None:
    state = V3AgentState(session_id="s1")
    state.add_observation(
        V3Observation(
            source="geant4_payload_builder_tool",
            status=V3ObservationStatus.OK,
            data={"simulation_spec": {}},
        )
    )
    turn = V3TurnInput(
        session_id="s1",
        user_text="change energy to 2 MeV",
        metadata={"config_overrides": {"source_energy_mev": 2.0}},
    )

    understanding = build_v3_turn_understanding(turn, state)
    payload = understanding.to_dict()

    assert payload["dialogue_act"] == "revise"
    assert payload["referenced_state"] == "payload"
    assert payload["requested_changes"] == [
        {"field": "source_energy_mev", "value": 2.0, "unit": "MeV", "evidence": "config_overrides"}
    ]


def test_empty_turn_understanding_is_safe_fallback() -> None:
    turn = V3TurnInput(session_id="s1", user_text="what can you do?")

    understanding = build_v3_turn_understanding(turn, None)

    assert understanding.dialogue_act == "ask"
    assert understanding.confirmation == "not_applicable"
    assert understanding.risk_intent == "read_only"


def test_llm_turn_understanding_prompt_uses_context_pack_not_raw_state() -> None:
    state = V3AgentState(session_id="s1", goal="old lead goal")
    state.metadata["raw_secret"] = "do-not-copy"
    state.add_observation(
        V3Observation(
            source="geant4_runtime_tool",
            status=V3ObservationStatus.OK,
            data={
                "runtime_payload": {
                    "geometry": {"material": "G4_WATER", "structure": "single_box", "raw_secret": "raw-leak-token"},
                    "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                    "large_raw_payload": {"nested": ["not copied"]},
                },
                "result_summary": {
                    "run": {"events_requested": 1000, "events_completed": 1000},
                    "configuration": {"material": "G4_WATER", "particle": "proton", "source_type": "beam"},
                    "scoring": {"target": {"target_edep_total_mev": 10.0}},
                },
            },
        )
    )
    turn = V3TurnInput(session_id="s1", user_text="explain this", metadata={})
    provider = LLMTurnUnderstandingProvider("fake.json")

    prompt = provider._prompt(turn, state)

    assert "V3ContextPack" in prompt
    assert "G4_WATER" in prompt
    assert "proton" in prompt
    assert "100.0" in prompt
    assert "do-not-copy" not in prompt
    assert "large_raw_payload" not in prompt
    assert "raw-leak-token" not in prompt


def test_llm_turn_understanding_parses_requested_changes_without_mutating_state() -> None:
    state = V3AgentState(session_id="s1")
    state.add_observation(V3Observation(source="geant4_payload_builder_tool", status=V3ObservationStatus.OK))
    turn = V3TurnInput(session_id="s1", user_text="change energy to 2 MeV", metadata={})
    provider = LLMTurnUnderstandingProvider("fake.json")

    provider._call_llm = lambda prompt: "{}"  # type: ignore[method-assign]
    provider._parse_llm_response = lambda raw: {  # type: ignore[method-assign]
        "dialogue_act": "revise",
        "user_goal": "change source energy",
        "referenced_state": "payload",
        "requested_changes": [{"field": "source_energy_mev", "value": 2.0, "unit": "MeV", "evidence": "user said 2 MeV"}],
        "confirmation": "not_applicable",
        "risk_intent": "draft_only",
        "confidence": 0.91,
        "reason": "explicit edit",
    }

    understanding = provider.understand(turn, state)

    assert understanding.source == "llm"
    assert understanding.dialogue_act == "revise"
    assert understanding.referenced_state == "payload"
    assert understanding.requested_changes[0].field == "source_energy_mev"
    assert understanding.requested_changes[0].value == 2.0
