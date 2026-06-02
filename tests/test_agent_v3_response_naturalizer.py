from __future__ import annotations

from core.agent_v3.response_naturalizer import V3ResponseNaturalizer


def _response() -> dict:
    return {
        "display_message": "Runtime config ready: G4_WATER target, 100 MeV, 1000 events.",
        "dialogue_act": "payload_draft_presented",
        "answer_parts": [
            {"kind": "summary", "text": "Runtime config ready: G4_WATER target, 100 MeV, 1000 events."},
            {"kind": "evidence", "items": [{"source": "geant4_payload_builder_tool", "status": "ok"}]},
            {"kind": "next_step", "items": [{"text": "Confirm run", "prefill": "confirm run"}]},
        ],
        "evidence_used": [{"source": "geant4_payload_builder_tool", "status": "ok"}],
        "context": {
            "schema_version": "geant4_agent_v3_context.v1",
            "session_id": "naturalizer-test",
            "goal": "old lead shielding goal",
            "phase": "payload_ready",
            "latest_payload": {
                "material": "G4_WATER",
                "particle": "proton",
                "source_energy_mev": 100.0,
                "events": 1000,
            },
            "latest_runtime_facts": {},
            "last_user_turn": "prepare 100 MeV proton in water",
        },
        "summary": {"phase": "payload_ready", "next_action": "confirm_run_or_modify_payload"},
        "state": {
            "metadata": {"raw_secret": "do-not-copy"},
            "observations": [{"data": {"runtime_payload": {"raw_secret": "raw-leak-token"}}}],
        },
        "observations": [{"source": "geant4_payload_builder_tool", "data": {"raw_secret": "observation-secret"}}],
    }


def test_naturalizer_prompt_uses_safe_context_not_raw_state_or_observations() -> None:
    naturalizer = V3ResponseNaturalizer("fake.json")

    prompt = naturalizer.build_prompt(_response(), locale="en-US")

    assert "V3ContextPack" not in prompt
    assert "G4_WATER" in prompt
    assert "proton" in prompt
    assert "100.0" in prompt
    assert "do-not-copy" not in prompt
    assert "raw-leak-token" not in prompt
    assert "observation-secret" not in prompt
    assert "runtime_payload" not in prompt


def test_naturalizer_accepts_grounded_message() -> None:
    naturalizer = V3ResponseNaturalizer("fake.json")
    naturalizer._call_llm = lambda prompt: '{"display_message":"The draft keeps the G4_WATER proton setup at 100 MeV and is waiting for confirmation."}'  # type: ignore[method-assign]

    result = naturalizer.naturalize(_response(), locale="en-US")

    assert result.ok is True
    assert result.used_llm is True
    assert "G4_WATER" in result.display_message
    assert "100 MeV" in result.display_message


def test_naturalizer_rejects_material_particle_and_energy_conflicts() -> None:
    naturalizer = V3ResponseNaturalizer("fake.json")
    naturalizer._call_llm = lambda prompt: '{"display_message":"This is a 1 MeV gamma beam through lead shielding."}'  # type: ignore[method-assign]

    result = naturalizer.naturalize(_response(), locale="en-US")

    assert result.ok is False
    assert result.display_message == _response()["display_message"]
    assert result.fallback_reason in {"material_conflict", "particle_conflict", "source_energy_conflict"}


def test_naturalizer_rejects_internal_marker_leaks() -> None:
    naturalizer = V3ResponseNaturalizer("fake.json")
    naturalizer._call_llm = lambda prompt: '{"display_message":"See state.metadata and raw trace for details."}'  # type: ignore[method-assign]

    result = naturalizer.naturalize(_response(), locale="en-US")

    assert result.ok is False
    assert result.fallback_reason == "raw_internal_marker_detected"
    assert result.to_dict()["fallback_category"] == "safety_rejected"


def test_naturalizer_classifies_network_errors() -> None:
    naturalizer = V3ResponseNaturalizer("fake.json")

    def fail(prompt: str) -> str:
        raise TimeoutError("network timeout")

    naturalizer._call_llm = fail  # type: ignore[method-assign]

    result = naturalizer.naturalize(_response(), locale="en-US")

    assert result.ok is False
    assert result.fallback_reason == "naturalizer_error:TimeoutError"
    assert result.to_dict()["fallback_category"] == "network_error"
