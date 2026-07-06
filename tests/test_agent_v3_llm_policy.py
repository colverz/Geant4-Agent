from core.agent_v3.contracts import V3TurnInput
from core.agent_v3.llm_policy import LLM_POLICY_SCHEMA_VERSION, V3LlmPolicy, llm_policy_from_turn, set_llm_policy


def test_llm_policy_prefers_structured_payload_over_compatibility_fields() -> None:
    policy = V3LlmPolicy.from_payload(
        {
            "llm_understanding_enabled": True,
            "llm_policy": {
                "understanding_enabled": False,
                "planning_enabled": False,
                "design_enabled": True,
            },
        }
    )

    assert policy.understanding_enabled is False
    assert policy.planning_enabled is False
    assert policy.design_enabled is True


def test_llm_policy_keeps_compatibility_mirrors_in_sync() -> None:
    turn = V3TurnInput(session_id="s1", user_text="draft")
    set_llm_policy(turn.metadata, V3LlmPolicy(planning_enabled=False, design_enabled=True))

    restored = llm_policy_from_turn(turn)

    assert restored.schema_version == LLM_POLICY_SCHEMA_VERSION
    assert turn.metadata["llm_policy"]["planning_enabled"] is False
    assert turn.metadata["llm_planning_enabled"] is False
    assert turn.metadata["llm_design_enabled"] is True
