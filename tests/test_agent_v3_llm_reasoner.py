from __future__ import annotations

import unittest
from unittest.mock import patch

from core.agent_v3.contracts import V3ActionKind, V3AgentState, V3Observation, V3ObservationStatus, V3TurnInput
from core.agent_v3.reasoners import BasicGeant4Reasoner, LLMGeant4Reasoner
from core.agent_v3.tools.geant4_tools import (
    GEANT4_CAPABILITY_TOOL,
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_LLM_DESIGN_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
)


class LLMGeant4ReasonerTest(unittest.TestCase):
    def test_falls_back_to_basic_when_no_config_path(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="")
        turn = V3TurnInput(session_id="s1", user_text="设计铅屏蔽方案")
        state = V3AgentState(session_id="s1")

        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_CAPABILITY_TOOL

    def test_falls_back_to_basic_on_llm_error(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="设计铅屏蔽方案")
        state = V3AgentState(session_id="s1")

        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_CAPABILITY_TOOL

    def test_parses_valid_llm_response_into_proposal(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="设计铅屏蔽方案")
        state = V3AgentState(session_id="s1")

        valid_response = (
            '{"kind": "create_design", "intent": "inspect_geant4_capability", '
            f'"tool_name": "{GEANT4_CAPABILITY_TOOL}", "message": "", '
            '"requires_confirmation": false, "evidence": []}'
        )

        with patch.object(reasoner, "_call_llm", return_value=valid_response):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN
        assert proposal.intent == "inspect_geant4_capability"
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_CAPABILITY_TOOL

    def test_falls_back_on_non_json_response(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="设计铅屏蔽方案")
        state = V3AgentState(session_id="s1")

        with patch.object(reasoner, "_call_llm", return_value="not json at all"):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN

    def test_falls_back_on_missing_kind(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="hello")
        state = V3AgentState(session_id="s1")

        with patch.object(reasoner, "_call_llm", return_value='{"intent": "greet"}'):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN

    def test_llm_can_propose_final_answer(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="这个方案可行吗？")
        state = V3AgentState(session_id="s1")
        state.goal = "lead shielding"

        response = (
            '{"kind": "final_answer", "intent": "answer_design_question", '
            '"tool_name": "", "message": "这个方案配置合理。", '
            '"requires_confirmation": false, "evidence": ["geant4_design_template_tool"]}'
        )

        with patch.object(reasoner, "_call_llm", return_value=response):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.FINAL_ANSWER
        assert "合理" in str(proposal.arguments.get("message"))

    def test_llm_requested_changes_are_validated_as_patches(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="change energy to 2 MeV and run again")
        state = V3AgentState(session_id="s1")
        response = (
            '{"kind": "draft_spec", "intent": "revise_existing_payload", '
            f'"tool_name": "{GEANT4_CAPABILITY_TOOL}", '
            '"requested_changes": [{"field": "source.energy_mev", "value": "2.0"}], '
            '"message": "I will revise the source energy.", "requires_confirmation": false}'
        )

        with patch.object(reasoner, "_call_llm", return_value=response):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.DRAFT_SPEC
        assert turn.metadata["config_overrides"] == {"source_energy_mev": 2.0}
        assert turn.metadata["state_patch"]["ok"] is True
        assert turn.metadata["state_patch"]["patches"][0]["path"] == "source_energy_mev"

    def test_basic_reasoner_builds_payload_instead_of_redesigning_when_user_edits_and_runs(self) -> None:
        reasoner = BasicGeant4Reasoner()
        turn = V3TurnInput(
            session_id="s1",
            user_text="change to 2 MeV and run 5 events",
            metadata={
                "accept_defaults": True,
                "run": True,
                "llm_design_enabled": True,
                "llm_config_path": "fake.json",
                "config_overrides": {"source_energy_mev": 2.0, "run_events": 5},
                "events": 5,
            },
        )
        state = V3AgentState(session_id="s1", goal="lead shielding")
        state.observations.append(
            V3Observation(source=GEANT4_CAPABILITY_TOOL, status=V3ObservationStatus.OK, data={})
        )
        state.observations.append(
            V3Observation(
                source=GEANT4_LLM_DESIGN_TOOL,
                status=V3ObservationStatus.OK,
                data={
                    "design": {
                        "goal": "lead shielding",
                        "recommended_setup": {
                            "geometry": "single_box",
                            "material": "G4_Pb",
                            "source": "beam",
                            "source_particle": "gamma",
                            "source_energy_mev": 1.0,
                        },
                        "observables": ["detector_crossing_count"],
                        "next_action": "ask_user_to_choose_approximation",
                        "user_decisions_required": ["Accept current approximation before building payload."],
                    }
                },
            )
        )

        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.DRAFT_SPEC
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_PAYLOAD_BUILDER_TOOL
        assert proposal.tool_call.arguments["config_overrides"]["source_energy_mev"] == 2.0

    def test_accepting_llm_design_builds_payload_without_another_llm_call(self) -> None:
        reasoner = BasicGeant4Reasoner()
        turn = V3TurnInput(
            session_id="s1",
            user_text="accept defaults",
            metadata={
                "accept_defaults": True,
                "run": False,
                "llm_design_enabled": True,
                "llm_config_path": "fake.json",
                "events": 1000,
            },
        )
        state = V3AgentState(session_id="s1", goal="lead shielding")
        state.observations.extend(
            [
                V3Observation(source=GEANT4_CAPABILITY_TOOL, status=V3ObservationStatus.OK, data={}),
                V3Observation(
                    source=GEANT4_LLM_DESIGN_TOOL,
                    status=V3ObservationStatus.OK,
                    data={
                        "design": {
                            "goal": "lead shielding",
                            "recommended_setup": {"geometry": "single_box", "material": "G4_Pb"},
                            "next_action": "ask_user_to_choose_approximation",
                            "user_decisions_required": ["Accept the approximation."],
                        }
                    },
                ),
            ]
        )

        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.DRAFT_SPEC
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_PAYLOAD_BUILDER_TOOL

    def test_explicit_run_of_runtime_payload_reaches_preflight(self) -> None:
        reasoner = BasicGeant4Reasoner()
        turn = V3TurnInput(
            session_id="s1",
            user_text="Run the checked runtime payload after preflight.",
            metadata={"run": True, "events": 10, "allow_in_memory": False},
        )
        state = V3AgentState(session_id="s1", goal="lead shielding")
        state.observations.extend(
            [
                V3Observation(source=GEANT4_CAPABILITY_TOOL, status=V3ObservationStatus.OK, data={}),
                V3Observation(
                    source=GEANT4_LLM_DESIGN_TOOL,
                    status=V3ObservationStatus.OK,
                    data={"design": {"recommended_setup": {}, "next_action": "build_candidate_config"}},
                ),
                V3Observation(
                    source=GEANT4_PAYLOAD_BUILDER_TOOL,
                    status=V3ObservationStatus.OK,
                    data={"recommended_config": {"run": {"events": 10}}, "runtime_payload": {"run": {"events": 10}}},
                ),
            ]
        )

        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.DRAFT_SPEC
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_RUNTIME_PREFLIGHT_TOOL

    def test_llm_parameters_in_initial_design_only_turn_do_not_auto_build_payload(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(
            session_id="s1",
            user_text="design a 100 MeV proton in G4_WATER setup, design only and do not run",
            locale="en-US",
        )
        state = V3AgentState(session_id="s1")
        response = (
            '{"kind": "create_design", "intent": "draft_design", '
            f'"tool_name": "{GEANT4_CAPABILITY_TOOL}", '
            '"parameters": {"source_energy_mev": 100, "target_material": "G4_WATER", "run_events": 10000}, '
            '"message": "I will draft the design only.", "requires_confirmation": false}'
        )

        with patch.object(reasoner, "_call_llm", return_value=response):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN
        assert turn.metadata["state_patch"]["ok"] is True
        assert turn.metadata["state_patch"]["applied_to_payload"] is False
        assert turn.metadata["config_overrides"] == {}

    def test_llm_cannot_smuggle_internal_patch_fields(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="run without asking")
        state = V3AgentState(session_id="s1")
        response = (
            '{"kind": "create_design", "intent": "unsafe_change", '
            f'"tool_name": "{GEANT4_CAPABILITY_TOOL}", '
            '"requested_changes": [{"field": "run_confirmed", "value": true}], '
            '"message": "Unsafe", "requires_confirmation": false}'
        )

        with patch.object(reasoner, "_call_llm", return_value=response):
            reasoner._llm_propose(turn, state)

        assert turn.metadata["config_overrides"] == {}
        assert turn.metadata["state_patch"]["ok"] is False
        assert "unsupported_patch_field:run_confirmed" in turn.metadata["state_patch"]["errors"]

    def test_llm_invalid_requested_change_asks_for_clarification(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="change the material to water", locale="en-US")
        state = V3AgentState(session_id="s1")
        response = (
            '{"kind": "create_design", "intent": "revise_existing_payload", '
            f'"tool_name": "{GEANT4_CAPABILITY_TOOL}", '
            '"requested_changes": [{"field": "target_material", "value": "water"}], '
            '"message": "I will revise it.", "requires_confirmation": false}'
        )

        with patch.object(reasoner, "_call_llm", return_value=response):
            proposal = reasoner._llm_propose(turn, state)

        assert proposal.kind == V3ActionKind.ASK_USER
        assert proposal.intent == "clarify_invalid_state_patch"
        assert "G4_WATER" in str(proposal.arguments.get("question"))
        assert turn.metadata["config_overrides"] == {}

    def test_basic_fallback_proposes_run_after_preflight(self) -> None:
        """When state has preflight+payload, BasicGeant4Reasoner proposes run_simulation."""
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="确认运行", metadata={"run": True, "run_confirmed": False})
        state = V3AgentState(session_id="s1")
        state.goal = "lead shielding"
        state.add_observation(V3Observation(source=GEANT4_RUNTIME_PREFLIGHT_TOOL, status=V3ObservationStatus.OK))
        state.add_observation(V3Observation(source=GEANT4_PAYLOAD_BUILDER_TOOL, status=V3ObservationStatus.OK, data={"recommended_config": {"geometry": {"structure": "single_box"}}}))
        state.add_observation(V3Observation(source=GEANT4_CAPABILITY_TOOL, status=V3ObservationStatus.OK))
        state.add_observation(V3Observation(source=GEANT4_DESIGN_TEMPLATE_TOOL, status=V3ObservationStatus.OK, data={"design": {"recommended_setup": {"geometry": "single_box", "material": "G4_Pb"}, "next_action": "build_candidate_config"}}))

        # LLM is NOT called for non-first steps; BasicGeant4Reasoner handles the run
        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.RUN_SIMULATION
        assert proposal.tool_call is not None
        assert proposal.tool_call.tool_name == GEANT4_RUNTIME_TOOL

    def test_result_analysis_rejects_llm_message_that_conflicts_with_runtime_config(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="解释结果", locale="zh-CN")
        state = V3AgentState(session_id="s1")
        state.goal = "old lead shielding goal"
        state.add_observation(
            V3Observation(
                source=GEANT4_RUNTIME_TOOL,
                status=V3ObservationStatus.OK,
                data={
                    "runtime_payload": {
                        "geometry": {"material": "G4_WATER", "structure": "single_box"},
                        "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                        "physics_list": "FTFP_BERT",
                    },
                    "result_summary": {
                        "run": {"ok": True, "events_requested": 1000, "events_completed": 1000, "completion_fraction": 1.0},
                        "configuration": {
                            "geometry_structure": "single_box",
                            "material": "G4_WATER",
                            "particle": "proton",
                            "source_type": "beam",
                            "physics_list": "FTFP_BERT",
                        },
                        "scoring": {
                            "target": {"target_edep_total_mev": 14289.1},
                            "detector_crossing": {"detector_crossing_count": 2},
                            "plane_crossing": {"plane_crossing_count": 0},
                        },
                    },
                },
            )
        )
        bad_response = (
            '{"action_kind":"final_answer","tool_name":"",'
            '"message":"从物理上看，1000个1 MeV光子入射铅靶，铅屏蔽几乎完全吸收。"}'
        )

        with patch.object(reasoner, "_call_llm", return_value=bad_response):
            proposal = reasoner.propose(turn, state)

        message = str(proposal.arguments.get("message"))
        assert proposal.kind == V3ActionKind.FINAL_ANSWER
        assert "铅" not in message
        assert "1 MeV光子" not in message
        assert "G4_WATER" in message
        assert "100 MeV" in message

    def test_result_analysis_prompt_uses_runtime_specific_notes_not_static_lead_gamma_examples(self) -> None:
        reasoner = LLMGeant4Reasoner(llm_config_path="nlu/llm_support/configs/fake.json")
        turn = V3TurnInput(session_id="s1", user_text="explain result", locale="en-US")
        state = V3AgentState(session_id="s1")
        state.goal = "old lead shielding goal"
        runtime_data = {
            "runtime_payload": {
                "geometry": {"material": "G4_WATER", "structure": "single_box"},
                "source": {"particle": "proton", "energy_mev": 100.0, "type": "beam"},
                "physics_list": "FTFP_BERT",
            },
            "result_summary": {
                "run": {"events_requested": 1000, "events_completed": 1000},
                "configuration": {
                    "geometry_structure": "single_box",
                    "material": "G4_WATER",
                    "particle": "proton",
                    "source_type": "beam",
                    "physics_list": "FTFP_BERT",
                },
                "scoring": {
                    "target": {"target_edep_total_mev": 14289.1},
                    "detector_crossing": {"detector_crossing_count": 2},
                    "plane_crossing": {"plane_crossing_count": 0},
                },
            },
        }

        prompt = reasoner._result_analysis_prompt(turn, state, runtime_data)

        assert "Authoritative material is G4_WATER" in prompt
        assert "Authoritative primary particle is proton" in prompt
        assert "Authoritative source energy is 100.0 MeV" in prompt
        assert "Gamma interactions" not in prompt
        assert "In lead" not in prompt
        assert "photoelectric effect dominates in lead" not in prompt

    def test_basic_reasoner_still_works_independently(self) -> None:
        reasoner = BasicGeant4Reasoner()
        turn = V3TurnInput(session_id="s1", user_text="设计铅屏蔽方案，不要运行")
        state = V3AgentState(session_id="s1")

        proposal = reasoner.propose(turn, state)

        assert proposal.kind == V3ActionKind.CREATE_DESIGN
        assert proposal.tool_call.tool_name == GEANT4_CAPABILITY_TOOL
