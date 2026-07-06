from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.agent_v3.contracts import V3AgentState, V3Observation, V3ObservationStatus
from core.agent_v3.service import V3_AGENT_TURN_SCHEMA_VERSION, V3AgentTurnService, build_turn_input
from core.agent_v3.tools.geant4_tools import GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_PAYLOAD_BUILDER_TOOL
from core.agent_v3.turn_understanding import V3RequestedChange, V3TurnUnderstanding
from ui.web.request_router import handle_post_request, is_supported_post_path


class V3AgentTurnServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._sessions_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_service(self) -> V3AgentTurnService:
        return V3AgentTurnService(sessions_dir=self._sessions_dir)

    def test_build_turn_input_accepts_pure_text_without_frontend_run_flag(self) -> None:
        turn, discovery = build_turn_input(
            {
                "session_id": "pure-language",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 3,
            }
        )

        self.assertIsNone(discovery)
        self.assertEqual(turn.session_id, "pure-language")
        self.assertEqual(turn.metadata["events"], 3)
        self.assertFalse(turn.metadata["run"])
        self.assertFalse(turn.metadata["run_confirmed"])
        self.assertFalse(turn.metadata["llm_design_enabled"])

    def test_build_turn_input_carries_llm_design_metadata(self) -> None:
        turn, _ = build_turn_input(
            {
                "session_id": "llm-design",
                "text": "帮我设计一个质子水箱剂量模拟",
                "llm_design_enabled": True,
                "llm_config_path": "nlu/llm_support/configs/local.json",
            }
        )

        self.assertTrue(turn.metadata["llm_design_enabled"])
        self.assertEqual(turn.metadata["llm_config_path"], "nlu/llm_support/configs/local.json")

    def test_build_turn_input_generates_unique_session_when_missing(self) -> None:
        first, _ = build_turn_input({"text": "hello"})
        second, _ = build_turn_input({"text": "hello"})

        self.assertTrue(first.session_id.startswith("v3-"))
        self.assertTrue(second.session_id.startswith("v3-"))
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertNotEqual(first.session_id, "default")

    def test_build_turn_input_does_not_extract_overrides(self) -> None:
        """config_overrides are now populated by LLMGeant4Reasoner, not regex."""
        turn, _ = build_turn_input(
            {
                "session_id": "no-extract",
                "text": "把刚才方案改成 2 MeV 再跑",
            }
        )
        self.assertEqual(turn.metadata["config_overrides"], {})

    def test_llm_turn_understanding_requested_change_drives_patch_and_payload_rebuild(self) -> None:
        service = self._make_service()
        state = V3AgentState(session_id="llm-revise-mainline", goal="proton water dose")
        state.add_observation(
            V3Observation(
                source=GEANT4_DESIGN_TEMPLATE_TOOL,
                status=V3ObservationStatus.OK,
                data={
                    "design": {
                        "goal": "proton water dose",
                        "recommended_setup": {"geometry": "single_box", "material": "G4_WATER", "source": "beam"},
                        "next_action": "build_candidate_config",
                    }
                },
            )
        )
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
        service.states[state.session_id] = state
        understanding = V3TurnUnderstanding(
            dialogue_act="revise",
            user_goal="change source energy to 2 MeV",
            referenced_state="payload",
            requested_changes=[
                V3RequestedChange(field="source_energy_mev", value=2.0, unit="MeV", evidence="user said 2 MeV")
            ],
            confirmation="not_applicable",
            risk_intent="draft_only",
            confidence=0.95,
            reason="explicit energy edit",
            source="llm",
        )

        with patch("core.agent_v3.service.LLMTurnUnderstandingProvider.understand", return_value=understanding):
            result = service.run_turn(
                {
                    "session_id": state.session_id,
                    "text": "change the source energy to 2 MeV",
                    "llm_config_path": "fake.json",
                    "lang": "en-US",
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["state"]["metadata"]["turn_understanding"]["source"], "llm")
        self.assertEqual(result["state"]["metadata"]["last_state_patch"]["config_overrides"]["source_energy_mev"], 2.0)
        self.assertEqual(result["state"]["metadata"]["last_state_patch_apply"]["stale_sources"], ["commit_gate", "geant4_payload_builder_tool", "geant4_runtime_preflight_tool", "geant4_runtime_tool"])
        self.assertTrue(result["summary"]["has_payload"])
        self.assertEqual(result["context"]["latest_payload"]["source_energy_mev"], 2.0)

    def test_current_configuration_question_answers_design_not_runtime_result(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "current-config-design",
                "text": "Design a gamma shielding setup, do not run.",
                "events": 5,
            }
        )

        result = service.run_turn(
            {
                "session_id": "current-config-design",
                "text": "What is currently configured?",
                "lang": "en",
            }
        )

        self.assertEqual(result["dialogue_act"], "configuration_answered")
        self.assertIn("Current design draft", result["display_message"])
        self.assertIn("G4_Pb", result["display_message"])
        self.assertNotIn("No runtime result", result["display_message"])

    def test_build_turn_input_keeps_confirmation_event_structured(self) -> None:
        turn, _ = build_turn_input(
            {
                "session_id": "event-confirm",
                "text": "confirm run",
                "confirmation_event": {"action_id": "run-1", "decision": "confirm"},
            }
        )

        self.assertFalse(turn.metadata["run_confirmed"])
        self.assertEqual(turn.metadata["confirmation_event"], {"action_id": "run-1", "decision": "confirm"})

    def test_build_turn_input_can_disable_llm_understanding_without_disabling_design(self) -> None:
        turn, _ = build_turn_input(
            {
                "session_id": "design-only-llm",
                "text": "draft a simulation",
                "llm_understanding_enabled": False,
                "llm_planning_enabled": False,
                "llm_design_enabled": True,
                "llm_config_path": "model.local.json",
            }
        )

        self.assertFalse(turn.metadata["llm_understanding_enabled"])
        self.assertFalse(turn.metadata["llm_planning_enabled"])
        self.assertTrue(turn.metadata["llm_design_enabled"])
        self.assertEqual(turn.metadata["llm_policy"]["schema_version"], "geant4_agent_v3_llm_policy.v1")

    def test_build_turn_input_ignores_public_run_confirmed_flag(self) -> None:
        turn, _ = build_turn_input(
            {
                "session_id": "external-run-confirmed",
                "text": "run now",
                "run": True,
                "run_confirmed": True,
                "allow_in_memory": True,
            }
        )

        self.assertTrue(turn.metadata["run"])
        self.assertFalse(turn.metadata["run_confirmed"])
        self.assertNotIn("execution_authorization", turn.metadata)

    def test_run_turn_uses_language_run_intent_and_waits_for_confirmation(self) -> None:
        service = self._make_service()
        result = service.run_turn(
            {
                "session_id": "confirm-flow",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 3,
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema_version"], V3_AGENT_TURN_SCHEMA_VERSION)
        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        self.assertIn("确认", result["answer"]["message"])
        self.assertEqual(result["observations"][-1]["source"], "commit_gate")
        self.assertIsNotNone(result["pending_action"])
        self.assertEqual(result["pending_action"]["kind"], "run_simulation")
        self.assertTrue(result["pending_action"]["requires_confirmation"])
        self.assertTrue(result["pending_action"]["action_id"].startswith("v3-action-"))
        self.assertEqual(result["pending_action"]["created_turn_id"], result["state"]["metadata"]["last_turn_id"])

        state_status, state_payload = service.get_state_payload("confirm-flow", lang="zh")
        self.assertEqual(state_status, 200)
        self.assertEqual(state_payload["summary"]["phase"], "await_confirmation")
        self.assertTrue(state_payload["summary"]["runtime_ready"])
        self.assertEqual(state_payload["summary"]["runtime_ready_reason"], "awaiting_user_confirmation")
        self.assertTrue(state_payload["summary"]["has_preflight"])
        self.assertEqual(state_payload["summary"]["preflight_status"], "ok")
        self.assertEqual(state_payload["summary"]["next_action"], "confirm_or_cancel_pending_action")

    def test_confirmation_turn_executes_pending_runtime_action(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "confirm-flow",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        result = service.run_turn(
            {
                "session_id": "confirm-flow",
                "text": "确认运行",
                "events": 2,
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "observed")
        self.assertEqual(result["observations"][-1]["source"], "geant4_runtime_tool")
        self.assertIsNone(result["pending_action"])
        self.assertIn("run_runtime", result["state"]["active_plan"])
        self.assertTrue(result["summary"]["has_runtime_result"])
        self.assertEqual(result["context"]["latest_runtime_facts"]["particle"], "gamma")
        self.assertEqual(result["state"]["metadata"]["turn_understanding"]["confirmation"], "confirmed")
        authorization = result["state"]["metadata"]["last_execution_authorization"]
        self.assertEqual(authorization["schema_version"], "geant4_agent_v3_execution_authorization.v1")
        self.assertTrue(authorization["action_id"].startswith("v3-action-"))
        self.assertEqual(authorization["confirmed_by"], "user_text")
        self.assertTrue(authorization["authorized_tool_call_hash"])

    def test_confirmation_event_executes_pending_runtime_action(self) -> None:
        service = self._make_service()
        first = service.run_turn(
            {
                "session_id": "confirm-event-flow",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        self.assertEqual(first["terminated_reason"], "waiting_confirmation")

        result = service.run_turn(
            {
                "session_id": "confirm-event-flow",
                "text": "confirm run",
                "events": 2,
                "allow_in_memory": True,
                "confirmation_event": {"action_id": first["pending_action"]["action_id"], "decision": "confirm"},
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "observed")
        self.assertIsNone(result["pending_action"])
        self.assertEqual(result["state"]["metadata"]["turn_understanding"]["source"], "explicit_event")
        authorization = result["state"]["metadata"]["last_execution_authorization"]
        self.assertEqual(authorization["action_id"], first["pending_action"]["action_id"])
        self.assertEqual(authorization["confirmed_by"], "confirmation_event")

    def test_short_confirmation_turn_executes_pending_runtime_action(self) -> None:
        service = self._make_service()
        first = service.run_turn(
            {
                "session_id": "short-confirm-flow",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        self.assertEqual(first["terminated_reason"], "waiting_confirmation")

        result = service.run_turn(
            {
                "session_id": "short-confirm-flow",
                "text": "确认",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "observed")
        self.assertEqual(result["observations"][-1]["source"], "geant4_runtime_tool")
        self.assertIsNone(result["pending_action"])

    def test_confirmation_without_pending_does_not_create_run_action_for_any_backend(self) -> None:
        service = self._make_service()

        for allow_in_memory in (False, True):
            with self.subTest(allow_in_memory=allow_in_memory):
                result = service.run_turn(
                    {
                        "session_id": f"confirm-no-pending-{allow_in_memory}",
                        "text": "confirm run",
                        "lang": "en",
                        "allow_in_memory": allow_in_memory,
                    }
                )

                self.assertTrue(result["ok"])
                self.assertEqual(result["terminated_reason"], "final_answer")
                self.assertIsNone(result["pending_action"])
                self.assertFalse(result["summary"]["has_payload"])
                self.assertFalse(result["summary"]["has_runtime_result"])
                self.assertNotIn("geant4_payload_builder_tool", [item["source"] for item in result["observations"]])
                self.assertNotIn("geant4_runtime_tool", [item["source"] for item in result["observations"]])
                self.assertIn("no pending run action", result["display_message"])
                self.assertEqual(result["state"]["metadata"]["turn_understanding"]["confirmation"], "confirmed")

    def test_confirmation_without_pending_after_payload_does_not_execute_or_auto_confirm(self) -> None:
        service = self._make_service()
        session_id = "confirm-no-pending-after-payload"
        service.run_turn(
            {
                "session_id": session_id,
                "text": "Design a gamma shielding setup, do not run.",
                "allow_in_memory": True,
            }
        )
        payload_result = service.run_turn(
            {
                "session_id": session_id,
                "text": "Accept defaults and build config.",
                "accept_defaults": True,
                "allow_in_memory": True,
            }
        )
        self.assertTrue(payload_result["summary"]["has_payload"])
        self.assertIsNone(payload_result["pending_action"])

        result = service.run_turn(
            {
                "session_id": session_id,
                "text": "confirm run",
                "lang": "en",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertIsNone(result["pending_action"])
        self.assertTrue(result["summary"]["has_payload"])
        self.assertFalse(result["summary"]["has_runtime_result"])
        self.assertNotIn("geant4_runtime_tool", [item["source"] for item in result["observations"]])
        self.assertIn("no pending run action", result["display_message"])

    def test_public_run_confirmed_request_does_not_bypass_pending_action(self) -> None:
        service = self._make_service()
        result = service.run_turn(
            {
                "session_id": "public-run-confirmed-blocked",
                "text": "run a default lead shielding gamma simulation",
                "run": True,
                "run_confirmed": True,
                "allow_in_memory": True,
                "events": 2,
            }
        )

        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        self.assertIsNotNone(result["pending_action"])
        self.assertFalse(result["summary"]["has_runtime_result"])
        self.assertNotIn("last_execution_authorization", result["state"]["metadata"])

    def test_cancel_turn_clears_pending_runtime_action(self) -> None:
        service = self._make_service()
        first = service.run_turn(
            {
                "session_id": "cancel-flow",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        self.assertIsNotNone(first["pending_action"])

        result = service.run_turn(
            {
                "session_id": "cancel-flow",
                "text": "取消运行，只保留方案",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertIsNone(result["pending_action"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertNotEqual(result["terminated_reason"], "observed")
        self.assertNotIn("await_confirmation", result["state"]["active_plan"])

    def test_cancel_event_clears_pending_runtime_action(self) -> None:
        service = self._make_service()
        first = service.run_turn(
            {
                "session_id": "cancel-event-flow",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        self.assertIsNotNone(first["pending_action"])

        result = service.run_turn(
            {
                "session_id": "cancel-event-flow",
                "text": "cancel run",
                "allow_in_memory": True,
                "confirmation_event": {"action_id": first["pending_action"]["action_id"], "decision": "cancel"},
            }
        )

        self.assertTrue(result["ok"])
        self.assertIsNone(result["pending_action"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertEqual(result["state"]["metadata"]["turn_understanding"]["confirmation"], "rejected")

    def test_mismatched_confirmation_event_does_not_execute_pending_action(self) -> None:
        service = self._make_service()
        first = service.run_turn(
            {
                "session_id": "confirm-event-mismatch",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        self.assertIsNotNone(first["pending_action"])

        result = service.run_turn(
            {
                "session_id": "confirm-event-mismatch",
                "text": "confirm run",
                "events": 2,
                "allow_in_memory": True,
                "confirmation_event": {"action_id": "wrong-action", "decision": "confirm"},
            }
        )

        self.assertTrue(result["ok"])
        self.assertNotEqual(result["terminated_reason"], "observed")
        self.assertIsNotNone(result["pending_action"])
        self.assertEqual(result["pending_action"]["action_id"], first["pending_action"]["action_id"])

    def test_result_followup_answers_runtime_observation_not_payload(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "result-followup",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": "result-followup",
                "text": "confirm run",
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": "result-followup",
                "text": "What does detector crossing mean in the result?",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertEqual(result["dialogue_act"], "runtime_result_answered")
        self.assertIn("detector_crossing_count", result["display_message"])
        self.assertNotIn("payload draft", result["display_message"].lower())

    def test_result_followup_without_runtime_does_not_create_design(self) -> None:
        service = self._make_service()
        result = service.run_turn(
            {
                "session_id": "result-followup-empty",
                "text": "What happened in the result?",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertIn("还没有可解释", result["display_message"])
        self.assertEqual(result["observations"], [])

    def test_modify_energy_rerun_rebuilds_payload_and_waits_for_confirmation(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "modify-rerun",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": "modify-rerun",
                "text": "确认",
                "events": 2,
                "allow_in_memory": True,
            }
        )

        # config_overrides injected directly (simulating LLM extraction)
        result = service.run_turn(
            {
                "session_id": "modify-rerun",
                "text": "把刚才方案改成 2 MeV 再跑",
                "events": 2,
                "allow_in_memory": True,
                "config_overrides": {"source_energy_mev": 2.0},
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        self.assertEqual(result["pending_action"]["kind"], "run_simulation")
        payload_observation = next(item for item in result["observations"] if item["source"] == "geant4_payload_builder_tool")
        self.assertEqual(payload_observation["data"]["applied_overrides"], {"source_energy_mev": 2.0})
        self.assertEqual(payload_observation["data"]["simulation_spec"]["source"]["energy_mev"], 2.0)
        self.assertEqual(result["state"]["open_questions"], [])
        self.assertEqual(
            result["state"]["metadata"]["last_state_patch"]["patches"][0]["path"],
            "source_energy_mev",
        )
        self.assertEqual(
            result["state"]["metadata"]["last_state_patch"]["config_overrides"],
            {"source_energy_mev": 2.0},
        )
        self.assertEqual(
            result["state"]["metadata"]["last_state_patch_apply"]["stale_sources"],
            [
                "commit_gate",
                "geant4_payload_builder_tool",
                "geant4_runtime_preflight_tool",
                "geant4_runtime_tool",
            ],
        )
        self.assertIn(
            "geant4_runtime_tool",
            [item["source"] for item in result["state"]["metadata"]["last_state_patch_apply"]["removed_observations"]],
        )

    def test_language_modify_energy_rerun_has_deterministic_fallback(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "modify-rerun-language",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": "modify-rerun-language",
                "text": "confirm run",
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": "modify-rerun-language",
                "text": "把刚才方案改成 2 MeV 再跑",
                "events": 2,
                "allow_in_memory": True,
                "llm_design_enabled": True,
                "llm_config_path": "missing-local-llm.json",
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        payload_observation = next(item for item in result["observations"] if item["source"] == "geant4_payload_builder_tool")
        self.assertEqual(payload_observation["data"]["applied_overrides"], {"source_energy_mev": 2.0})
        self.assertEqual(payload_observation["data"]["simulation_spec"]["source"]["energy_mev"], 2.0)
        self.assertEqual(result["state"]["metadata"]["last_state_patch"]["ok"], True)

    def test_analysis_followup_uses_runtime_fallback_when_llm_unavailable(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "analysis-fallback",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": "analysis-fallback",
                "text": "confirm run",
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": "analysis-fallback",
                "text": "分析结果，给我建议",
                "allow_in_memory": True,
                "llm_design_enabled": True,
                "llm_config_path": "missing-local-llm.json",
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["dialogue_act"], "runtime_result_answered")
        self.assertIn("target_edep_total_mev", result["display_message"])
        self.assertNotIn("payload draft", result["display_message"].lower())

    def test_result_suggestion_increase_events_rebuilds_payload_and_waits(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "suggestion-rerun",
                "text": "run a default lead shielding gamma simulation",
                "lang": "en-US",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        observed = service.run_turn(
            {
                "session_id": "suggestion-rerun",
                "text": "confirm run",
                "lang": "en-US",
                "allow_in_memory": True,
            }
        )
        prefill = next(
            item["prefill"]
            for item in observed["dialogue"]["next_suggestions"]
            if item["text"] == "Increase events and rerun"
        )

        result = service.run_turn(
            {
                "session_id": "suggestion-rerun",
                "text": prefill,
                "lang": "en-US",
                "allow_in_memory": True,
            }
        )

        self.assertEqual(prefill, "change event count to 20 events and run again")
        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        self.assertEqual(result["pending_action"]["kind"], "run_simulation")
        payload_observation = next(item for item in result["observations"] if item["source"] == "geant4_payload_builder_tool")
        self.assertEqual(payload_observation["data"]["applied_overrides"], {"run_events": 20})
        self.assertEqual(payload_observation["data"]["simulation_spec"]["run"]["events"], 20)

    def test_add_downstream_scoring_rerun_rebuilds_payload_and_waits(self) -> None:
        service = self._make_service()
        session_id = "suggestion-add-downstream-scoring"
        service.run_turn(
            {
                "session_id": session_id,
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": session_id,
                "text": "confirm run",
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": session_id,
                "text": "add downstream detector and plane scoring and run again",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        self.assertEqual(result["pending_action"]["kind"], "run_simulation")
        self.assertEqual(
            result["state"]["metadata"]["last_state_patch"]["config_overrides"],
            {"enable_downstream_scoring": True},
        )
        payload_observation = next(item for item in result["observations"] if item["source"] == "geant4_payload_builder_tool")
        self.assertEqual(payload_observation["data"]["applied_overrides"], {"enable_downstream_scoring": True})
        runtime_payload = payload_observation["data"]["runtime_payload"]
        self.assertTrue(runtime_payload["detector_enabled"])
        self.assertTrue(runtime_payload["scoring"]["detector_crossings"])
        self.assertTrue(runtime_payload["scoring"]["plane_crossings"])
        self.assertFalse(result["summary"]["has_runtime_result"])

    def test_auto_sweep_from_llm_is_saved_as_suggestion_not_executed_same_turn(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "auto-sweep-safety",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": "auto-sweep-safety",
                "text": "confirm run",
                "allow_in_memory": True,
            }
        )
        llm_response = (
            '{"action_kind":"final_answer","tool_name":"","message":"Try an energy sweep.",'
            '"auto_sweep":true,'
            '"sweep_suggestion":{"parameter":"source_energy_mev","values":[0.5,1.0,2.0],"label":"Energy"},'
            '"suggestions":[{"text":"Run sweep","prefill":"run sweep 0.5 1.0 2.0 MeV"}]}'
        )

        with patch("core.agent_v3.reasoners.LLMGeant4Reasoner._call_llm", return_value=llm_response):
            result = service.run_turn(
                {
                    "session_id": "auto-sweep-safety",
                    "text": "analyze result and suggest a sweep",
                    "allow_in_memory": True,
                    "llm_design_enabled": True,
                    "llm_config_path": "fake.json",
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertEqual(result["observations"], [])
        self.assertIn("suggested_next_actions", result["state"]["metadata"])

    def test_llm_runtime_proposal_is_blocked_without_payload_and_preflight(self) -> None:
        service = self._make_service()
        llm_response = (
            '{"action_kind":"run_simulation",'
            '"tool_name":"geant4_runtime_tool",'
            '"message":"I will run now.",'
            '"requires_confirmation":false}'
        )

        with patch("core.agent_v3.reasoners.LLMGeant4Reasoner._call_llm", return_value=llm_response):
            result = service.run_turn(
                {
                    "session_id": "llm-unsafe-runtime",
                    "text": "run immediately",
                    "allow_in_memory": True,
                    "llm_design_enabled": True,
                    "llm_config_path": "fake.json",
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "blocked")
        self.assertEqual(result["observations"][-1]["source"], "proposal_critic")
        self.assertEqual(result["observations"][-1]["data"]["reason"], "runtime_without_payload")
        self.assertIsNone(result["pending_action"])

    def test_modify_material_thickness_and_events_rebuilds_payload(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "modify-material-rerun",
                "text": "请按默认参数运行一个铅屏蔽 gamma 模拟",
                "events": 2,
                "allow_in_memory": True,
            }
        )

        # config_overrides injected directly (simulating LLM extraction)
        result = service.run_turn(
            {
                "session_id": "modify-material-rerun",
                "text": "把材料换成铜，厚度改成 20 mm，再跑 15 events",
                "allow_in_memory": True,
                "config_overrides": {
                    "run_events": 15,
                    "target_material": "G4_Cu",
                    "target_thickness_mm": 20.0,
                },
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "waiting_confirmation")
        payload_observation = next(item for item in result["observations"] if item["source"] == "geant4_payload_builder_tool")
        self.assertEqual(
            payload_observation["data"]["applied_overrides"],
            {
                "run_events": 15,
                "target_material": "G4_Cu",
                "target_thickness_mm": 20.0,
            },
        )
        self.assertEqual(payload_observation["data"]["simulation_spec"]["geometry"]["material"], "G4_Cu")
        self.assertEqual(payload_observation["data"]["simulation_spec"]["run"]["events"], 15)
        self.assertEqual(result["state"]["metadata"]["last_state_patch"]["config_overrides"]["run_events"], 15)

    def test_free_text_material_change_prefers_explicit_geant4_id(self) -> None:
        service = self._make_service()
        session_id = "modify-explicit-water-material"
        service.run_turn(
            {
                "session_id": session_id,
                "text": "Design a gamma shielding setup, do not run.",
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": session_id,
                "text": "Accept defaults and build config.",
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": session_id,
                "text": "Change target material to G4_WATER.",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertIn("G4_WATER", result["display_message"])
        payload_observation = next(item for item in result["observations"] if item["source"] == "geant4_payload_builder_tool")
        self.assertEqual(payload_observation["data"]["applied_overrides"]["target_material"], "G4_WATER")
        self.assertEqual(payload_observation["data"]["simulation_spec"]["geometry"]["material"], "G4_WATER")

    def test_free_text_material_change_without_value_does_not_infer_aluminum_from_material_word(self) -> None:
        service = self._make_service()
        session_id = "modify-material-no-value"
        service.run_turn(
            {
                "session_id": session_id,
                "text": "Design a gamma shielding setup, do not run.",
                "allow_in_memory": True,
            }
        )
        service.run_turn(
            {
                "session_id": session_id,
                "text": "Accept defaults and build config.",
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": session_id,
                "text": "Change target material.",
                "allow_in_memory": True,
            }
        )

        self.assertTrue(result["ok"])
        self.assertNotIn("G4_Al", result["display_message"])
        patch = result["state"]["metadata"].get("last_state_patch") or {}
        self.assertFalse(patch.get("config_overrides"))

    def test_invalid_config_override_is_not_applied_as_patch(self) -> None:
        service = self._make_service()
        service.run_turn(
            {
                "session_id": "invalid-patch",
                "text": "run a default lead shielding gamma simulation",
                "events": 2,
                "allow_in_memory": True,
            }
        )

        result = service.run_turn(
            {
                "session_id": "invalid-patch",
                "text": "try bad internal override",
                "allow_in_memory": True,
                "config_overrides": {"run_confirmed": True, "source_energy_mev": -2},
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["terminated_reason"], "waiting_user")
        self.assertEqual(result["dialogue_act"], "needs_user_input")
        self.assertIn("run_confirmed", result["display_message"])
        self.assertIsNone(result["pending_action"])
        patch = result["state"]["metadata"]["last_state_patch"]
        self.assertFalse(patch["ok"])
        self.assertEqual(patch["config_overrides"], {})
        self.assertIn("unsupported_patch_field:run_confirmed", patch["errors"])

    def test_request_router_exposes_v3_turn_endpoint(self) -> None:
        self.assertTrue(is_supported_post_path("/api/v3/agent/turn"))
        status, body = handle_post_request(
            "/api/v3/agent/turn",
            {
                "session_id": "router-v3",
                "text": "只给我一个铅屏蔽 gamma 模拟方案，不要运行",
            },
            legacy_sessions=None,
            solve_fn=lambda payload: {"unused": True},
            step_fn=lambda payload: {"unused": True},
        )

        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["schema_version"], V3_AGENT_TURN_SCHEMA_VERSION)

    def test_request_router_exposes_v3_state_endpoint(self) -> None:
        session_id = "router-v3-state"
        common = {"legacy_sessions": None, "solve_fn": lambda payload: {"unused": True}, "step_fn": lambda payload: {"unused": True}}
        status, first = handle_post_request(
            "/api/v3/agent/turn",
            {"session_id": session_id, "text": "design a lead shielding gamma setup"},
            **common,
        )
        self.assertEqual(status, 200)
        self.assertTrue(first["ok"])

        state_status, state_body = handle_post_request(
            "/api/v3/agent/state",
            {"session_id": session_id, "lang": "en"},
            **common,
        )

        self.assertEqual(state_status, 200)
        self.assertTrue(state_body["ok"])
        self.assertEqual(state_body["summary"]["schema_version"], "geant4_agent_v3_state_summary.v1")
        self.assertEqual(state_body["context"]["schema_version"], "geant4_agent_v3_context.v1")

    def test_v3_state_endpoint_returns_404_for_missing_session(self) -> None:
        status, body = handle_post_request(
            "/api/v3/agent/state",
            {"session_id": "missing-v3-session"},
            legacy_sessions=None,
            solve_fn=lambda payload: {"unused": True},
            step_fn=lambda payload: {"unused": True},
        )

        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "session_not_found")


if __name__ == "__main__":
    unittest.main()
