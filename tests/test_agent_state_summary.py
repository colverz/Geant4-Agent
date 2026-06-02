from __future__ import annotations

import unittest

from core.agent.state_summary import AGENT_STATE_SUMMARY_SCHEMA_VERSION, build_agent_state_summary
from core.orchestrator.session_manager import get_session_config_summary, process_turn, reset_session


class AgentStateSummaryTest(unittest.TestCase):
    def test_summary_marks_missing_information(self) -> None:
        summary = build_agent_state_summary(
            lang="en",
            intent="config_mutation",
            safety_class="config_mutation",
            terminal_state="read_only_answer",
            missing_fields=["source.energy"],
            missing_fields_friendly=["source energy"],
            llm_used=False,
            fallback_reason="E_LLM_DISABLED",
        )

        self.assertEqual(summary["schema_version"], AGENT_STATE_SUMMARY_SCHEMA_VERSION)
        self.assertEqual(summary["status"], "needs_information")
        self.assertEqual(summary["next_action"], "ask_clarifying_question")
        self.assertTrue(summary["llm"]["degraded"])
        self.assertIn("source energy", summary["user_visible_summary"])

    def test_chinese_summary_is_readable(self) -> None:
        summary = build_agent_state_summary(
            lang="zh",
            intent="config_mutation",
            safety_class="config_mutation",
            terminal_state="read_only_answer",
            missing_fields=["source.energy"],
            missing_fields_friendly=["源能量"],
            llm_used=True,
        )

        message = summary["user_visible_summary"]
        self.assertIn("我已经理解", message)
        self.assertNotRegex(message, r"[\u5bb8\u93b4\u8930\u9429\u95b0\u6b7f\u947b\u951b\u9286\u20ac]")

    def test_process_turn_returns_summary_for_ready_config(self) -> None:
        session_id = "agent-state-summary-ready"
        reset_session(session_id)
        try:
            out = process_turn(
                {
                    "session_id": session_id,
                    "text": (
                        "10 mm x 20 mm x 30 mm copper box target; "
                        "gamma point source 1 MeV at (0,0,-20) mm along +z; "
                        "physics FTFP_BERT; output json."
                    ),
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
                lang="en",
            )

            summary = out["agent_state_summary"]
            self.assertEqual(summary["schema_version"], AGENT_STATE_SUMMARY_SCHEMA_VERSION)
            self.assertEqual(summary["status"], "ready_to_run")
            self.assertTrue(summary["runtime_ready"])
            self.assertIn("geometry.structure", summary["applied_paths"])
            self.assertEqual(out["internal_trace"]["agent"]["agent_state_summary"], summary)
        finally:
            reset_session(session_id)

    def test_process_turn_returns_guarded_runtime_summary(self) -> None:
        session_id = "agent-state-summary-runtime-guard"
        reset_session(session_id)
        try:
            out = process_turn(
                {
                    "session_id": session_id,
                    "text": "run 10 events now",
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
                lang="en",
            )

            summary = out["agent_state_summary"]
            self.assertEqual(summary["status"], "runtime_action_guarded")
            self.assertTrue(summary["runtime_action_guarded"])
            self.assertIn("run_beam", summary["tool_calls_blocked"])
            self.assertEqual(summary["next_action"], "ask_user_to_confirm_runtime_action")
        finally:
            reset_session(session_id)

    def test_config_summary_returns_agent_state_summary(self) -> None:
        session_id = "agent-state-summary-config"
        reset_session(session_id)
        try:
            process_turn(
                {
                    "session_id": session_id,
                    "text": "10 mm copper box with gamma point source 1 MeV, physics FTFP_BERT, output json.",
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": True,
                    "geometry_pipeline": "v2",
                    "source_pipeline": "v2",
                    "enable_compare": False,
                },
                ollama_config_path="",
                lang="en",
            )
            summary_response = get_session_config_summary(session_id, lang="en")
            summary = summary_response["agent_state_summary"]

            self.assertEqual(summary["intent"], "read_config")
            self.assertEqual(summary["safety_class"], "read_only")
            self.assertFalse(summary["runtime_action_guarded"])
        finally:
            reset_session(session_id)


if __name__ == "__main__":
    unittest.main()
