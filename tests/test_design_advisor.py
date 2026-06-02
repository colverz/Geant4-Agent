from __future__ import annotations

import unittest

from core.agent.design_advisor import DESIGN_ADVICE_SCHEMA_VERSION, build_design_advice
from core.agent.simulation_design import build_simulation_design_candidate
from core.orchestrator.session_manager import process_turn, reset_session


class DesignAdvisorTest(unittest.TestCase):
    def test_supported_design_advice_marks_candidate_config_ready(self) -> None:
        candidate = build_simulation_design_candidate(
            "Use a 1 MeV gamma beam through lead shielding and measure transmission at a silicon detector."
        ).to_dict()
        recommended_config = {
            "geometry": {"structure": "single_box"},
            "materials": {"selected_materials": ["G4_Pb"]},
            "source": {"type": "beam", "particle": "gamma", "energy": 1.0},
            "physics": {"physics_list": "FTFP_BERT"},
            "output": {"format": "json"},
        }

        advice = build_design_advice(candidate, recommended_config=recommended_config, lang="en")

        self.assertEqual(advice["schema_version"], DESIGN_ADVICE_SCHEMA_VERSION)
        self.assertEqual(advice["status"], "candidate_config_ready")
        self.assertTrue(advice["primary_option"]["runnable_with_current_runtime"])
        self.assertIn("runtime preflight", " ".join(advice["recommended_next_steps"]).lower())
        self.assertEqual(advice["unsupported_capabilities"], [])

    def test_approximation_design_advice_requires_user_decision(self) -> None:
        candidate = build_simulation_design_candidate(
            "Compare detector transmission for a steel pipe section with corrosion thinning."
        ).to_dict()

        advice = build_design_advice(candidate, recommended_config={}, lang="en")

        self.assertEqual(advice["status"], "needs_user_decision")
        self.assertTrue(advice["primary_option"]["requires_user_approval"])
        self.assertTrue(advice["user_decisions_required"])
        self.assertFalse(advice["primary_option"]["runnable_with_current_runtime"])

    def test_chinese_design_advice_is_readable(self) -> None:
        candidate = build_simulation_design_candidate("设计一个铅屏蔽伽马透射实验").to_dict()
        advice = build_design_advice(
            candidate,
            recommended_config={"geometry": {"structure": "single_box"}},
            lang="zh",
        )

        summary = advice["user_visible_summary"]
        self.assertIn("仿真方案", summary)
        self.assertNotRegex(summary, r"[\u5bb8\u93b4\u8930\u9429\u95b0\u6b7f\u947b\u951b\u9286\u20ac]")

    def test_process_turn_simulation_design_returns_design_advice(self) -> None:
        session_id = "design-advisor-process-turn"
        reset_session(session_id)
        try:
            out = process_turn(
                {
                    "session_id": session_id,
                    "text": "Design a lead shielding gamma transmission benchmark with a silicon detector.",
                    "enable_simulation_design": True,
                    "llm_router": False,
                    "llm_question": False,
                    "normalize_input": False,
                },
                ollama_config_path="",
                lang="en",
            )

            advice = out["design_advice"]
            self.assertEqual(advice["schema_version"], DESIGN_ADVICE_SCHEMA_VERSION)
            self.assertIn(advice["status"], {"candidate_config_ready", "design_ready", "needs_user_decision"})
            self.assertEqual(out["internal_trace"]["design_advice"], advice)
            self.assertEqual(out["internal_trace"]["agent"]["agent_state_summary"], out["agent_state_summary"])
        finally:
            reset_session(session_id)


if __name__ == "__main__":
    unittest.main()
