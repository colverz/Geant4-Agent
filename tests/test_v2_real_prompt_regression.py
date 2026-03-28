from __future__ import annotations

import unittest

from core.orchestrator.session_manager import process_turn, reset_session


class V2RealPromptRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session("v2-real-prompt")

    def _run(self, text: str) -> dict:
        return process_turn(
            {
                "session_id": "v2-real-prompt",
                "text": text,
                "llm_router": False,
                "llm_question": False,
                "normalize_input": True,
                "geometry_pipeline": "v2",
                "source_pipeline": "v2",
                "enable_compare": False,
            },
            ollama_config_path="",
        )

    def test_complete_box_point_prompt_is_runtime_ready_in_v2(self) -> None:
        out = self._run(
            "10 mm x 20 mm x 30 mm copper box target; "
            "gamma point source 1 MeV at (0,0,-20) mm along +z; "
            "physics FTFP_BERT; output json."
        )
        self.assertTrue(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_box")
        self.assertEqual(out["config"]["source"]["type"], "point")
        self.assertEqual(out["config"]["source"]["particle"], "gamma")
        self.assertTrue(out["slot_debug"]["geometry_v2"]["runtime_ready"])
        self.assertTrue(out["slot_debug"]["source_v2"]["runtime_ready"])

    def test_cylinder_beam_prompt_keeps_geometry_and_asks_for_direction(self) -> None:
        out = self._run(
            "copper cylinder radius 15 mm half length 40 mm; "
            "gamma beam 5 MeV from (0,0,-250) mm; "
            "physics Shielding; output json."
        )
        self.assertFalse(out["is_complete"])
        self.assertEqual(out["config"]["geometry"]["structure"], "single_tubs")
        self.assertNotEqual(out["config"]["source"].get("type"), "beam")
        self.assertIn("source.direction", out["missing_fields"])
        self.assertIn("source.direction", out["asked_fields"])
        self.assertTrue(out["slot_debug"]["geometry_v2"]["runtime_ready"])
        self.assertFalse(out["slot_debug"]["source_v2"]["runtime_ready"])


if __name__ == "__main__":
    unittest.main()
