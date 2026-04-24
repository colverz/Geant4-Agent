from __future__ import annotations

import unittest
from unittest import mock

from planner.runtime_result import build_runtime_result_message, naturalize_runtime_result_message


def _report() -> dict:
    return {
        "ok": True,
        "events_requested": 4,
        "events_completed": 4,
        "completion_fraction": 1.0,
        "configuration": {
            "geometry_structure": "single_box",
            "material": "G4_Cu",
            "particle": "gamma",
            "physics_list": "FTFP_BERT",
        },
        "key_metrics": {
            "target_edep_total_mev": 1.5,
            "target_hit_events": 2,
            "detector_crossing_count": 1,
            "plane_crossing_count": 0,
        },
        "run_summary_path": "F:/tmp/run_summary.json",
    }


class RuntimeResultExplanationTest(unittest.TestCase):
    def test_build_runtime_result_message_is_grounded(self) -> None:
        message = build_runtime_result_message(_report(), lang="en")

        self.assertIn("The simulation run completed.", message)
        self.assertIn("Events: 4 / 4", message)
        self.assertIn("target_edep_total_mev=1.5", message)
        self.assertIn("F:/tmp/run_summary.json", message)

    def test_build_runtime_result_message_zh(self) -> None:
        message = build_runtime_result_message(_report(), lang="zh")

        self.assertIn("模拟运行已完成", message)
        self.assertIn("4 / 4", message)
        self.assertIn("target_edep_total_mev=1.5", message)

    def test_naturalize_runtime_result_message_uses_deterministic_by_default(self) -> None:
        result = naturalize_runtime_result_message(_report(), lang="en")

        self.assertEqual(result["source"], "deterministic")
        self.assertIn("Events: 4 / 4", result["message"])

    def test_naturalize_runtime_result_message_accepts_grounded_llm_rewrite(self) -> None:
        with mock.patch(
            "planner.runtime_result.chat",
            return_value={
                "response": (
                    "The run completed with 4 / 4 events. "
                    "The target_edep_total_mev value is 1.5, detector_crossing_count is 1, "
                    "and plane_crossing_count is 0."
                )
            },
        ):
            result = naturalize_runtime_result_message(_report(), lang="en", use_llm=True)

        self.assertEqual(result["source"], "llm")
        self.assertIn("4 / 4", result["message"])

    def test_naturalize_runtime_result_message_rejects_new_numbers(self) -> None:
        with mock.patch(
            "planner.runtime_result.chat",
            return_value={"response": "The run completed and the dose is probably 99 Gy."},
        ):
            result = naturalize_runtime_result_message(_report(), lang="en", use_llm=True)

        self.assertEqual(result["source"], "deterministic")
        self.assertEqual(result["fallback_reason"], "invalid_llm_output")
        self.assertIn("target_edep_total_mev=1.5", result["message"])

    def test_naturalize_runtime_result_message_falls_back_on_llm_error(self) -> None:
        with mock.patch("planner.runtime_result.chat", side_effect=RuntimeError("offline")):
            result = naturalize_runtime_result_message(_report(), lang="en", use_llm=True)

        self.assertEqual(result["source"], "deterministic")
        self.assertEqual(result["fallback_reason"], "llm_failed")


if __name__ == "__main__":
    unittest.main()
