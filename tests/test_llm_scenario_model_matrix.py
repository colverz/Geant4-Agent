from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tools.evaluate_llm_scenario_model_matrix import (
    MODEL_MATRIX_SCHEMA_VERSION,
    evaluate_llm_scenario_model_matrix,
)


def _report(*, model: str, accuracy: float, fallback_count: int = 0, profile_mismatch_count: int = 0) -> dict:
    total = 4
    passed = int(round(accuracy * total))
    failed = total - passed
    return {
        "name": "llm_scenario_parsing",
        "mode": "live_llm",
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "min_accuracy": 0.75,
        "meets_threshold": failed == 0 and accuracy >= 0.75,
        "model_override": model,
        "failures": [],
        "known_gap_count": 0,
        "live_summary": {
            "llm_used_count": total - fallback_count,
            "fallback_count": fallback_count,
            "profile_mismatch_count": profile_mismatch_count,
            "lang_counts": {"en": 2, "zh": 2},
            "slot_prompt_profiles": {f"slot_extract_{model}_profile": total},
            "semantic_prompt_profiles": {},
        },
        "results": [],
    }


class LlmScenarioModelMatrixTest(unittest.TestCase):
    def test_model_matrix_ranks_models_and_flags_hidden_fallback(self) -> None:
        reports = [
            _report(model="deepseek-v4-flash", accuracy=1.0),
            _report(model="deepseek-v4-pro", accuracy=1.0, fallback_count=1),
        ]

        with mock.patch(
            "tools.evaluate_llm_scenario_model_matrix.evaluate_llm_scenario_parsing",
            side_effect=reports,
        ) as fake_eval:
            matrix = evaluate_llm_scenario_model_matrix(
                casebank_path=Path("docs/eval/llm_scenario_live_casebank.json"),
                live_llm=True,
                llm_config_path="dummy.json",
                models=["deepseek-v4-flash", "deepseek-v4-pro"],
                min_accuracy=0.75,
                max_cases=4,
            )

        self.assertEqual(matrix["schema_version"], MODEL_MATRIX_SCHEMA_VERSION)
        self.assertFalse(matrix["ok"])
        self.assertEqual(matrix["model_count"], 2)
        self.assertEqual(matrix["best_model"], "deepseek-v4-flash")
        self.assertEqual(matrix["hidden_fallback_models"], ["deepseek-v4-pro"])
        self.assertEqual(matrix["profile_mismatch_models"], [])
        self.assertEqual(fake_eval.call_count, 2)
        self.assertEqual(matrix["model_summaries"][0]["llm_usage_rate"], 1.0)
        self.assertEqual(matrix["model_summaries"][1]["fallback_rate"], 0.25)

    def test_model_matrix_flags_profile_mismatch(self) -> None:
        with mock.patch(
            "tools.evaluate_llm_scenario_model_matrix.evaluate_llm_scenario_parsing",
            return_value=_report(model="deepseek-v4-flash", accuracy=1.0, profile_mismatch_count=2),
        ):
            matrix = evaluate_llm_scenario_model_matrix(
                live_llm=True,
                llm_config_path="dummy.json",
                models=["deepseek-v4-flash"],
            )

        self.assertFalse(matrix["ok"])
        self.assertEqual(matrix["profile_mismatch_models"], ["deepseek-v4-flash"])
        self.assertEqual(matrix["model_summaries"][0]["profile_mismatch_rate"], 0.5)

    def test_model_matrix_defaults_to_offline_label_without_live_llm(self) -> None:
        with mock.patch(
            "tools.evaluate_llm_scenario_model_matrix.evaluate_llm_scenario_parsing",
            return_value=_report(model="", accuracy=1.0, fallback_count=4),
        ):
            matrix = evaluate_llm_scenario_model_matrix(live_llm=False, models=None)

        self.assertTrue(matrix["ok"])
        self.assertEqual(matrix["mode"], "offline_v2")
        self.assertEqual(matrix["model_summaries"][0]["model"], "offline_v2")
        self.assertEqual(matrix["hidden_fallback_models"], [])


if __name__ == "__main__":
    unittest.main()
