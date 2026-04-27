from __future__ import annotations

import unittest

from tools.evaluate_llm_scenario_parsing import evaluate_llm_scenario_parsing


class LlmScenarioParsingBenchmarkTest(unittest.TestCase):
    def test_offline_v2_parser_reaches_core_runtime_contract(self) -> None:
        report = evaluate_llm_scenario_parsing()

        self.assertEqual(report["mode"], "offline_v2")
        self.assertEqual(report["failed"], 0)
        self.assertGreaterEqual(report["total"], 3)
        self.assertGreaterEqual(report["known_gap_count"], 1)
        for result in report["results"]:
            self.assertFalse(result["errors"])

    def test_live_llm_mode_requires_explicit_config_path(self) -> None:
        report = evaluate_llm_scenario_parsing(live_llm=True, llm_config_path="")

        self.assertEqual(report["mode"], "live_llm")
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["failures"][0]["errors"], ["missing_llm_config_path"])


if __name__ == "__main__":
    unittest.main()
