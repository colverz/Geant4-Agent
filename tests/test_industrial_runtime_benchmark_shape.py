from __future__ import annotations

import json
import unittest
from pathlib import Path


BENCHMARK_PATH = Path("docs/eval/industrial_runtime_benchmark.json")


class IndustrialRuntimeBenchmarkShapeTest(unittest.TestCase):
    def test_industrial_runtime_benchmark_shape_is_strong(self) -> None:
        benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

        self.assertEqual(benchmark["schema_version"], "geant4_agent_industrial_runtime_benchmark.v1")
        self.assertTrue(benchmark["official_pass_requires"]["real_geant4_runtime"])
        self.assertTrue(benchmark["official_pass_requires"]["golden_numeric_metrics"])
        self.assertFalse(benchmark["official_pass_requires"]["llm_as_judge"])

        cases = benchmark["cases"]
        self.assertGreaterEqual(len(cases), 20)
        domains = {case["domain"] for case in cases}
        self.assertGreaterEqual(
            domains,
            {
                "industrial_ndt",
                "shielding",
                "medical_phantom",
                "detector_response",
                "beam_source",
                "multi_turn_engineering",
                "unsupported_boundary",
            },
        )

    def test_official_cases_require_real_runtime_and_golden_metrics(self) -> None:
        benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

        official_cases = [case for case in benchmark["cases"] if case.get("golden_required") is True]
        self.assertGreaterEqual(len(official_cases), 18)
        for case in official_cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["required_runtime"], "real_geant4")
                self.assertEqual(case["llm_role"], "candidate_config_only")
                self.assertIsInstance(case.get("raw_dialogue"), list)
                self.assertGreater(len(case["raw_dialogue"]), 0)
                self.assertIsInstance(case.get("scenario_spec"), dict)
                self.assertIsInstance(case.get("golden_metrics"), dict)
                self.assertGreater(len(case["golden_metrics"]), 0)
                for metric_name, metric in case["golden_metrics"].items():
                    self.assertIsInstance(metric_name, str)
                    self.assertIsInstance(metric, dict)
                    self.assertIn("expected", metric)
                    self.assertIn("tolerance", metric)

    def test_unsupported_cases_are_explicit_capability_gaps(self) -> None:
        benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

        unsupported_cases = [case for case in benchmark["cases"] if case["domain"] == "unsupported_boundary"]
        self.assertGreaterEqual(len(unsupported_cases), 2)
        for case in unsupported_cases:
            with self.subTest(case=case["id"]):
                self.assertFalse(case["golden_required"])
                self.assertEqual(case["expected_status"], "unsupported_capability")
                self.assertEqual(case["llm_role"], "capability_gap_explanation_only")


if __name__ == "__main__":
    unittest.main()
