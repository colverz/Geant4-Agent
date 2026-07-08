from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.evaluate_industrial_runtime_benchmark import (
    evaluate_industrial_runtime_benchmark,
    validate_industrial_benchmark_shape,
)


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

    def test_shape_validator_accepts_current_manifest(self) -> None:
        report = validate_industrial_benchmark_shape(BENCHMARK_PATH)

        self.assertEqual(report["failed"], 0)
        self.assertGreaterEqual(report["total"], 20)
        self.assertIn("industrial_ndt", report["domain_counts"])

    def test_semantic_requirements_use_typed_detector_policy(self) -> None:
        benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        semantic_cases = [case for case in benchmark["cases"] if "semantic_requirements" in case]

        self.assertGreaterEqual(len(semantic_cases), 4)
        for case in semantic_cases:
            with self.subTest(case=case["id"]):
                semantic = case["semantic_requirements"]
                self.assertIn(semantic["detector"]["policy"], {"required", "optional", "forbidden"})
                self.assertTrue(semantic["required_materials"])
                self.assertTrue(semantic["required_scoring"])

    def test_official_evaluator_refuses_to_pass_without_real_runtime(self) -> None:
        report = evaluate_industrial_runtime_benchmark(BENCHMARK_PATH, env={})

        self.assertFalse(report["ok"])
        self.assertFalse(report["runtime_gate"]["env_enabled"])
        self.assertFalse(report["runtime_gate"]["runtime_command_configured"])
        self.assertGreater(report["not_evaluable"], 0)
        self.assertGreater(report["unsupported"], 0)
        categories = report["summary"]["failure_categories"]
        self.assertIn("runtime_unavailable_and_missing_golden", categories)
        self.assertIn("unsupported_capability", categories)
        self.assertEqual(report["passed"], 0)

    def test_official_evaluator_still_refuses_when_runtime_exists_but_goldens_are_missing(self) -> None:
        report = evaluate_industrial_runtime_benchmark(
            BENCHMARK_PATH,
            env={
                "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK": "1",
                "GEANT4_RUNTIME_COMMAND_JSON": '["fake-geant4"]',
            },
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["runtime_gate"]["real_runtime_ready"])
        self.assertGreater(report["not_evaluable"], 0)
        self.assertIn("missing_golden", report["summary"]["failure_categories"])
        self.assertEqual(report["passed"], 0)


if __name__ == "__main__":
    unittest.main()
