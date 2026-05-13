from __future__ import annotations

import unittest
from pathlib import Path

from tools.analyze_industrial_benchmark_failures import analyze_industrial_benchmark_report
from tools.create_industrial_golden import generate_industrial_golden
from tools.evaluate_industrial_runtime_benchmark import evaluate_industrial_runtime_benchmark


BENCHMARK_PATH = Path("docs/eval/industrial_runtime_benchmark.json")


class IndustrialRuntimeBenchmarkToolsTest(unittest.TestCase):
    def test_golden_generation_refuses_without_real_runtime(self) -> None:
        report = generate_industrial_golden(BENCHMARK_PATH, case_id="shielding_lead_gamma_transmission", env={})

        self.assertFalse(report["ok"])
        self.assertFalse(report["runtime_gate"]["real_runtime_ready"])
        self.assertEqual(report["generated"], 0)
        self.assertEqual(report["blocked"], 1)
        self.assertEqual(report["case_results"][0]["failure_category"], "runtime_unavailable")
        self.assertIn("missing_runtime_command", report["case_results"][0]["reasons"])

    def test_golden_generation_refuses_to_fabricate_when_spec_compiler_is_missing(self) -> None:
        report = generate_industrial_golden(
            BENCHMARK_PATH,
            case_id="shielding_lead_gamma_transmission",
            env={
                "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK": "1",
                "GEANT4_RUNTIME_COMMAND_JSON": '["fake-real-wrapper"]',
            },
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["runtime_gate"]["real_runtime_ready"])
        self.assertEqual(report["generated"], 0)
        self.assertEqual(report["not_evaluable"], 1)
        result = report["case_results"][0]
        self.assertEqual(result["failure_category"], "spec_compile_error")
        self.assertIn("scenario_runtime_mapping_not_implemented", result["reasons"])
        self.assertFalse(result["golden_generated"])

    def test_golden_generation_reports_unknown_case(self) -> None:
        report = generate_industrial_golden(BENCHMARK_PATH, case_id="does_not_exist", env={})

        self.assertFalse(report["ok"])
        self.assertEqual(report["summary"]["failure_categories"]["case_not_found"], 1)
        self.assertEqual(report["errors"][0]["error"], "case_not_found")

    def test_failure_analysis_groups_current_hard_blockers(self) -> None:
        report = evaluate_industrial_runtime_benchmark(BENCHMARK_PATH, env={})
        analysis = analyze_industrial_benchmark_report(report)

        self.assertFalse(analysis["ok"])
        self.assertEqual(analysis["passed"], 0)
        self.assertIn("runtime_unavailable_and_missing_golden", analysis["failure_categories"])
        self.assertIn("unsupported_capability", analysis["failure_categories"])
        blockers = {item["failure_category"]: item for item in analysis["top_blockers"]}
        self.assertIn("runtime_unavailable_and_missing_golden", blockers)
        self.assertIn("Configure real runtime first", blockers["runtime_unavailable_and_missing_golden"]["next_action"])
        self.assertIn("industrial_ndt", analysis["domain_failures"])

    def test_failure_analysis_identifies_spec_compiler_as_next_blocker_after_runtime_and_goldens(self) -> None:
        report = {
            "name": "industrial_runtime_benchmark",
            "case_results": [
                {
                    "id": "case_a",
                    "domain": "shielding",
                    "status": "not_evaluable",
                    "failure_category": "spec_compile_error",
                },
                {
                    "id": "case_b",
                    "domain": "medical_phantom",
                    "status": "not_evaluable",
                    "failure_category": "missing_metric",
                },
            ],
        }
        analysis = analyze_industrial_benchmark_report(report)

        self.assertFalse(analysis["ok"])
        self.assertEqual(analysis["failure_categories"]["spec_compile_error"], 1)
        self.assertEqual(analysis["failure_categories"]["missing_metric"], 1)
        self.assertEqual(analysis["top_blockers"][0]["failure_category"], "spec_compile_error")
        self.assertIn("deterministic scenario-to-runtime", analysis["top_blockers"][0]["next_action"])


if __name__ == "__main__":
    unittest.main()
