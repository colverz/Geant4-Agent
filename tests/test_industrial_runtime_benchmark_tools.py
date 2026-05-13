from __future__ import annotations

import unittest
from pathlib import Path

from tools.analyze_industrial_benchmark_failures import analyze_industrial_benchmark_report
from tools.create_industrial_golden import generate_industrial_golden
from tools.evaluate_industrial_runtime_benchmark import evaluate_industrial_runtime_benchmark
from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime, summarize_compile_results


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

    def test_golden_generation_refuses_to_fabricate_after_successful_compile(self) -> None:
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
        self.assertEqual(result["failure_category"], "runtime_error")
        self.assertIn("industrial_golden_runtime_execution_not_implemented", result["reasons"])
        self.assertEqual(result["compile_report"]["status"], "compiled")
        self.assertTrue(result["compile_report"]["runtime_payload_available"])
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
        self.assertIn("compile_summary", report["summary"])
        self.assertGreater(report["summary"]["compile_summary"]["status_counts"]["compiled"], 0)
        self.assertGreater(report["summary"]["compile_summary"]["status_counts"]["unsupported_capability"], 0)
        self.assertIn("compile_summary", analysis)
        self.assertGreater(len(analysis["compile_blockers"]), 0)

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

    def test_runtime_compiler_builds_payload_for_currently_supported_industrial_case(self) -> None:
        case = _case_by_id("shielding_lead_gamma_transmission")
        benchmark = _benchmark()
        result = compile_industrial_case_to_runtime(case, runtime_defaults=benchmark["runtime_defaults"])

        self.assertEqual(result["status"], "compiled")
        self.assertEqual(result["runtime_payload"]["geometry"]["material"], "G4_Pb")
        self.assertEqual(result["runtime_payload"]["source"]["particle"], "gamma")
        self.assertEqual(result["runtime_payload"]["source"]["energy_mev"], 1.0)
        self.assertTrue(result["runtime_payload"]["detector"]["enabled"])
        self.assertEqual(result["metric_plan"]["unsupported"], {})
        self.assertIn("transmission_factor", result["metric_plan"]["supported"])

    def test_runtime_compiler_exposes_structural_gaps_instead_of_simplifying_them(self) -> None:
        case = _case_by_id("ndt_aluminum_block_void_contrast")
        result = compile_industrial_case_to_runtime(case, runtime_defaults=_benchmark()["runtime_defaults"])

        self.assertEqual(result["status"], "unsupported_capability")
        self.assertIn("embedded_void_geometry_not_supported_by_current_single_volume_runtime", result["unsupported_features"])
        self.assertFalse(result.get("runtime_payload"))

    def test_runtime_compiler_marks_metric_gaps_for_partial_runtime_support(self) -> None:
        case = _case_by_id("medical_proton_water_depth_dose")
        result = compile_industrial_case_to_runtime(case, runtime_defaults=_benchmark()["runtime_defaults"])

        self.assertEqual(result["status"], "compiled_with_gaps")
        self.assertTrue(result["runtime_payload"])
        self.assertIn("peak_depth_mm", result["metric_plan"]["unsupported"])
        self.assertIn("depth_bin_edep_hash", result["metric_plan"]["unsupported"])
        summary = summarize_compile_results([result])
        self.assertEqual(summary["status_counts"]["compiled_with_gaps"], 1)
        self.assertEqual(summary["unsupported_metrics"]["peak_depth_mm"], 1)


if __name__ == "__main__":
    unittest.main()


def _benchmark() -> dict:
    import json

    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _case_by_id(case_id: str) -> dict:
    for case in _benchmark()["cases"]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"case not found: {case_id}")
