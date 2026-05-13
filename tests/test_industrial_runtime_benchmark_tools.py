from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.analyze_industrial_benchmark_failures import analyze_industrial_benchmark_report
from tools.create_industrial_golden import generate_industrial_golden
from tools.evaluate_industrial_runtime_benchmark import evaluate_industrial_runtime_benchmark
from tools.industrial_runtime_executor import (
    compare_industrial_metrics,
    execute_industrial_case,
    extract_industrial_metrics,
)
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
        self.assertIn("run_beam_failed", result["reasons"])
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

    def test_metric_extractor_supports_direct_and_derived_metrics(self) -> None:
        case = _case_by_id("shielding_lead_gamma_transmission")
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=_benchmark()["runtime_defaults"])
        result = extract_industrial_metrics(
            {
                "run": {"events_completed": 4},
                "scoring": {
                    "detector_crossing": {"detector_crossing_count": 2},
                    "roles": {"detector": {"edep_total_mev": 0.25}},
                },
            },
            compiled["metric_plan"],
        )

        self.assertEqual(result["missing_metrics"], [])
        self.assertEqual(result["actual_metrics"]["detector_crossing_count"], 2)
        self.assertAlmostEqual(result["actual_metrics"]["detector_edep_total_mev"], 0.25)
        self.assertAlmostEqual(result["actual_metrics"]["transmission_factor"], 0.5)

    def test_metric_compare_uses_numeric_tolerance(self) -> None:
        comparison = compare_industrial_metrics(
            {"detector_crossing_count": 10, "transmission_factor": 0.49},
            {
                "detector_crossing_count": {"expected": 10, "tolerance": 0},
                "transmission_factor": {"expected": 0.5, "tolerance": 0.02},
            },
        )

        self.assertTrue(comparison["ok"])
        self.assertAlmostEqual(comparison["metric_diff"]["transmission_factor"]["delta"], -0.01)

    def test_executor_requires_local_process_for_official_runtime(self) -> None:
        execution = execute_industrial_case(
            _case_by_id("shielding_lead_gamma_transmission"),
            runtime_defaults=_benchmark()["runtime_defaults"],
            env={},
        )

        self.assertEqual(execution["status"], "not_evaluable")
        self.assertEqual(execution["failure_category"], "runtime_unavailable")
        self.assertIn("local_process_runtime_required", execution["errors"])

    def test_golden_generation_writes_file_after_fake_local_runtime_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            script = _write_fake_runtime_script(root, artifact_dir)
            golden_dir = root / "golden"
            report = generate_industrial_golden(
                BENCHMARK_PATH,
                case_id="shielding_lead_gamma_transmission",
                golden_dir=golden_dir,
                env={
                    "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK": "1",
                    "GEANT4_RUNTIME_COMMAND_JSON": json.dumps([sys.executable, str(script)]),
                },
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["generated"], 1)
            golden_path = Path(report["case_results"][0]["golden_file"])
            golden = json.loads(golden_path.read_text(encoding="utf-8"))

        self.assertEqual(golden["case_id"], "shielding_lead_gamma_transmission")
        self.assertEqual(golden["metrics"]["detector_crossing_count"]["expected"], 2500)
        self.assertAlmostEqual(golden["metrics"]["detector_edep_total_mev"]["expected"], 0.75)
        self.assertAlmostEqual(golden["metrics"]["transmission_factor"]["expected"], 0.25)
        self.assertEqual(golden["review"]["status"], "unreviewed")

    def test_official_evaluator_runs_and_compares_against_generated_golden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            script = _write_fake_runtime_script(root, artifact_dir)
            env = {
                "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK": "1",
                "GEANT4_RUNTIME_COMMAND_JSON": json.dumps([sys.executable, str(script)]),
            }
            golden_dir = root / "golden"
            golden_report = generate_industrial_golden(
                BENCHMARK_PATH,
                case_id="shielding_lead_gamma_transmission",
                golden_dir=golden_dir,
                env=env,
            )
            self.assertTrue(golden_report["ok"])

            report = evaluate_industrial_runtime_benchmark(BENCHMARK_PATH, env=env, golden_dir=golden_dir)

        lead_case = next(item for item in report["case_results"] if item["id"] == "shielding_lead_gamma_transmission")
        self.assertEqual(lead_case["status"], "passed")
        self.assertEqual(lead_case["failure_category"], None)
        self.assertEqual(lead_case["actual_metrics"]["detector_crossing_count"], 2500)
        self.assertTrue(lead_case["metric_diff"]["transmission_factor"]["passed"])


def _benchmark() -> dict:
    import json

    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _case_by_id(case_id: str) -> dict:
    for case in _benchmark()["cases"]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"case not found: {case_id}")


def _write_fake_runtime_script(root: Path, artifact_dir: Path) -> Path:
    script = root / "fake_geant4_runtime.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                f"artifact_dir = Path({str(artifact_dir)!r})",
                "summary = {",
                "  'run_ok': True,",
                "  'events_requested': 10000,",
                "  'events_completed': 10000,",
                "  'geometry_structure': 'single_box',",
                "  'material': 'G4_Pb',",
                "  'particle': 'gamma',",
                "  'source_type': 'beam',",
                "  'source_position_mm': [0, 0, -100],",
                "  'source_direction': [0, 0, 1],",
                "  'physics_list': 'FTFP_BERT',",
                "  'events': 10000,",
                "  'mode': 'batch',",
                "  'run_seed': 1337,",
                "  'scoring': {",
                "    'target_edep_enabled': True,",
                "    'target_edep_total_mev': 5.0,",
                "    'detector_crossings_enabled': True,",
                "    'detector_crossing_count': 2500,",
                "    'detector_crossing_events': 2500,",
                "    'volume_stats': {",
                "      'LeadShield': {'edep_total_mev': 5.0, 'hit_events': 100, 'step_count': 100, 'track_entries': 100},",
                "      'Detector': {'edep_total_mev': 0.75, 'hit_events': 20, 'crossing_count': 2500, 'step_count': 50, 'track_entries': 20}",
                "    }",
                "  },",
                "  'detector': {",
                "    'enabled': True,",
                "    'volume_name': 'Detector',",
                "    'material': 'G4_Si',",
                "    'position_mm': [0, 0, 50],",
                "    'size_mm': [20, 20, 2]",
                "  }",
                "}",
                "(artifact_dir / 'run_summary.json').write_text(json.dumps(summary), encoding='utf-8')",
                "print(f'artifact_dir={artifact_dir}')",
            ]
        ),
        encoding="utf-8",
    )
    return script


if __name__ == "__main__":
    unittest.main()
