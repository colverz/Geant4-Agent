from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_geant4_agent_benchmark import (
    _config_delta_errors,
    evaluate_benchmark_dry_run,
    validate_benchmark_coverage,
    validate_benchmark_shape,
)


class Geant4AgentBenchmarkShapeTest(unittest.TestCase):
    def test_agentic_benchmark_v1_shape_passes(self) -> None:
        report = validate_benchmark_shape(Path("docs/eval/agentic_benchmark_v1.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreaterEqual(report["total"], 5)
        self.assertIn("tool_guard", report["suite_counts"])
        self.assertIn("workflow_trace", report["capability_counts"])

    def test_agentic_benchmark_v1_dry_run_passes(self) -> None:
        report = evaluate_benchmark_dry_run(Path("docs/eval/agentic_benchmark_v1.json"))

        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["passed"], report["total"])
        summary = report["config_delta_summary"]
        self.assertGreater(summary["cases"], 0)
        self.assertGreater(summary["expected_final_values_total"], 0)
        self.assertEqual(summary["expected_final_value_accuracy"], 1.0)
        self.assertEqual(summary["forbidden_final_value_guard_rate"], 1.0)
        self.assertGreater(summary["allowed_apply_paths_cases"], 0)
        self.assertEqual(summary["applied_path_precision"], 1.0)
        self.assertEqual(summary["unexpected_applied_path_rate"], 0.0)
        quantitative_summary = report["quantitative_result_summary"]
        self.assertGreater(quantitative_summary["cases"], 0)
        self.assertEqual(quantitative_summary["expected_metric_value_accuracy"], 1.0)
        self.assertEqual(quantitative_summary["expected_metric_range_rate"], 1.0)
        self.assertEqual(quantitative_summary["non_negative_metric_rate"], 1.0)
        self.assertEqual(quantitative_summary["relation_pass_rate"], 1.0)
        self.assertEqual(report["suite_summary"]["quantitative_runtime"]["pass_rate"], 1.0)
        self.assertEqual(report["suite_summary"]["tool_guard"]["pass_rate"], 1.0)
        self.assertEqual(report["capability_summary"]["workflow_trace"]["pass_rate"], 1.0)
        self.assertEqual(report["capability_summary"]["quantitative_result"]["pass_rate"], 1.0)
        self.assertIn("adversarial", report["difficulty_summary"])
        route_summary = report["model_route_summary"]
        self.assertGreater(route_summary["cases"], 0)
        self.assertEqual(route_summary["runtime_allowed_count"], 0)
        self.assertIn("cheap_model_ok", route_summary["label_counts"])
        self.assertIn("human_confirmation_required", route_summary["label_counts"])

    def test_agentic_benchmark_v1_coverage_passes(self) -> None:
        report = validate_benchmark_coverage(Path("docs/eval/agentic_benchmark_v1.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)

    def test_unknown_fields_are_rejected(self) -> None:
        cases = [
            {
                "id": "bad-case",
                "suite": "trajectory",
                "difficulty": "smoke",
                "lang": "en",
                "turns": [
                    {
                        "text": "run now",
                        "expected_trace": {
                            "intent": "run_requested",
                            "action_safety_class": "expensive_runtime",
                            "unsupported_future_field": True,
                        },
                    }
                ],
                "expected_config": {"must_set_paths": ["source.energy"]},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertGreater(report["failed"], 0)
        errors = [failure["error"] for failure in report["failures"]]
        self.assertIn("unsupported_key:expected_config", errors)
        self.assertIn("unsupported_key:unsupported_future_field", errors)

    def test_multi_turn_shape_is_allowed(self) -> None:
        cases = [
            {
                "id": "multi-turn-probe",
                "suite": "trajectory",
                "difficulty": "standard",
                "lang": "en",
                "capabilities": ["intent_routing", "workflow_trace"],
                "turns": [
                    {
                        "text": "What is configured?",
                        "expected_trace": {
                            "intent": "read_config",
                            "action_safety_class": "read_only",
                        },
                    },
                    {
                        "text": "run 10 events",
                        "expected_trace": {
                            "intent": "run_requested",
                            "action_safety_class": "expensive_runtime",
                            "must_block_tools": ["run_beam"],
                        },
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertEqual(report["failed"], 0)

    def test_invalid_model_route_shape_is_rejected(self) -> None:
        cases = [
            {
                "id": "bad-routing",
                "suite": "routing",
                "difficulty": "standard",
                "lang": "en",
                "capabilities": ["model_routing"],
                "turns": [{"text": "What is configured?"}],
                "expected_model_route": {
                    "label": "auto_run_everything",
                    "must_not_allow_runtime": "yes",
                    "unsupported_field": True,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertGreater(report["failed"], 0)
        errors = [failure["error"] for failure in report["failures"]]
        self.assertIn("invalid_label:auto_run_everything", errors)
        self.assertIn("must_not_allow_runtime_not_bool", errors)
        self.assertIn("unsupported_key:unsupported_field", errors)

    def test_invalid_runtime_turn_index_shape_is_rejected(self) -> None:
        cases = [
            {
                "id": "bad-runtime-turn-index",
                "suite": "runtime",
                "difficulty": "standard",
                "lang": "en",
                "turns": [{"text": "configure a copper box"}],
                "expected_runtime": {
                    "after_turn_index": -1,
                    "must_have_runtime_payload": True,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertGreater(report["failed"], 0)
        errors = [failure["error"] for failure in report["failures"]]
        self.assertIn("after_turn_index_not_non_negative_int", errors)

    def test_invalid_result_answer_sample_report_is_rejected(self) -> None:
        cases = [
            {
                "id": "bad-result-sample",
                "suite": "result_qa",
                "difficulty": "standard",
                "lang": "en",
                "turns": [{"text": "What was the latest result?"}],
                "expected_result_answer": {
                    "question": "What was the latest result?",
                    "sample_report": "invented",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertGreater(report["failed"], 0)
        errors = [failure["error"] for failure in report["failures"]]
        self.assertIn("invalid_sample_report:invented", errors)

    def test_invalid_quantitative_result_shape_is_rejected(self) -> None:
        cases = [
            {
                "id": "bad-quantitative-result",
                "suite": "quantitative_runtime",
                "difficulty": "standard",
                "lang": "en",
                "capabilities": ["quantitative_result"],
                "turns": [{"text": "What was the target edep?"}],
                "expected_quantitative_result": {
                    "sample_report": "invented",
                    "required_metric_keys": "key_metrics.target_edep_total_mev",
                    "expected_metric_values": ["key_metrics.target_edep_total_mev", 1.0],
                    "expected_metric_ranges": {
                        "key_metrics.target_edep_total_mev": {
                            "min": "zero",
                            "max": True,
                            "unsupported_bound": 2,
                        },
                        "key_metrics.detector_crossing_count": {},
                        "key_metrics.plane_crossing_count": {
                            "min": 2,
                            "max": 1,
                        },
                    },
                    "expected_relations": [
                        {
                            "left": "key_metrics.target_edep_mean_mev_per_event",
                            "op": "multiply_magic",
                        }
                    ],
                    "unsupported_field": True,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertGreater(report["failed"], 0)
        errors = [failure["error"] for failure in report["failures"]]
        self.assertIn("invalid_sample_report:invented", errors)
        self.assertIn("required_metric_keys_not_list", errors)
        self.assertIn("expected_metric_values_not_object", errors)
        self.assertIn("metric_range_min_not_number", errors)
        self.assertIn("metric_range_max_not_number", errors)
        self.assertIn("metric_range_missing_bound", errors)
        self.assertIn("metric_range_min_gt_max", errors)
        self.assertIn("unsupported_key:unsupported_bound", errors)
        self.assertIn("invalid_op:multiply_magic", errors)
        self.assertIn("unsupported_key:unsupported_field", errors)

    def test_invalid_config_delta_shape_is_rejected(self) -> None:
        cases = [
            {
                "id": "bad-config-delta",
                "suite": "core",
                "difficulty": "standard",
                "lang": "en",
                "capabilities": ["config_extraction"],
                "turns": [{"text": "Set source energy to 1 MeV."}],
                "expected_config_delta": {
                    "must_apply_paths": "source.energy",
                    "allowed_apply_paths": {"source.energy": True},
                    "expected_final_values": ["source.energy", 1.0],
                    "unsupported_field": True,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = validate_benchmark_shape(path)

        self.assertGreater(report["failed"], 0)
        errors = [failure["error"] for failure in report["failures"]]
        self.assertIn("must_apply_paths_not_list", errors)
        self.assertIn("allowed_apply_paths_not_list", errors)
        self.assertIn("expected_final_values_not_object", errors)
        self.assertIn("unsupported_key:unsupported_field", errors)

    def test_config_delta_allowed_apply_paths_catches_unexpected_mutation(self) -> None:
        errors = _config_delta_errors(
            {
                "must_apply_paths": ["source.energy"],
                "allowed_apply_paths": ["source.energy"],
            },
            final_config={},
            outputs=[
                {
                    "nlu_turn_trace": {
                        "applied_paths": ["source.energy", "physics.physics_list"],
                    }
                }
            ],
            case_id="precision-probe",
        )

        self.assertIn("unexpected_applied_path:physics.physics_list", [error["error"] for error in errors])


if __name__ == "__main__":
    unittest.main()
