from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_geant4_agent_benchmark import (
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


if __name__ == "__main__":
    unittest.main()
