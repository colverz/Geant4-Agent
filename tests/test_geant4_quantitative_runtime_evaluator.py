from __future__ import annotations

import unittest
from unittest import mock

from core.runtime.types import Geant4RuntimePhase, RuntimeStateSnapshot
from tools.evaluate_geant4_agent_benchmark import _sample_runtime_report
from tools.evaluate_geant4_quantitative_runtime import evaluate_live_quantitative_runtime


class _FakeLocalAdapter:
    def snapshot(self) -> RuntimeStateSnapshot:
        return RuntimeStateSnapshot(
            connected=True,
            runtime_phase=Geant4RuntimePhase.INITIALIZED,
            available_actions=["run_beam", "summarize_last_result"],
            metadata={"adapter": "local_process", "command": ["fake-geant4"]},
        )


def _runtime_report_with_different_edep() -> dict:
    report = _sample_runtime_report()
    report["key_metrics"]["target_edep_total_mev"] = 2.0
    report["key_metrics"]["target_edep_mean_mev_per_event"] = 0.5
    report["result_summary"]["scoring"]["target"]["target_edep_total_mev"] = 2.0
    return report


class Geant4QuantitativeRuntimeEvaluatorTest(unittest.TestCase):
    def test_live_quantitative_runtime_is_skipped_by_default(self) -> None:
        report = evaluate_live_quantitative_runtime(env={})

        self.assertTrue(report["skipped"])
        self.assertEqual(report["skip_reason"], "live_runtime_not_enabled")
        self.assertEqual(report["failed"], 0)

    def test_live_quantitative_runtime_requires_runtime_command(self) -> None:
        report = evaluate_live_quantitative_runtime(env={"GEANT4_BENCHMARK_LIVE_RUNTIME": "1"})

        self.assertTrue(report["skipped"])
        self.assertEqual(report["skip_reason"], "missing_runtime_command")
        self.assertEqual(report["failed"], 0)

    def test_live_quantitative_runtime_uses_ranges_not_fixture_exact_values(self) -> None:
        env = {
            "GEANT4_BENCHMARK_LIVE_RUNTIME": "1",
            "GEANT4_RUNTIME_COMMAND_JSON": '["fake-geant4"]',
        }
        with mock.patch(
            "tools.evaluate_geant4_quantitative_runtime.build_geant4_adapter_from_env",
            return_value=_FakeLocalAdapter(),
        ), mock.patch(
            "tools.evaluate_geant4_quantitative_runtime._run_runtime_smoke",
            return_value=_runtime_report_with_different_edep(),
        ):
            report = evaluate_live_quantitative_runtime(env=env, events=4)

        self.assertFalse(report["skipped"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["passed"], report["total"])
        self.assertGreater(report["total"], 0)
        quantitative_summary = report["quantitative_result_summary"]
        self.assertEqual(quantitative_summary["expected_metric_values_total"], 0)
        self.assertEqual(quantitative_summary["expected_metric_range_rate"], 1.0)
        self.assertEqual(quantitative_summary["relation_pass_rate"], 1.0)

    def test_live_quantitative_runtime_reports_range_failure(self) -> None:
        bad_report = _sample_runtime_report()
        bad_report["key_metrics"]["target_edep_total_mev"] = 100.0
        bad_report["result_summary"]["scoring"]["target"]["target_edep_total_mev"] = 100.0
        env = {
            "GEANT4_BENCHMARK_LIVE_RUNTIME": "1",
            "GEANT4_RUNTIME_COMMAND_JSON": '["fake-geant4"]',
        }
        with mock.patch(
            "tools.evaluate_geant4_quantitative_runtime.build_geant4_adapter_from_env",
            return_value=_FakeLocalAdapter(),
        ), mock.patch(
            "tools.evaluate_geant4_quantitative_runtime._run_runtime_smoke",
            return_value=bad_report,
        ):
            report = evaluate_live_quantitative_runtime(env=env, events=4)

        self.assertGreater(report["failed"], 0)
        errors = [
            error["error"]
            for failure in report["failures"]
            for error in failure.get("errors", [])
        ]
        self.assertTrue(any(error.startswith("metric_range:key_metrics.target_edep_total_mev") for error in errors))


if __name__ == "__main__":
    unittest.main()
