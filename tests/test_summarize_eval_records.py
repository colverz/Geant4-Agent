from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.eval_report_io import save_eval_output
from tools.summarize_eval_records import main, summarize_eval_records


class SummarizeEvalRecordsTest(unittest.TestCase):
    def test_summarize_eval_records_extracts_compact_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            save_eval_output(
                {
                    "ok": True,
                    "report": {
                        "name": "geant4_agent_benchmark_dry_run",
                        "total": 2,
                        "passed": 2,
                        "failed": 0,
                        "config_delta_summary": {
                            "expected_final_value_accuracy": 1.0,
                            "applied_path_precision": 1.0,
                        },
                    },
                },
                outdir=outdir,
                tool="geant4_agent_benchmark_dry_run",
                run_id="dry-run",
            )
            save_eval_output(
                {
                    "name": "llm_scenario_model_matrix",
                    "ok": False,
                    "reports": [
                        {
                            "name": "llm_scenario_parsing",
                            "mode": "live_llm",
                            "model_override": "offline_v2",
                            "failures": [
                                {
                                    "id": "case-1",
                                    "errors": [
                                        "runtime.energy:expected=2.0:actual=1.0",
                                        {"section": "expected_runtime", "error": "missing_payload_key:particle"},
                                    ],
                                }
                            ],
                        }
                    ],
                    "model_summaries": [
                        {
                            "model": "offline_v2",
                            "accuracy": 0.5,
                            "failed": 1,
                            "fallback_count": 2,
                            "profile_mismatch_count": 0,
                            "llm_used_count": 0,
                            "elapsed_seconds": 12.5,
                            "seconds_per_case": 3.125,
                        }
                    ],
                    "failed_models": ["offline_v2"],
                },
                outdir=outdir,
                tool="llm_scenario_model_matrix",
                run_id="matrix",
            )

            summary = summarize_eval_records(outdir)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        dry_run = next(record for record in summary["records"] if record["run_id"] == "dry-run")
        matrix = next(record for record in summary["records"] if record["run_id"] == "matrix")
        self.assertEqual(dry_run["key_metrics"]["applied_path_precision"], 1.0)
        self.assertEqual(matrix["model_summaries"][0]["model"], "offline_v2")
        self.assertEqual(matrix["model_summaries"][0]["elapsed_seconds"], 12.5)
        self.assertEqual(matrix["model_summaries"][0]["seconds_per_case"], 3.125)
        self.assertEqual(matrix["failed_models"], ["offline_v2"])
        self.assertEqual(matrix["failure_count"], 1)
        self.assertEqual(matrix["failure_summary"][0]["id"], "case-1")
        self.assertEqual(matrix["failure_summary"][0]["source"], "offline_v2")
        self.assertIn("missing_payload_key:particle", matrix["failure_summary"][0]["errors"][1])

    def test_latest_only_filters_to_latest_pointer_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            save_eval_output({"ok": True, "report": {"name": "probe"}}, outdir=outdir, tool="probe", run_id="old")
            save_eval_output({"ok": True, "report": {"name": "probe"}}, outdir=outdir, tool="probe", run_id="new")

            summary = summarize_eval_records(outdir, latest_only=True)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["records"][0]["run_id"], "new")

    def test_cli_summary_does_not_fail_on_failed_records_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            save_eval_output({"ok": False, "report": {"name": "probe"}}, outdir=outdir, tool="probe", run_id="failed")

            with mock.patch("sys.argv", ["summarize_eval_records.py", "--outdir", str(outdir), "--json"]):
                self.assertEqual(main(), 0)
            with mock.patch(
                "sys.argv",
                ["summarize_eval_records.py", "--outdir", str(outdir), "--json", "--fail-on-failed-record"],
            ):
                self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
