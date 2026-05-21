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
            save_eval_output(
                {
                    "ok": True,
                    "report": {
                        "schema_version": "geant4_agent_industrial_llm_runtime_stage.v1",
                        "ok": True,
                        "case_results": [
                            {
                                "id": "shielding",
                                "domain": "shielding",
                                "status": "passed",
                                "actual_metrics": {
                                    "detector_crossing_count": 4659,
                                    "detector_edep_total_mev": 53.9888,
                                    "transmission_factor": 0.4659,
                                },
                            }
                        ],
                        "stage_summary": {
                            "passed": 1,
                            "failed": 0,
                            "not_evaluable": 0,
                            "llm_contract_passed": 1,
                            "runtime_completed": 1,
                            "nlu_boundary": {
                                "no_bert_prior_pass_rate": 1.0,
                                "backend_check_pass_rate": 1.0,
                            },
                            "candidate_boundary": {
                                "cases": 1,
                                "role_counts": {"candidate_config_only": 1},
                                "schema_counts": {"geant4_agent_llm_candidate_contract.v1": 1},
                                "requires_confirmation_cases": 1,
                                "uncertainty_cases": 0,
                                "assumption_count": 4,
                                "physics_rationale_count": 3,
                            },
                            "contract_alignment": {
                                "applied_cases": 1,
                                "correction_count": 9,
                                "completion_count": 6,
                                "override_count": 3,
                                "risk_correction_count": 2,
                                "correction_categories": {
                                    "material_role": 2,
                                    "runtime_default": 1,
                                },
                            },
                            "simulation_design": {
                                "supported_count": 2,
                                "approximation_required_count": 1,
                                "unsupported_count": 1,
                                "user_decision_required_count": 2,
                            },
                        },
                    },
                },
                outdir=outdir,
                tool="industrial_llm_runtime_stage",
                run_id="industrial",
            )
            save_eval_output(
                {
                    "ok": True,
                    "report": {
                        "schema_version": "geant4_agent_industrial_runtime_stage.v1",
                        "ok": True,
                        "stage_summary": {
                            "evaluation_status": {
                                "passed": 4,
                                "failed": 0,
                                "not_evaluable": 0,
                                "unsupported": 0,
                            }
                        },
                    },
                },
                outdir=outdir,
                tool="industrial_runtime_stage",
                run_id="runtime-stage",
            )

            summary = summarize_eval_records(outdir)

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["ok_count"], 3)
        self.assertEqual(summary["failed_count"], 1)
        dry_run = next(record for record in summary["records"] if record["run_id"] == "dry-run")
        matrix = next(record for record in summary["records"] if record["run_id"] == "matrix")
        industrial = next(record for record in summary["records"] if record["run_id"] == "industrial")
        runtime_stage = next(record for record in summary["records"] if record["run_id"] == "runtime-stage")
        self.assertEqual(dry_run["key_metrics"]["applied_path_precision"], 1.0)
        self.assertEqual(matrix["model_summaries"][0]["model"], "offline_v2")
        self.assertEqual(matrix["model_summaries"][0]["elapsed_seconds"], 12.5)
        self.assertEqual(matrix["model_summaries"][0]["seconds_per_case"], 3.125)
        self.assertEqual(matrix["failed_models"], ["offline_v2"])
        self.assertEqual(matrix["failure_count"], 1)
        self.assertEqual(matrix["failure_summary"][0]["id"], "case-1")
        self.assertEqual(matrix["failure_summary"][0]["source"], "offline_v2")
        self.assertIn("missing_payload_key:particle", matrix["failure_summary"][0]["errors"][1])
        self.assertEqual(industrial["key_metrics"]["runtime_completed"], 1)
        self.assertEqual(industrial["key_metrics"]["no_bert_prior_pass_rate"], 1.0)
        self.assertEqual(industrial["key_metrics"]["candidate_boundary.cases"], 1)
        self.assertEqual(industrial["key_metrics"]["candidate_boundary.role.candidate_config_only"], 1)
        self.assertEqual(industrial["key_metrics"]["candidate_boundary.requires_confirmation_cases"], 1)
        self.assertEqual(industrial["key_metrics"]["candidate_boundary.assumption_count"], 4)
        self.assertEqual(industrial["key_metrics"]["contract_alignment.correction_count"], 9)
        self.assertEqual(industrial["key_metrics"]["contract_alignment.completion_count"], 6)
        self.assertEqual(industrial["key_metrics"]["contract_alignment.override_count"], 3)
        self.assertEqual(industrial["key_metrics"]["contract_alignment.risk_correction_count"], 2)
        self.assertEqual(industrial["key_metrics"]["contract_alignment.material_role"], 2)
        self.assertEqual(industrial["key_metrics"]["simulation_design.supported_count"], 2)
        self.assertEqual(industrial["key_metrics"]["simulation_design.approximation_required_count"], 1)
        self.assertEqual(industrial["key_metrics"]["simulation_design.unsupported_count"], 1)
        self.assertEqual(industrial["key_metrics"]["simulation_design.user_decision_required_count"], 2)
        self.assertEqual(industrial["key_metrics"]["actual.detector_crossing_count"], 4659)
        self.assertEqual(industrial["key_metrics"]["actual.transmission_factor"], 0.4659)
        self.assertEqual(runtime_stage["key_metrics"]["evaluation.passed"], 4)
        self.assertEqual(runtime_stage["key_metrics"]["evaluation.not_evaluable"], 0)

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
