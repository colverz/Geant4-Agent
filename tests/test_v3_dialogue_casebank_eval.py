from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_v3_dialogue_casebank import evaluate_v3_dialogue_casebank, run_v3_dialogue_adapter


class V3DialogueCasebankEvalTest(unittest.TestCase):
    def test_default_casebank_runs_with_metrics_and_trajectory(self) -> None:
        report = evaluate_v3_dialogue_casebank()

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema_version"], "geant4_agent_v3_dialogue_casebank_eval.v1")
        self.assertGreaterEqual(report["case_count"], 1)
        self.assertEqual(report["failed_count"], 0)
        self.assertIn("metrics", report)
        self.assertGreaterEqual(report["metrics"]["turn_count"], report["case_count"])
        self.assertIn("dialogue_quality_warnings", report["metrics"])
        self.assertIn("dialogue_quality_warning_turns", report["metrics"])
        self.assertIsInstance(report["metrics"]["dialogue_quality_warning_turns"], list)
        self.assertTrue(report["cases"][0]["trajectory"])
        self.assertIn("dialogue_quality", report["cases"][0]["trajectory"][0])

    def test_casebank_reports_expectation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            casebank = Path(tmpdir) / "bad_casebank.json"
            casebank.write_text(
                json.dumps(
                    [
                        {
                            "id": "bad-expectation",
                            "lang": "en",
                            "turns": [
                                {
                                    "text": "What does detector_crossing_count mean?",
                                    "expect": {"dialogue_act_in": ["runtime_result_answered"]},
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = evaluate_v3_dialogue_casebank(casebank)

        self.assertFalse(report["ok"])
        self.assertEqual(report["failed_count"], 1)
        self.assertIn("dialogue_act", report["cases"][0]["failures"][0])
        self.assertEqual(report["cases"][0]["trajectory"][0]["dialogue_act"], "final_answer")

    def test_casebank_eval_can_save_report_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = evaluate_v3_dialogue_casebank(outdir=Path(tmpdir), run_id="v3-dialogue-test")

            record = report["eval_record"]
            report_path = Path(record["report_path"])
            latest_path = Path(record["latest_path"])

            self.assertEqual(record["tool"], "v3-dialogue-casebank")
            self.assertTrue(report_path.exists())
            self.assertTrue(latest_path.exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["eval_record"]["run_id"], "v3-dialogue-test")

    def test_adapter_payload_returns_strict_output_shape(self) -> None:
        output = run_v3_dialogue_adapter(
            {
                "id": "adapter-probe",
                "prompts": ["What does detector_crossing_count mean?"],
            }
        )

        self.assertEqual(output["status"], "completed")
        self.assertTrue(output["message"])
        self.assertIsInstance(output["trajectory"], list)
        self.assertEqual(output["metadata"]["adapter"], "v3-dialogue-casebank")
        self.assertEqual(output["metadata"]["case_id"], "adapter-probe")

    def test_adapter_cli_writes_exactly_one_stdout_json_object(self) -> None:
        payload = {"id": "adapter-cli", "prompts": ["What does detector_crossing_count mean?"]}
        completed = subprocess.run(
            [sys.executable, "tools/evaluate_v3_dialogue_casebank.py", "--adapter"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        stdout = completed.stdout.strip()
        self.assertTrue(stdout.startswith("{") and stdout.endswith("}"))
        parsed = json.loads(stdout)
        self.assertEqual(parsed["status"], "completed")
        self.assertIsInstance(parsed["trajectory"], list)


if __name__ == "__main__":
    unittest.main()
