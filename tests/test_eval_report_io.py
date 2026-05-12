from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.eval_report_io import EVAL_REPORT_SCHEMA_VERSION, save_eval_output


class EvalReportIoTest(unittest.TestCase):
    def test_save_eval_output_writes_report_and_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = {"ok": True, "report": {"name": "probe", "passed": 1, "failed": 0}}
            saved = save_eval_output(output, outdir=Path(tmpdir), tool="probe tool", run_id="run 001")

            record = saved["eval_record"]
            self.assertEqual(record["schema_version"], EVAL_REPORT_SCHEMA_VERSION)
            self.assertEqual(record["run_id"], "run-001")
            self.assertEqual(record["tool"], "probe-tool")
            report_path = Path(record["report_path"])
            latest_path = Path(record["latest_path"])
            self.assertTrue(report_path.exists())
            self.assertTrue(latest_path.exists())

            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["eval_record"]["run_id"], "run-001")
            self.assertEqual(latest_payload["report_path"], str(report_path))
            self.assertTrue(latest_payload["ok"])

    def test_save_eval_output_is_noop_without_outdir(self) -> None:
        output = {"ok": False}

        self.assertIs(save_eval_output(output, outdir=None, tool="probe"), output)


if __name__ == "__main__":
    unittest.main()
