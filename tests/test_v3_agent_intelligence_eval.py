from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_v3_agent_intelligence import (
    DEFAULT_TASKS_PATH,
    evaluate_v3_agent_intelligence,
    grade_intelligence_trial,
    load_intelligence_tasks,
)


class V3AgentIntelligenceEvalTest(unittest.TestCase):
    def test_default_intelligence_tasks_pass(self) -> None:
        report = evaluate_v3_agent_intelligence(DEFAULT_TASKS_PATH)

        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["task_count"], 2)
        self.assertEqual(report["failed_task_count"], 0)
        self.assertGreaterEqual(report["metrics"]["llm_turn_count"], 1)
        self.assertGreaterEqual(report["metrics"]["state_patch_turn_count"], 2)

    def test_task_corpus_contains_no_dictionary_fallback_guard(self) -> None:
        tasks = load_intelligence_tasks(DEFAULT_TASKS_PATH)

        guarded = [
            task
            for task in tasks
            for invariant in task.get("invariants", [])
            if isinstance(invariant, dict) and invariant.get("type") == "no_controlled_fallback"
        ]
        self.assertTrue(guarded)

    def test_grader_catches_unwanted_fallback(self) -> None:
        task = {
            "id": "bad-fallback",
            "invariants": [{"type": "no_controlled_fallback", "turn": 1}],
        }
        trial = {
            "trajectory": [
                {
                    "turn_understanding": {"source": "fallback"},
                    "summary": {},
                    "pending_action": {},
                }
            ]
        }

        grade = grade_intelligence_trial(task, trial)

        self.assertFalse(grade["pass"])
        self.assertIn("no_controlled_fallback:source=fallback", grade["failures"])

    def test_eval_cli_writes_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/evaluate_v3_agent_intelligence.py", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["schema_version"], "geant4_agent_v3_intelligence_eval.v1")

    def test_eval_can_save_report_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = evaluate_v3_agent_intelligence(outdir=Path(tmpdir), run_id="v3-intelligence-test")

            record = report["eval_record"]
            self.assertEqual(record["tool"], "v3-agent-intelligence")
            self.assertTrue(Path(record["report_path"]).exists())
            self.assertTrue(Path(record["latest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
