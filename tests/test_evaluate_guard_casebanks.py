from __future__ import annotations

from pathlib import Path
import unittest

from tools.evaluate_guard_casebanks import (
    evaluate_agentic_behavior,
    evaluate_multiturn_guard,
    evaluate_runtime_result_qa,
    evaluate_session_behavior_guard,
    evaluate_workflow_guard,
    validate_casebank_shapes,
)


class EvaluateGuardCasebanksTest(unittest.TestCase):
    def test_casebank_shapes_are_valid(self) -> None:
        report = validate_casebank_shapes()

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)

    def test_workflow_guard_casebank_passes(self) -> None:
        report = evaluate_workflow_guard(Path("docs/eval/workflow_guard_casebank.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)

    def test_workflow_guard_failures_include_agent_decision_and_graph_errors(self) -> None:
        import json
        import tempfile

        cases = [
            {
                "id": "intent_failure_probe",
                "text": "run 10 events now",
                "lang": "en",
                "expected_intent": "read_config",
                "expected_safety": "read_only",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "casebank.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = evaluate_workflow_guard(path)

        self.assertEqual(report["failed"], 1)
        failure = report["failures"][0]
        self.assertIn("decision", failure)
        self.assertEqual(failure["decision"]["intent"], "run_requested")
        self.assertIn("intent", failure["errors"])

    def test_runtime_result_qa_casebank_passes(self) -> None:
        report = evaluate_runtime_result_qa(Path("docs/eval/runtime_result_qa_casebank.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)

    def test_multiturn_guard_casebank_passes(self) -> None:
        report = evaluate_multiturn_guard(Path("docs/eval/multiturn_guard_casebank.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)

    def test_session_behavior_guard_casebank_passes(self) -> None:
        report = evaluate_session_behavior_guard(Path("docs/eval/session_behavior_casebank.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)

    def test_agentic_behavior_casebank_passes(self) -> None:
        report = evaluate_agentic_behavior(Path("docs/eval/agentic_behavior_casebank.json"))

        self.assertEqual(report["failed"], 0)
        self.assertGreater(report["total"], 0)


if __name__ == "__main__":
    unittest.main()
