from __future__ import annotations

from pathlib import Path
import unittest

from tools.evaluate_guard_casebanks import (
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


if __name__ == "__main__":
    unittest.main()
