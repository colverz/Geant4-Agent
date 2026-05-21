from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.review_industrial_golden import review_industrial_golden, validate_industrial_golden_for_review


class IndustrialGoldenReviewTest(unittest.TestCase):
    def test_review_refuses_missing_reviewer(self) -> None:
        report = review_industrial_golden(case_id="case_a", reviewer=" ")

        self.assertFalse(report["ok"])
        self.assertEqual(report["failure_category"], "missing_reviewer")

    def test_validation_requires_complete_numeric_metrics(self) -> None:
        validation = validate_industrial_golden_for_review(
            {
                "schema_version": "geant4_agent_industrial_golden.v1",
                "case_id": "case_a",
                "runtime_fingerprint": {
                    "runtime_payload_hash": "abc",
                    "physics_list": "FTFP_BERT",
                    "seed": 1337,
                    "events": 10000,
                    "threads": 1,
                },
                "metrics": {
                    "detector_crossing_count": {"expected": 100, "tolerance": 0},
                    "bad_metric": {"expected": None, "tolerance": 0.0},
                },
                "review": {"status": "unreviewed"},
            }
        )

        self.assertFalse(validation["ok"])
        self.assertIn("metric_expected_not_numeric:bad_metric", validation["errors"])
        self.assertIn("missing_runtime_fingerprint:geant4_version", validation["warnings"])

    def test_review_marks_unreviewed_golden_without_changing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_dir = Path(tmpdir)
            golden_path = _write_unreviewed_golden(golden_dir, "case_a")
            before = json.loads(golden_path.read_text(encoding="utf-8"))

            report = review_industrial_golden(
                case_id="case_a",
                golden_dir=golden_dir,
                reviewer="test-reviewer",
                notes="Physics setup and runtime fingerprint checked.",
                evidence="unit-test",
            )
            after = json.loads(golden_path.read_text(encoding="utf-8"))

        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "reviewed")
        self.assertEqual(after["metrics"], before["metrics"])
        self.assertEqual(after["review"]["status"], "reviewed")
        self.assertEqual(after["review"]["reviewer"], "test-reviewer")
        self.assertEqual(after["review"]["previous_status"], "unreviewed")
        self.assertIn("metrics_hash", after["review"])

    def test_dry_run_validates_without_writing_review_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_dir = Path(tmpdir)
            golden_path = _write_unreviewed_golden(golden_dir, "case_a")

            report = review_industrial_golden(
                case_id="case_a",
                golden_dir=golden_dir,
                reviewer="test-reviewer",
                dry_run=True,
            )
            after = json.loads(golden_path.read_text(encoding="utf-8"))

        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "would_review")
        self.assertEqual(after["review"]["status"], "unreviewed")
        self.assertIn("metrics_hash", report)

    def test_review_blocks_already_reviewed_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            golden_dir = Path(tmpdir)
            _write_unreviewed_golden(golden_dir, "case_a")
            first = review_industrial_golden(case_id="case_a", golden_dir=golden_dir, reviewer="test")
            second = review_industrial_golden(case_id="case_a", golden_dir=golden_dir, reviewer="test")

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["failure_category"], "already_reviewed")


def _write_unreviewed_golden(golden_dir: Path, case_id: str) -> Path:
    golden_dir.mkdir(parents=True, exist_ok=True)
    path = golden_dir / f"{case_id}.golden.json"
    payload = {
        "schema_version": "geant4_agent_industrial_golden.v1",
        "case_id": case_id,
        "created_at_utc": "2026-05-15T00:00:00Z",
        "runtime_fingerprint": {
            "geant4_version": "$Name: geant4-11-04 [MT]$",
            "runtime_payload_hash": "abc",
            "physics_list": "FTFP_BERT",
            "seed": 1337,
            "events": 10000,
            "threads": 1,
            "platform": "Windows",
        },
        "runtime_payload_hash": "abc",
        "metrics": {
            "detector_crossing_count": {"expected": 2500, "tolerance": 0},
            "detector_edep_total_mev": {"expected": 0.75, "tolerance": 0.0},
            "transmission_factor": {"expected": 0.25, "tolerance": 0.0},
        },
        "artifact_dir": "",
        "run_summary_path": "",
        "review": {"status": "unreviewed", "reviewer": "", "notes": ""},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
