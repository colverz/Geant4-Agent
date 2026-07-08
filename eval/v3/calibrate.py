from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.v3.graders.behavior_grader import grade_v3_behavior


V3_CALIBRATION_RESULT_SCHEMA_VERSION = "geant4_agent_v3_grader_calibration.v1"
DEFAULT_CALIBRATION_PATH = Path("eval/v3/calibration/behavior_grader.jsonl")


def calibrate_behavior_grader(path: Path | str = DEFAULT_CALIBRATION_PATH) -> dict[str, Any]:
    cases = load_calibration_cases(Path(path))
    results: list[dict[str, Any]] = []
    for case in cases:
        grade = grade_v3_behavior(case.get("task"), case.get("trial_result"))
        expected = bool(case.get("expected_pass"))
        actual = bool(grade.get("pass"))
        results.append(
            {
                "id": str(case.get("id") or ""),
                "ok": actual == expected,
                "expected_pass": expected,
                "actual_pass": actual,
                "score": grade.get("score"),
                "failures": list(grade.get("failures") or []),
            }
        )
    false_positives = [item["id"] for item in results if item["actual_pass"] and not item["expected_pass"]]
    false_negatives = [item["id"] for item in results if not item["actual_pass"] and item["expected_pass"]]
    return {
        "schema_version": V3_CALIBRATION_RESULT_SCHEMA_VERSION,
        "ok": all(item["ok"] for item in results),
        "calibration_path": str(path),
        "case_count": len(results),
        "matched_count": sum(1 for item in results if item["ok"]),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "cases": results,
    }


def load_calibration_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"calibration_case_not_object:{line_no}")
        if not item.get("id"):
            raise ValueError(f"calibration_case_id_missing:{line_no}")
        if "expected_pass" not in item:
            raise ValueError(f"calibration_expected_pass_missing:{line_no}")
        cases.append(item)
    return cases


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Calibrate the deterministic v3 behavior grader with positive and negative controls.")
    parser.add_argument("--cases", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = calibrate_behavior_grader(args.cases)
    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            f"ok={report['ok']} matched={report['matched_count']}/{report['case_count']} "
            f"false_positive={report['false_positive_count']} false_negative={report['false_negative_count']}"
        )
    return 0 if report.get("ok") else 1


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
