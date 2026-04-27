from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from planner.runtime_intent import classify_user_runtime_intent
from planner.runtime_result import build_runtime_result_question_answer


DEFAULT_WORKFLOW_CASEBANK = Path("docs/eval/workflow_guard_casebank.json")
DEFAULT_RUNTIME_QA_CASEBANK = Path("docs/eval/runtime_result_qa_casebank.json")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_runtime_report() -> dict[str, Any]:
    return {
        "ok": True,
        "events_requested": 4,
        "events_completed": 4,
        "completion_fraction": 1.0,
        "configuration": {
            "geometry_structure": "single_box",
            "material": "G4_Cu",
            "source_type": "point",
            "particle": "gamma",
            "physics_list": "FTFP_BERT",
        },
        "key_metrics": {
            "target_edep_total_mev": 1.5,
            "target_hit_events": 2,
            "detector_crossing_count": 1,
            "plane_crossing_count": 0,
        },
        "artifact_dir": "F:/tmp/artifacts",
        "run_summary_path": "F:/tmp/run_summary.json",
        "result_summary": {
            "source": {
                "primary_count": 4,
                "sampled_position_mean_mm": [0.0, 0.0, -20.0],
                "sampled_direction_mean": [0.0, 0.0, 1.0],
            }
        },
    }


def evaluate_workflow_guard(path: Path) -> dict[str, Any]:
    cases = _load_json(path)
    failures: list[dict[str, Any]] = []
    for case in cases:
        result = classify_user_runtime_intent(case["text"], case["lang"])
        if result.intent.value != case["expected_intent"] or result.action_safety_class.value != case["expected_safety"]:
            failures.append(
                {
                    "id": case["id"],
                    "expected_intent": case["expected_intent"],
                    "actual_intent": result.intent.value,
                    "expected_safety": case["expected_safety"],
                    "actual_safety": result.action_safety_class.value,
                }
            )
    return {"name": "workflow_guard", "total": len(cases), "failed": len(failures), "failures": failures}


def evaluate_runtime_result_qa(path: Path, report: dict[str, Any] | None = None) -> dict[str, Any]:
    cases = _load_json(path)
    report = report or _default_runtime_report()
    failures: list[dict[str, Any]] = []
    for case in cases:
        message = build_runtime_result_question_answer(case["question"], report, lang=case["lang"])
        missing = [text for text in case.get("expected_substrings", []) if text not in message]
        forbidden = [text for text in case.get("forbidden_substrings", []) if text in message]
        if missing or forbidden:
            failures.append(
                {
                    "id": case["id"],
                    "missing": missing,
                    "forbidden_present": forbidden,
                    "message": message,
                }
            )
    return {"name": "runtime_result_qa", "total": len(cases), "failed": len(failures), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate lightweight user-safety and grounded-result casebanks.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW_CASEBANK)
    parser.add_argument("--runtime-qa", type=Path, default=DEFAULT_RUNTIME_QA_CASEBANK)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    reports = [
        evaluate_workflow_guard(args.workflow),
        evaluate_runtime_result_qa(args.runtime_qa),
    ]
    failed = sum(item["failed"] for item in reports)
    output = {"ok": failed == 0, "failed": failed, "reports": reports}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print(f"{report['name']}: {report['total'] - report['failed']} / {report['total']} passed")
            for failure in report["failures"]:
                print(f"  FAIL {failure['id']}: {failure}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
