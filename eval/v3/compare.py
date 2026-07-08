from __future__ import annotations

import json
import sys
from typing import Any

from eval.v3.graders.behavior_grader import V3_BEHAVIOR_GRADE_SCHEMA_VERSION


V3_COMPARE_REQUEST_SCHEMA_VERSION = "geant4_agent_v3_compare_request.v1"
V3_COMPARE_RESULT_SCHEMA_VERSION = "geant4_agent_v3_compare_result.v1"


def compare_v3_grades(baseline: Any, candidate: Any) -> dict[str, Any]:
    baseline_map = _grade_map(baseline)
    candidate_map = _grade_map(candidate)
    shared = sorted(set(baseline_map) & set(candidate_map))
    missing = sorted(set(baseline_map) - set(candidate_map))
    added = sorted(set(candidate_map) - set(baseline_map))
    regressions = [_delta(key, baseline_map[key], candidate_map[key]) for key in shared if baseline_map[key]["pass"] and not candidate_map[key]["pass"]]
    fixes = [_delta(key, baseline_map[key], candidate_map[key]) for key in shared if not baseline_map[key]["pass"] and candidate_map[key]["pass"]]
    candidate_failures = [_summary(key, grade) for key, grade in sorted(candidate_map.items()) if not grade["pass"]]
    return {
        "schema_version": V3_COMPARE_RESULT_SCHEMA_VERSION,
        "ok": not missing and not regressions and not candidate_failures,
        "comparable_count": len(shared),
        "missing_candidate_trials": missing,
        "added_candidate_trials": added,
        "regressions": regressions,
        "fixes": fixes,
        "candidate_failures": candidate_failures,
        "score_delta_by_slice": _slice_deltas(baseline_map, candidate_map),
    }


def compare_v3_grades_payload(payload: Any) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    if raw.get("schema_version") != V3_COMPARE_REQUEST_SCHEMA_VERSION:
        return {
            "schema_version": V3_COMPARE_RESULT_SCHEMA_VERSION,
            "ok": False,
            "error": "compare_request_schema_invalid",
        }
    try:
        return compare_v3_grades(raw.get("baseline"), raw.get("candidate"))
    except ValueError as exc:
        return {"schema_version": V3_COMPARE_RESULT_SCHEMA_VERSION, "ok": False, "error": str(exc)}


def _grade_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("grades"), list):
        grades = value["grades"]
    else:
        grades = value if isinstance(value, list) else []
    result: dict[str, dict[str, Any]] = {}
    for index, grade in enumerate(grades, start=1):
        if not isinstance(grade, dict) or grade.get("schema_version") != V3_BEHAVIOR_GRADE_SCHEMA_VERSION:
            raise ValueError(f"invalid_grade:{index}")
        key = _key(grade)
        if key in result:
            raise ValueError(f"duplicate_grade:{key}")
        result[key] = grade
    return result


def _key(grade: dict[str, Any]) -> str:
    return f"{grade.get('taskId')}::{grade.get('trialIndex')}::{grade.get('variant')}"


def _delta(key: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial": key,
        "slice": _slice(candidate) or _slice(baseline),
        "baseline_pass": bool(baseline.get("pass")),
        "candidate_pass": bool(candidate.get("pass")),
        "baseline_score": float(baseline.get("score") or 0.0),
        "candidate_score": float(candidate.get("score") or 0.0),
        "candidate_failures": list(candidate.get("failures") or []),
    }


def _summary(key: str, grade: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial": key,
        "slice": _slice(grade),
        "score": float(grade.get("score") or 0.0),
        "failures": list(grade.get("failures") or []),
    }


def _slice(grade: dict[str, Any]) -> str:
    metadata = grade.get("metadata") if isinstance(grade.get("metadata"), dict) else {}
    return str(metadata.get("slice") or "unknown")


def _slice_deltas(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, float]:
    slices: dict[str, dict[str, list[float]]] = {}
    for label, grades in (("baseline", baseline), ("candidate", candidate)):
        for grade in grades.values():
            bucket = slices.setdefault(_slice(grade), {"baseline": [], "candidate": []})
            bucket[label].append(float(grade.get("score") or 0.0))
    result: dict[str, float] = {}
    for name, scores in sorted(slices.items()):
        baseline_avg = sum(scores["baseline"]) / len(scores["baseline"]) if scores["baseline"] else 0.0
        candidate_avg = sum(scores["candidate"]) / len(scores["candidate"]) if scores["candidate"] else 0.0
        result[name] = round(candidate_avg - baseline_avg, 6)
    return result


def main() -> int:
    _configure_stdio()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        result = {"schema_version": V3_COMPARE_RESULT_SCHEMA_VERSION, "ok": False, "error": "invalid_json"}
    else:
        result = compare_v3_grades_payload(payload)
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


def _configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
