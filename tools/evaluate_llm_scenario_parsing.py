from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session
from mcp.geant4.runtime_payload import build_runtime_payload
from tools.evaluate_simulation_scenarios import DEFAULT_SCENARIO_CASEBANK


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_cases(cases: Any, max_cases: int | None) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        return []
    normalized = [case for case in cases if isinstance(case, dict)]
    if max_cases is None or max_cases <= 0:
        return normalized
    return normalized[:max_cases]


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _compare_expected(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}:expected_dict:actual={actual!r}")
            return
        for key, expected_value in expected.items():
            _compare_expected(expected_value, actual.get(key), f"{path}.{key}" if path else key, errors)
        return
    if isinstance(expected, float):
        if not _float_equal(actual, expected):
            errors.append(f"{path}:expected={expected!r}:actual={actual!r}")
        return
    if actual != expected:
        errors.append(f"{path}:expected={expected!r}:actual={actual!r}")


def _process_case(case: dict[str, Any], *, live_llm: bool, llm_config_path: str) -> dict[str, Any]:
    case_id = str(case.get("id") or "unknown")
    prompt = str(case.get("prompt") or "").strip()
    parser_expected = case.get("parser_expected", {}) if isinstance(case.get("parser_expected"), dict) else {}
    errors: list[str] = []
    if not prompt:
        return {"id": case_id, "errors": ["missing_prompt"], "known_gaps": parser_expected.get("known_gaps", [])}

    session_id = f"llm-scenario-parsing-{case_id}-{'live' if live_llm else 'offline'}"
    reset_session(session_id)
    try:
        out = process_turn(
            {
                "session_id": session_id,
                "text": prompt,
                "llm_router": live_llm,
                "llm_question": False,
                "normalize_input": True,
                "geometry_pipeline": "v2",
                "source_pipeline": "v2",
                "enable_compare": False,
                "autofix": True,
            },
            ollama_config_path=llm_config_path,
            lang="en",
        )
        if out.get("error"):
            errors.append(f"process_turn_error:{out['error']}")
        llm_used = bool(out.get("llm_used"))
        if live_llm and not llm_used:
            errors.append(f"live_llm_not_used:fallback={out.get('fallback_reason')!r}")

        if "is_complete" in parser_expected and bool(out.get("is_complete")) != bool(parser_expected["is_complete"]):
            errors.append(f"is_complete:expected={parser_expected['is_complete']!r}:actual={out.get('is_complete')!r}")

        runtime_payload = build_runtime_payload(out.get("config", {}))
        expected_runtime = parser_expected.get("runtime")
        if isinstance(expected_runtime, dict):
            _compare_expected(expected_runtime, runtime_payload, "runtime", errors)

        return {
            "id": case_id,
            "errors": errors,
            "known_gaps": parser_expected.get("known_gaps", []),
            "llm_used": llm_used,
            "fallback_reason": out.get("fallback_reason"),
        }
    finally:
        reset_session(session_id)


def evaluate_llm_scenario_parsing(
    path: Path = DEFAULT_SCENARIO_CASEBANK,
    *,
    live_llm: bool = False,
    llm_config_path: str = "",
    min_accuracy: float = 1.0,
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = _slice_cases(_load_json(path), max_cases)
    if live_llm and not llm_config_path:
        return {
            "name": "llm_scenario_parsing",
            "mode": "live_llm",
            "total": len(cases),
            "passed": 0,
            "failed": 1,
            "accuracy": 0.0,
            "min_accuracy": min_accuracy,
            "meets_threshold": False,
            "failures": [{"id": "<setup>", "errors": ["missing_llm_config_path"]}],
            "known_gap_count": 0,
        }

    results = [_process_case(case, live_llm=live_llm, llm_config_path=llm_config_path) for case in cases]
    failures = [{"id": result["id"], "errors": result["errors"]} for result in results if result["errors"]]
    total = len(results)
    failed = len(failures)
    passed = total - failed
    accuracy = (passed / total) if total else 0.0
    known_gap_count = sum(len(result.get("known_gaps") or []) for result in results)
    return {
        "name": "llm_scenario_parsing",
        "mode": "live_llm" if live_llm else "offline_v2",
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "min_accuracy": min_accuracy,
        "meets_threshold": failed == 0 and accuracy >= min_accuracy,
        "failures": failures,
        "known_gap_count": known_gap_count,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate natural-language scenario parsing into the runtime bridge contract.")
    parser.add_argument("--casebank", type=Path, default=DEFAULT_SCENARIO_CASEBANK)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--live-llm", action="store_true", help="Opt in to the configured live LLM path.")
    parser.add_argument("--llm-config", default=os.environ.get("GEANT4_LLM_CONFIG", ""))
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit evaluated cases for low-cost live smoke runs.")
    args = parser.parse_args()

    env_live = os.environ.get("GEANT4_LLM_SCENARIO", "").strip().lower() in {"1", "true", "yes", "on"}
    live_llm = bool(args.live_llm or env_live)
    report = evaluate_llm_scenario_parsing(
        args.casebank,
        live_llm=live_llm,
        llm_config_path=str(args.llm_config or ""),
        min_accuracy=args.min_accuracy,
        max_cases=args.max_cases or None,
    )
    output = {
        "ok": bool(report["meets_threshold"]),
        "failed": report["failed"],
        "accuracy": report["accuracy"],
        "min_accuracy": report["min_accuracy"],
        "reports": [report],
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(
            f"{report['name']}[{report['mode']}]: "
            f"{report['passed']} / {report['total']} passed "
            f"(accuracy={report['accuracy']:.3f}, min={report['min_accuracy']:.3f})"
        )
        if report.get("known_gap_count"):
            print(f"  known gaps documented: {report['known_gap_count']}")
        for failure in report["failures"]:
            print(f"  FAIL {failure['id']}: {failure}")
    return 0 if report["meets_threshold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
