from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from core.interpreter import run_interpreter_v2


DEFAULT_CASES = [
    {
        "id": "energy_change_and_run_guard",
        "prompt": "Change source energy to 10 MeV and run 10 events now.",
        "context_summary": "source.energy_mev=1",
        "expect_update_path": "source.energy_mev",
        "expect_guarded_action": "run_beam",
    },
    {
        "id": "unsupported_let_not_patch",
        "prompt": "Can you score LET for this proton setup?",
        "context_summary": "supported_scoring=target_edep,detector_crossings,plane_crossings; unsupported_scoring=let",
        "expect_no_candidate_updates": True,
    },
]


def _load_cases(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return list(DEFAULT_CASES)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("casebank must be a JSON list")
    return [case for case in payload if isinstance(case, dict)]


def _check_case(case: dict[str, Any], result: Any) -> list[str]:
    errors: list[str] = []
    if not result.ok:
        errors.extend(result.validation.errors or [result.fallback_reason or "unknown_failure"])
        return errors
    payload = result.payload
    updates = payload.get("candidate_updates") if isinstance(payload.get("candidate_updates"), list) else []
    guarded = payload.get("guarded_actions") if isinstance(payload.get("guarded_actions"), list) else []
    expected_path = case.get("expect_update_path")
    if expected_path and not any(isinstance(item, dict) and item.get("path") == expected_path for item in updates):
        errors.append(f"missing_update_path:{expected_path}")
    expected_action = case.get("expect_guarded_action")
    if expected_action and not any(isinstance(item, dict) and item.get("action") == expected_action for item in guarded):
        errors.append(f"missing_guarded_action:{expected_action}")
    if case.get("expect_no_candidate_updates") and updates:
        errors.append("unexpected_candidate_updates")
    return errors


def evaluate_interpreter_v2_live(
    *,
    config_path: str,
    casebank_path: str | Path | None = None,
    model_override: str = "",
    max_cases: int | None = None,
) -> dict[str, Any]:
    if not config_path:
        return {
            "ok": False,
            "mode": "live_llm",
            "total": 0,
            "passed": 0,
            "failed": 1,
            "failures": [{"id": "<setup>", "errors": ["missing_llm_config_path"]}],
        }
    previous_model_override = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
    if model_override:
        os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = model_override
    try:
        cases = _load_cases(casebank_path)
        if max_cases is not None:
            cases = cases[: max(0, int(max_cases))]
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for case in cases:
            result = run_interpreter_v2(
                str(case.get("prompt", "") or ""),
                str(case.get("context_summary", "") or ""),
                config_path=config_path,
                temperature=0.0,
            )
            errors = _check_case(case, result)
            item = {
                "id": str(case.get("id", "")),
                "ok": not errors,
                "errors": errors,
                "prompt_profile_id": result.prompt_profile_id,
                "fallback_reason": result.fallback_reason,
                "payload": result.payload,
            }
            results.append(item)
            if errors:
                failures.append({"id": item["id"], "errors": errors})
        total = len(results)
        passed = sum(1 for item in results if item["ok"])
        failed = total - passed
        return {
            "ok": failed == 0,
            "mode": "live_llm",
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy": (passed / total) if total else 0.0,
            "model_override": model_override or os.environ.get("GEANT4_LLM_MODEL_OVERRIDE", ""),
            "failures": failures,
            "results": results,
        }
    finally:
        if model_override:
            if previous_model_override is None:
                os.environ.pop("GEANT4_LLM_MODEL_OVERRIDE", None)
            else:
                os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = previous_model_override


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in live LLM smoke test for interpreter v2 path/evidence contract.")
    parser.add_argument("--llm-config", default=os.environ.get("GEANT4_LLM_CONFIG", ""))
    parser.add_argument("--casebank", default="")
    parser.add_argument("--model-override", default=os.environ.get("GEANT4_LLM_MODEL_OVERRIDE", ""))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate_interpreter_v2_live(
        config_path=str(args.llm_config or ""),
        casebank_path=args.casebank or None,
        model_override=str(args.model_override or ""),
        max_cases=args.max_cases,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"ok={report['ok']} passed={report['passed']} failed={report['failed']} total={report['total']}")
        for failure in report.get("failures", []):
            print(f"- {failure['id']}: {failure['errors']}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
