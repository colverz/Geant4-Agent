from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.agent_v3.service import V3AgentTurnService
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output


V3_DIALOGUE_CASEBANK_EVAL_SCHEMA_VERSION = "geant4_agent_v3_dialogue_casebank_eval.v1"
DEFAULT_CASEBANK_PATH = Path("docs/eval/v3_dialogue_casebank.json")


def evaluate_v3_dialogue_casebank(
    casebank_path: Path | str = DEFAULT_CASEBANK_PATH,
    *,
    live_llm: bool = False,
    llm_config: str = "",
    naturalize: bool = False,
    allow_in_memory: bool = False,
    outdir: Path | str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    cases = _load_cases(Path(casebank_path))
    case_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        service = V3AgentTurnService(sessions_dir=Path(tmpdir))
        for case in cases:
            case_results.append(
                _run_case(
                    service,
                    case,
                    live_llm=live_llm,
                    llm_config=llm_config,
                    naturalize=naturalize,
                    allow_in_memory=allow_in_memory,
                )
            )
        service.reset()

    output = {
        "schema_version": V3_DIALOGUE_CASEBANK_EVAL_SCHEMA_VERSION,
        "ok": all(item["ok"] for item in case_results),
        "casebank": str(casebank_path),
        "mode": "live_llm" if live_llm else "deterministic",
        "naturalize": bool(naturalize),
        "case_count": len(case_results),
        "passed_count": sum(1 for item in case_results if item["ok"]),
        "failed_count": sum(1 for item in case_results if not item["ok"]),
        "metrics": _suite_metrics(case_results),
        "cases": case_results,
    }
    return save_eval_output(output, outdir=outdir, tool="v3-dialogue-casebank", run_id=run_id)


def run_v3_dialogue_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    case = payload.get("case") if isinstance(payload.get("case"), dict) else None
    if case is None:
        prompts = payload.get("prompts") if isinstance(payload.get("prompts"), list) else []
        case = {
            "id": str(payload.get("id") or "adapter-case"),
            "lang": str(payload.get("lang") or "en"),
            "turns": [{"text": str(prompt or "")} for prompt in prompts],
        }
    live_llm = bool(payload.get("live_llm"))
    llm_config = str(payload.get("llm_config") or "")
    naturalize = bool(payload.get("naturalize"))
    allow_in_memory = bool(payload.get("allow_in_memory"))
    with tempfile.TemporaryDirectory() as tmpdir:
        service = V3AgentTurnService(sessions_dir=Path(tmpdir))
        result = _run_case(
            service,
            case,
            live_llm=live_llm,
            llm_config=llm_config,
            naturalize=naturalize,
            allow_in_memory=allow_in_memory,
        )
        service.reset()
    final_message = ""
    if result.get("trajectory"):
        final_message = str(result["trajectory"][-1].get("display_message") or "")
    return {
        "status": "completed" if result.get("ok") else "failed",
        "message": final_message or ("case passed" if result.get("ok") else "case failed"),
        "trajectory": result.get("trajectory") or [],
        "metadata": {
            "schema_version": "geant4_agent_v3_dialogue_adapter_output.v1",
            "adapter": "v3-dialogue-casebank",
            "case_id": result.get("id"),
            "ok": bool(result.get("ok")),
            "failures": result.get("failures") or [],
            "metrics": result.get("metrics") or {},
        },
    }


def _run_case(
    service: V3AgentTurnService,
    case: dict[str, Any],
    *,
    live_llm: bool,
    llm_config: str,
    naturalize: bool,
    allow_in_memory: bool,
) -> dict[str, Any]:
    case_id = str(case.get("id") or "unnamed-case")
    session_id = f"v3-dialogue-casebank-{case_id}"
    lang = str(case.get("lang") or "en").lower()
    locale = "zh-CN" if lang.startswith("zh") else "en-US"
    failures: list[str] = []
    trajectory: list[dict[str, Any]] = []
    turns = case.get("turns") if isinstance(case.get("turns"), list) else []
    for index, turn_spec in enumerate(turns, start=1):
        if not isinstance(turn_spec, dict):
            failures.append(f"turn_{index}:not_object")
            continue
        request = _request_for_turn(
            session_id,
            turn_spec,
            locale=locale,
            live_llm=live_llm,
            llm_config=llm_config,
            naturalize=naturalize,
            allow_in_memory=allow_in_memory,
        )
        response = service.run_turn(request)
        trajectory.append(_trajectory_entry(index, turn_spec, response))
        failures.extend(_grade_turn(index, response, _dict(turn_spec.get("expect"))))
    return {
        "id": case_id,
        "ok": not failures,
        "tags": list(case.get("tags") or []) if isinstance(case.get("tags"), list) else [],
        "failures": failures,
        "trajectory": trajectory,
        "metrics": _case_metrics(trajectory),
    }


def _request_for_turn(
    session_id: str,
    turn_spec: dict[str, Any],
    *,
    locale: str,
    live_llm: bool,
    llm_config: str,
    naturalize: bool,
    allow_in_memory: bool,
) -> dict[str, Any]:
    request = {
        "session_id": session_id,
        "text": str(turn_spec.get("text") or ""),
        "locale": locale,
        "lang": "zh" if locale.startswith("zh") else "en",
        "events": int(turn_spec.get("events") or 5),
        "allow_in_memory": bool(allow_in_memory),
        "llm_design_enabled": bool(live_llm),
        "llm_result_enabled": bool(live_llm),
        "llm_naturalize_enabled": bool(naturalize),
    }
    if live_llm or naturalize:
        request["llm_config_path"] = llm_config
    overrides = _dict(turn_spec.get("request"))
    request.update(overrides)
    return request


def _trajectory_entry(index: int, turn_spec: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(response.get("summary"))
    quality = _dict(response.get("dialogue_quality"))
    naturalization = _dict(response.get("naturalization"))
    return {
        "turn": index,
        "user_text": str(turn_spec.get("text") or ""),
        "ok": bool(response.get("ok")),
        "terminated_reason": str(response.get("terminated_reason") or ""),
        "dialogue_act": str(response.get("dialogue_act") or ""),
        "display_message": str(response.get("display_message") or ""),
        "summary": {
            "phase": summary.get("phase"),
            "runtime_ready": summary.get("runtime_ready"),
            "runtime_ready_reason": summary.get("runtime_ready_reason"),
            "has_payload": summary.get("has_payload"),
            "has_runtime_result": summary.get("has_runtime_result"),
            "needs_confirmation": summary.get("needs_confirmation"),
        },
        "pending_action": _pending_action_brief(response.get("pending_action")),
        "dialogue_quality": {
            "ok": bool(quality.get("ok")),
            "score": quality.get("score"),
            "warnings": list(quality.get("warnings") or []) if isinstance(quality.get("warnings"), list) else [],
        },
        "naturalization": {
            "reported": bool(naturalization),
            "ok": bool(naturalization.get("ok")),
            "fallback_reason": str(naturalization.get("fallback_reason") or ""),
            "fallback_category": str(naturalization.get("fallback_category") or ""),
        },
        "observations": _observation_brief(response),
    }


def _grade_turn(index: int, response: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    summary = _dict(response.get("summary"))
    checks = {
        "dialogue_act": response.get("dialogue_act"),
        "terminated_reason": response.get("terminated_reason"),
        "has_pending_action": response.get("pending_action") is not None,
        "has_payload": bool(summary.get("has_payload")),
        "has_runtime_result": bool(summary.get("has_runtime_result")),
        "runtime_ready": bool(summary.get("runtime_ready")),
        "phase": summary.get("phase"),
        "runtime_ready_reason": summary.get("runtime_ready_reason"),
    }
    for key, actual in checks.items():
        expected_key = key
        expected_in_key = f"{key}_in"
        if expected_key in expect and actual != expect[expected_key]:
            failures.append(f"turn_{index}:{key}:expected={expect[expected_key]!r}:actual={actual!r}")
        if expected_in_key in expect and actual not in expect[expected_in_key]:
            failures.append(f"turn_{index}:{key}:expected_in={expect[expected_in_key]!r}:actual={actual!r}")
    quality = _dict(response.get("dialogue_quality"))
    if expect.get("dialogue_quality_ok", True) and quality and not quality.get("ok"):
        failures.append(f"turn_{index}:dialogue_quality_failed:{quality.get('warnings')!r}")
    if "naturalization_ok" in expect:
        naturalization = _dict(response.get("naturalization"))
        if bool(naturalization.get("ok")) != bool(expect["naturalization_ok"]):
            failures.append(f"turn_{index}:naturalization_ok:expected={expect['naturalization_ok']!r}:actual={naturalization.get('ok')!r}")
    display = str(response.get("display_message") or "")
    for item in expect.get("display_contains") if isinstance(expect.get("display_contains"), list) else []:
        if str(item) not in display:
            failures.append(f"turn_{index}:display_missing:{item!r}")
    for item in expect.get("display_not_contains") if isinstance(expect.get("display_not_contains"), list) else []:
        if str(item) in display:
            failures.append(f"turn_{index}:display_forbidden:{item!r}")
    if "min_dialogue_quality_score" in expect:
        try:
            min_score = float(expect["min_dialogue_quality_score"])
            actual = float(quality.get("score") or 0.0)
            if actual < min_score:
                failures.append(f"turn_{index}:dialogue_quality_score_below_min:expected>={min_score}:actual={actual}")
        except (TypeError, ValueError):
            failures.append(f"turn_{index}:invalid_min_dialogue_quality_score")
    return failures


def _case_metrics(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    quality_scores = [
        float(_dict(turn.get("dialogue_quality")).get("score") or 0.0)
        for turn in trajectory
        if _dict(turn.get("dialogue_quality"))
    ]
    return {
        "turn_count": len(trajectory),
        "dialogue_quality_ok_count": sum(1 for turn in trajectory if _dict(turn.get("dialogue_quality")).get("ok")),
        "min_dialogue_quality_score": min(quality_scores, default=0.0),
        "naturalization_reported_count": sum(1 for turn in trajectory if _dict(turn.get("naturalization")).get("reported")),
        "naturalization_ok_count": sum(1 for turn in trajectory if _dict(turn.get("naturalization")).get("ok")),
    }


def _suite_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories = [turn for case in case_results for turn in case.get("trajectory", []) if isinstance(turn, dict)]
    warnings: dict[str, int] = {}
    warning_turns: list[dict[str, Any]] = []
    fallbacks: dict[str, int] = {}
    fallback_categories: dict[str, int] = {}
    for case in case_results:
        case_id = str(case.get("id") or "")
        for turn in case.get("trajectory", []) if isinstance(case.get("trajectory"), list) else []:
            if not isinstance(turn, dict):
                continue
            quality = _dict(turn.get("dialogue_quality"))
            turn_warnings = [str(item) for item in quality.get("warnings") or [] if str(item)] if isinstance(quality.get("warnings"), list) else []
            for warning in turn_warnings:
                warnings[warning] = warnings.get(warning, 0) + 1
            if turn_warnings:
                warning_turns.append(
                    {
                        "case_id": case_id,
                        "turn": turn.get("turn"),
                        "dialogue_act": turn.get("dialogue_act"),
                        "warnings": turn_warnings,
                    }
                )
            naturalization = _dict(turn.get("naturalization"))
            reason = str(naturalization.get("fallback_reason") or "")
            if reason:
                fallbacks[reason] = fallbacks.get(reason, 0) + 1
            category = str(naturalization.get("fallback_category") or "")
            if category:
                fallback_categories[category] = fallback_categories.get(category, 0) + 1
    scores = [float(_dict(turn.get("dialogue_quality")).get("score") or 0.0) for turn in trajectories]
    return {
        "turn_count": len(trajectories),
        "dialogue_quality_ok_count": sum(1 for turn in trajectories if _dict(turn.get("dialogue_quality")).get("ok")),
        "min_dialogue_quality_score": min(scores, default=0.0),
        "dialogue_quality_warnings": warnings,
        "dialogue_quality_warning_count": sum(warnings.values()),
        "dialogue_quality_warning_turns": warning_turns,
        "naturalization_reported_count": sum(1 for turn in trajectories if _dict(turn.get("naturalization")).get("reported")),
        "naturalization_ok_count": sum(1 for turn in trajectories if _dict(turn.get("naturalization")).get("ok")),
        "naturalization_fallback_reasons": fallbacks,
        "naturalization_fallback_categories": fallback_categories,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("v3_dialogue_casebank_must_be_list")
    return [item for item in data if isinstance(item, dict)]


def _pending_action_brief(raw: Any) -> dict[str, Any]:
    pending = _dict(raw)
    if not pending:
        return {}
    return {
        "kind": pending.get("kind"),
        "intent": pending.get("intent"),
        "risk_level": pending.get("risk_level"),
        "requires_confirmation": bool(pending.get("requires_confirmation")),
    }


def _observation_brief(response: dict[str, Any]) -> list[dict[str, Any]]:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    return [
        {
            "source": item.get("source"),
            "status": item.get("status"),
            "not_evaluable_reason": item.get("not_evaluable_reason"),
        }
        for item in observations
        if isinstance(item, dict)
    ]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Evaluate v3 dialogue casebank with structured trajectory metrics.")
    parser.add_argument("--adapter", action="store_true", help="Read one adapter-input JSON object from stdin and write one adapter-output JSON object to stdout.")
    parser.add_argument("--casebank", default=str(DEFAULT_CASEBANK_PATH))
    parser.add_argument("--live-llm", action="store_true")
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--naturalize", action="store_true")
    parser.add_argument("--allow-in-memory", action="store_true")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.adapter:
        payload = json.load(sys.stdin)
        output = run_v3_dialogue_adapter(payload if isinstance(payload, dict) else {})
        json.dump(output, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0 if output.get("status") == "completed" else 1
    report = evaluate_v3_dialogue_casebank(
        args.casebank,
        live_llm=bool(args.live_llm),
        llm_config=str(args.llm_config or ""),
        naturalize=bool(args.naturalize),
        allow_in_memory=bool(args.allow_in_memory),
        outdir=Path(args.outdir) if args.outdir else None,
        run_id=str(args.run_id or "") or None,
    )
    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"ok={report['ok']} passed={report['passed_count']} failed={report['failed_count']}")
    return 0 if report.get("ok") else 1


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
