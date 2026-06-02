from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.agent_v3.service import V3AgentTurnService


V3_LIVE_LLM_DIALOGUE_SCHEMA_VERSION = "geant4_agent_v3_live_llm_dialogue.v1"


def _ensure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _safe_print(*args: Any, **kwargs: Any) -> None:
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode("ascii", errors="replace").decode("ascii"))
            else:
                safe_args.append(arg)
        try:
            builtins.print(*safe_args, **kwargs)
        except Exception:
            pass
DEFAULT_DIALOGUE = [
    "我想评估 1 MeV gamma 穿过铅屏蔽后的透射效果，请先给出 Geant4 方案，不要运行。",
    "把刚才方案改成 2 MeV，再跑 5 events。",
    "取消运行，只保留方案。",
]

DEFAULT_DIALOGUE_CLEAN = [
    "我想评估 1 MeV gamma 穿过铅屏蔽后的透射效果，请先给出 Geant4 方案，不要运行。",
    "把刚才方案改成 2 MeV，再跑 5 events。",
    "取消运行，只保留方案。",
]


def run_v3_live_llm_dialogue(
    *,
    live_llm: bool = False,
    llm_config: str = "",
    texts: list[str] | None = None,
    session_id: str = "v3-live-llm-dialogue",
    events: int = 5,
    auto_discover_runtime: bool = True,
    allow_in_memory: bool = False,
    naturalize: bool = False,
    include_full_responses: bool = True,
    sessions_dir: str = "",
    verbose: bool = False,
) -> dict[str, Any]:
    if not live_llm:
        return _skip("live_llm_not_enabled")
    if not llm_config:
        return _fail("llm_config_required")

    temp_sessions = tempfile.TemporaryDirectory() if not sessions_dir else None
    service = V3AgentTurnService(
        sessions_dir=Path(sessions_dir) if sessions_dir else Path(temp_sessions.name)
    )
    service.reset(session_id)
    raw_dialogue: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    llm_raw_responses: list[dict[str, str]] = []

    if verbose:
        _log_header("V3 Live LLM Dialogue", session_id, llm_config)

    for index, text in enumerate(texts or DEFAULT_DIALOGUE_CLEAN, start=1):
        if verbose:
            _log_turn_start(index, text)

        request = _build_request(session_id, text, events, auto_discover_runtime, allow_in_memory, llm_config, naturalize)
        response = service.run_turn(request)
        responses.append(response)

        if verbose:
            _log_trace(response)
            _log_observations(response)
            _log_dialogue(response)

        raw_dialogue.extend(_dialogue_entries(index, text, response))
        for item in _llm_raw_responses(response):
            llm_raw_responses.append({"turn": str(index), **item})

    result = {
        "schema_version": V3_LIVE_LLM_DIALOGUE_SCHEMA_VERSION,
        "ok": all(bool(item.get("ok")) for item in responses),
        "skipped": False,
        "session_id": session_id,
        "llm_config": str(Path(llm_config)),
        "raw_dialogue": raw_dialogue,
        "llm_raw_responses": llm_raw_responses,
        "metrics": _dialogue_eval_metrics(responses, naturalize=naturalize),
    }
    if include_full_responses:
        result["responses"] = responses

    if verbose:
        _log_summary(result)

    return result


# ── verbose logging helpers ──────────────────────────────────────────

def _log_header(session_id: str, llm_config: str, title: str = "") -> None:
    _safe_print(f"\n{'='*60}")
    _safe_print(f"  {title or 'V3 Agent Live LLM Dialogue'}")
    _safe_print(f"  session: {session_id}")
    _safe_print(f"  config:  {llm_config}")
    _safe_print(f"{'='*60}")


def _log_turn_start(index: int, text: str) -> None:
    _safe_print(f"\n  [Turn {index}] USER: {text}")


def _log_trace(response: dict[str, Any]) -> None:
    trace = response.get("trace")
    if not isinstance(trace, list) or not trace:
        return
    _safe_print(f"    chain ({len(trace)} steps):")
    for event in trace:
        if not isinstance(event, dict):
            continue
        step = event.get("step", "?")
        phase = event.get("phase", "?")
        summary = event.get("summary", "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        marker = _phase_marker(phase)
        extra = _trace_extra(phase, data)
        _safe_print(f"      {marker} step {step} [{phase}] {summary}{extra}")


def _log_observations(response: dict[str, Any]) -> None:
    observations = response.get("observations")
    if not isinstance(observations, list) or not observations:
        return
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        source = obs.get("source", "?")
        status = obs.get("status", "?")
        not_eval = obs.get("not_evaluable_reason", "")
        marker = _status_marker(status)
        detail = f" ({not_eval})" if not_eval else ""
        _safe_print(f"      {marker} tool: {source} → {status}{detail}")


def _log_dialogue(response: dict[str, Any]) -> None:
    act = response.get("dialogue_act", "")
    reason = response.get("terminated_reason", "")
    display = response.get("display_message", "")[:200]
    pending = response.get("pending_action")
    pa_info = ""
    if isinstance(pending, dict):
        pa_info = f" [pending: {pending.get('kind', '?')}]"
    _safe_print(f"      >> dialogue_act={act} reason={reason}{pa_info}")
    _safe_print(f"      >> display: {display}")


def _log_summary(result: dict[str, Any]) -> None:
    _safe_print(f"\n{'='*60}")
    _safe_print(f"  result: {'OK' if result.get('ok') else 'FAILED'}")
    llm_resps = result.get("llm_raw_responses") or []
    if llm_resps:
        _safe_print(f"  LLM calls: {len(llm_resps)}")
        for item in llm_resps:
            raw = str(item.get("raw_response", ""))[:150]
            _safe_print(f"    turn {item.get('turn')}: {item.get('ok')} | {raw}")
    _safe_print(f"{'='*60}\n")


def _phase_marker(phase: str) -> str:
    return {
        "perceive": "[PERCEIVE]",
        "reason": "[REASON]",
        "review_constraints": "[REVIEW]",
        "commit_gate": "[GATE]",
        "act": "[ACT]",
        "observe": "[OBSERVE]",
    }.get(phase, "  ")


def _status_marker(status: str) -> str:
    return {"ok": "[OK]", "failed": "[FAIL]", "blocked": "[BLOCKED]", "not_evaluable": "[NOT_EVAL]"}.get(status, "[?]")


def _trace_extra(phase: str, data: dict[str, Any]) -> str:
    if phase == "reason" and data.get("proposal"):
        proposal = data["proposal"] if isinstance(data["proposal"], dict) else {}
        kind = proposal.get("kind", "?")
        intent = proposal.get("intent", "")
        tool_name = ""
        tc = proposal.get("tool_call")
        if isinstance(tc, dict):
            tool_name = tc.get("tool_name", "")
        tool_str = f" → {tool_name}" if tool_name else ""
        return f" [{kind}: {intent}{tool_str}]"
    if phase == "observe" and data.get("observation"):
        obs = data["observation"] if isinstance(data["observation"], dict) else {}
        msg = str(obs.get("message") or "")[:80]
        return f" [{msg}]" if msg else ""
    return ""


# ── helpers ──────────────────────────────────────────────────────────

def _skip(reason: str) -> dict[str, Any]:
    return {
        "schema_version": V3_LIVE_LLM_DIALOGUE_SCHEMA_VERSION,
        "ok": True,
        "skipped": True,
        "skip_reason": reason,
        "raw_dialogue": [],
    }


def _fail(reason: str) -> dict[str, Any]:
    return {
        "schema_version": V3_LIVE_LLM_DIALOGUE_SCHEMA_VERSION,
        "ok": False,
        "skipped": True,
        "skip_reason": reason,
        "raw_dialogue": [],
    }


def _build_request(
    session_id: str,
    text: str,
    events: int,
    auto_discover: bool,
    allow_in_memory: bool,
    llm_config: str,
    naturalize: bool = False,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "text": text,
        "lang": "zh-CN",
        "locale": "zh-CN",
        "events": events,
        "auto_discover_runtime": auto_discover,
        "allow_in_memory": allow_in_memory,
        "llm_design_enabled": True,
        "llm_result_enabled": True,
        "llm_naturalize_enabled": bool(naturalize),
        "llm_config_path": llm_config,
    }


def _dialogue_entries(index: int, text: str, response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "turn": index,
            "role": "user",
            "content": text,
            "request": {"text": text, "events": response.get("state", {}).get("metadata", {}).get("events", "?")},
        },
        {
            "turn": index,
            "role": "agent",
            "content": response.get("display_message") or response.get("answer", {}).get("message", ""),
            "raw_message": response.get("raw_message") or response.get("answer", {}).get("message", ""),
            "dialogue_act": response.get("dialogue_act", ""),
            "terminated_reason": response.get("terminated_reason", ""),
            "dialogue_quality_ok": bool(_dict(response.get("dialogue_quality")).get("ok")),
            "naturalization": _naturalization_brief(response),
            "pending_action": response.get("pending_action"),
            "observations": _observation_brief(response),
        },
    ]


def _observation_brief(response: dict[str, Any]) -> list[dict[str, Any]]:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    return [
        {
            "source": item.get("source"),
            "status": item.get("status"),
            "message": item.get("message"),
            "not_evaluable_reason": item.get("not_evaluable_reason"),
        }
        for item in observations
        if isinstance(item, dict)
    ]


def _llm_raw_responses(response: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("source") != "geant4_llm_design_tool":
            continue
        data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
        llm = data.get("llm") if isinstance(data.get("llm"), dict) else {}
        raw = str(llm.get("raw_response") or "")
        if raw:
            out.append({
                "source": "geant4_llm_design_tool",
                "ok": str(bool(llm.get("ok"))).lower(),
                "prompt_profile_id": str(llm.get("prompt_profile_id") or ""),
                "raw_response": raw,
            })
    return out


def _dialogue_eval_metrics(responses: list[dict[str, Any]], *, naturalize: bool) -> dict[str, Any]:
    quality_reports = [_dict(response.get("dialogue_quality")) for response in responses]
    naturalization_reports = [_dict(response.get("naturalization")) for response in responses if isinstance(response.get("naturalization"), dict)]
    warnings: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    fallback_categories: dict[str, int] = {}
    for report in quality_reports:
        for warning in report.get("warnings") if isinstance(report.get("warnings"), list) else []:
            key = str(warning)
            warnings[key] = warnings.get(key, 0) + 1
    for report in naturalization_reports:
        reason = str(report.get("fallback_reason") or "")
        if reason:
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        category = str(report.get("fallback_category") or "")
        if category:
            fallback_categories[category] = fallback_categories.get(category, 0) + 1
    return {
        "schema_version": "geant4_agent_v3_dialogue_eval_metrics.v1",
        "turn_count": len(responses),
        "ok_turn_count": sum(1 for response in responses if response.get("ok")),
        "dialogue_quality": {
            "reported_count": len(quality_reports),
            "ok_count": sum(1 for report in quality_reports if report.get("ok")),
            "min_score": min((float(report.get("score") or 0.0) for report in quality_reports), default=0.0),
            "warnings": warnings,
        },
        "naturalization": {
            "enabled": bool(naturalize),
            "reported_count": len(naturalization_reports),
            "ok_count": sum(1 for report in naturalization_reports if report.get("ok")),
            "fallback_count": sum(1 for report in naturalization_reports if report and not report.get("ok")),
            "fallback_reasons": fallback_reasons,
            "fallback_categories": fallback_categories,
        },
    }


def _naturalization_brief(response: dict[str, Any]) -> dict[str, Any]:
    naturalization = _dict(response.get("naturalization"))
    if not naturalization:
        return {}
    return {
        "ok": bool(naturalization.get("ok")),
        "used_llm": bool(naturalization.get("used_llm")),
        "fallback_reason": str(naturalization.get("fallback_reason") or ""),
        "fallback_category": str(naturalization.get("fallback_category") or ""),
        "prompt_profile_id": str(naturalization.get("prompt_profile_id") or ""),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run an opt-in v3 live LLM dialogue smoke.")
    parser.add_argument("--live-llm", action="store_true", help="Actually call the configured live LLM.")
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--session-id", default="v3-live-llm-dialogue")
    parser.add_argument("--text", action="append", default=None, help="Dialogue turn text. Can be repeated.")
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--allow-in-memory", action="store_true")
    parser.add_argument("--naturalize", action="store_true", help="Enable optional v3 response naturalization and report its metrics.")
    parser.add_argument("--no-auto-discover-runtime", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Omit full nested responses from JSON output.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed chain-of-reasoning log.")
    args = parser.parse_args()
    result = run_v3_live_llm_dialogue(
        live_llm=args.live_llm,
        llm_config=args.llm_config,
        texts=args.text,
        session_id=args.session_id,
        events=args.events,
        auto_discover_runtime=not args.no_auto_discover_runtime,
        allow_in_memory=args.allow_in_memory,
        naturalize=args.naturalize,
        include_full_responses=not args.compact,
        verbose=args.verbose,
    )
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    elif not args.verbose:
        for item in result.get("raw_dialogue", []):
            _safe_print(f"[turn {item.get('turn')}] {item.get('role')}: {item.get('content')}")
        if result.get("llm_raw_responses"):
            _safe_print("\nLLM raw responses:")
            for item in result["llm_raw_responses"]:
                _safe_print(item["raw_response"])
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
