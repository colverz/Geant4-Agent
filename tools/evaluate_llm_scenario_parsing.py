from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session
from mcp.geant4.runtime_payload import build_runtime_payload
from tools.evaluate_simulation_scenarios import DEFAULT_SCENARIO_CASEBANK

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


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


def _runtime_payload_ready(runtime_payload: dict[str, Any]) -> bool:
    required = ["structure", "material", "source_type", "particle", "energy", "physics_list"]
    return all(runtime_payload.get(key) not in {None, ""} for key in required)


def _case_lang(case: dict[str, Any]) -> str:
    explicit = str(case.get("lang") or "").strip().lower()
    if explicit in {"zh", "en"}:
        return explicit
    prompt = str(case.get("prompt") or "")
    return "zh" if _CJK_PATTERN.search(prompt) else "en"


def _profile_matches_lang(profile_id: Any, lang: str) -> bool:
    if not profile_id:
        return False
    return f"_{lang}_" in str(profile_id)


def _trajectory_from_output(out: dict[str, Any], runtime_payload: dict[str, Any]) -> dict[str, Any]:
    slot_debug = out.get("slot_debug") if isinstance(out.get("slot_debug"), dict) else {}
    return {
        "llm_used": bool(out.get("llm_used")),
        "fallback_reason": out.get("fallback_reason"),
        "is_complete": bool(out.get("is_complete")),
        "runtime_payload_ready": _runtime_payload_ready(runtime_payload),
        "dialogue_action": out.get("dialogue_action"),
        "pipelines": out.get("pipelines"),
        "slot_prompt_profile_id": slot_debug.get("prompt_profile_id"),
        "semantic_prompt_profile_id": (out.get("internal_trace") or {}).get("prompt_profile_id")
        if isinstance(out.get("internal_trace"), dict)
        else None,
    }


def _check_agent_expected(expected: dict[str, Any], trajectory: dict[str, Any], errors: list[str]) -> None:
    if expected.get("must_use_llm") is True and not trajectory["llm_used"]:
        errors.append(f"agent.llm_used:expected=True:actual={trajectory['llm_used']!r}")
    if expected.get("forbid_fallback") is True and trajectory.get("fallback_reason"):
        errors.append(f"agent.fallback_reason:expected=None:actual={trajectory.get('fallback_reason')!r}")
    if expected.get("must_be_complete") is True and not trajectory["is_complete"]:
        errors.append(f"agent.is_complete:expected=True:actual={trajectory['is_complete']!r}")
    if expected.get("must_have_runtime_payload") is True and not trajectory["runtime_payload_ready"]:
        errors.append("agent.runtime_payload_ready:expected=True:actual=False")
    if expected.get("must_apply_session") is True and "apply_session" not in set(trajectory.get("node_sequence") or []):
        errors.append("agent.node_sequence:missing=apply_session")
    if expected.get("must_pass_validate") is True and "validate" not in set(trajectory.get("node_sequence") or []):
        errors.append("agent.node_sequence:missing=validate")
    if expected.get("must_not_call_runtime") is True:
        blocked = set(trajectory.get("tool_calls_blocked") or [])
        allowed = set(trajectory.get("tool_calls_allowed") or [])
        if "run_beam" in allowed or "viewer_open" in allowed:
            errors.append("agent.tool_calls_allowed:runtime_call_present")
        if trajectory.get("action_safety_class") == "config_mutation" and "run_beam" not in blocked:
            errors.append("agent.tool_calls_blocked:missing=run_beam")
    if expected.get("must_detect_composite_runtime_intent") is True:
        composite = trajectory.get("composite_intent") or {}
        if not composite.get("requires_staged_runtime_guard"):
            errors.append("agent.composite_intent.requires_staged_runtime_guard:expected=True:actual=False")
    for item in expected.get("context_must_include_unsupported", []) or []:
        unsupported = trajectory.get("unsupported_capabilities") or {}
        found = any(item in values for values in unsupported.values() if isinstance(values, list))
        if not found:
            errors.append(f"agent.context.unsupported:missing={item}")
    for item in expected.get("context_must_include_supported", []) or []:
        supported = trajectory.get("supported_capabilities") or {}
        found = any(item in values for values in supported.values() if isinstance(values, list))
        if not found:
            errors.append(f"agent.context.supported:missing={item}")
    for source_type in expected.get("context_forbid_knowledge_source_types", []) or []:
        actual_types = set(trajectory.get("knowledge_source_types") or [])
        if source_type in actual_types:
            errors.append(f"agent.context.knowledge_source_type:forbidden={source_type}")


def _process_case(case: dict[str, Any], *, live_llm: bool, llm_config_path: str) -> dict[str, Any]:
    case_id = str(case.get("id") or "unknown")
    prompt = str(case.get("prompt") or "").strip()
    lang = _case_lang(case)
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
            lang=lang,
        )
        if out.get("error"):
            errors.append(f"process_turn_error:{out['error']}")
        llm_used = bool(out.get("llm_used"))
        if live_llm and not llm_used:
            errors.append(f"live_llm_not_used:fallback={out.get('fallback_reason')!r}")

        if "is_complete" in parser_expected and bool(out.get("is_complete")) != bool(parser_expected["is_complete"]):
            errors.append(f"is_complete:expected={parser_expected['is_complete']!r}:actual={out.get('is_complete')!r}")

        runtime_payload = build_runtime_payload(out.get("config", {}))
        turn_trace = out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}
        context_pack = out.get("context_pack") if isinstance(out.get("context_pack"), dict) else {}
        trajectory = _trajectory_from_output(out, runtime_payload)
        trajectory.update(
            {
                "lang": lang,
                "node_sequence": list(turn_trace.get("node_sequence") or []),
                "terminal_state": turn_trace.get("terminal_state"),
                "action_safety_class": turn_trace.get("action_safety_class"),
                "applied_paths": list(turn_trace.get("applied_paths") or []),
                "confirmation_required": bool(turn_trace.get("confirmation_required")),
                "tool_calls_allowed": list(turn_trace.get("tool_calls_allowed") or []),
                "tool_calls_blocked": list(turn_trace.get("tool_calls_blocked") or []),
                "composite_intent": dict(turn_trace.get("composite_intent") or {}),
                "guarded_runtime_intent_pending": bool(turn_trace.get("guarded_runtime_intent_pending")),
                "context_pack_hash": context_pack.get("context_pack_hash"),
                "context_intent": context_pack.get("intent"),
                "supported_capabilities": dict(context_pack.get("supported_capabilities") or {}),
                "unsupported_capabilities": dict(context_pack.get("unsupported_capabilities") or {}),
                "knowledge_source_types": [
                    item.get("source_type")
                    for item in context_pack.get("retrieved_knowledge", [])
                    if isinstance(item, dict)
                ],
            }
        )
        expected_runtime = parser_expected.get("runtime")
        if isinstance(expected_runtime, dict):
            _compare_expected(expected_runtime, runtime_payload, "runtime", errors)
        agent_expected = case.get("agent_expected")
        if isinstance(agent_expected, dict):
            _check_agent_expected(agent_expected, trajectory, errors)
        if live_llm and llm_used:
            slot_profile = trajectory.get("slot_prompt_profile_id")
            if not _profile_matches_lang(slot_profile, lang):
                errors.append(f"agent.slot_prompt_profile_language:expected={lang}:actual={slot_profile!r}")

        return {
            "id": case_id,
            "errors": errors,
            "known_gaps": parser_expected.get("known_gaps", []),
            "lang": lang,
            "llm_used": llm_used,
            "fallback_reason": out.get("fallback_reason"),
            "trajectory": trajectory,
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
    model_override: str = "",
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

    previous_model_override = os.environ.get("GEANT4_LLM_MODEL_OVERRIDE")
    if model_override:
        os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = model_override
    try:
        results = [_process_case(case, live_llm=live_llm, llm_config_path=llm_config_path) for case in cases]
    finally:
        if model_override:
            if previous_model_override is None:
                os.environ.pop("GEANT4_LLM_MODEL_OVERRIDE", None)
            else:
                os.environ["GEANT4_LLM_MODEL_OVERRIDE"] = previous_model_override
    failures = [{"id": result["id"], "errors": result["errors"]} for result in results if result["errors"]]
    total = len(results)
    failed = len(failures)
    passed = total - failed
    accuracy = (passed / total) if total else 0.0
    known_gap_count = sum(len(result.get("known_gaps") or []) for result in results)
    slot_profiles: dict[str, int] = {}
    semantic_profiles: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    profile_mismatch_count = 0
    fallback_count = 0
    llm_used_count = 0
    for result in results:
        lang = str(result.get("lang") or "")
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if result.get("llm_used"):
            llm_used_count += 1
        if result.get("fallback_reason"):
            fallback_count += 1
        trajectory = result.get("trajectory") if isinstance(result.get("trajectory"), dict) else {}
        slot_profile = str(trajectory.get("slot_prompt_profile_id") or "")
        semantic_profile = str(trajectory.get("semantic_prompt_profile_id") or "")
        if slot_profile:
            slot_profiles[slot_profile] = slot_profiles.get(slot_profile, 0) + 1
            if lang and not _profile_matches_lang(slot_profile, lang):
                profile_mismatch_count += 1
        if semantic_profile:
            semantic_profiles[semantic_profile] = semantic_profiles.get(semantic_profile, 0) + 1
    return {
        "name": "llm_scenario_parsing",
        "mode": "live_llm" if live_llm else "offline_v2",
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "min_accuracy": min_accuracy,
        "meets_threshold": failed == 0 and accuracy >= min_accuracy,
        "model_override": model_override or os.environ.get("GEANT4_LLM_MODEL_OVERRIDE", ""),
        "failures": failures,
        "known_gap_count": known_gap_count,
        "live_summary": {
            "llm_used_count": llm_used_count,
            "fallback_count": fallback_count,
            "profile_mismatch_count": profile_mismatch_count,
            "lang_counts": dict(sorted(lang_counts.items())),
            "slot_prompt_profiles": dict(sorted(slot_profiles.items())),
            "semantic_prompt_profiles": dict(sorted(semantic_profiles.items())),
        },
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
    parser.add_argument("--model-override", default=os.environ.get("GEANT4_LLM_MODEL_OVERRIDE", ""))
    args = parser.parse_args()

    env_live = os.environ.get("GEANT4_LLM_SCENARIO", "").strip().lower() in {"1", "true", "yes", "on"}
    live_llm = bool(args.live_llm or env_live)
    report = evaluate_llm_scenario_parsing(
        args.casebank,
        live_llm=live_llm,
        llm_config_path=str(args.llm_config or ""),
        min_accuracy=args.min_accuracy,
        max_cases=args.max_cases or None,
        model_override=str(args.model_override or ""),
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
