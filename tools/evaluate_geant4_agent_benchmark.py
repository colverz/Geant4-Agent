from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session
from mcp.geant4.runtime_payload import build_runtime_payload
from planner.runtime_result import build_runtime_result_question_answer


DEFAULT_BENCHMARK_PATH = Path("docs/eval/agentic_benchmark_v1.json")

VALID_SUITES = {"core", "trajectory", "grounding", "tool_guard", "runtime", "result_qa", "live_llm", "routing"}
VALID_DIFFICULTIES = {"smoke", "standard", "adversarial", "expert", "live"}
VALID_LANGS = {"en", "zh"}
VALID_CAPABILITIES = {
    "intent_routing",
    "config_extraction",
    "grounding",
    "confirmation_policy",
    "workflow_trace",
    "tool_guard",
    "runtime_readiness",
    "result_grounding",
    "llm_reliability",
    "model_routing",
}
VALID_INTENTS = {"read_config", "read_summary", "config_mutation", "run_requested", "viewer_requested", "normal_chat"}
VALID_SAFETY = {"read_only", "config_mutation", "expensive_runtime"}
VALID_TERMINALS = {
    "read_only_answer",
    "mutation_applied",
    "waiting_confirmation",
    "rejected",
    "unsupported",
    "runtime_action_guarded",
    "error",
}
VALID_NODES = {
    "start",
    "route_intent",
    "build_context",
    "interpret",
    "check_grounding",
    "normalize_patch",
    "validate",
    "confirmation_policy",
    "wait_confirmation",
    "apply_session",
    "runtime_guard",
    "answer",
    "end",
}
VALID_TOOLS = {"run_beam", "viewer_open"}
VALID_MODEL_ROUTE_LABELS = {
    "no_llm_required",
    "cheap_model_ok",
    "strong_model_candidate",
    "escalate_after_validation_failure",
    "human_confirmation_required",
}

TOP_LEVEL_KEYS = {
    "id",
    "suite",
    "difficulty",
    "lang",
    "turns",
    "capabilities",
    "requires_live_llm",
    "requires_real_runtime",
    "expected_runtime",
    "expected_result_answer",
    "expected_model_route",
    "forbidden",
}
TURN_KEYS = {"text", "lang", "expected_trace"}
TRACE_KEYS = {
    "intent",
    "action_safety_class",
    "terminal_state",
    "must_include_nodes",
    "must_not_include_nodes",
    "must_block_tools",
    "guarded_runtime_intent_pending",
    "must_not_apply_session",
    "must_not_call_runtime",
}
RUNTIME_KEYS = {"must_have_runtime_payload", "required_payload_keys", "expected_payload_values"}
RESULT_ANSWER_KEYS = {"question", "must_include", "must_not_include", "must_remain_read_only"}
MODEL_ROUTE_KEYS = {"label", "must_not_allow_runtime", "rationale_contains"}
FORBIDDEN_KEYS = {"runtime_side_effects", "session_mutation", "unsupported_capability_as_supported"}
REQUIRED_TOP_LEVEL_KEYS = {"id", "suite", "difficulty", "lang", "turns"}
MIN_V1_SUITE_COUNTS = {
    "core": 1,
    "grounding": 1,
    "result_qa": 1,
    "runtime": 1,
    "tool_guard": 2,
    "trajectory": 1,
}
MIN_V1_DIFFICULTY_COUNTS = {"smoke": 2, "standard": 2, "adversarial": 2}
MIN_V1_CAPABILITY_COUNTS = {
    "intent_routing": 3,
    "workflow_trace": 5,
    "tool_guard": 2,
    "runtime_readiness": 1,
    "grounding": 1,
    "result_grounding": 1,
    "confirmation_policy": 1,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_ids(items: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        item_id = str(item.get("id", ""))
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    return sorted(duplicates)


def _add_unknown_key_errors(
    failures: list[dict[str, Any]],
    *,
    case_id: str,
    section: str,
    payload: dict[str, Any],
    allowed: set[str],
) -> None:
    for key in sorted(set(payload) - allowed):
        failures.append({"id": case_id, "section": section, "error": f"unsupported_key:{key}"})


def _validate_string_list(
    failures: list[dict[str, Any]],
    *,
    case_id: str,
    section: str,
    field: str,
    value: Any,
    allowed: set[str] | None = None,
) -> None:
    if not isinstance(value, list):
        failures.append({"id": case_id, "section": section, "error": f"{field}_not_list"})
        return
    for item in value:
        if not isinstance(item, str):
            failures.append({"id": case_id, "section": section, "error": f"{field}_item_not_string"})
            continue
        if allowed is not None and item not in allowed:
            failures.append({"id": case_id, "section": section, "error": f"invalid_{field}_item:{item}"})


def validate_benchmark_shape(path: Path = DEFAULT_BENCHMARK_PATH) -> dict[str, Any]:
    data = _load_json(path)
    failures: list[dict[str, Any]] = []
    suite_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    capability_counts: dict[str, int] = {}

    if not isinstance(data, list):
        return {
            "name": "geant4_agent_benchmark_shape",
            "total": 0,
            "failed": 1,
            "failures": [{"id": "<root>", "section": "root", "error": "not_list"}],
            "suite_counts": {},
            "difficulty_counts": {},
            "capability_counts": {},
        }

    for duplicate in _duplicate_ids([item for item in data if isinstance(item, dict)]):
        failures.append({"id": duplicate, "section": "root", "error": "duplicate_id"})

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            failures.append({"id": f"<item:{index}>", "section": "root", "error": "not_object"})
            continue
        case_id = str(item.get("id") or f"<item:{index}>")
        _add_unknown_key_errors(failures, case_id=case_id, section="root", payload=item, allowed=TOP_LEVEL_KEYS)
        for missing in sorted(REQUIRED_TOP_LEVEL_KEYS - set(item)):
            failures.append({"id": case_id, "section": "root", "error": f"missing_key:{missing}"})

        suite = item.get("suite")
        if suite not in VALID_SUITES:
            failures.append({"id": case_id, "section": "root", "error": f"invalid_suite:{suite}"})
        elif isinstance(suite, str):
            suite_counts[suite] = suite_counts.get(suite, 0) + 1

        difficulty = item.get("difficulty")
        if difficulty not in VALID_DIFFICULTIES:
            failures.append({"id": case_id, "section": "root", "error": f"invalid_difficulty:{difficulty}"})
        elif isinstance(difficulty, str):
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        if item.get("lang") not in VALID_LANGS:
            failures.append({"id": case_id, "section": "root", "error": f"invalid_lang:{item.get('lang')}"})

        if "capabilities" in item:
            _validate_string_list(
                failures,
                case_id=case_id,
                section="root.capabilities",
                field="capabilities",
                value=item["capabilities"],
                allowed=VALID_CAPABILITIES,
            )
            if isinstance(item.get("capabilities"), list):
                for capability in item["capabilities"]:
                    if isinstance(capability, str) and capability in VALID_CAPABILITIES:
                        capability_counts[capability] = capability_counts.get(capability, 0) + 1

        for bool_key in ("requires_live_llm", "requires_real_runtime"):
            if bool_key in item and not isinstance(item[bool_key], bool):
                failures.append({"id": case_id, "section": "root", "error": f"{bool_key}_not_bool"})

        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            failures.append({"id": case_id, "section": "turns", "error": "turns_empty_or_not_list"})
        elif turns:
            for turn_index, turn in enumerate(turns):
                section = f"turns[{turn_index}]"
                if not isinstance(turn, dict):
                    failures.append({"id": case_id, "section": section, "error": "not_object"})
                    continue
                _add_unknown_key_errors(failures, case_id=case_id, section=section, payload=turn, allowed=TURN_KEYS)
                if not str(turn.get("text") or "").strip():
                    failures.append({"id": case_id, "section": section, "error": "missing_text"})
                if "lang" in turn and turn["lang"] not in VALID_LANGS:
                    failures.append({"id": case_id, "section": section, "error": f"invalid_lang:{turn['lang']}"})
                expected_trace = turn.get("expected_trace")
                if expected_trace is not None:
                    if not isinstance(expected_trace, dict):
                        failures.append({"id": case_id, "section": f"{section}.expected_trace", "error": "not_object"})
                    else:
                        _validate_trace(failures, case_id=case_id, trace=expected_trace)

        if "expected_runtime" in item:
            expected_runtime = item["expected_runtime"]
            if not isinstance(expected_runtime, dict):
                failures.append({"id": case_id, "section": "expected_runtime", "error": "not_object"})
            else:
                _validate_runtime(failures, case_id=case_id, runtime=expected_runtime)

        if "expected_result_answer" in item:
            expected_result_answer = item["expected_result_answer"]
            if not isinstance(expected_result_answer, dict):
                failures.append({"id": case_id, "section": "expected_result_answer", "error": "not_object"})
            else:
                _validate_result_answer(failures, case_id=case_id, expected=expected_result_answer)

        if "expected_model_route" in item:
            expected_model_route = item["expected_model_route"]
            if not isinstance(expected_model_route, dict):
                failures.append({"id": case_id, "section": "expected_model_route", "error": "not_object"})
            else:
                _validate_model_route(failures, case_id=case_id, expected=expected_model_route)

        if "forbidden" in item:
            forbidden = item["forbidden"]
            if not isinstance(forbidden, dict):
                failures.append({"id": case_id, "section": "forbidden", "error": "not_object"})
            else:
                _add_unknown_key_errors(
                    failures,
                    case_id=case_id,
                    section="forbidden",
                    payload=forbidden,
                    allowed=FORBIDDEN_KEYS,
                )
                for key, value in forbidden.items():
                    if key in FORBIDDEN_KEYS and not isinstance(value, bool):
                        failures.append({"id": case_id, "section": "forbidden", "error": f"{key}_not_bool"})

    return {
        "name": "geant4_agent_benchmark_shape",
        "total": len(data),
        "failed": len(failures),
        "failures": failures,
        "suite_counts": dict(sorted(suite_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
    }


def validate_benchmark_coverage(path: Path = DEFAULT_BENCHMARK_PATH) -> dict[str, Any]:
    shape_report = validate_benchmark_shape(path)
    failures: list[dict[str, Any]] = []
    if shape_report["failed"]:
        failures.append({"section": "shape", "error": "shape_validation_failed"})

    for suite, minimum in MIN_V1_SUITE_COUNTS.items():
        actual = int(shape_report["suite_counts"].get(suite, 0))
        if actual < minimum:
            failures.append({"section": "suite_counts", "error": f"{suite}:min={minimum}:actual={actual}"})
    for difficulty, minimum in MIN_V1_DIFFICULTY_COUNTS.items():
        actual = int(shape_report["difficulty_counts"].get(difficulty, 0))
        if actual < minimum:
            failures.append({"section": "difficulty_counts", "error": f"{difficulty}:min={minimum}:actual={actual}"})
    for capability, minimum in MIN_V1_CAPABILITY_COUNTS.items():
        actual = int(shape_report["capability_counts"].get(capability, 0))
        if actual < minimum:
            failures.append({"section": "capability_counts", "error": f"{capability}:min={minimum}:actual={actual}"})

    return {
        "name": "geant4_agent_benchmark_coverage",
        "total": (
            len(MIN_V1_SUITE_COUNTS)
            + len(MIN_V1_DIFFICULTY_COUNTS)
            + len(MIN_V1_CAPABILITY_COUNTS)
            + 1
        ),
        "failed": len(failures),
        "failures": failures,
        "minimums": {
            "suite_counts": MIN_V1_SUITE_COUNTS,
            "difficulty_counts": MIN_V1_DIFFICULTY_COUNTS,
            "capability_counts": MIN_V1_CAPABILITY_COUNTS,
        },
        "shape_report": shape_report,
    }


def _validate_trace(failures: list[dict[str, Any]], *, case_id: str, trace: dict[str, Any]) -> None:
    _add_unknown_key_errors(failures, case_id=case_id, section="expected_trace", payload=trace, allowed=TRACE_KEYS)
    if "intent" in trace and trace["intent"] not in VALID_INTENTS:
        failures.append({"id": case_id, "section": "expected_trace", "error": f"invalid_intent:{trace['intent']}"})
    if "action_safety_class" in trace and trace["action_safety_class"] not in VALID_SAFETY:
        failures.append(
            {"id": case_id, "section": "expected_trace", "error": f"invalid_action_safety_class:{trace['action_safety_class']}"}
        )
    if "terminal_state" in trace and trace["terminal_state"] not in VALID_TERMINALS:
        failures.append({"id": case_id, "section": "expected_trace", "error": f"invalid_terminal_state:{trace['terminal_state']}"})
    for field in ("must_include_nodes", "must_not_include_nodes"):
        if field in trace:
            _validate_string_list(
                failures,
                case_id=case_id,
                section="expected_trace",
                field=field,
                value=trace[field],
                allowed=VALID_NODES,
            )
    if "must_block_tools" in trace:
        _validate_string_list(
            failures,
            case_id=case_id,
            section="expected_trace",
            field="must_block_tools",
            value=trace["must_block_tools"],
            allowed=VALID_TOOLS,
        )
    for bool_key in (
        "guarded_runtime_intent_pending",
        "must_not_apply_session",
        "must_not_call_runtime",
    ):
        if bool_key in trace and not isinstance(trace[bool_key], bool):
            failures.append({"id": case_id, "section": "expected_trace", "error": f"{bool_key}_not_bool"})


def _validate_runtime(failures: list[dict[str, Any]], *, case_id: str, runtime: dict[str, Any]) -> None:
    _add_unknown_key_errors(failures, case_id=case_id, section="expected_runtime", payload=runtime, allowed=RUNTIME_KEYS)
    if "must_have_runtime_payload" in runtime and not isinstance(runtime["must_have_runtime_payload"], bool):
        failures.append({"id": case_id, "section": "expected_runtime", "error": "must_have_runtime_payload_not_bool"})
    if "required_payload_keys" in runtime:
        _validate_string_list(
            failures,
            case_id=case_id,
            section="expected_runtime",
            field="required_payload_keys",
            value=runtime["required_payload_keys"],
        )
    if "expected_payload_values" in runtime and not isinstance(runtime["expected_payload_values"], dict):
        failures.append({"id": case_id, "section": "expected_runtime", "error": "expected_payload_values_not_object"})


def _validate_result_answer(failures: list[dict[str, Any]], *, case_id: str, expected: dict[str, Any]) -> None:
    _add_unknown_key_errors(failures, case_id=case_id, section="expected_result_answer", payload=expected, allowed=RESULT_ANSWER_KEYS)
    if "question" in expected and not isinstance(expected["question"], str):
        failures.append({"id": case_id, "section": "expected_result_answer", "error": "question_not_string"})
    for field in ("must_include", "must_not_include"):
        if field in expected:
            _validate_string_list(
                failures,
                case_id=case_id,
                section="expected_result_answer",
                field=field,
                value=expected[field],
            )
    if "must_remain_read_only" in expected and not isinstance(expected["must_remain_read_only"], bool):
        failures.append({"id": case_id, "section": "expected_result_answer", "error": "must_remain_read_only_not_bool"})


def _validate_model_route(failures: list[dict[str, Any]], *, case_id: str, expected: dict[str, Any]) -> None:
    _add_unknown_key_errors(failures, case_id=case_id, section="expected_model_route", payload=expected, allowed=MODEL_ROUTE_KEYS)
    label = expected.get("label")
    if label not in VALID_MODEL_ROUTE_LABELS:
        failures.append({"id": case_id, "section": "expected_model_route", "error": f"invalid_label:{label}"})
    if "must_not_allow_runtime" in expected and not isinstance(expected["must_not_allow_runtime"], bool):
        failures.append({"id": case_id, "section": "expected_model_route", "error": "must_not_allow_runtime_not_bool"})
    if "rationale_contains" in expected:
        _validate_string_list(
            failures,
            case_id=case_id,
            section="expected_model_route",
            field="rationale_contains",
            value=expected["rationale_contains"],
        )


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return _float_equal(left, right)
    return left == right


def _runtime_payload_ready(payload: dict[str, Any], required_keys: list[str] | None = None) -> bool:
    required = required_keys or ["structure", "material", "source_type", "particle", "energy", "physics_list"]
    return all(payload.get(key) not in {None, ""} for key in required)


def _trace_errors(expected: dict[str, Any], actual: dict[str, Any], *, case_id: str, turn_index: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    section = f"turns[{turn_index}].expected_trace"
    if "intent" in expected and actual.get("intent") != expected["intent"]:
        failures.append({"id": case_id, "section": section, "error": f"intent:expected={expected['intent']}:actual={actual.get('intent')}"})
    if "action_safety_class" in expected and actual.get("action_safety_class") != expected["action_safety_class"]:
        failures.append(
            {
                "id": case_id,
                "section": section,
                "error": f"action_safety_class:expected={expected['action_safety_class']}:actual={actual.get('action_safety_class')}",
            }
        )
    if "terminal_state" in expected and actual.get("terminal_state") != expected["terminal_state"]:
        failures.append(
            {
                "id": case_id,
                "section": section,
                "error": f"terminal_state:expected={expected['terminal_state']}:actual={actual.get('terminal_state')}",
            }
        )

    nodes = set(actual.get("node_sequence") or [])
    for node in expected.get("must_include_nodes", []) or []:
        if node not in nodes:
            failures.append({"id": case_id, "section": section, "error": f"missing_node:{node}"})
    for node in expected.get("must_not_include_nodes", []) or []:
        if node in nodes:
            failures.append({"id": case_id, "section": section, "error": f"forbidden_node:{node}"})

    blocked = set(actual.get("tool_calls_blocked") or [])
    for tool_name in expected.get("must_block_tools", []) or []:
        if tool_name not in blocked:
            failures.append({"id": case_id, "section": section, "error": f"missing_blocked_tool:{tool_name}"})
    if "guarded_runtime_intent_pending" in expected and bool(actual.get("guarded_runtime_intent_pending")) != bool(
        expected["guarded_runtime_intent_pending"]
    ):
        failures.append(
            {
                "id": case_id,
                "section": section,
                "error": (
                    "guarded_runtime_intent_pending:"
                    f"expected={bool(expected['guarded_runtime_intent_pending'])}:actual={bool(actual.get('guarded_runtime_intent_pending'))}"
                ),
            }
        )
    if expected.get("must_not_apply_session"):
        if "apply_session" in nodes:
            failures.append({"id": case_id, "section": section, "error": "forbidden_apply_session_node"})
        if actual.get("applied_paths"):
            failures.append({"id": case_id, "section": section, "error": f"forbidden_applied_paths:{actual.get('applied_paths')}"})
    if expected.get("must_not_call_runtime"):
        allowed = set(actual.get("tool_calls_allowed") or [])
        runtime_allowed = sorted(allowed & VALID_TOOLS)
        if runtime_allowed:
            failures.append({"id": case_id, "section": section, "error": f"runtime_tool_allowed:{runtime_allowed}"})
    return failures


def _runtime_errors(expected: dict[str, Any], payload: dict[str, Any], *, case_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required_keys = list(expected.get("required_payload_keys") or [])
    if expected.get("must_have_runtime_payload") and not _runtime_payload_ready(payload, required_keys or None):
        failures.append({"id": case_id, "section": "expected_runtime", "error": "runtime_payload_not_ready"})
    for key in required_keys:
        if payload.get(key) in {None, ""}:
            failures.append({"id": case_id, "section": "expected_runtime", "error": f"missing_payload_key:{key}"})
    expected_values = expected.get("expected_payload_values") or {}
    if isinstance(expected_values, dict):
        for key, value in expected_values.items():
            actual = payload.get(key)
            if not _values_equal(actual, value):
                failures.append({"id": case_id, "section": "expected_runtime", "error": f"payload_value:{key}:expected={value!r}:actual={actual!r}"})
    return failures


def _forbidden_errors(case: dict[str, Any], outputs: list[dict[str, Any]], *, case_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    forbidden = case.get("forbidden") if isinstance(case.get("forbidden"), dict) else {}
    if forbidden.get("runtime_side_effects"):
        for index, out in enumerate(outputs):
            trace = out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}
            allowed = set(trace.get("tool_calls_allowed") or [])
            if allowed & VALID_TOOLS:
                failures.append({"id": case_id, "section": f"turns[{index}].forbidden", "error": f"runtime_side_effect_allowed:{sorted(allowed & VALID_TOOLS)}"})
    if forbidden.get("session_mutation"):
        for index, out in enumerate(outputs):
            trace = out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}
            if trace.get("applied_paths"):
                failures.append({"id": case_id, "section": f"turns[{index}].forbidden", "error": f"session_mutation:{trace.get('applied_paths')}"})
    if forbidden.get("unsupported_capability_as_supported"):
        for index, out in enumerate(outputs):
            trace = out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}
            if trace.get("runtime_payload_ready"):
                failures.append({"id": case_id, "section": f"turns[{index}].forbidden", "error": "unsupported_capability_runtime_ready"})
    return failures


def _sample_runtime_report() -> dict[str, Any]:
    return {
        "ok": True,
        "events_requested": 4,
        "events_completed": 4,
        "completion_fraction": 1.0,
        "configuration": {
            "geometry_structure": "single_box",
            "material": "G4_Cu",
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


def _result_answer_errors(case: dict[str, Any], *, case_id: str, lang: str) -> list[dict[str, Any]]:
    expected = case.get("expected_result_answer") if isinstance(case.get("expected_result_answer"), dict) else {}
    if not expected:
        return []
    question = str(expected.get("question") or "")
    answer = build_runtime_result_question_answer(question, _sample_runtime_report(), lang=lang)
    failures: list[dict[str, Any]] = []
    for expected_text in expected.get("must_include", []) or []:
        if expected_text not in answer:
            failures.append({"id": case_id, "section": "expected_result_answer", "error": f"missing_answer_text:{expected_text}"})
    for forbidden_text in expected.get("must_not_include", []) or []:
        if forbidden_text in answer:
            failures.append({"id": case_id, "section": "expected_result_answer", "error": f"forbidden_answer_text:{forbidden_text}"})
    return failures


def _model_route_decision(case: dict[str, Any], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [out.get("nlu_turn_trace") for out in outputs if isinstance(out.get("nlu_turn_trace"), dict)]
    terminals = {str(trace.get("terminal_state") or "") for trace in traces}
    intents = {str(trace.get("intent") or "") for trace in traces}
    safety_classes = {str(trace.get("action_safety_class") or "") for trace in traces}
    has_guarded_runtime_pending = any(bool(trace.get("guarded_runtime_intent_pending")) for trace in traces)
    capabilities = {str(capability) for capability in case.get("capabilities", []) if isinstance(capability, str)}
    difficulty = str(case.get("difficulty") or "")

    label = "no_llm_required"
    rationale = "read-only or deterministic guarded action does not require model interpretation"

    if "runtime_action_guarded" in terminals or "expensive_runtime" in safety_classes or has_guarded_runtime_pending:
        label = "human_confirmation_required"
        rationale = "expensive_runtime action is guarded and must not be authorized by model routing"
    elif terminals.intersection({"waiting_confirmation", "rejected"}):
        label = "human_confirmation_required"
        rationale = "confirmation_policy requires explicit user confirmation before mutation is applied"
    elif terminals.intersection({"unsupported", "error"}):
        label = "escalate_after_validation_failure"
        rationale = "validation failure requires escalation instead of silent model fallback"
    elif "config_mutation" in intents:
        if not any(trace.get("applied_paths") for trace in traces):
            label = "escalate_after_validation_failure"
            rationale = "validation failure produced no applied configuration paths and requires escalation"
            return {"label": label, "runtime_allowed": False, "rationale": rationale}
        if difficulty in {"expert", "live"} or "llm_reliability" in capabilities:
            label = "strong_model_candidate"
            rationale = "higher difficulty interpretation should be routed to a stronger model candidate"
        else:
            label = "cheap_model_ok"
            rationale = "standard structured configuration can use a cheap model before validation"

    return {"label": label, "runtime_allowed": False, "rationale": rationale}


def _model_route_errors(case: dict[str, Any], outputs: list[dict[str, Any]], *, case_id: str) -> list[dict[str, Any]]:
    expected = case.get("expected_model_route") if isinstance(case.get("expected_model_route"), dict) else {}
    if not expected:
        return []
    actual = _model_route_decision(case, outputs)
    failures: list[dict[str, Any]] = []
    if actual["label"] != expected.get("label"):
        failures.append(
            {
                "id": case_id,
                "section": "expected_model_route",
                "error": f"label:expected={expected.get('label')}:actual={actual['label']}",
            }
        )
    if expected.get("must_not_allow_runtime") is True and actual["runtime_allowed"]:
        failures.append({"id": case_id, "section": "expected_model_route", "error": "runtime_was_allowed"})
    rationale = str(actual.get("rationale") or "")
    for expected_text in expected.get("rationale_contains", []) or []:
        if expected_text not in rationale:
            failures.append(
                {
                    "id": case_id,
                    "section": "expected_model_route",
                    "error": f"missing_rationale_text:{expected_text}",
                }
            )
    return failures


def evaluate_benchmark_dry_run(path: Path = DEFAULT_BENCHMARK_PATH) -> dict[str, Any]:
    shape_report = validate_benchmark_shape(path)
    if shape_report["failed"]:
        return {
            "name": "geant4_agent_benchmark_dry_run",
            "total": 0,
            "failed": 1,
            "failures": [{"id": "<shape>", "section": "shape", "error": "shape_validation_failed"}],
            "shape_report": shape_report,
        }

    cases = _load_json(path)
    failures: list[dict[str, Any]] = []
    passed = 0
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or f"case-{case_index}")
        session_id = f"geant4-agent-benchmark-{case_id}"
        reset_session(session_id)
        outputs: list[dict[str, Any]] = []
        case_failures: list[dict[str, Any]] = []
        try:
            for turn_index, turn in enumerate(case.get("turns") or []):
                if not isinstance(turn, dict):
                    continue
                out = process_turn(
                    {
                        "session_id": session_id,
                        "text": str(turn.get("text") or ""),
                        "llm_router": False,
                        "llm_question": False,
                        "normalize_input": True,
                        "geometry_pipeline": "v2",
                        "source_pipeline": "v2",
                        "enable_compare": False,
                        "autofix": True,
                    },
                    ollama_config_path="",
                    lang=str(turn.get("lang") or case.get("lang") or "en"),
                )
                outputs.append(out)
                trace = out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}
                expected_trace = turn.get("expected_trace") if isinstance(turn.get("expected_trace"), dict) else {}
                case_failures.extend(_trace_errors(expected_trace, trace, case_id=case_id, turn_index=turn_index))

            final_config = outputs[-1].get("config", {}) if outputs else {}
            runtime_payload = build_runtime_payload(final_config)
            expected_runtime = case.get("expected_runtime") if isinstance(case.get("expected_runtime"), dict) else {}
            if expected_runtime:
                case_failures.extend(_runtime_errors(expected_runtime, runtime_payload, case_id=case_id))
            case_failures.extend(_forbidden_errors(case, outputs, case_id=case_id))
            case_failures.extend(_result_answer_errors(case, case_id=case_id, lang=str(case.get("lang") or "en")))
            case_failures.extend(_model_route_errors(case, outputs, case_id=case_id))
        finally:
            reset_session(session_id)

        if case_failures:
            failures.append(
                {
                    "id": case_id,
                    "errors": case_failures,
                    "last_trace": (outputs[-1].get("nlu_turn_trace") if outputs and isinstance(outputs[-1], dict) else {}),
                }
            )
        else:
            passed += 1

    total = len(cases) if isinstance(cases, list) else 0
    return {
        "name": "geant4_agent_benchmark_dry_run",
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "shape_report": shape_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or dry-run Geant4Agent benchmark v1.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--coverage", action="store_true", help="Check minimum V1 suite/difficulty/capability coverage.")
    parser.add_argument("--dry-run", action="store_true", help="Execute deterministic process_turn/runtime-payload grading.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        report = evaluate_benchmark_dry_run(args.benchmark)
    elif args.coverage:
        report = validate_benchmark_coverage(args.benchmark)
    else:
        report = validate_benchmark_shape(args.benchmark)
    output = {"ok": report["failed"] == 0, "report": report}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"{report['name']}: {report['total'] - report['failed']} / {report['total']} passed")
        for failure in report["failures"]:
            print(f"  FAIL {failure}")
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
