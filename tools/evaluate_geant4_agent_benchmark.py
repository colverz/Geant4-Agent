from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.orchestrator.session_manager import process_turn, reset_session
from mcp.geant4.runtime_payload import build_runtime_payload
from planner.runtime_result import build_runtime_result_question_answer


DEFAULT_BENCHMARK_PATH = Path("docs/eval/agentic_benchmark_v1.json")

VALID_SUITES = {
    "core",
    "trajectory",
    "grounding",
    "tool_guard",
    "runtime",
    "result_qa",
    "quantitative_runtime",
    "live_llm",
    "routing",
}
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
    "quantitative_result",
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
    "expected_config_delta",
    "expected_runtime",
    "expected_result_answer",
    "expected_quantitative_result",
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
RUNTIME_KEYS = {"after_turn_index", "must_have_runtime_payload", "required_payload_keys", "expected_payload_values"}
CONFIG_DELTA_KEYS = {"must_apply_paths", "must_not_apply_paths", "expected_final_values", "forbidden_final_values"}
RESULT_ANSWER_KEYS = {"question", "sample_report", "must_include", "must_not_include", "must_remain_read_only"}
QUANTITATIVE_RESULT_KEYS = {
    "sample_report",
    "required_metric_keys",
    "expected_metric_values",
    "non_negative_metric_keys",
    "expected_relations",
}
QUANTITATIVE_RELATION_KEYS = {"left", "op", "right", "numerator", "denominator"}
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

        if "expected_config_delta" in item:
            expected_config_delta = item["expected_config_delta"]
            if not isinstance(expected_config_delta, dict):
                failures.append({"id": case_id, "section": "expected_config_delta", "error": "not_object"})
            else:
                _validate_config_delta(failures, case_id=case_id, expected=expected_config_delta)

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

        if "expected_quantitative_result" in item:
            expected_quantitative_result = item["expected_quantitative_result"]
            if not isinstance(expected_quantitative_result, dict):
                failures.append({"id": case_id, "section": "expected_quantitative_result", "error": "not_object"})
            else:
                _validate_quantitative_result(failures, case_id=case_id, expected=expected_quantitative_result)

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
    if "after_turn_index" in runtime and (not isinstance(runtime["after_turn_index"], int) or runtime["after_turn_index"] < 0):
        failures.append({"id": case_id, "section": "expected_runtime", "error": "after_turn_index_not_non_negative_int"})
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


def _validate_config_delta(failures: list[dict[str, Any]], *, case_id: str, expected: dict[str, Any]) -> None:
    _add_unknown_key_errors(failures, case_id=case_id, section="expected_config_delta", payload=expected, allowed=CONFIG_DELTA_KEYS)
    for field in ("must_apply_paths", "must_not_apply_paths"):
        if field in expected:
            _validate_string_list(
                failures,
                case_id=case_id,
                section="expected_config_delta",
                field=field,
                value=expected[field],
            )
    for field in ("expected_final_values", "forbidden_final_values"):
        if field in expected and not isinstance(expected[field], dict):
            failures.append({"id": case_id, "section": "expected_config_delta", "error": f"{field}_not_object"})


def _validate_result_answer(failures: list[dict[str, Any]], *, case_id: str, expected: dict[str, Any]) -> None:
    _add_unknown_key_errors(failures, case_id=case_id, section="expected_result_answer", payload=expected, allowed=RESULT_ANSWER_KEYS)
    if "question" in expected and not isinstance(expected["question"], str):
        failures.append({"id": case_id, "section": "expected_result_answer", "error": "question_not_string"})
    if "sample_report" in expected and expected["sample_report"] not in {"default", "none"}:
        failures.append({"id": case_id, "section": "expected_result_answer", "error": f"invalid_sample_report:{expected['sample_report']}"})
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


def _validate_quantitative_result(failures: list[dict[str, Any]], *, case_id: str, expected: dict[str, Any]) -> None:
    _add_unknown_key_errors(
        failures,
        case_id=case_id,
        section="expected_quantitative_result",
        payload=expected,
        allowed=QUANTITATIVE_RESULT_KEYS,
    )
    if "sample_report" in expected and expected["sample_report"] not in {"default"}:
        failures.append(
            {
                "id": case_id,
                "section": "expected_quantitative_result",
                "error": f"invalid_sample_report:{expected['sample_report']}",
            }
        )
    for field in ("required_metric_keys", "non_negative_metric_keys"):
        if field in expected:
            _validate_string_list(
                failures,
                case_id=case_id,
                section="expected_quantitative_result",
                field=field,
                value=expected[field],
            )
    if "expected_metric_values" in expected and not isinstance(expected["expected_metric_values"], dict):
        failures.append(
            {
                "id": case_id,
                "section": "expected_quantitative_result",
                "error": "expected_metric_values_not_object",
            }
        )
    relations = expected.get("expected_relations")
    if relations is None:
        return
    if not isinstance(relations, list):
        failures.append({"id": case_id, "section": "expected_quantitative_result", "error": "expected_relations_not_list"})
        return
    for index, relation in enumerate(relations):
        section = f"expected_quantitative_result.expected_relations[{index}]"
        if not isinstance(relation, dict):
            failures.append({"id": case_id, "section": section, "error": "not_object"})
            continue
        _add_unknown_key_errors(failures, case_id=case_id, section=section, payload=relation, allowed=QUANTITATIVE_RELATION_KEYS)
        op = relation.get("op")
        if op not in {"equals", "equals_division"}:
            failures.append({"id": case_id, "section": section, "error": f"invalid_op:{op}"})
        for key in ("left", "right", "numerator", "denominator"):
            if key in relation and not isinstance(relation[key], str):
                failures.append({"id": case_id, "section": section, "error": f"{key}_not_string"})


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


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


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


def _config_delta_errors(
    expected: dict[str, Any],
    final_config: dict[str, Any],
    outputs: list[dict[str, Any]],
    *,
    case_id: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    applied_paths = {
        str(path)
        for out in outputs
        for path in ((out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}).get("applied_paths") or [])
    }
    for path in expected.get("must_apply_paths", []) or []:
        if path not in applied_paths:
            failures.append({"id": case_id, "section": "expected_config_delta", "error": f"missing_applied_path:{path}"})
    for path in expected.get("must_not_apply_paths", []) or []:
        if path in applied_paths:
            failures.append({"id": case_id, "section": "expected_config_delta", "error": f"forbidden_applied_path:{path}"})
    expected_values = expected.get("expected_final_values") or {}
    if isinstance(expected_values, dict):
        for path, value in expected_values.items():
            actual = _get_path(final_config, str(path))
            if not _values_equal(actual, value):
                failures.append({"id": case_id, "section": "expected_config_delta", "error": f"final_value:{path}:expected={value!r}:actual={actual!r}"})
    forbidden_values = expected.get("forbidden_final_values") or {}
    if isinstance(forbidden_values, dict):
        for path, value in forbidden_values.items():
            actual = _get_path(final_config, str(path))
            if _values_equal(actual, value):
                failures.append({"id": case_id, "section": "expected_config_delta", "error": f"forbidden_final_value:{path}:actual={actual!r}"})
    return failures


def _new_config_delta_summary() -> dict[str, Any]:
    return {
        "cases": 0,
        "must_apply_paths_total": 0,
        "must_apply_paths_passed": 0,
        "must_not_apply_paths_total": 0,
        "must_not_apply_paths_passed": 0,
        "expected_final_values_total": 0,
        "expected_final_values_passed": 0,
        "forbidden_final_values_total": 0,
        "forbidden_final_values_passed": 0,
    }


def _new_quantitative_result_summary() -> dict[str, Any]:
    return {
        "cases": 0,
        "required_metric_keys_total": 0,
        "required_metric_keys_passed": 0,
        "expected_metric_values_total": 0,
        "expected_metric_values_passed": 0,
        "non_negative_metric_keys_total": 0,
        "non_negative_metric_keys_passed": 0,
        "expected_relations_total": 0,
        "expected_relations_passed": 0,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _update_config_delta_summary(
    summary: dict[str, Any],
    expected: dict[str, Any],
    final_config: dict[str, Any],
    outputs: list[dict[str, Any]],
) -> None:
    if not expected:
        return
    summary["cases"] += 1
    applied_paths = {
        str(path)
        for out in outputs
        for path in ((out.get("nlu_turn_trace") if isinstance(out.get("nlu_turn_trace"), dict) else {}).get("applied_paths") or [])
    }
    for path in expected.get("must_apply_paths", []) or []:
        summary["must_apply_paths_total"] += 1
        if path in applied_paths:
            summary["must_apply_paths_passed"] += 1
    for path in expected.get("must_not_apply_paths", []) or []:
        summary["must_not_apply_paths_total"] += 1
        if path not in applied_paths:
            summary["must_not_apply_paths_passed"] += 1
    expected_values = expected.get("expected_final_values") or {}
    if isinstance(expected_values, dict):
        for path, value in expected_values.items():
            summary["expected_final_values_total"] += 1
            if _values_equal(_get_path(final_config, str(path)), value):
                summary["expected_final_values_passed"] += 1
    forbidden_values = expected.get("forbidden_final_values") or {}
    if isinstance(forbidden_values, dict):
        for path, value in forbidden_values.items():
            summary["forbidden_final_values_total"] += 1
            if not _values_equal(_get_path(final_config, str(path)), value):
                summary["forbidden_final_values_passed"] += 1


def _finalize_config_delta_summary(summary: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(summary)
    finalized["must_apply_path_recall"] = _ratio(summary["must_apply_paths_passed"], summary["must_apply_paths_total"])
    finalized["must_not_apply_path_guard_rate"] = _ratio(summary["must_not_apply_paths_passed"], summary["must_not_apply_paths_total"])
    finalized["expected_final_value_accuracy"] = _ratio(
        summary["expected_final_values_passed"],
        summary["expected_final_values_total"],
    )
    finalized["forbidden_final_value_guard_rate"] = _ratio(
        summary["forbidden_final_values_passed"],
        summary["forbidden_final_values_total"],
    )
    return finalized


def _relation_passes(relation: dict[str, Any], report: dict[str, Any]) -> bool:
    op = relation.get("op")
    left = _get_path(report, str(relation.get("left") or ""))
    if op == "equals":
        right = _get_path(report, str(relation.get("right") or ""))
        return _values_equal(left, right)
    if op == "equals_division":
        numerator = _get_path(report, str(relation.get("numerator") or ""))
        denominator = _get_path(report, str(relation.get("denominator") or ""))
        try:
            expected = float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        return _float_equal(left, expected)
    return False


def _quantitative_result_errors(expected: dict[str, Any], report: dict[str, Any], *, case_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for path in expected.get("required_metric_keys", []) or []:
        if _get_path(report, str(path)) is None:
            failures.append({"id": case_id, "section": "expected_quantitative_result", "error": f"missing_metric:{path}"})
    expected_values = expected.get("expected_metric_values") or {}
    if isinstance(expected_values, dict):
        for path, value in expected_values.items():
            actual = _get_path(report, str(path))
            if not _values_equal(actual, value):
                failures.append(
                    {
                        "id": case_id,
                        "section": "expected_quantitative_result",
                        "error": f"metric_value:{path}:expected={value!r}:actual={actual!r}",
                    }
                )
    for path in expected.get("non_negative_metric_keys", []) or []:
        actual = _get_path(report, str(path))
        try:
            if float(actual) < 0:
                failures.append({"id": case_id, "section": "expected_quantitative_result", "error": f"negative_metric:{path}:{actual!r}"})
        except (TypeError, ValueError):
            failures.append({"id": case_id, "section": "expected_quantitative_result", "error": f"metric_not_numeric:{path}:{actual!r}"})
    for index, relation in enumerate(expected.get("expected_relations", []) or []):
        if not isinstance(relation, dict):
            continue
        if not _relation_passes(relation, report):
            failures.append(
                {
                    "id": case_id,
                    "section": "expected_quantitative_result",
                    "error": f"relation_failed:{index}:{relation!r}",
                }
            )
    return failures


def _update_quantitative_result_summary(summary: dict[str, Any], expected: dict[str, Any], report: dict[str, Any]) -> None:
    if not expected:
        return
    summary["cases"] += 1
    for path in expected.get("required_metric_keys", []) or []:
        summary["required_metric_keys_total"] += 1
        if _get_path(report, str(path)) is not None:
            summary["required_metric_keys_passed"] += 1
    expected_values = expected.get("expected_metric_values") or {}
    if isinstance(expected_values, dict):
        for path, value in expected_values.items():
            summary["expected_metric_values_total"] += 1
            if _values_equal(_get_path(report, str(path)), value):
                summary["expected_metric_values_passed"] += 1
    for path in expected.get("non_negative_metric_keys", []) or []:
        summary["non_negative_metric_keys_total"] += 1
        try:
            if float(_get_path(report, str(path))) >= 0:
                summary["non_negative_metric_keys_passed"] += 1
        except (TypeError, ValueError):
            pass
    for relation in expected.get("expected_relations", []) or []:
        if not isinstance(relation, dict):
            continue
        summary["expected_relations_total"] += 1
        if _relation_passes(relation, report):
            summary["expected_relations_passed"] += 1


def _finalize_quantitative_result_summary(summary: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(summary)
    finalized["required_metric_key_rate"] = _ratio(summary["required_metric_keys_passed"], summary["required_metric_keys_total"])
    finalized["expected_metric_value_accuracy"] = _ratio(summary["expected_metric_values_passed"], summary["expected_metric_values_total"])
    finalized["non_negative_metric_rate"] = _ratio(summary["non_negative_metric_keys_passed"], summary["non_negative_metric_keys_total"])
    finalized["relation_pass_rate"] = _ratio(summary["expected_relations_passed"], summary["expected_relations_total"])
    return finalized


def _record_bucket(summary: dict[str, dict[str, int]], key: str, *, passed: bool) -> None:
    if not key:
        return
    bucket = summary.setdefault(key, {"total": 0, "passed": 0, "failed": 0})
    bucket["total"] += 1
    if passed:
        bucket["passed"] += 1
    else:
        bucket["failed"] += 1


def _finalize_bucket_summary(summary: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, bucket in sorted(summary.items()):
        total = int(bucket.get("total", 0))
        passed = int(bucket.get("passed", 0))
        failed = int(bucket.get("failed", 0))
        finalized[key] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": _ratio(passed, total),
        }
    return finalized


def _new_model_route_summary() -> dict[str, Any]:
    return {
        "cases": 0,
        "label_counts": {},
        "runtime_allowed_count": 0,
        "expected_label_counts": {},
    }


def _update_model_route_summary(summary: dict[str, Any], case: dict[str, Any], outputs: list[dict[str, Any]]) -> None:
    capabilities = {str(capability) for capability in case.get("capabilities", []) if isinstance(capability, str)}
    if "model_routing" not in capabilities:
        return
    summary["cases"] += 1
    actual = _model_route_decision(case, outputs)
    label = str(actual.get("label") or "")
    if label:
        label_counts = summary.setdefault("label_counts", {})
        label_counts[label] = int(label_counts.get(label, 0)) + 1
    if actual.get("runtime_allowed"):
        summary["runtime_allowed_count"] = int(summary.get("runtime_allowed_count", 0)) + 1
    expected = case.get("expected_model_route") if isinstance(case.get("expected_model_route"), dict) else {}
    expected_label = str(expected.get("label") or "")
    if expected_label:
        expected_counts = summary.setdefault("expected_label_counts", {})
        expected_counts[expected_label] = int(expected_counts.get(expected_label, 0)) + 1


def _finalize_model_route_summary(summary: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(summary)
    finalized["label_counts"] = dict(sorted((summary.get("label_counts") or {}).items()))
    finalized["expected_label_counts"] = dict(sorted((summary.get("expected_label_counts") or {}).items()))
    return finalized


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
            "target_edep_mean_mev_per_event": 0.375,
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
            },
            "scoring": {
                "target": {
                    "target_edep_total_mev": 1.5,
                    "target_edep_mean_mev_per_event": 0.375,
                }
            },
        },
    }


def _result_answer_errors(case: dict[str, Any], *, case_id: str, lang: str) -> list[dict[str, Any]]:
    expected = case.get("expected_result_answer") if isinstance(case.get("expected_result_answer"), dict) else {}
    if not expected:
        return []
    question = str(expected.get("question") or "")
    sample_report = None if expected.get("sample_report") == "none" else _sample_runtime_report()
    answer = build_runtime_result_question_answer(question, sample_report, lang=lang)
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
    config_delta_summary = _new_config_delta_summary()
    quantitative_result_summary = _new_quantitative_result_summary()
    suite_summary: dict[str, dict[str, int]] = {}
    difficulty_summary: dict[str, dict[str, int]] = {}
    capability_summary: dict[str, dict[str, int]] = {}
    model_route_summary = _new_model_route_summary()
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
            expected_config_delta = case.get("expected_config_delta") if isinstance(case.get("expected_config_delta"), dict) else {}
            if expected_config_delta:
                _update_config_delta_summary(config_delta_summary, expected_config_delta, final_config, outputs)
                case_failures.extend(_config_delta_errors(expected_config_delta, final_config, outputs, case_id=case_id))
            expected_runtime = case.get("expected_runtime") if isinstance(case.get("expected_runtime"), dict) else {}
            if expected_runtime:
                runtime_output_index = int(expected_runtime.get("after_turn_index", len(outputs) - 1))
                runtime_config = outputs[runtime_output_index].get("config", {}) if 0 <= runtime_output_index < len(outputs) else {}
                runtime_payload = build_runtime_payload(runtime_config)
                case_failures.extend(_runtime_errors(expected_runtime, runtime_payload, case_id=case_id))
            case_failures.extend(_forbidden_errors(case, outputs, case_id=case_id))
            case_failures.extend(_result_answer_errors(case, case_id=case_id, lang=str(case.get("lang") or "en")))
            expected_quantitative_result = (
                case.get("expected_quantitative_result")
                if isinstance(case.get("expected_quantitative_result"), dict)
                else {}
            )
            if expected_quantitative_result:
                sample_report = _sample_runtime_report()
                _update_quantitative_result_summary(quantitative_result_summary, expected_quantitative_result, sample_report)
                case_failures.extend(
                    _quantitative_result_errors(
                        expected_quantitative_result,
                        sample_report,
                        case_id=case_id,
                    )
                )
            case_failures.extend(_model_route_errors(case, outputs, case_id=case_id))
            _update_model_route_summary(model_route_summary, case, outputs)
        finally:
            reset_session(session_id)

        case_passed = not case_failures
        _record_bucket(suite_summary, str(case.get("suite") or ""), passed=case_passed)
        _record_bucket(difficulty_summary, str(case.get("difficulty") or ""), passed=case_passed)
        for capability in case.get("capabilities") or []:
            if isinstance(capability, str):
                _record_bucket(capability_summary, capability, passed=case_passed)

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
        "config_delta_summary": _finalize_config_delta_summary(config_delta_summary),
        "quantitative_result_summary": _finalize_quantitative_result_summary(quantitative_result_summary),
        "suite_summary": _finalize_bucket_summary(suite_summary),
        "difficulty_summary": _finalize_bucket_summary(difficulty_summary),
        "capability_summary": _finalize_bucket_summary(capability_summary),
        "model_route_summary": _finalize_model_route_summary(model_route_summary),
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
