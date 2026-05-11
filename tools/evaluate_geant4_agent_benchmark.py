from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
FORBIDDEN_KEYS = {"runtime_side_effects", "session_mutation", "unsupported_capability_as_supported"}
REQUIRED_TOP_LEVEL_KEYS = {"id", "suite", "difficulty", "lang", "turns"}


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
        elif len(turns) > 1:
            failures.append({"id": case_id, "section": "turns", "error": "multi_turn_not_supported_in_v1_shape"})
        elif isinstance(turns[0], dict):
            turn = turns[0]
            _add_unknown_key_errors(failures, case_id=case_id, section="turns[0]", payload=turn, allowed=TURN_KEYS)
            if not str(turn.get("text") or "").strip():
                failures.append({"id": case_id, "section": "turns[0]", "error": "missing_text"})
            if "lang" in turn and turn["lang"] not in VALID_LANGS:
                failures.append({"id": case_id, "section": "turns[0]", "error": f"invalid_lang:{turn['lang']}"})
            expected_trace = turn.get("expected_trace")
            if expected_trace is not None:
                if not isinstance(expected_trace, dict):
                    failures.append({"id": case_id, "section": "turns[0].expected_trace", "error": "not_object"})
                else:
                    _validate_trace(failures, case_id=case_id, trace=expected_trace)
        elif turns:
            failures.append({"id": case_id, "section": "turns[0]", "error": "not_object"})

        if "expected_runtime" in item:
            expected_runtime = item["expected_runtime"]
            if not isinstance(expected_runtime, dict):
                failures.append({"id": case_id, "section": "expected_runtime", "error": "not_object"})
            else:
                _validate_runtime(failures, case_id=case_id, runtime=expected_runtime)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Geant4Agent benchmark v1 shape.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

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
