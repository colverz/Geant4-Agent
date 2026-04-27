from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from planner.runtime_intent import action_for_runtime_intent, classify_user_runtime_intent
from planner.runtime_result import build_runtime_result_question_answer
from core.orchestrator.path_ops import set_path
from core.orchestrator.session_manager import get_or_create_session, reset_session
from ui.web.request_router import handle_post_request


DEFAULT_WORKFLOW_CASEBANK = Path("docs/eval/workflow_guard_casebank.json")
DEFAULT_RUNTIME_QA_CASEBANK = Path("docs/eval/runtime_result_qa_casebank.json")
DEFAULT_MULTITURN_CASEBANK = Path("docs/eval/multiturn_guard_casebank.json")
DEFAULT_SESSION_BEHAVIOR_CASEBANK = Path("docs/eval/session_behavior_casebank.json")

VALID_INTENTS = {
    "read_config",
    "read_summary",
    "config_mutation",
    "run_requested",
    "viewer_requested",
    "normal_chat",
}
VALID_SAFETY = {"read_only", "config_mutation", "expensive_runtime"}
VALID_ACTIONS = {"config_summary", "runtime_summary", "step_async", "guarded_runtime_ui", "read_only_chat"}
VALID_LANGS = {"zh", "en"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_keys(item: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - set(item.keys()))


def _duplicate_ids(items: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        item_id = str(item.get("id", ""))
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    return sorted(duplicates)


def validate_casebank_shapes(
    *,
    workflow_path: Path = DEFAULT_WORKFLOW_CASEBANK,
    runtime_qa_path: Path = DEFAULT_RUNTIME_QA_CASEBANK,
    multiturn_path: Path = DEFAULT_MULTITURN_CASEBANK,
    session_behavior_path: Path = DEFAULT_SESSION_BEHAVIOR_CASEBANK,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    workflow = _load_json(workflow_path)
    runtime_qa = _load_json(runtime_qa_path)
    multiturn = _load_json(multiturn_path)
    session_behavior = _load_json(session_behavior_path)

    def add_failure(casebank: str, item_id: str, error: str) -> None:
        failures.append({"casebank": casebank, "id": item_id, "error": error})

    for casebank, items, required in (
        ("workflow_guard", workflow, {"id", "text", "lang", "expected_intent", "expected_safety"}),
        ("runtime_result_qa", runtime_qa, {"id", "question", "lang", "expected_substrings", "forbidden_substrings"}),
        (
            "session_behavior_guard",
            session_behavior,
            {
                "id",
                "text",
                "lang",
                "expected_intent",
                "expected_action",
                "expect_session_turn_delta",
                "expect_step_job",
                "expect_runtime_run",
                "expect_config_mutation_allowed",
            },
        ),
    ):
        if not isinstance(items, list):
            add_failure(casebank, "<root>", "not_list")
            continue
        for duplicate in _duplicate_ids(items):
            add_failure(casebank, duplicate, "duplicate_id")
        for item in items:
            if not isinstance(item, dict):
                add_failure(casebank, "<item>", "not_object")
                continue
            item_id = str(item.get("id", "<missing_id>"))
            for missing in _missing_keys(item, required):
                add_failure(casebank, item_id, f"missing_key:{missing}")
            if "lang" in item and item["lang"] not in VALID_LANGS:
                add_failure(casebank, item_id, f"invalid_lang:{item['lang']}")
            if "expected_intent" in item and item["expected_intent"] not in VALID_INTENTS:
                add_failure(casebank, item_id, f"invalid_intent:{item['expected_intent']}")
            if "expected_safety" in item and item["expected_safety"] not in VALID_SAFETY:
                add_failure(casebank, item_id, f"invalid_safety:{item['expected_safety']}")
            if "expected_action" in item and item["expected_action"] not in VALID_ACTIONS:
                add_failure(casebank, item_id, f"invalid_action:{item['expected_action']}")
            if casebank == "runtime_result_qa":
                if not isinstance(item.get("expected_substrings"), list):
                    add_failure(casebank, item_id, "expected_substrings_not_list")
                if not isinstance(item.get("forbidden_substrings"), list):
                    add_failure(casebank, item_id, "forbidden_substrings_not_list")

    if not isinstance(multiturn, list):
        add_failure("multiturn_guard", "<root>", "not_list")
    else:
        for duplicate in _duplicate_ids(multiturn):
            add_failure("multiturn_guard", duplicate, "duplicate_id")
        for flow in multiturn:
            if not isinstance(flow, dict):
                add_failure("multiturn_guard", "<flow>", "not_object")
                continue
            flow_id = str(flow.get("id", "<missing_id>"))
            for missing in _missing_keys(flow, {"id", "lang", "turns"}):
                add_failure("multiturn_guard", flow_id, f"missing_key:{missing}")
            if "lang" in flow and flow["lang"] not in VALID_LANGS:
                add_failure("multiturn_guard", flow_id, f"invalid_lang:{flow['lang']}")
            turns = flow.get("turns")
            if not isinstance(turns, list):
                add_failure("multiturn_guard", flow_id, "turns_not_list")
                continue
            for index, turn in enumerate(turns):
                turn_id = f"{flow_id}[{index}]"
                if not isinstance(turn, dict):
                    add_failure("multiturn_guard", turn_id, "turn_not_object")
                    continue
                for missing in _missing_keys(turn, {"text", "expected_intent", "expected_safety", "expected_action"}):
                    add_failure("multiturn_guard", turn_id, f"missing_key:{missing}")
                if "lang" in turn and turn["lang"] not in VALID_LANGS:
                    add_failure("multiturn_guard", turn_id, f"invalid_lang:{turn['lang']}")
                if "expected_intent" in turn and turn["expected_intent"] not in VALID_INTENTS:
                    add_failure("multiturn_guard", turn_id, f"invalid_intent:{turn['expected_intent']}")
                if "expected_safety" in turn and turn["expected_safety"] not in VALID_SAFETY:
                    add_failure("multiturn_guard", turn_id, f"invalid_safety:{turn['expected_safety']}")
                if "expected_action" in turn and turn["expected_action"] not in VALID_ACTIONS:
                    add_failure("multiturn_guard", turn_id, f"invalid_action:{turn['expected_action']}")

    total = (
        (len(workflow) if isinstance(workflow, list) else 0)
        + (len(runtime_qa) if isinstance(runtime_qa, list) else 0)
        + (len(session_behavior) if isinstance(session_behavior, list) else 0)
        + sum(len(flow.get("turns", [])) for flow in multiturn if isinstance(flow, dict) and isinstance(flow.get("turns"), list))
    )
    return {"name": "casebank_schema", "total": total, "failed": len(failures), "failures": failures}


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


def evaluate_multiturn_guard(path: Path) -> dict[str, Any]:
    flows = _load_json(path)
    failures: list[dict[str, Any]] = []
    total = 0
    for flow in flows:
        lang = str(flow.get("lang", "en"))
        for index, turn in enumerate(flow.get("turns", [])):
            total += 1
            result = classify_user_runtime_intent(turn["text"], turn.get("lang", lang))
            action = action_for_runtime_intent(result.intent).value
            errors: dict[str, Any] = {}
            if result.intent.value != turn["expected_intent"]:
                errors["intent"] = {"expected": turn["expected_intent"], "actual": result.intent.value}
            if result.action_safety_class.value != turn["expected_safety"]:
                errors["safety"] = {
                    "expected": turn["expected_safety"],
                    "actual": result.action_safety_class.value,
                }
            if action != turn["expected_action"]:
                errors["action"] = {"expected": turn["expected_action"], "actual": action}
            if errors:
                failures.append({"id": flow["id"], "turn_index": index, "text": turn["text"], "errors": errors})
    return {"name": "multiturn_guard", "total": total, "failed": len(failures), "failures": failures}


def _seed_guard_session(session_id: str) -> int:
    reset_session(session_id)
    state = get_or_create_session(session_id)
    set_path(state.config, "geometry.structure", "single_box")
    set_path(state.config, "source.type", "point")
    set_path(state.config, "source.particle", "gamma")
    return int(state.turn_id)


def _fail_step_fn(_payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    raise AssertionError("read-only guard path must not call step_fn")


def _ok_step_fn(payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    return {"ok": True, "session_id": payload.get("session_id"), "guard_test": True}


def evaluate_session_behavior_guard(path: Path) -> dict[str, Any]:
    cases = _load_json(path)
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        errors: dict[str, Any] = {}
        session_id = f"casebank-session-behavior-{index}"
        turn_before = _seed_guard_session(session_id)
        result = classify_user_runtime_intent(case["text"], case["lang"])
        action = action_for_runtime_intent(result.intent).value

        if result.intent.value != case["expected_intent"]:
            errors["intent"] = {"expected": case["expected_intent"], "actual": result.intent.value}
        if action != case["expected_action"]:
            errors["action"] = {"expected": case["expected_action"], "actual": action}

        step_job_created = False
        runtime_run_called = False
        status = 200
        body: dict[str, Any] = {}

        try:
            if action == "config_summary":
                status, body = handle_post_request(
                    "/api/config/summary",
                    {"session_id": session_id, "lang": case["lang"]},
                    legacy_sessions={},
                    solve_fn=lambda _payload: {"unexpected": "solve"},
                    step_fn=_fail_step_fn,
                )
            elif action == "runtime_summary":
                status, body = handle_post_request(
                    "/api/geant4/summary",
                    {"lang": case["lang"], "question": case["text"]},
                    legacy_sessions={},
                    solve_fn=lambda _payload: {"unexpected": "solve"},
                    step_fn=_fail_step_fn,
                )
            elif action == "step_async":
                status, body = handle_post_request(
                    "/api/step_async",
                    {"session_id": session_id, "text": case["text"], "lang": case["lang"]},
                    legacy_sessions={},
                    solve_fn=lambda _payload: {"unexpected": "solve"},
                    step_fn=_ok_step_fn,
                )
                step_job_created = bool(body.get("job_id"))
            elif action == "guarded_runtime_ui":
                # Guarded runtime requests are intentionally not dispatched to run/viewer APIs from chat.
                runtime_run_called = False
            elif action == "read_only_chat":
                status, body = 200, {"action_safety_class": "read_only"}
            else:
                errors["unknown_action"] = action
        except Exception as exc:
            errors["exception"] = str(exc)

        turn_after = int(get_or_create_session(session_id).turn_id)
        actual_turn_delta = turn_after - turn_before
        if actual_turn_delta != int(case["expect_session_turn_delta"]):
            errors["session_turn_delta"] = {
                "expected": case["expect_session_turn_delta"],
                "actual": actual_turn_delta,
            }
        if step_job_created != bool(case["expect_step_job"]):
            errors["step_job"] = {"expected": case["expect_step_job"], "actual": step_job_created}
        if runtime_run_called != bool(case["expect_runtime_run"]):
            errors["runtime_run"] = {"expected": case["expect_runtime_run"], "actual": runtime_run_called}
        if (action == "step_async") != bool(case["expect_config_mutation_allowed"]):
            errors["config_mutation_allowed"] = {
                "expected": case["expect_config_mutation_allowed"],
                "actual": action == "step_async",
            }
        if action in {"config_summary", "runtime_summary"} and body.get("action_safety_class") != "read_only":
            errors["action_safety_class"] = {"expected": "read_only", "actual": body.get("action_safety_class")}
        expected_statuses = {400 if action == "runtime_summary" else 200}
        if action == "runtime_summary":
            expected_statuses.add(200)
        if status not in expected_statuses:
            errors["status"] = {"expected": sorted(expected_statuses), "actual": status}

        if errors:
            failures.append({"id": case["id"], "text": case["text"], "errors": errors})
    return {"name": "session_behavior_guard", "total": len(cases), "failed": len(failures), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate lightweight user-safety and grounded-result casebanks.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW_CASEBANK)
    parser.add_argument("--runtime-qa", type=Path, default=DEFAULT_RUNTIME_QA_CASEBANK)
    parser.add_argument("--multiturn", type=Path, default=DEFAULT_MULTITURN_CASEBANK)
    parser.add_argument("--session-behavior", type=Path, default=DEFAULT_SESSION_BEHAVIOR_CASEBANK)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    reports = [
        validate_casebank_shapes(
            workflow_path=args.workflow,
            runtime_qa_path=args.runtime_qa,
            multiturn_path=args.multiturn,
            session_behavior_path=args.session_behavior,
        ),
        evaluate_workflow_guard(args.workflow),
        evaluate_runtime_result_qa(args.runtime_qa),
        evaluate_multiturn_guard(args.multiturn),
        evaluate_session_behavior_guard(args.session_behavior),
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
