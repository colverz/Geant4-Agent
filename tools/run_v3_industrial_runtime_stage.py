from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from core.agent_v3.service import V3AgentTurnService
from core.agent_v3.tools import GEANT4_LLM_DESIGN_TOOL, GEANT4_PAYLOAD_BUILDER_TOOL, GEANT4_RUNTIME_TOOL
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_industrial_runtime_benchmark import (
    DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    INDUSTRIAL_RUNTIME_ENV,
    RUNTIME_COMMAND_ENVS,
    _golden_metrics,
    _golden_status,
    validate_industrial_benchmark_shape,
)
from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime
from tools.industrial_runtime_contract import (
    V3IndustrialCandidateRequirements,
    compare_candidate_runtime_contract,
    compare_v3_candidate_runtime_contract,
)
from tools.industrial_runtime_executor import compare_industrial_metrics, extract_industrial_metrics


V3_INDUSTRIAL_RUNTIME_STAGE_SCHEMA_VERSION = "geant4_agent_v3_industrial_runtime_stage.v1"
_SAFE_RUNTIME_ENV_KEYS = {
    INDUSTRIAL_RUNTIME_ENV,
    "GEANT4_RUNTIME_COMMAND_JSON",
    "GEANT4_RUNTIME_COMMAND",
    "GEANT4_ROOT",
    "GEANT4_DATA_DIR",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case.get("id")): case for case in benchmark.get("cases", []) if isinstance(case, dict)}


def _selected_case_ids(benchmark: dict[str, Any], requested: list[str]) -> list[str]:
    cases = _case_map(benchmark)
    if requested:
        return [case_id for case_id in requested if case_id in cases]
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    return [
        case_id
        for case_id, case in cases.items()
        if compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults).get("status") == "compiled"
    ]


def _runtime_gate(env: dict[str, str]) -> dict[str, Any]:
    enabled = str(env.get(INDUSTRIAL_RUNTIME_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}
    command_configured = any(str(env.get(name, "")).strip() for name in RUNTIME_COMMAND_ENVS)
    return {
        "env_enabled": enabled,
        "runtime_command_configured": command_configured,
        "real_runtime_ready": enabled and command_configured,
    }


def _runtime_policy(env: dict[str, str]) -> dict[str, Any]:
    return {
        "allow_in_memory": False,
        "backend_preference": "local_process",
        "source": "v3_industrial_runtime_stage",
        "env": {key: str(env[key]) for key in _SAFE_RUNTIME_ENV_KEYS if str(env.get(key, "")).strip()},
    }


def _latest_observation(response: dict[str, Any], source: str) -> dict[str, Any] | None:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    for observation in reversed(observations):
        if isinstance(observation, dict) and observation.get("source") == source:
            return observation
    return None


def _observation_data(observation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    data = observation.get("data")
    return data if isinstance(data, dict) else {}


def _turn_record(label: str, response: dict[str, Any]) -> dict[str, Any]:
    metadata = ((response.get("state") or {}).get("metadata") or {}) if isinstance(response.get("state"), dict) else {}
    understanding = metadata.get("turn_understanding") if isinstance(metadata.get("turn_understanding"), dict) else {}
    pending = response.get("pending_action") if isinstance(response.get("pending_action"), dict) else None
    return {
        "label": label,
        "ok": bool(response.get("ok")),
        "terminated_reason": response.get("terminated_reason"),
        "understanding_source": understanding.get("source"),
        "dialogue_act": response.get("dialogue_act"),
        "observation_sources": [
            item.get("source") for item in response.get("observations", []) if isinstance(item, dict)
        ],
        "pending_action_id": pending.get("action_id") if pending else None,
    }


def _base_request(
    *,
    session_id: str,
    text: str,
    events: int,
    runtime_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "text": text,
        "locale": "en",
        "events": events,
        "allow_in_memory": False,
        "runtime_policy": runtime_policy,
    }


def _not_evaluable(case: dict[str, Any], category: str, reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": "not_evaluable",
        "failure_category": category,
        "reasons": list(dict.fromkeys(reasons)),
        **extra,
    }


def _failed(case: dict[str, Any], category: str, reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": "failed",
        "failure_category": category,
        "reasons": list(dict.fromkeys(reasons)),
        **extra,
    }


def _run_v3_case(
    service: V3AgentTurnService,
    case: dict[str, Any],
    compiled: dict[str, Any],
    *,
    llm_config_path: str,
    runtime_policy: dict[str, Any],
    golden_dir: Path,
    allow_unreviewed_goldens: bool,
) -> dict[str, Any]:
    if compiled.get("run_mode") == "paired_run":
        return _not_evaluable(
            case,
            "v3_paired_runtime_not_supported",
            ["paired_case_requires_first_class_v3_multi_run_workflow"],
        )

    expected_payload = compiled.get("runtime_payload") if isinstance(compiled.get("runtime_payload"), dict) else {}
    events = int((expected_payload.get("run") or {}).get("events") or 10000)
    session_id = f"v3-industrial-{case.get('id')}"
    trajectory: list[dict[str, Any]] = []
    dialogue = "\n".join(str(item) for item in case.get("raw_dialogue") or [] if str(item).strip())
    request = _base_request(
        session_id=session_id,
        text=dialogue,
        events=events,
        runtime_policy=runtime_policy,
    )
    request.update(
        {
            "accept_defaults": True,
            "llm_understanding_enabled": False,
            "llm_planning_enabled": False,
            "llm_design_enabled": True,
            "llm_config_path": llm_config_path,
        }
    )
    design_response = service.run_turn(request)
    trajectory.append(_turn_record("candidate", design_response))
    design_observation = _latest_observation(design_response, GEANT4_LLM_DESIGN_TOOL)
    llm_report = _observation_data(design_observation).get("llm")
    llm_report = llm_report if isinstance(llm_report, dict) else {}
    if not llm_report.get("used") or not llm_report.get("ok"):
        return _not_evaluable(
            case,
            "llm_unavailable",
            [str(llm_report.get("fallback_reason") or "v3_llm_design_not_used")],
            trajectory=trajectory,
            llm_report={
                "used": bool(llm_report.get("used")),
                "ok": bool(llm_report.get("ok")),
                "fallback_reason": llm_report.get("fallback_reason"),
                "prompt_profile_id": llm_report.get("prompt_profile_id"),
            },
            runtime_attempted=False,
        )
    payload_observation = _latest_observation(design_response, GEANT4_PAYLOAD_BUILDER_TOOL)

    if payload_observation is None:
        payload_request = _base_request(
            session_id=session_id,
            text="Accept the current design and build the runtime payload without running it.",
            events=events,
            runtime_policy=runtime_policy,
        )
        payload_request["accept_defaults"] = True
        payload_response = service.run_turn(payload_request)
        trajectory.append(_turn_record("build_payload", payload_response))
        payload_observation = _latest_observation(payload_response, GEANT4_PAYLOAD_BUILDER_TOOL)

    payload_data = _observation_data(payload_observation)
    candidate_payload = payload_data.get("runtime_payload") if isinstance(payload_data.get("runtime_payload"), dict) else {}
    if not candidate_payload:
        return _failed(
            case,
            "v3_candidate_payload_missing",
            ["v3_did_not_produce_runtime_payload"],
            trajectory=trajectory,
        )

    requirements = V3IndustrialCandidateRequirements.from_case(case, expected_payload)
    contract = compare_v3_candidate_runtime_contract(
        candidate_payload,
        expected_payload,
        requirements=requirements,
    )
    canonical_alignment = compare_candidate_runtime_contract(candidate_payload, expected_payload)
    if not contract["ok"]:
        return _failed(
            case,
            "v3_candidate_contract_mismatch",
            [f"mismatch:{item['field']}" for item in contract["mismatches"]],
            candidate_contract=contract,
            canonical_alignment=canonical_alignment,
            trajectory=trajectory,
            runtime_attempted=False,
        )

    run_request = _base_request(
        session_id=session_id,
        text="Run the checked runtime payload after preflight.",
        events=events,
        runtime_policy=runtime_policy,
    )
    run_request.update({"run": True, "accept_defaults": True})
    preflight_response = service.run_turn(run_request)
    trajectory.append(_turn_record("preflight", preflight_response))
    pending = preflight_response.get("pending_action") if isinstance(preflight_response.get("pending_action"), dict) else {}
    action_id = str(pending.get("action_id") or "")
    if not action_id:
        return _failed(
            case,
            "v3_confirmation_gate_missing",
            ["preflight_did_not_create_pending_action"],
            candidate_contract=contract,
            canonical_alignment=canonical_alignment,
            trajectory=trajectory,
            runtime_attempted=False,
        )

    confirm_request = _base_request(
        session_id=session_id,
        text="Confirm the checked runtime action.",
        events=events,
        runtime_policy=runtime_policy,
    )
    confirm_request["confirmation_event"] = {"action_id": action_id, "decision": "confirm"}
    runtime_response = service.run_turn(confirm_request)
    trajectory.append(_turn_record("confirm_and_run", runtime_response))
    runtime_observation = _latest_observation(runtime_response, GEANT4_RUNTIME_TOOL)
    runtime_data = _observation_data(runtime_observation)
    if not runtime_observation or runtime_observation.get("status") != "ok":
        return _not_evaluable(
            case,
            "runtime_error",
            [str((runtime_observation or {}).get("message") or "v3_runtime_observation_missing")],
            candidate_contract=contract,
            canonical_alignment=canonical_alignment,
            trajectory=trajectory,
            runtime_attempted=True,
            runtime_adapter=runtime_data.get("adapter"),
        )
    if runtime_data.get("adapter") != "local_process":
        return _failed(
            case,
            "non_real_runtime",
            [f"adapter:{runtime_data.get('adapter')!r}"],
            candidate_contract=contract,
            canonical_alignment=canonical_alignment,
            trajectory=trajectory,
            runtime_attempted=True,
        )

    result_summary = runtime_data.get("result_summary") if isinstance(runtime_data.get("result_summary"), dict) else {}
    metric_plan = compiled.get("metric_plan") if isinstance(compiled.get("metric_plan"), dict) else {}
    extracted = extract_industrial_metrics(result_summary, metric_plan)
    actual_metrics = extracted["actual_metrics"]
    if extracted["missing_metrics"]:
        return _not_evaluable(
            case,
            "missing_metric",
            list(extracted["missing_metrics"]),
            candidate_contract=contract,
            canonical_alignment=canonical_alignment,
            trajectory=trajectory,
            runtime_attempted=True,
            runtime_adapter="local_process",
            actual_metrics=actual_metrics,
        )

    if not canonical_alignment["ok"]:
        return {
            "id": case.get("id"),
            "domain": case.get("domain"),
            "status": "passed",
            "failure_category": None,
            "reasons": [],
            "comparison_scope": "semantic_contract_and_real_runtime",
            "candidate_contract": contract,
            "canonical_alignment": canonical_alignment,
            "trajectory": trajectory,
            "runtime_attempted": True,
            "runtime_adapter": "local_process",
            "actual_metrics": actual_metrics,
            "golden_comparison": {
                "performed": False,
                "reason": "candidate_is_physically_valid_but_not_canonical_payload",
            },
        }

    golden_status = _golden_status(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens)
    if not golden_status["ready"]:
        return _not_evaluable(
            case,
            "unreviewed_golden" if golden_status.get("review_required") else "missing_golden",
            ["reviewed_golden_required"],
            candidate_contract=contract,
            canonical_alignment=canonical_alignment,
            trajectory=trajectory,
            runtime_attempted=True,
            runtime_adapter="local_process",
            actual_metrics=actual_metrics,
            golden_status=golden_status,
        )

    comparison = compare_industrial_metrics(
        actual_metrics,
        _golden_metrics(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens),
    )
    reasons = comparison["missing_metrics"] + comparison["mismatched_metrics"] + comparison["unchecked_metrics"]
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": "passed" if comparison["ok"] else "failed",
        "failure_category": None if comparison["ok"] else "metric_mismatch",
        "reasons": reasons,
        "candidate_contract": contract,
        "canonical_alignment": canonical_alignment,
        "comparison_scope": "canonical_golden",
        "trajectory": trajectory,
        "runtime_attempted": True,
        "runtime_adapter": "local_process",
        "actual_metrics": actual_metrics,
        "metric_diff": comparison["metric_diff"],
        "golden_comparison": {"performed": True, "ok": comparison["ok"]},
        "golden_status": golden_status,
    }


def run_v3_industrial_runtime_stage(
    *,
    benchmark_path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    case_ids: list[str] | None = None,
    llm_config_path: str = "",
    allow_unreviewed_goldens: bool = False,
    env: dict[str, str] | None = None,
    service: V3AgentTurnService | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    shape_report = validate_industrial_benchmark_shape(benchmark_path)
    benchmark = _load_json(benchmark_path) if shape_report["failed"] == 0 else {"cases": [], "runtime_defaults": {}}
    selected = _selected_case_ids(benchmark, case_ids or [])
    runtime_gate = _runtime_gate(env_map)
    llm_ready = bool(llm_config_path and Path(llm_config_path).exists())
    cases = _case_map(benchmark)
    policy = _runtime_policy(env_map)
    results: list[dict[str, Any]] = []

    def execute(active_service: V3AgentTurnService) -> None:
        for case_id in selected:
            case = cases[case_id]
            compiled = compile_industrial_case_to_runtime(
                case,
                runtime_defaults=benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {},
            )
            if compiled.get("status") != "compiled":
                results.append(_not_evaluable(case, str(compiled.get("failure_category") or "compile_error"), list(compiled.get("unsupported_features") or [])))
            elif not llm_ready:
                results.append(_not_evaluable(case, "llm_unavailable", ["missing_llm_config_path"]))
            elif not runtime_gate["real_runtime_ready"]:
                results.append(_not_evaluable(case, "runtime_unavailable", ["real_local_process_runtime_required"]))
            else:
                try:
                    results.append(
                        _run_v3_case(
                            active_service,
                            case,
                            compiled,
                            llm_config_path=llm_config_path,
                            runtime_policy=policy,
                            golden_dir=golden_dir,
                            allow_unreviewed_goldens=allow_unreviewed_goldens,
                        )
                    )
                except Exception as exc:
                    results.append(_failed(case, "v3_stage_exception", [f"{type(exc).__name__}: {exc}"]))

    if service is not None:
        execute(service)
    else:
        with tempfile.TemporaryDirectory(prefix="geant4-v3-industrial-sessions-") as sessions_dir:
            execute(V3AgentTurnService(sessions_dir=Path(sessions_dir)))

    counts = Counter(str(item.get("status")) for item in results)
    failures = Counter(str(item.get("failure_category")) for item in results if item.get("failure_category"))
    return {
        "schema_version": V3_INDUSTRIAL_RUNTIME_STAGE_SCHEMA_VERSION,
        "ok": bool(results) and counts.get("passed", 0) == len(results),
        "benchmark_path": str(benchmark_path),
        "golden_dir": str(golden_dir),
        "runtime_gate": runtime_gate,
        "llm_gate": {"configured": llm_ready, "config_name": Path(llm_config_path).name if llm_ready else None},
        "golden_policy": {"review_required": not allow_unreviewed_goldens},
        "shape_report": shape_report,
        "selected_case_ids": selected,
        "case_results": results,
        "stage_summary": {
            "status_counts": dict(sorted(counts.items())),
            "failure_categories": dict(sorted(failures.items())),
            "candidate_contract_passed": sum(1 for item in results if (item.get("candidate_contract") or {}).get("ok")),
            "runtime_attempted": sum(1 for item in results if item.get("runtime_attempted")),
            "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0),
            "not_evaluable": counts.get("not_evaluable", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v3-first industrial LLM and real Geant4 acceptance stage.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--allow-unreviewed-goldens", action="store_true")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_EVAL_REPORT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_v3_industrial_runtime_stage(
        benchmark_path=args.benchmark,
        golden_dir=args.golden_dir,
        case_ids=args.case_id,
        llm_config_path=args.llm_config,
        allow_unreviewed_goldens=args.allow_unreviewed_goldens,
    )
    saved = save_eval_output(report, outdir=args.outdir, tool="v3-industrial-runtime-stage", run_id=args.run_id or None)
    if args.json:
        print(json.dumps(saved, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(saved["stage_summary"], ensure_ascii=False, indent=2))
    return 0 if saved["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
