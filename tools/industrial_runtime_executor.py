from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import build_geant4_adapter_from_env
from mcp.geant4.server import Geant4McpServer
from tools.industrial_runtime_compiler import (
    INDUSTRIAL_RUNTIME_COMPILER_SCHEMA_VERSION,
    compile_industrial_case_to_runtime,
)

INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION = "geant4_agent_industrial_runtime_execution.v1"
INDUSTRIAL_GOLDEN_SCHEMA_VERSION = "geant4_agent_industrial_golden.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _numeric(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _runtime_fingerprint(runtime_payload: dict[str, Any], run_payload: dict[str, Any]) -> dict[str, Any]:
    simulation_result = run_payload.get("simulation_result") if isinstance(run_payload.get("simulation_result"), dict) else {}
    result_summary = run_payload.get("result_summary") if isinstance(run_payload.get("result_summary"), dict) else {}
    run = result_summary.get("run") if isinstance(result_summary.get("run"), dict) else {}
    encoded = json.dumps(runtime_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return {
        "geant4_version": simulation_result.get("geant4_version"),
        "runtime_wrapper_hash": None,
        "runtime_payload_hash": hashlib.sha256(encoded).hexdigest(),
        "physics_list": _get_path(runtime_payload, "physics.list") or runtime_payload.get("physics_list"),
        "seed": run.get("seed") or _get_path(runtime_payload, "run.seed"),
        "events": run.get("events_requested") or _get_path(runtime_payload, "run.events"),
        "threads": 1,
        "platform": platform.platform(),
    }


def extract_industrial_metrics(result_summary: dict[str, Any], metric_plan: dict[str, Any]) -> dict[str, Any]:
    supported = metric_plan.get("supported") if isinstance(metric_plan.get("supported"), dict) else {}
    actual: dict[str, Any] = {}
    missing: list[str] = []
    for metric, spec in supported.items():
        if not isinstance(spec, dict):
            missing.append(str(metric))
            continue
        kind = spec.get("kind")
        if kind == "direct":
            value = _numeric(_get_path({"result_summary": result_summary}, str(spec.get("path") or "")))
        elif kind == "derived":
            inputs = spec.get("inputs") if isinstance(spec.get("inputs"), list) else []
            input_values: dict[str, float | int] = {}
            for name in inputs:
                if name == "events_completed":
                    raw = _get_path({"result_summary": result_summary}, "result_summary.run.events_completed")
                elif name == "detector_crossing_count":
                    raw = _get_path(
                        {"result_summary": result_summary},
                        "result_summary.scoring.detector_crossing.detector_crossing_count",
                    )
                else:
                    raw = None
                numeric = _numeric(raw)
                if numeric is not None:
                    input_values[str(name)] = numeric
            denominator = float(input_values.get("events_completed", 0) or 0)
            numerator = float(input_values.get("detector_crossing_count", 0) or 0)
            value = numerator / denominator if denominator > 0 else None
        else:
            value = None
        if value is None:
            missing.append(str(metric))
        else:
            actual[str(metric)] = value
    return {"actual_metrics": actual, "missing_metrics": missing}


def compare_industrial_metrics(actual_metrics: dict[str, Any], golden_metrics: dict[str, Any]) -> dict[str, Any]:
    diffs: dict[str, Any] = {}
    missing: list[str] = []
    mismatches: list[str] = []
    unchecked: list[str] = []
    for metric_name, golden in golden_metrics.items():
        if not isinstance(golden, dict) or golden.get("expected") is None:
            unchecked.append(str(metric_name))
            continue
        actual = _numeric(actual_metrics.get(metric_name))
        expected = _numeric(golden.get("expected"))
        tolerance = _numeric(golden.get("tolerance"))
        if tolerance is None:
            tolerance = 0.0
        if actual is None:
            missing.append(str(metric_name))
            continue
        if expected is None:
            unchecked.append(str(metric_name))
            continue
        delta = float(actual) - float(expected)
        passed = abs(delta) <= float(tolerance)
        if not passed:
            mismatches.append(str(metric_name))
        diffs[str(metric_name)] = {
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
            "delta": delta,
            "passed": passed,
        }
    return {
        "ok": not missing and not mismatches and not unchecked,
        "metric_diff": diffs,
        "missing_metrics": missing,
        "mismatched_metrics": mismatches,
        "unchecked_metrics": unchecked,
    }


def execute_industrial_case(
    case: dict[str, Any],
    *,
    runtime_defaults: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    adapter: Any | None = None,
    require_local_process: bool = True,
) -> dict[str, Any]:
    compile_result = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
    if compile_result.get("status") not in {"compiled", "compiled_with_gaps"}:
        return {
            "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
            "case_id": case.get("id"),
            "status": "not_evaluable",
            "failure_category": compile_result.get("failure_category") or "spec_compile_error",
            "errors": list(compile_result.get("unsupported_features") or []),
            "compile_result": compile_result,
        }

    metric_plan = compile_result.get("metric_plan") if isinstance(compile_result.get("metric_plan"), dict) else {}
    unsupported_metrics = metric_plan.get("unsupported") if isinstance(metric_plan.get("unsupported"), dict) else {}
    if unsupported_metrics:
        return {
            "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
            "case_id": case.get("id"),
            "status": "not_evaluable",
            "failure_category": "missing_metric",
            "errors": [f"unsupported_metric:{metric}" for metric in unsupported_metrics],
            "compile_result": compile_result,
        }

    runtime_adapter = adapter or build_geant4_adapter_from_env(env or os.environ)
    snapshot = runtime_adapter.snapshot()
    adapter_name = snapshot.metadata.get("adapter") if isinstance(snapshot.metadata, dict) else None
    if require_local_process and adapter_name != "local_process":
        return {
            "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
            "case_id": case.get("id"),
            "status": "not_evaluable",
            "failure_category": "runtime_unavailable",
            "errors": ["local_process_runtime_required"],
            "compile_result": compile_result,
            "runtime_adapter": adapter_name or "<unknown>",
        }

    server = Geant4McpServer(adapter=runtime_adapter)
    config = compile_result["config"]
    events = int((runtime_defaults or {}).get("events", 10000) or 10000)

    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": config}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed(case, compile_result, "apply_config_patch_failed", apply_obs)

    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed(case, compile_result, "initialize_run_failed", init_obs)

    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": events}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        return _runtime_failed(case, compile_result, "run_beam_failed", run_obs)

    result_summary = run_obs.payload.get("result_summary")
    if not isinstance(result_summary, dict):
        return {
            "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
            "case_id": case.get("id"),
            "status": "failed",
            "failure_category": "missing_metric",
            "errors": ["missing_result_summary"],
            "compile_result": compile_result,
            "run_payload": run_obs.payload,
        }

    metrics = extract_industrial_metrics(result_summary, metric_plan)
    if metrics["missing_metrics"]:
        return {
            "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
            "case_id": case.get("id"),
            "status": "failed",
            "failure_category": "missing_metric",
            "errors": [f"missing_metric:{metric}" for metric in metrics["missing_metrics"]],
            "compile_result": compile_result,
            "run_payload": run_obs.payload,
            **metrics,
        }

    return {
        "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
        "case_id": case.get("id"),
        "status": "completed",
        "failure_category": None,
        "compile_result": compile_result,
        "runtime_fingerprint": _runtime_fingerprint(compile_result["runtime_payload"], run_obs.payload),
        "run_payload": run_obs.payload,
        "result_summary": result_summary,
        **metrics,
    }


def build_industrial_golden_payload(
    case: dict[str, Any],
    execution_result: dict[str, Any],
    *,
    artifact_dir: str | None = None,
    run_summary_path: str | None = None,
) -> dict[str, Any]:
    if execution_result.get("status") != "completed":
        raise ValueError("cannot_build_golden_from_incomplete_execution")
    return {
        "schema_version": INDUSTRIAL_GOLDEN_SCHEMA_VERSION,
        "case_id": case.get("id"),
        "created_at_utc": _utc_now(),
        "runtime_fingerprint": execution_result.get("runtime_fingerprint") or {},
        "runtime_payload_hash": (execution_result.get("runtime_fingerprint") or {}).get("runtime_payload_hash"),
        "metrics": {
            metric: {"expected": value, "tolerance": _default_tolerance(value)}
            for metric, value in sorted((execution_result.get("actual_metrics") or {}).items())
        },
        "artifact_dir": artifact_dir or _artifact_value(execution_result, "artifact_dir"),
        "run_summary_path": run_summary_path or _artifact_value(execution_result, "run_summary_path"),
        "review": {
            "status": "unreviewed",
            "reviewer": "",
            "notes": "Generated by industrial runtime golden tool; review before using as official baseline.",
        },
    }


def _default_tolerance(value: Any) -> float | int:
    if isinstance(value, int):
        return 0
    return 0.0


def _artifact_value(execution_result: dict[str, Any], key: str) -> str:
    run_payload = execution_result.get("run_payload") if isinstance(execution_result.get("run_payload"), dict) else {}
    simulation_result = (
        run_payload.get("simulation_result") if isinstance(run_payload.get("simulation_result"), dict) else {}
    )
    value = simulation_result.get(key)
    return str(value) if value else ""


def _runtime_failed(
    case: dict[str, Any],
    compile_result: dict[str, Any],
    reason: str,
    observation: Any,
) -> dict[str, Any]:
    return {
        "schema_version": INDUSTRIAL_RUNTIME_EXECUTION_SCHEMA_VERSION,
        "case_id": case.get("id"),
        "status": "failed",
        "failure_category": "runtime_error",
        "errors": [reason, *list(getattr(observation, "errors", []) or [])],
        "compile_result": compile_result,
        "runtime_observation": {
            "status": str(getattr(observation, "status", "")),
            "message": getattr(observation, "message", ""),
            "payload": getattr(observation, "payload", {}),
        },
    }
