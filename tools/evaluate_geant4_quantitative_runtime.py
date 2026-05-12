from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from core.simulation import build_runtime_smoke_report
from mcp.geant4.adapter import InMemoryGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.server import Geant4McpServer
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_geant4_agent_benchmark import (
    DEFAULT_BENCHMARK_PATH,
    _finalize_quantitative_result_summary,
    _load_json,
    _new_quantitative_result_summary,
    _quantitative_result_errors,
    _update_quantitative_result_summary,
    validate_benchmark_shape,
)

LIVE_RUNTIME_ENV = "GEANT4_BENCHMARK_LIVE_RUNTIME"
RUNTIME_COMMAND_ENVS = ("GEANT4_RUNTIME_COMMAND_JSON", "GEANT4_RUNTIME_COMMAND")


def _live_runtime_enabled(env: dict[str, str]) -> bool:
    return str(env.get(LIVE_RUNTIME_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_command_configured(env: dict[str, str]) -> bool:
    return any(str(env.get(name, "")).strip() for name in RUNTIME_COMMAND_ENVS)


def _benchmark_runtime_patch() -> dict[str, Any]:
    return {
        "geometry": {
            "structure": "single_box",
            "root_name": "Target",
            "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
        },
        "materials": {
            "selected_materials": ["G4_Cu"],
            "volume_material_map": {"Target": "G4_Cu"},
        },
        "source": {
            "type": "point",
            "particle": "gamma",
            "energy": 1.0,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "physics_list": {"name": "FTFP_BERT"},
        "scoring": {"target_edep": True},
    }


def _live_quantitative_expected(expected: dict[str, Any]) -> dict[str, Any]:
    """Real runtime grading must use stable facts, not fixture-only exact values."""

    return {
        key: value
        for key, value in expected.items()
        if key
        in {
            "sample_report",
            "required_metric_keys",
            "expected_metric_ranges",
            "non_negative_metric_keys",
            "expected_relations",
        }
    }


def _run_runtime_smoke(adapter: Any, *, events: int) -> dict[str, Any]:
    server = Geant4McpServer(adapter=adapter)
    patch = _benchmark_runtime_patch()

    validate_obs = server.call_tool(ToolCallRequest(tool_name="validate_config", arguments={"config": patch, "events": events}))
    if validate_obs.status != RuntimeActionStatus.COMPLETED or not (validate_obs.payload or {}).get("ok"):
        raise RuntimeError(f"validate_config failed: {validate_obs.errors or validate_obs.message}")

    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": patch}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        raise RuntimeError(f"apply_config_patch failed: {apply_obs.errors or apply_obs.message}")

    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        raise RuntimeError(f"initialize_run failed: {init_obs.errors or init_obs.message}")

    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": events}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        raise RuntimeError(f"run_beam failed: {run_obs.errors or run_obs.message}")

    summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
    if summary_obs.status != RuntimeActionStatus.COMPLETED:
        raise RuntimeError(f"summarize_last_result failed: {summary_obs.errors or summary_obs.message}")

    return build_runtime_smoke_report(events=events, run_payload=run_obs.payload, summary_payload=summary_obs.payload)


def _adapter_metadata(adapter: Any) -> dict[str, Any]:
    try:
        snapshot = adapter.snapshot()
    except Exception:
        return {"adapter": type(adapter).__name__}
    metadata = getattr(snapshot, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {"adapter": type(adapter).__name__}


def evaluate_live_quantitative_runtime(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    *,
    env: dict[str, str] | None = None,
    events: int = 4,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    if not _live_runtime_enabled(env_map):
        return _skip_report("live_runtime_not_enabled", events=events)
    if not _runtime_command_configured(env_map):
        return _skip_report("missing_runtime_command", events=events)

    shape_report = validate_benchmark_shape(benchmark_path)
    if shape_report["failed"]:
        return {
            "name": "geant4_agent_live_quantitative_runtime",
            "enabled": True,
            "skipped": False,
            "skip_reason": None,
            "events": events,
            "total": 0,
            "passed": 0,
            "failed": 1,
            "failures": [{"id": "<shape>", "section": "shape", "error": "shape_validation_failed"}],
            "shape_report": shape_report,
        }

    adapter = build_geant4_adapter_from_env(env_map)
    if isinstance(adapter, InMemoryGeant4Adapter):
        return _skip_report("in_memory_adapter", events=events)

    cases = _live_quantitative_cases(_load_json(benchmark_path))
    if not cases:
        return _skip_report("no_live_quantitative_cases", events=events)

    try:
        runtime_report = _run_runtime_smoke(adapter, events=max(1, int(events)))
    except Exception as exc:
        return {
            "name": "geant4_agent_live_quantitative_runtime",
            "enabled": True,
            "skipped": False,
            "skip_reason": None,
            "events": max(1, int(events)),
            "adapter": _adapter_metadata(adapter),
            "total": len(cases),
            "passed": 0,
            "failed": 1,
            "failures": [{"id": "<runtime>", "section": "runtime", "error": f"{type(exc).__name__}: {exc}"}],
            "quantitative_result_summary": _finalize_quantitative_result_summary(_new_quantitative_result_summary()),
        }

    failures: list[dict[str, Any]] = []
    passed = 0
    summary = _new_quantitative_result_summary()
    for case in cases:
        case_id = str(case.get("id") or "unknown")
        expected = _live_quantitative_expected(case["expected_quantitative_result"])
        _update_quantitative_result_summary(summary, expected, runtime_report)
        case_errors = _quantitative_result_errors(expected, runtime_report, case_id=case_id)
        if case_errors:
            failures.append({"id": case_id, "errors": case_errors})
        else:
            passed += 1

    return {
        "name": "geant4_agent_live_quantitative_runtime",
        "enabled": True,
        "skipped": False,
        "skip_reason": None,
        "events": max(1, int(events)),
        "adapter": _adapter_metadata(adapter),
        "total": len(cases),
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "quantitative_result_summary": _finalize_quantitative_result_summary(summary),
        "runtime_smoke_report": runtime_report,
    }


def _live_quantitative_cases(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        return []
    selected: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if "quantitative_result" not in (case.get("capabilities") or []):
            continue
        expected = case.get("expected_quantitative_result")
        if not isinstance(expected, dict):
            continue
        live_expected = _live_quantitative_expected(expected)
        if any(live_expected.get(key) for key in ("required_metric_keys", "expected_metric_ranges", "non_negative_metric_keys", "expected_relations")):
            selected.append(case)
    return selected


def _skip_report(reason: str, *, events: int) -> dict[str, Any]:
    return {
        "name": "geant4_agent_live_quantitative_runtime",
        "enabled": reason not in {"live_runtime_not_enabled"},
        "skipped": True,
        "skip_reason": reason,
        "events": max(1, int(events)),
        "total": 0,
        "passed": 0,
        "failed": 0,
        "failures": [],
        "quantitative_result_summary": _finalize_quantitative_result_summary(_new_quantitative_result_summary()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in live Geant4 quantitative benchmark evaluator.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=None, help="Optional directory for a full JSON eval record.")
    parser.add_argument("--run-id", default="", help="Optional stable run id for saved eval records.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate_live_quantitative_runtime(args.benchmark, events=args.events)
    output = {"ok": report["failed"] == 0, "report": report}
    if args.outdir:
        output = save_eval_output(
            output,
            outdir=args.outdir or DEFAULT_EVAL_REPORT_DIR,
            tool=report["name"],
            run_id=args.run_id or None,
        )
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if report.get("skipped"):
            print(f"{report['name']}: skipped ({report['skip_reason']})")
        else:
            print(f"{report['name']}: {report['passed']} / {report['total']} passed")
            for failure in report["failures"]:
                print(f"  FAIL {failure}")
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
