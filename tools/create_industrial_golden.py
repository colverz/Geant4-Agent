from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_industrial_runtime_benchmark import (
    DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    INDUSTRIAL_BENCHMARK_SCHEMA_VERSION,
    INDUSTRIAL_RUNTIME_ENV,
    RUNTIME_COMMAND_ENVS,
    _load_json,
    _runtime_command_configured,
    _runtime_enabled,
    validate_industrial_benchmark_shape,
)
from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime, summarize_compile_results
from tools.industrial_runtime_executor import build_industrial_golden_payload, execute_industrial_case

INDUSTRIAL_GOLDEN_SCHEMA_VERSION = "geant4_agent_industrial_golden.v1"
DEFAULT_INDUSTRIAL_GOLDEN_DIR = Path("docs/eval/golden/industrial_runtime")


def _case_is_official(case: dict[str, Any]) -> bool:
    return case.get("golden_required") is True and case.get("domain") != "unsupported_boundary"


def _selected_cases(benchmark: dict[str, Any], case_id: str | None) -> list[dict[str, Any]]:
    cases = [case for case in benchmark.get("cases", []) if isinstance(case, dict)]
    if not case_id:
        return [case for case in cases if _case_is_official(case)]
    return [case for case in cases if case.get("id") == case_id]


def _golden_path(golden_dir: Path, case_id: str) -> Path:
    return golden_dir / f"{case_id}.golden.json"


def _runtime_gate(env: dict[str, str]) -> dict[str, Any]:
    enabled = _runtime_enabled(env)
    configured = _runtime_command_configured(env)
    reasons: list[str] = []
    if not enabled:
        reasons.append(f"{INDUSTRIAL_RUNTIME_ENV}_not_enabled")
    if not configured:
        reasons.append("missing_runtime_command")
    return {
        "env_enabled": enabled,
        "runtime_command_configured": configured,
        "real_runtime_ready": enabled and configured,
        "reasons": reasons,
        "required_env": [INDUSTRIAL_RUNTIME_ENV, *RUNTIME_COMMAND_ENVS],
    }


def _not_generated_result(
    case: dict[str, Any],
    *,
    status: str,
    failure_category: str,
    reasons: list[str],
    golden_file: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": status,
        "failure_category": failure_category,
        "reasons": list(dict.fromkeys(reasons)),
        "golden_generated": False,
    }
    if golden_file is not None:
        result["golden_file"] = str(golden_file)
    return result


def generate_industrial_golden(
    benchmark_path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    *,
    case_id: str | None = None,
    golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    env: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate reviewed industrial golden metrics from a real Geant4 runtime.

    The current implementation intentionally refuses to fabricate goldens. It
    validates benchmark shape, runtime opt-in, and deterministic scenario
    compilation before reporting the remaining execution blocker.
    """

    env_map = dict(os.environ if env is None else env)
    shape_report = validate_industrial_benchmark_shape(benchmark_path)
    if shape_report["failed"]:
        return {
            "name": "industrial_golden_generation",
            "schema_version": INDUSTRIAL_GOLDEN_SCHEMA_VERSION,
            "ok": False,
            "benchmark_schema_version": INDUSTRIAL_BENCHMARK_SCHEMA_VERSION,
            "shape_report": shape_report,
            "runtime_gate": _runtime_gate(env_map),
            "total": 0,
            "generated": 0,
            "blocked": 0,
            "not_evaluable": 0,
            "skipped": 0,
            "case_results": [],
            "summary": {"failure_categories": {"shape": 1}},
        }

    benchmark = _load_json(benchmark_path)
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    selected = _selected_cases(benchmark, case_id)
    if case_id and not selected:
        return {
            "name": "industrial_golden_generation",
            "schema_version": INDUSTRIAL_GOLDEN_SCHEMA_VERSION,
            "ok": False,
            "benchmark_schema_version": INDUSTRIAL_BENCHMARK_SCHEMA_VERSION,
            "shape_report": shape_report,
            "runtime_gate": _runtime_gate(env_map),
            "total": 0,
            "generated": 0,
            "blocked": 0,
            "not_evaluable": 0,
            "skipped": 0,
            "case_results": [],
            "summary": {"failure_categories": {"case_not_found": 1}},
            "errors": [{"id": case_id, "error": "case_not_found"}],
        }

    gate = _runtime_gate(env_map)
    case_results: list[dict[str, Any]] = []
    compile_results: list[dict[str, Any]] = []
    for case in selected:
        case_name = str(case.get("id"))
        compile_result = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        compile_results.append(compile_result)
        if not _case_is_official(case):
            result = _not_generated_result(
                case,
                status="skipped",
                failure_category="unsupported_capability",
                reasons=["golden_generation_only_supports_official_runtime_cases"],
            )
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        golden_file = _golden_path(golden_dir, case_name)
        if golden_file.exists() and not force:
            case_results.append(
                _not_generated_result(
                    case,
                    status="blocked",
                    failure_category="golden_exists",
                    reasons=["golden_file_exists_use_force_to_refresh"],
                    golden_file=golden_file,
                )
            )
            continue

        if not gate["real_runtime_ready"]:
            result = _not_generated_result(
                case,
                status="blocked",
                failure_category="runtime_unavailable",
                reasons=list(gate["reasons"]),
                golden_file=golden_file,
            )
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        if compile_result.get("status") == "unsupported_capability":
            result = _not_generated_result(
                case,
                status="not_evaluable",
                failure_category="spec_compile_error",
                reasons=list(compile_result.get("unsupported_features") or ["runtime_blueprint_not_available_for_case"]),
                golden_file=golden_file,
            )
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        metric_plan = compile_result.get("metric_plan") if isinstance(compile_result.get("metric_plan"), dict) else {}
        unsupported_metrics = metric_plan.get("unsupported") if isinstance(metric_plan.get("unsupported"), dict) else {}
        if unsupported_metrics:
            result = _not_generated_result(
                case,
                status="not_evaluable",
                failure_category="missing_metric",
                reasons=[f"unsupported_metric:{metric}" for metric in unsupported_metrics],
                golden_file=golden_file,
            )
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        execution = execute_industrial_case(case, runtime_defaults=runtime_defaults, env=env_map)
        if execution.get("status") != "completed":
            result = _not_generated_result(
                case,
                status="not_evaluable",
                failure_category=str(execution.get("failure_category") or "runtime_error"),
                reasons=list(execution.get("errors") or ["industrial_runtime_execution_failed"]),
                golden_file=golden_file,
            )
            result["compile_report"] = _compile_report_preview(compile_result)
            result["execution_report"] = _execution_report_preview(execution)
            case_results.append(result)
            continue

        golden_file.parent.mkdir(parents=True, exist_ok=True)
        golden_payload = build_industrial_golden_payload(case, execution)
        golden_file.write_text(json.dumps(golden_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        case_results.append(
            {
                "id": case.get("id"),
                "domain": case.get("domain"),
                "status": "generated",
                "failure_category": None,
                "reasons": [],
                "golden_generated": True,
                "golden_file": str(golden_file),
                "actual_metrics": execution.get("actual_metrics") or {},
                "compile_report": _compile_report_preview(compile_result),
                "execution_report": _execution_report_preview(execution),
            }
        )

    status_counts = Counter(str(item.get("status")) for item in case_results)
    failure_categories = Counter(
        str(item.get("failure_category")) for item in case_results if item.get("failure_category")
    )
    return {
        "name": "industrial_golden_generation",
        "schema_version": INDUSTRIAL_GOLDEN_SCHEMA_VERSION,
        "ok": status_counts.get("generated", 0) > 0
        and status_counts.get("blocked", 0) == 0
        and status_counts.get("not_evaluable", 0) == 0,
        "benchmark_schema_version": INDUSTRIAL_BENCHMARK_SCHEMA_VERSION,
        "shape_report": shape_report,
        "runtime_gate": gate,
        "golden_dir": str(golden_dir),
        "total": len(case_results),
        "generated": status_counts.get("generated", 0),
        "blocked": status_counts.get("blocked", 0),
        "not_evaluable": status_counts.get("not_evaluable", 0),
        "skipped": status_counts.get("skipped", 0),
        "case_results": case_results,
        "summary": {
            "status_counts": dict(sorted(status_counts.items())),
            "failure_categories": dict(sorted(failure_categories.items())),
            "compile_summary": summarize_compile_results(compile_results),
        },
    }


def _compile_report_preview(compile_result: dict[str, Any]) -> dict[str, Any]:
    metric_plan = compile_result.get("metric_plan") if isinstance(compile_result.get("metric_plan"), dict) else {}
    unsupported = metric_plan.get("unsupported") if isinstance(metric_plan.get("unsupported"), dict) else {}
    supported = metric_plan.get("supported") if isinstance(metric_plan.get("supported"), dict) else {}
    runtime_payload = compile_result.get("runtime_payload") if isinstance(compile_result.get("runtime_payload"), dict) else {}
    return {
        "schema_version": compile_result.get("schema_version"),
        "status": compile_result.get("status"),
        "failure_category": compile_result.get("failure_category"),
        "unsupported_features": list(compile_result.get("unsupported_features") or []),
        "supported_metric_count": len(supported),
        "unsupported_metrics": sorted(unsupported.keys()),
        "runtime_payload_keys": sorted(runtime_payload.keys()),
        "runtime_payload_available": bool(runtime_payload),
    }


def _execution_report_preview(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": execution.get("schema_version"),
        "status": execution.get("status"),
        "failure_category": execution.get("failure_category"),
        "errors": list(execution.get("errors") or []),
        "actual_metrics": execution.get("actual_metrics") or {},
        "missing_metrics": execution.get("missing_metrics") or [],
        "runtime_fingerprint": execution.get("runtime_fingerprint") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate industrial benchmark golden metrics from a real Geant4 runtime."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = generate_industrial_golden(
        args.benchmark,
        case_id=args.case_id or None,
        golden_dir=args.golden_dir,
        force=args.force,
    )
    output = {"ok": report["ok"], "report": report}
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
        print(
            f"{report['name']}: generated={report['generated']} "
            f"blocked={report['blocked']} not_evaluable={report['not_evaluable']} "
            f"skipped={report['skipped']}"
        )
        for category, count in report["summary"].get("failure_categories", {}).items():
            print(f"  {category}: {count}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
