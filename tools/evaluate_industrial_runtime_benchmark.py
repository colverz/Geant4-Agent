from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime, summarize_compile_results
from tools.industrial_runtime_executor import compare_industrial_metrics, execute_industrial_case

INDUSTRIAL_BENCHMARK_SCHEMA_VERSION = "geant4_agent_industrial_runtime_benchmark.v1"
DEFAULT_INDUSTRIAL_BENCHMARK_PATH = Path("docs/eval/industrial_runtime_benchmark.json")
DEFAULT_INDUSTRIAL_GOLDEN_DIR = Path("docs/eval/golden/industrial_runtime")
INDUSTRIAL_RUNTIME_ENV = "GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK"
RUNTIME_COMMAND_ENVS = ("GEANT4_RUNTIME_COMMAND_JSON", "GEANT4_RUNTIME_COMMAND")
REVIEWED_GOLDEN_STATUS = "reviewed"

REQUIRED_DOMAINS = {
    "industrial_ndt",
    "shielding",
    "medical_phantom",
    "detector_response",
    "beam_source",
    "multi_turn_engineering",
    "unsupported_boundary",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_enabled(env: dict[str, str]) -> bool:
    return str(env.get(INDUSTRIAL_RUNTIME_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_command_configured(env: dict[str, str]) -> bool:
    return any(str(env.get(name, "")).strip() for name in RUNTIME_COMMAND_ENVS)


def _golden_file_path(case: dict[str, Any], golden_dir: Path) -> Path:
    return golden_dir / f"{case.get('id')}.golden.json"


def _golden_payload(case: dict[str, Any], golden_dir: Path | None = None) -> dict[str, Any] | None:
    if golden_dir is not None:
        path = _golden_file_path(case, golden_dir)
        if path.exists():
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _golden_review_status(golden_payload: dict[str, Any] | None) -> str | None:
    review = golden_payload.get("review") if isinstance(golden_payload, dict) else None
    if not isinstance(review, dict):
        return None
    status = review.get("status")
    return str(status).strip().lower() if status else None


def _golden_metrics(
    case: dict[str, Any],
    golden_dir: Path | None = None,
    *,
    allow_unreviewed: bool = False,
) -> dict[str, Any]:
    payload = _golden_payload(case, golden_dir)
    if payload is not None:
        review_status = _golden_review_status(payload)
        if review_status == REVIEWED_GOLDEN_STATUS or allow_unreviewed:
            metrics = payload.get("metrics") if isinstance(payload, dict) else {}
            return metrics if isinstance(metrics, dict) else {}
        return {}
    metrics = case.get("golden_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _golden_status(
    case: dict[str, Any],
    golden_dir: Path | None = None,
    *,
    allow_unreviewed: bool = False,
) -> dict[str, Any]:
    payload = _golden_payload(case, golden_dir)
    review_status = _golden_review_status(payload)
    review_required = payload is not None and review_status != REVIEWED_GOLDEN_STATUS and not allow_unreviewed
    metrics = _golden_metrics(case, golden_dir, allow_unreviewed=allow_unreviewed)
    if not isinstance(metrics, dict) or not metrics:
        missing = ["<all>"]
        if review_required:
            missing = ["<reviewed_golden>"]
        result = {
            "ready": False,
            "missing_metrics": missing,
            "review_status": review_status,
            "review_required": review_required,
        }
        if golden_dir is not None:
            result["golden_file"] = str(_golden_file_path(case, golden_dir))
        return result
    missing: list[str] = []
    for metric_name, metric in metrics.items():
        if not isinstance(metric, dict):
            missing.append(str(metric_name))
            continue
        if metric.get("expected") is None:
            missing.append(str(metric_name))
            continue
        if "tolerance" not in metric:
            missing.append(str(metric_name))
    if review_required:
        missing.append("<reviewed_golden>")
    result = {
        "ready": not missing and not review_required,
        "missing_metrics": missing,
        "review_status": review_status,
        "review_required": review_required,
    }
    if golden_dir is not None:
        result["golden_file"] = str(_golden_file_path(case, golden_dir))
    return result


def validate_industrial_benchmark_shape(path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    try:
        benchmark = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "name": "industrial_runtime_benchmark_shape",
            "total": 0,
            "failed": 1,
            "failures": [{"id": "<file>", "error": f"{type(exc).__name__}: {exc}"}],
        }

    if not isinstance(benchmark, dict):
        failures.append({"id": "<root>", "error": "root_not_object"})
        cases: list[dict[str, Any]] = []
    else:
        if benchmark.get("schema_version") != INDUSTRIAL_BENCHMARK_SCHEMA_VERSION:
            failures.append({"id": "<root>", "error": "schema_version_mismatch"})
        official = benchmark.get("official_pass_requires")
        if not isinstance(official, dict):
            failures.append({"id": "<root>", "error": "official_pass_requires_not_object"})
        else:
            for key in ("real_geant4_runtime", "golden_numeric_metrics", "single_thread_default"):
                if official.get(key) is not True:
                    failures.append({"id": "<root>", "error": f"official_requirement_not_true:{key}"})
            if official.get("llm_as_judge") is not False:
                failures.append({"id": "<root>", "error": "llm_as_judge_must_be_false"})
        raw_cases = benchmark.get("cases")
        cases = raw_cases if isinstance(raw_cases, list) else []
        if not isinstance(raw_cases, list):
            failures.append({"id": "<root>", "error": "cases_not_list"})

    domains = {case.get("domain") for case in cases if isinstance(case, dict)}
    missing_domains = sorted(REQUIRED_DOMAINS - domains)
    if missing_domains:
        failures.append({"id": "<coverage>", "error": f"missing_domains:{','.join(missing_domains)}"})
    if len(cases) < 20:
        failures.append({"id": "<coverage>", "error": "case_count_below_20"})

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            failures.append({"id": f"<case:{index}>", "error": "case_not_object"})
            continue
        case_id = str(case.get("id") or f"<case:{index}>")
        for key in ("id", "domain", "task", "raw_dialogue", "llm_role", "required_runtime", "scenario_spec"):
            if key not in case:
                failures.append({"id": case_id, "error": f"missing_key:{key}"})
        if not isinstance(case.get("raw_dialogue"), list) or not case.get("raw_dialogue"):
            failures.append({"id": case_id, "error": "raw_dialogue_must_be_non_empty_list"})
        if not isinstance(case.get("scenario_spec"), dict):
            failures.append({"id": case_id, "error": "scenario_spec_not_object"})
        if case.get("golden_required") is True:
            if case.get("required_runtime") != "real_geant4":
                failures.append({"id": case_id, "error": "official_case_must_require_real_geant4"})
            if case.get("llm_role") != "candidate_config_only":
                failures.append({"id": case_id, "error": "official_case_llm_role_must_be_candidate_config_only"})
            if not isinstance(case.get("golden_metrics"), dict) or not case.get("golden_metrics"):
                failures.append({"id": case_id, "error": "official_case_missing_golden_metrics"})
        elif case.get("domain") == "unsupported_boundary":
            if case.get("expected_status") != "unsupported_capability":
                failures.append({"id": case_id, "error": "unsupported_case_must_mark_expected_status"})
        else:
            failures.append({"id": case_id, "error": "non_official_case_must_be_unsupported_boundary"})

    return {
        "name": "industrial_runtime_benchmark_shape",
        "total": len(cases),
        "failed": len(failures),
        "failures": failures,
        "domain_counts": dict(sorted(Counter(str(case.get("domain")) for case in cases if isinstance(case, dict)).items())),
    }


def _case_not_evaluable_result(
    case: dict[str, Any],
    *,
    reasons: list[str],
    failure_category: str,
    golden_dir: Path | None = None,
    allow_unreviewed_goldens: bool = False,
) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "domain": case.get("domain"),
        "status": "not_evaluable",
        "failure_category": failure_category,
        "reasons": list(dict.fromkeys(reasons)),
        "required_runtime": case.get("required_runtime"),
        "golden_status": _golden_status(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens),
    }


def evaluate_industrial_runtime_benchmark(
    path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    *,
    env: dict[str, str] | None = None,
    golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    allow_unreviewed_goldens: bool = False,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    shape_report = validate_industrial_benchmark_shape(path)
    if shape_report["failed"]:
        return {
            "name": "industrial_runtime_benchmark",
            "schema_version": INDUSTRIAL_BENCHMARK_SCHEMA_VERSION,
            "ok": False,
            "shape_report": shape_report,
            "total": 0,
            "passed": 0,
            "failed": 1,
            "not_evaluable": 0,
            "unsupported": 0,
            "case_results": [],
            "summary": {"failure_categories": {"shape": 1}},
        }

    benchmark = _load_json(path)
    requested_ids = {str(case_id) for case_id in (case_ids or []) if str(case_id)}
    cases = [
        case
        for case in benchmark["cases"]
        if not requested_ids or str(case.get("id") or "") in requested_ids
    ]
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    runtime_ready = _runtime_enabled(env_map) and _runtime_command_configured(env_map)
    runtime_reasons: list[str] = []
    if not _runtime_enabled(env_map):
        runtime_reasons.append(f"{INDUSTRIAL_RUNTIME_ENV}_not_enabled")
    if not _runtime_command_configured(env_map):
        runtime_reasons.append("missing_runtime_command")

    case_results: list[dict[str, Any]] = []
    compile_results: list[dict[str, Any]] = []
    for case in cases:
        compile_result = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        compile_results.append(compile_result)
        if case.get("domain") == "unsupported_boundary":
            case_results.append(
                {
                    "id": case.get("id"),
                    "domain": case.get("domain"),
                    "status": "unsupported_capability",
                    "failure_category": "unsupported_capability",
                    "reasons": list(case.get("capability_pressure") or []),
                    "expected_status": case.get("expected_status"),
                    "compile_status": compile_result.get("status"),
                    "compile_report": _compile_report_preview(compile_result),
                }
            )
            continue

        reasons: list[str] = []
        golden = _golden_status(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens)
        compile_status = str(compile_result.get("status") or "")
        if not runtime_ready:
            reasons.extend(runtime_reasons)
        if not golden["ready"]:
            if golden.get("review_required"):
                reasons.append("unreviewed_golden_metrics")
            else:
                reasons.append("missing_golden_metrics")
        if compile_status == "unsupported_capability":
            reasons.extend(str(item) for item in compile_result.get("unsupported_features") or [])
        if reasons:
            category = "runtime_unavailable" if runtime_reasons else "missing_golden"
            if runtime_reasons and not golden["ready"]:
                category = "runtime_unavailable_and_missing_golden"
            if golden.get("review_required") and not runtime_reasons:
                category = "unreviewed_golden"
            if not runtime_reasons and golden["ready"] and compile_status == "unsupported_capability":
                category = "spec_compile_error"
            result = _case_not_evaluable_result(
                case,
                reasons=reasons,
                failure_category=category,
                golden_dir=golden_dir,
                allow_unreviewed_goldens=allow_unreviewed_goldens,
            )
            result["compile_status"] = compile_result.get("status")
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        if compile_status == "unsupported_capability":
            result = _case_not_evaluable_result(
                case,
                reasons=list(compile_result.get("unsupported_features") or ["runtime_blueprint_not_available_for_case"]),
                failure_category="spec_compile_error",
                golden_dir=golden_dir,
                allow_unreviewed_goldens=allow_unreviewed_goldens,
            )
            result["compile_status"] = compile_result.get("status")
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        metric_plan = compile_result.get("metric_plan") if isinstance(compile_result.get("metric_plan"), dict) else {}
        unsupported_metrics = metric_plan.get("unsupported") if isinstance(metric_plan.get("unsupported"), dict) else {}
        if unsupported_metrics:
            result = _case_not_evaluable_result(
                case,
                reasons=[f"unsupported_metric:{metric}" for metric in unsupported_metrics],
                failure_category="missing_metric",
                golden_dir=golden_dir,
                allow_unreviewed_goldens=allow_unreviewed_goldens,
            )
            result["compile_status"] = compile_result.get("status")
            result["compile_report"] = _compile_report_preview(compile_result)
            case_results.append(result)
            continue

        execution = execute_industrial_case(case, runtime_defaults=runtime_defaults, env=env_map)
        if execution.get("status") != "completed":
            result = _case_not_evaluable_result(
                case,
                reasons=list(execution.get("errors") or ["industrial_runtime_execution_failed"]),
                failure_category=str(execution.get("failure_category") or "runtime_error"),
                golden_dir=golden_dir,
                allow_unreviewed_goldens=allow_unreviewed_goldens,
            )
            result["compile_status"] = compile_result.get("status")
            result["compile_report"] = _compile_report_preview(compile_result)
            result["execution_report"] = _execution_report_preview(execution)
            case_results.append(result)
            continue

        comparison = compare_industrial_metrics(
            execution.get("actual_metrics") or {},
            _golden_metrics(case, golden_dir, allow_unreviewed=allow_unreviewed_goldens),
        )
        status = "passed" if comparison["ok"] else "failed"
        failure_category = None
        if comparison["missing_metrics"]:
            failure_category = "missing_metric"
        elif comparison["mismatched_metrics"] or comparison["unchecked_metrics"]:
            failure_category = "metric_mismatch"
        case_results.append(
            {
                "id": case.get("id"),
                "domain": case.get("domain"),
                "status": status,
                "failure_category": failure_category,
                "reasons": list(comparison["missing_metrics"])
                + list(comparison["mismatched_metrics"])
                + list(comparison["unchecked_metrics"]),
                "required_runtime": case.get("required_runtime"),
                "golden_status": golden,
                "compile_status": compile_result.get("status"),
                "compile_report": _compile_report_preview(compile_result),
                "execution_report": _execution_report_preview(execution),
                "actual_metrics": execution.get("actual_metrics") or {},
                "metric_diff": comparison["metric_diff"],
            }
        )

    status_counts = Counter(str(item.get("status")) for item in case_results)
    failure_categories = Counter(str(item.get("failure_category")) for item in case_results if item.get("failure_category"))
    domain_counts = Counter(str(item.get("domain")) for item in case_results)
    return {
        "name": "industrial_runtime_benchmark",
        "schema_version": INDUSTRIAL_BENCHMARK_SCHEMA_VERSION,
        "ok": status_counts.get("passed", 0) > 0
        and status_counts.get("not_evaluable", 0) == 0
        and status_counts.get("unsupported_capability", 0) == 0,
        "shape_report": shape_report,
        "runtime_gate": {
            "env_enabled": _runtime_enabled(env_map),
            "runtime_command_configured": _runtime_command_configured(env_map),
            "real_runtime_ready": runtime_ready,
        },
        "golden_policy": {
            "golden_dir": str(golden_dir),
            "review_required": not allow_unreviewed_goldens,
            "allow_unreviewed_goldens": allow_unreviewed_goldens,
        },
        "total": len(case_results),
        "passed": status_counts.get("passed", 0),
        "failed": status_counts.get("failed", 0),
        "not_evaluable": status_counts.get("not_evaluable", 0),
        "unsupported": status_counts.get("unsupported_capability", 0),
        "case_results": case_results,
        "summary": {
            "status_counts": dict(sorted(status_counts.items())),
            "failure_categories": dict(sorted(failure_categories.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
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
    parser = argparse.ArgumentParser(description="Evaluate the industrial runtime benchmark acceptance gate.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument(
        "--allow-unreviewed-goldens",
        action="store_true",
        help="Wiring/dev mode only. Official benchmark pass requires reviewed golden files.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate_industrial_runtime_benchmark(
        args.benchmark,
        golden_dir=args.golden_dir,
        allow_unreviewed_goldens=args.allow_unreviewed_goldens,
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
            f"{report['name']}: passed={report['passed']} "
            f"not_evaluable={report['not_evaluable']} unsupported={report['unsupported']}"
        )
        for category, count in report["summary"]["failure_categories"].items():
            print(f"  {category}: {count}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
