from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, EVAL_REPORT_SCHEMA_VERSION


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _payload_report(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    return report if isinstance(report, dict) else payload


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _compact_model_summaries(report: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = report.get("model_summaries")
    if not isinstance(summaries, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "model": item.get("model"),
                "accuracy": item.get("accuracy"),
                "failed": item.get("failed"),
                "fallback_count": item.get("fallback_count"),
                "profile_mismatch_count": item.get("profile_mismatch_count"),
                "llm_used_count": item.get("llm_used_count"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "seconds_per_case": item.get("seconds_per_case"),
            }
        )
    return compact


def _stringify_failure_error(error: Any) -> str:
    if isinstance(error, dict):
        section = error.get("section")
        message = error.get("error") or error.get("message") or error
        return f"{section}:{message}" if section else str(message)
    return str(error)


def _compact_failures(report: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []

    def add_failure(failure: dict[str, Any], *, source: str | None = None) -> None:
        if len(compact) >= limit:
            return
        errors = failure.get("errors")
        error_items = errors if isinstance(errors, list) else []
        item: dict[str, Any] = {
            "id": failure.get("id"),
            "error_count": len(error_items),
            "errors": [_stringify_failure_error(error) for error in error_items[:3]],
        }
        if source:
            item["source"] = source
        compact.append(item)

    for failure in report.get("failures") or []:
        if isinstance(failure, dict):
            add_failure(failure)
    for child in report.get("reports") or []:
        if not isinstance(child, dict):
            continue
        source = str(child.get("model_override") or child.get("mode") or child.get("name") or "")
        for failure in child.get("failures") or []:
            if isinstance(failure, dict):
                add_failure(failure, source=source or None)
    for result in report.get("case_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("status") == "passed":
            continue
        add_failure(
            {
                "id": result.get("id"),
                "errors": result.get("reasons") or result.get("errors") or [result.get("failure_category")],
            },
            source=str(result.get("domain") or "") or None,
        )
    return compact


def _compact_key_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("total", "passed", "failed", "accuracy"):
        value = _safe_number(report.get(key))
        if value is not None:
            metrics[key] = value
    config_delta = report.get("config_delta_summary")
    if isinstance(config_delta, dict):
        for key in ("expected_final_value_accuracy", "applied_path_precision", "unexpected_applied_path_rate"):
            if key in config_delta:
                metrics[key] = config_delta[key]
    quantitative = report.get("quantitative_result_summary")
    if isinstance(quantitative, dict):
        for key in ("expected_metric_value_accuracy", "expected_metric_range_rate", "relation_pass_rate"):
            if key in quantitative:
                metrics[key] = quantitative[key]
    stage = report.get("stage_summary")
    if isinstance(stage, dict):
        for key in ("passed", "failed", "not_evaluable", "llm_contract_passed", "runtime_completed"):
            if key in stage:
                metrics[key] = stage[key]
        eval_status = stage.get("evaluation_status")
        if isinstance(eval_status, dict):
            for key in ("passed", "failed", "not_evaluable", "unsupported"):
                if key in eval_status:
                    metrics[f"evaluation.{key}"] = eval_status[key]
        nlu_boundary = stage.get("nlu_boundary")
        if isinstance(nlu_boundary, dict):
            for key in ("no_bert_prior_pass_rate", "backend_check_pass_rate"):
                if key in nlu_boundary:
                    metrics[key] = nlu_boundary[key]
        candidate_boundary = stage.get("candidate_boundary")
        if isinstance(candidate_boundary, dict):
            for key in ("cases", "requires_confirmation_cases", "uncertainty_cases", "assumption_count", "physics_rationale_count"):
                if key in candidate_boundary:
                    metrics[f"candidate_boundary.{key}"] = candidate_boundary[key]
            role_counts = candidate_boundary.get("role_counts")
            if isinstance(role_counts, dict):
                for key, value in sorted(role_counts.items()):
                    if _safe_number(value) is not None:
                        metrics[f"candidate_boundary.role.{key}"] = value
        alignment = stage.get("contract_alignment")
        if isinstance(alignment, dict):
            for key in ("applied_cases", "correction_count", "completion_count", "override_count", "risk_correction_count"):
                if key in alignment:
                    metrics[f"contract_alignment.{key}"] = alignment[key]
            categories = alignment.get("correction_categories")
            if isinstance(categories, dict):
                for key, value in sorted(categories.items()):
                    if _safe_number(value) is not None:
                        metrics[f"contract_alignment.{key}"] = value
        simulation_design = stage.get("simulation_design")
        if isinstance(simulation_design, dict):
            for key in (
                "supported_count",
                "approximation_required_count",
                "unsupported_count",
                "user_decision_required_count",
            ):
                if key in simulation_design:
                    metrics[f"simulation_design.{key}"] = simulation_design[key]
    case_results = report.get("case_results")
    if isinstance(case_results, list) and len(case_results) == 1 and isinstance(case_results[0], dict):
        actual_metrics = case_results[0].get("actual_metrics")
        if isinstance(actual_metrics, dict):
            for key, value in sorted(actual_metrics.items()):
                if _safe_number(value) is not None:
                    metrics[f"actual.{key}"] = value
    return metrics


def summarize_eval_record(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not payload:
        return None
    record = payload.get("eval_record")
    if not isinstance(record, dict) or record.get("schema_version") != EVAL_REPORT_SCHEMA_VERSION:
        return None
    report = _payload_report(payload)
    git = record.get("git") if isinstance(record.get("git"), dict) else {}
    return {
        "run_id": record.get("run_id"),
        "tool": record.get("tool"),
        "created_at_utc": record.get("created_at_utc"),
        "ok": bool(payload.get("ok") if "ok" in payload else report.get("ok")),
        "report_path": str(path),
        "git_commit": git.get("commit"),
        "git_dirty": bool(git.get("dirty")),
        "git_changed_line_count": git.get("changed_line_count"),
        "key_metrics": _compact_key_metrics(report),
        "failure_count": _failure_count(report),
        "failure_summary": _compact_failures(report),
        "model_summaries": _compact_model_summaries(report),
        "failed_models": report.get("failed_models", []),
        "hidden_fallback_models": report.get("hidden_fallback_models", []),
        "profile_mismatch_models": report.get("profile_mismatch_models", []),
    }


def _failure_count(report: dict[str, Any]) -> int:
    failed = _safe_number(report.get("failed"))
    if failed is not None:
        return int(failed)
    stage = report.get("stage_summary")
    if isinstance(stage, dict):
        failed = _safe_number(stage.get("failed"))
        if failed is not None:
            return int(failed)
    return len(_compact_failures(report))


def summarize_eval_records(outdir: Path = DEFAULT_EVAL_REPORT_DIR, *, latest_only: bool = False) -> dict[str, Any]:
    paths = sorted(Path(outdir).glob("*.json"))
    latest_payloads: list[dict[str, Any]] = []
    latest_run_ids: set[str] = set()
    for path in paths:
        if not path.name.endswith(".latest.json"):
            continue
        payload = _load_json(path)
        if isinstance(payload, dict) and payload.get("run_id"):
            latest_payloads.append(payload)
            latest_run_ids.add(str(payload["run_id"]))

    records: list[dict[str, Any]] = []
    for path in paths:
        if path.name.endswith(".latest.json"):
            continue
        record = summarize_eval_record(path)
        if record is None:
            continue
        if latest_only and str(record.get("run_id")) not in latest_run_ids:
            continue
        records.append(record)

    records.sort(key=lambda item: str(item.get("created_at_utc") or ""))
    return {
        "name": "geant4_agent_eval_record_summary",
        "outdir": str(outdir),
        "latest_only": latest_only,
        "total": len(records),
        "ok_count": sum(1 for record in records if record.get("ok")),
        "failed_count": sum(1 for record in records if not record.get("ok")),
        "latest_pointers": latest_payloads,
        "records": records,
    }


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Eval Records: {summary['total']}",
        "",
        "| created_at_utc | tool | run_id | ok | failures | key metrics |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary["records"]:
        metrics = record.get("key_metrics") if isinstance(record.get("key_metrics"), dict) else {}
        metric_text = ", ".join(f"{key}={value}" for key, value in metrics.items()) or "-"
        lines.append(
            "| "
            f"{record.get('created_at_utc') or '-'} | "
            f"{record.get('tool') or '-'} | "
            f"{record.get('run_id') or '-'} | "
            f"{record.get('ok')} | "
            f"{record.get('failure_count') or 0} | "
            f"{metric_text} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize saved Geant4Agent eval JSON records.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_EVAL_REPORT_DIR)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--fail-on-failed-record", action="store_true", help="Return non-zero if any summarized record failed.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize_eval_records(args.outdir, latest_only=bool(args.latest_only))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(_format_markdown(summary))
    return 1 if args.fail_on_failed_record and summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
