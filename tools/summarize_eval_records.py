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
        "failure_count": int(report.get("failed") or 0) if _safe_number(report.get("failed")) is not None else len(_compact_failures(report)),
        "failure_summary": _compact_failures(report),
        "model_summaries": _compact_model_summaries(report),
        "failed_models": report.get("failed_models", []),
        "hidden_fallback_models": report.get("hidden_fallback_models", []),
        "profile_mismatch_models": report.get("profile_mismatch_models", []),
    }


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
