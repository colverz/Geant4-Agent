from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_llm_scenario_parsing import DEFAULT_SCENARIO_CASEBANK, evaluate_llm_scenario_parsing

MODEL_MATRIX_SCHEMA_VERSION = "geant4_agent_llm_scenario_model_matrix.v1"


def _model_label(model: str, *, live_llm: bool) -> str:
    value = str(model or "").strip()
    if value:
        return value
    return "<config_model>" if live_llm else "offline_v2"


def _summarize_report(
    report: dict[str, Any],
    *,
    model: str,
    live_llm: bool,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    live_summary = report.get("live_summary") if isinstance(report.get("live_summary"), dict) else {}
    total = int(report.get("total") or 0)
    fallback_count = int(live_summary.get("fallback_count") or 0)
    profile_mismatch_count = int(live_summary.get("profile_mismatch_count") or 0)
    llm_used_count = int(live_summary.get("llm_used_count") or 0)
    return {
        "model": _model_label(model, live_llm=live_llm),
        "mode": report.get("mode"),
        "total": total,
        "passed": int(report.get("passed") or 0),
        "failed": int(report.get("failed") or 0),
        "accuracy": float(report.get("accuracy") or 0.0),
        "meets_threshold": bool(report.get("meets_threshold")),
        "llm_used_count": llm_used_count,
        "llm_usage_rate": round(llm_used_count / total, 6) if total else None,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 6) if total else None,
        "profile_mismatch_count": profile_mismatch_count,
        "profile_mismatch_rate": round(profile_mismatch_count / total, 6) if total else None,
        "elapsed_seconds": round(elapsed_seconds, 6) if elapsed_seconds is not None else None,
        "seconds_per_case": round(elapsed_seconds / total, 6) if elapsed_seconds is not None and total else None,
        "lang_counts": live_summary.get("lang_counts", {}),
        "slot_prompt_profiles": live_summary.get("slot_prompt_profiles", {}),
        "semantic_prompt_profiles": live_summary.get("semantic_prompt_profiles", {}),
    }


def _rank_models(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        summaries,
        key=lambda item: (
            float(item.get("accuracy") or 0.0),
            -int(item.get("fallback_count") or 0),
            -int(item.get("profile_mismatch_count") or 0),
        ),
        reverse=True,
    )
    return [
        {
            "rank": index + 1,
            "model": item["model"],
            "accuracy": item["accuracy"],
            "fallback_count": item["fallback_count"],
            "profile_mismatch_count": item["profile_mismatch_count"],
            "meets_threshold": item["meets_threshold"],
        }
        for index, item in enumerate(ranked)
    ]


def evaluate_llm_scenario_model_matrix(
    *,
    casebank_path: Path = DEFAULT_SCENARIO_CASEBANK,
    live_llm: bool = False,
    llm_config_path: str = "",
    models: list[str] | None = None,
    min_accuracy: float = 1.0,
    max_cases: int | None = None,
) -> dict[str, Any]:
    selected_models = list(models or [""])
    reports: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    matrix_started = time.perf_counter()
    for model in selected_models:
        started = time.perf_counter()
        report = evaluate_llm_scenario_parsing(
            casebank_path,
            live_llm=live_llm,
            llm_config_path=llm_config_path,
            min_accuracy=min_accuracy,
            max_cases=max_cases,
            model_override=model,
        )
        elapsed_seconds = time.perf_counter() - started
        reports.append(report)
        summaries.append(
            _summarize_report(
                report,
                model=model,
                live_llm=live_llm,
                elapsed_seconds=elapsed_seconds,
            )
        )

    failed_reports = [summary for summary in summaries if not summary["meets_threshold"]]
    hidden_fallback_reports = [
        summary
        for summary in summaries
        if live_llm and int(summary["fallback_count"] or 0) > 0
    ]
    profile_mismatch_reports = [summary for summary in summaries if int(summary["profile_mismatch_count"] or 0) > 0]
    ranking = _rank_models(summaries)
    return {
        "name": "llm_scenario_model_matrix",
        "schema_version": MODEL_MATRIX_SCHEMA_VERSION,
        "mode": "live_llm" if live_llm else "offline_v2",
        "casebank": str(casebank_path),
        "min_accuracy": min_accuracy,
        "max_cases": max_cases,
        "elapsed_seconds": round(time.perf_counter() - matrix_started, 6),
        "ok": not failed_reports and not hidden_fallback_reports and not profile_mismatch_reports,
        "model_count": len(summaries),
        "model_summaries": summaries,
        "ranking": ranking,
        "best_model": ranking[0]["model"] if ranking else None,
        "failed_models": [summary["model"] for summary in failed_reports],
        "hidden_fallback_models": [summary["model"] for summary in hidden_fallback_reports],
        "profile_mismatch_models": [summary["model"] for summary in profile_mismatch_reports],
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare one or more models on the LLM scenario parsing benchmark.")
    parser.add_argument("--casebank", type=Path, default=DEFAULT_SCENARIO_CASEBANK)
    parser.add_argument("--live-llm", action="store_true", help="Opt in to configured live LLM calls.")
    parser.add_argument("--llm-config", default=os.environ.get("GEANT4_LLM_CONFIG", ""))
    parser.add_argument("--model", action="append", default=[], help="Model override. Can be provided multiple times.")
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=None, help="Optional directory for a full JSON eval record.")
    parser.add_argument("--run-id", default="", help="Optional stable run id for saved eval records.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    env_live = os.environ.get("GEANT4_LLM_SCENARIO", "").strip().lower() in {"1", "true", "yes", "on"}
    report = evaluate_llm_scenario_model_matrix(
        casebank_path=args.casebank,
        live_llm=bool(args.live_llm or env_live),
        llm_config_path=str(args.llm_config or ""),
        models=list(args.model or []) or None,
        min_accuracy=float(args.min_accuracy),
        max_cases=args.max_cases or None,
    )
    if args.outdir:
        report = save_eval_output(
            report,
            outdir=args.outdir or DEFAULT_EVAL_REPORT_DIR,
            tool=report["name"],
            run_id=args.run_id or None,
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{report['name']}[{report['mode']}]: ok={report['ok']} models={report['model_count']}")
        for summary in report["model_summaries"]:
            print(
                "  "
                f"{summary['model']}: accuracy={summary['accuracy']:.3f} "
                f"fallback={summary['fallback_count']} profile_mismatch={summary['profile_mismatch_count']} "
                f"elapsed={summary.get('elapsed_seconds')}s"
            )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
