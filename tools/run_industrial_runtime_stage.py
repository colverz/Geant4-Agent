from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tools.analyze_industrial_benchmark_failures import analyze_industrial_benchmark_report
from tools.create_industrial_golden import DEFAULT_INDUSTRIAL_GOLDEN_DIR, generate_industrial_golden
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_industrial_runtime_benchmark import (
    DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    evaluate_industrial_runtime_benchmark,
    validate_industrial_benchmark_shape,
)
from tools.industrial_runtime_compiler import compile_industrial_case_to_runtime, summarize_compile_results

INDUSTRIAL_RUNTIME_STAGE_SCHEMA_VERSION = "geant4_agent_industrial_runtime_stage.v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case.get("id")): case for case in benchmark.get("cases", []) if isinstance(case, dict)}


def _selected_case_ids(benchmark: dict[str, Any], case_ids: list[str]) -> list[str]:
    cases = _case_map(benchmark)
    if case_ids:
        return [case_id for case_id in case_ids if case_id in cases]
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    selected: list[str] = []
    for case_id, case in cases.items():
        compiled = compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        if compiled.get("status") == "compiled":
            selected.append(case_id)
    return selected


def _compile_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    runtime_defaults = benchmark.get("runtime_defaults") if isinstance(benchmark.get("runtime_defaults"), dict) else {}
    results = [
        compile_industrial_case_to_runtime(case, runtime_defaults=runtime_defaults)
        for case in benchmark.get("cases", [])
        if isinstance(case, dict)
    ]
    summary = summarize_compile_results(results)
    compiled_cases = [result["case_id"] for result in results if result.get("status") == "compiled"]
    compiled_with_gaps = [result["case_id"] for result in results if result.get("status") == "compiled_with_gaps"]
    unsupported_cases = [result["case_id"] for result in results if result.get("status") == "unsupported_capability"]
    return {
        **summary,
        "compiled_cases": compiled_cases,
        "compiled_with_gaps_cases": compiled_with_gaps,
        "unsupported_cases": unsupported_cases,
    }


def _runtime_gate(env: dict[str, str]) -> dict[str, Any]:
    enabled = str(env.get("GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK", "")).strip().lower() in {"1", "true", "yes", "on"}
    runtime_command = bool(str(env.get("GEANT4_RUNTIME_COMMAND_JSON") or env.get("GEANT4_RUNTIME_COMMAND") or "").strip())
    return {
        "env_enabled": enabled,
        "runtime_command_configured": runtime_command,
        "real_runtime_ready": enabled and runtime_command,
    }


def run_industrial_runtime_stage(
    *,
    benchmark_path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    case_ids: list[str] | None = None,
    generate_goldens: bool = False,
    run_evaluation: bool = True,
    allow_unreviewed_goldens: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    shape_report = validate_industrial_benchmark_shape(benchmark_path)
    benchmark = _load_json(benchmark_path) if shape_report["failed"] == 0 else {"cases": []}
    selected = _selected_case_ids(benchmark, case_ids or [])

    golden_reports: list[dict[str, Any]] = []
    if generate_goldens:
        for case_id in selected:
            golden_reports.append(
                generate_industrial_golden(
                    benchmark_path,
                    case_id=case_id,
                    golden_dir=golden_dir,
                    env=env_map,
                )
            )

    evaluation_report = (
        evaluate_industrial_runtime_benchmark(
            benchmark_path,
            env=env_map,
            golden_dir=golden_dir,
            allow_unreviewed_goldens=allow_unreviewed_goldens,
        )
        if run_evaluation
        else None
    )
    failure_analysis = analyze_industrial_benchmark_report(evaluation_report) if evaluation_report else None

    generated = sum(int(report.get("generated", 0) or 0) for report in golden_reports)
    blocked = sum(int(report.get("blocked", 0) or 0) for report in golden_reports)
    not_evaluable = sum(int(report.get("not_evaluable", 0) or 0) for report in golden_reports)
    compile_summary = _compile_summary(benchmark)
    stage_ok = bool(evaluation_report and evaluation_report.get("ok"))
    return {
        "schema_version": INDUSTRIAL_RUNTIME_STAGE_SCHEMA_VERSION,
        "ok": stage_ok,
        "benchmark_path": str(benchmark_path),
        "golden_dir": str(golden_dir),
        "runtime_gate": _runtime_gate(env_map),
        "shape_report": shape_report,
        "compile_summary": compile_summary,
        "selected_case_ids": selected,
        "golden_generation": {
            "requested": generate_goldens,
            "generated": generated,
            "blocked": blocked,
            "not_evaluable": not_evaluable,
            "review_required_for_official_eval": not allow_unreviewed_goldens,
            "allow_unreviewed_goldens": allow_unreviewed_goldens,
            "reports": golden_reports,
        },
        "evaluation": evaluation_report,
        "failure_analysis": failure_analysis,
        "stage_summary": _stage_summary(
            runtime_gate=_runtime_gate(env_map),
            compile_summary=compile_summary,
            golden_generated=generated,
            golden_blocked=blocked,
            golden_not_evaluable=not_evaluable,
            evaluation_report=evaluation_report,
            failure_analysis=failure_analysis,
        ),
    }


def _stage_summary(
    *,
    runtime_gate: dict[str, Any],
    compile_summary: dict[str, Any],
    golden_generated: int,
    golden_blocked: int,
    golden_not_evaluable: int,
    evaluation_report: dict[str, Any] | None,
    failure_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    eval_status = {
        "passed": int(evaluation_report.get("passed", 0) or 0) if evaluation_report else 0,
        "failed": int(evaluation_report.get("failed", 0) or 0) if evaluation_report else 0,
        "not_evaluable": int(evaluation_report.get("not_evaluable", 0) or 0) if evaluation_report else 0,
        "unsupported": int(evaluation_report.get("unsupported", 0) or 0) if evaluation_report else 0,
    }
    top_blockers = []
    if failure_analysis:
        top_blockers = [
            {
                "kind": item.get("failure_category") or item.get("kind"),
                "count": item.get("count"),
                "next_action": item.get("next_action"),
            }
            for item in (failure_analysis.get("top_blockers") or [])[:3]
        ]
        if not top_blockers:
            top_blockers = [
                {
                    "kind": item.get("kind"),
                    "count": item.get("count"),
                    "next_action": item.get("next_action"),
                }
                for item in (failure_analysis.get("compile_blockers") or [])[:3]
            ]
    return {
        "runtime_ready": bool(runtime_gate.get("real_runtime_ready")),
        "compile_status_counts": compile_summary.get("status_counts", {}),
        "compiled_case_count": len(compile_summary.get("compiled_cases") or []),
        "golden_generated": golden_generated,
        "golden_blocked": golden_blocked,
        "golden_not_evaluable": golden_not_evaluable,
        "review_required_for_official_eval": True,
        "evaluation_status": eval_status,
        "top_blockers": top_blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the industrial runtime benchmark stage workflow.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--generate-goldens", action="store_true")
    parser.add_argument(
        "--allow-unreviewed-goldens",
        action="store_true",
        help="Wiring/dev mode only. Official stage evaluation requires reviewed golden files.",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_industrial_runtime_stage(
        benchmark_path=args.benchmark,
        golden_dir=args.golden_dir,
        case_ids=list(args.case_id or []),
        generate_goldens=args.generate_goldens,
        run_evaluation=not args.no_eval,
        allow_unreviewed_goldens=args.allow_unreviewed_goldens,
    )
    output = {"ok": report["ok"], "report": report}
    if args.outdir:
        output = save_eval_output(
            output,
            outdir=args.outdir or DEFAULT_EVAL_REPORT_DIR,
            tool="industrial_runtime_stage",
            run_id=args.run_id or None,
        )
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        summary = report["stage_summary"]
        print("Industrial runtime stage summary")
        print(f"runtime_ready={summary['runtime_ready']}")
        print(f"compile_status_counts={summary['compile_status_counts']}")
        print(f"selected_case_ids={', '.join(report['selected_case_ids'])}")
        print(
            "golden_generation="
            f"generated:{summary['golden_generated']} "
            f"blocked:{summary['golden_blocked']} "
            f"not_evaluable:{summary['golden_not_evaluable']}"
        )
        print(f"review_required_for_official_eval={summary['review_required_for_official_eval']}")
        print(f"evaluation_status={summary['evaluation_status']}")
        if summary["top_blockers"]:
            print("top_blockers:")
            for item in summary["top_blockers"]:
                print(f"  - {item['kind']}: {item['count']} | {item['next_action']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
