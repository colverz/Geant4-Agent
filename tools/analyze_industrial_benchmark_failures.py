from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from tools.evaluate_industrial_runtime_benchmark import (
    DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    evaluate_industrial_runtime_benchmark,
)


FAILURE_ACTIONS = {
    "runtime_unavailable": "Configure GEANT4_INDUSTRIAL_RUNTIME_BENCHMARK=1 and a real GEANT4_RUNTIME_COMMAND_JSON/GEANT4_RUNTIME_COMMAND.",
    "runtime_unavailable_and_missing_golden": "Configure real runtime first, then run industrial golden generation for every official case.",
    "missing_golden": "Run tools/create_industrial_golden.py against a pinned real Geant4 runtime and review generated golden files.",
    "unreviewed_golden": "Review generated golden files and set review.status to reviewed before official evaluation.",
    "spec_compile_error": "Implement deterministic scenario-to-runtime compilation for this benchmark family; do not ask the LLM to judge physics.",
    "runtime_error": "Inspect real Geant4 artifacts, wrapper stderr, and generated runtime payload.",
    "missing_metric": "Extend runtime result extraction so every required golden metric is present in structured output.",
    "metric_mismatch": "Compare runtime fingerprint and metric diffs; update code or regenerate goldens only after review.",
    "llm_config_error": "Tighten prompt profile, validator, or config repair before runtime launch.",
    "result_qa_hallucination": "Constrain result explanation to actual RuntimeSmokeReport and metric diff fields.",
    "unsupported_capability": "Keep as explicit capability gap until the project supports this industrial feature.",
    "shape": "Fix benchmark manifest schema before evaluating behavior.",
    "golden_exists": "Use --force only when intentionally refreshing reviewed golden metrics.",
}


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
        return payload["report"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("report_json_must_be_object")


def _case_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("case_results")
    return rows if isinstance(rows, list) else []


def analyze_industrial_benchmark_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in _case_rows(report) if isinstance(row, dict)]
    status_counts = Counter(str(row.get("status")) for row in rows)
    failure_categories = Counter(
        str(row.get("failure_category")) for row in rows if row.get("failure_category")
    )
    domain_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    domain_failures: dict[str, Counter[str]] = defaultdict(Counter)
    representative_cases: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        domain = str(row.get("domain") or "<unknown>")
        status = str(row.get("status") or "<unknown>")
        category = str(row.get("failure_category") or "<none>")
        domain_statuses[domain][status] += 1
        domain_failures[domain][category] += 1
        if len(representative_cases[category]) < 5:
            representative_cases[category].append(str(row.get("id")))

    blocker_order = [
        "shape",
        "runtime_unavailable_and_missing_golden",
        "runtime_unavailable",
        "missing_golden",
        "spec_compile_error",
        "runtime_error",
        "missing_metric",
        "metric_mismatch",
        "llm_config_error",
        "result_qa_hallucination",
        "unsupported_capability",
    ]
    top_blockers: list[dict[str, Any]] = []
    for category in blocker_order:
        count = failure_categories.get(category, 0)
        if not count:
            continue
        top_blockers.append(
            {
                "failure_category": category,
                "count": count,
                "example_cases": representative_cases.get(category, []),
                "next_action": FAILURE_ACTIONS.get(category, "Investigate this failure category."),
            }
        )
    for category, count in sorted(failure_categories.items()):
        if category in blocker_order:
            continue
        top_blockers.append(
            {
                "failure_category": category,
                "count": count,
                "example_cases": representative_cases.get(category, []),
                "next_action": FAILURE_ACTIONS.get(category, "Investigate this failure category."),
            }
        )

    next_actions = [item["next_action"] for item in top_blockers[:3]]
    compile_summary = _compile_summary(report)
    compile_blockers = _compile_blockers(compile_summary)
    return {
        "name": "industrial_benchmark_failure_analysis",
        "ok": not top_blockers and bool(rows),
        "source_report": report.get("name"),
        "total": len(rows),
        "passed": status_counts.get("passed", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_categories": dict(sorted(failure_categories.items())),
        "domain_statuses": {domain: dict(sorted(counts.items())) for domain, counts in sorted(domain_statuses.items())},
        "domain_failures": {
            domain: dict(sorted(counts.items())) for domain, counts in sorted(domain_failures.items())
        },
        "compile_summary": compile_summary,
        "compile_blockers": compile_blockers,
        "top_blockers": top_blockers,
        "next_actions": next_actions,
    }


def _compile_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    compile_summary = summary.get("compile_summary") if isinstance(summary.get("compile_summary"), dict) else {}
    return compile_summary


def _compile_blockers(compile_summary: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    status_counts = compile_summary.get("status_counts") if isinstance(compile_summary.get("status_counts"), dict) else {}
    if status_counts:
        for status in ("unsupported_capability", "compiled_with_gaps"):
            count = int(status_counts.get(status, 0) or 0)
            if count:
                action = (
                    "Implement missing geometry/source/scoring capabilities before these cases can produce official goldens."
                    if status == "unsupported_capability"
                    else "Extend structured metric extraction or derived metric calculation for compiled cases with metric gaps."
                )
                blockers.append({"kind": status, "count": count, "next_action": action})
    features = compile_summary.get("unsupported_features") if isinstance(compile_summary.get("unsupported_features"), dict) else {}
    metrics = compile_summary.get("unsupported_metrics") if isinstance(compile_summary.get("unsupported_metrics"), dict) else {}
    for name, count in sorted(features.items(), key=lambda item: (-int(item[1]), str(item[0])))[:5]:
        blockers.append(
            {
                "kind": "unsupported_feature",
                "name": str(name),
                "count": int(count),
                "next_action": "Add this capability to the runtime payload/compiler or keep the benchmark case explicitly unsupported.",
            }
        )
    for name, count in sorted(metrics.items(), key=lambda item: (-int(item[1]), str(item[0])))[:5]:
        blockers.append(
            {
                "kind": "unsupported_metric",
                "name": str(name),
                "count": int(count),
                "next_action": "Add this metric to structured Geant4 result extraction or deterministic derived metric calculation.",
            }
        )
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze industrial benchmark failure categories and next actions.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = _load_report(args.report) if args.report else evaluate_industrial_runtime_benchmark(args.benchmark)
    analysis = analyze_industrial_benchmark_report(report)
    output = {"ok": analysis["ok"], "analysis": analysis}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(
            f"{analysis['name']}: passed={analysis['passed']} "
            f"total={analysis['total']} blockers={len(analysis['top_blockers'])}"
        )
        for blocker in analysis["top_blockers"]:
            print(f"  {blocker['failure_category']}: {blocker['count']} cases")
            print(f"    next: {blocker['next_action']}")
    return 0 if analysis["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
