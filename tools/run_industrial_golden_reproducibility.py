from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from tools.create_industrial_golden import (
    DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    generate_industrial_golden,
)
from tools.eval_report_io import DEFAULT_EVAL_REPORT_DIR, save_eval_output
from tools.evaluate_industrial_runtime_benchmark import DEFAULT_INDUSTRIAL_BENCHMARK_PATH
from tools.review_industrial_golden import validate_industrial_golden_for_review


INDUSTRIAL_GOLDEN_REPRODUCIBILITY_SCHEMA_VERSION = "geant4_agent_industrial_golden_reproducibility.v1"
DEFAULT_CANDIDATE_ROOT = Path("runtime_artifacts/industrial_golden_candidates")
_FINGERPRINT_KEYS = (
    "geant4_version",
    "runtime_payload_hash",
    "physics_list",
    "seed",
    "events",
    "threads",
)


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _metric_values(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return {
        str(name): metric.get("expected")
        for name, metric in sorted(metrics.items())
        if isinstance(metric, dict)
    }


def _fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("runtime_fingerprint") if isinstance(payload.get("runtime_fingerprint"), dict) else {}
    return {key: raw.get(key) for key in _FINGERPRINT_KEYS}


def compare_industrial_golden_candidates(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if len(payloads) < 2:
        return {
            "ok": False,
            "reproducible": False,
            "failure_category": "insufficient_repeats",
            "differences": [],
        }
    baseline = payloads[0]
    baseline_case_id = str(baseline.get("case_id") or "")
    baseline_metrics = _metric_values(baseline)
    baseline_fingerprint = _fingerprint(baseline)
    differences: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads[1:], start=2):
        if str(payload.get("case_id") or "") != baseline_case_id:
            differences.append({"repeat": index, "field": "case_id", "expected": baseline_case_id, "actual": payload.get("case_id")})
        current_fingerprint = _fingerprint(payload)
        for key in _FINGERPRINT_KEYS:
            if current_fingerprint.get(key) != baseline_fingerprint.get(key):
                differences.append(
                    {
                        "repeat": index,
                        "field": f"runtime_fingerprint.{key}",
                        "expected": baseline_fingerprint.get(key),
                        "actual": current_fingerprint.get(key),
                    }
                )
        current_metrics = _metric_values(payload)
        for metric in sorted(set(baseline_metrics) | set(current_metrics)):
            if current_metrics.get(metric) != baseline_metrics.get(metric):
                differences.append(
                    {
                        "repeat": index,
                        "field": f"metrics.{metric}",
                        "expected": baseline_metrics.get(metric),
                        "actual": current_metrics.get(metric),
                    }
                )
    return {
        "ok": not differences,
        "reproducible": not differences,
        "failure_category": None if not differences else "nondeterministic",
        "case_id": baseline_case_id,
        "repeat_count": len(payloads),
        "runtime_fingerprint": baseline_fingerprint,
        "metrics": baseline_metrics,
        "differences": differences,
    }


def _default_case_ids(golden_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix(".golden.json") for path in golden_dir.glob("*.golden.json"))


def run_industrial_golden_reproducibility(
    *,
    benchmark_path: Path = DEFAULT_INDUSTRIAL_BENCHMARK_PATH,
    official_golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    case_ids: list[str] | None = None,
    repeat: int = 3,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    run_id: str = "",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected = list(dict.fromkeys(case_ids or _default_case_ids(official_golden_dir)))
    actual_run_id = run_id.strip() or _utc_run_id()
    run_root = candidate_root / actual_run_id
    if repeat < 2:
        return {
            "schema_version": INDUSTRIAL_GOLDEN_REPRODUCIBILITY_SCHEMA_VERSION,
            "ok": False,
            "failure_category": "repeat_must_be_at_least_two",
            "repeat": repeat,
            "selected_case_ids": selected,
        }
    env_map = dict(os.environ if env is None else env)
    generation_reports: list[dict[str, Any]] = []
    candidate_paths: dict[str, list[Path]] = {case_id: [] for case_id in selected}
    for repeat_index in range(1, repeat + 1):
        repeat_dir = run_root / f"repeat-{repeat_index:02d}"
        for case_id in selected:
            report = generate_industrial_golden(
                benchmark_path,
                case_id=case_id,
                golden_dir=repeat_dir,
                env=env_map,
                force=True,
            )
            generation_reports.append({"repeat": repeat_index, "case_id": case_id, "report": report})
            candidate_path = repeat_dir / f"{case_id}.golden.json"
            if report.get("generated") == 1 and candidate_path.exists():
                candidate_paths[case_id].append(candidate_path)

    case_results: list[dict[str, Any]] = []
    for case_id in selected:
        paths = candidate_paths[case_id]
        payloads = [_load_json(path) for path in paths]
        validation = [validate_industrial_golden_for_review(payload) for payload in payloads]
        if len(paths) != repeat or any(not item.get("ok") for item in validation):
            case_results.append(
                {
                    "case_id": case_id,
                    "ok": False,
                    "reproducible": False,
                    "failure_category": "candidate_generation_incomplete",
                    "candidate_paths": [str(path) for path in paths],
                    "validation": validation,
                }
            )
            continue
        comparison = compare_industrial_golden_candidates(payloads)
        case_results.append(
            {
                **comparison,
                "candidate_paths": [str(path) for path in paths],
                "review_candidate": str(paths[0]),
                "validation": validation,
            }
        )
    report = {
        "schema_version": INDUSTRIAL_GOLDEN_REPRODUCIBILITY_SCHEMA_VERSION,
        "ok": bool(case_results) and all(bool(item.get("ok")) for item in case_results),
        "run_id": actual_run_id,
        "benchmark_path": str(benchmark_path),
        "official_golden_dir": str(official_golden_dir),
        "candidate_root": str(run_root),
        "repeat": repeat,
        "selected_case_ids": selected,
        "case_results": case_results,
        "generation_reports": generation_reports,
        "summary": {
            "total": len(case_results),
            "reproducible": sum(1 for item in case_results if item.get("reproducible")),
            "failed": sum(1 for item in case_results if not item.get("ok")),
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated real-Geant4 golden candidates and verify exact reproducibility.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_INDUSTRIAL_BENCHMARK_PATH)
    parser.add_argument("--official-golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_industrial_golden_reproducibility(
        benchmark_path=args.benchmark,
        official_golden_dir=args.official_golden_dir,
        case_ids=list(args.case_id or []),
        repeat=args.repeat,
        candidate_root=args.candidate_root,
        run_id=args.run_id,
    )
    output: dict[str, Any] = {"ok": report.get("ok"), "report": report}
    if args.outdir:
        output = save_eval_output(
            output,
            outdir=args.outdir or DEFAULT_EVAL_REPORT_DIR,
            tool="industrial_golden_reproducibility",
            run_id=report.get("run_id"),
        )
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        summary = report.get("summary") or {}
        print(f"industrial_golden_reproducibility: reproducible={summary.get('reproducible', 0)} failed={summary.get('failed', 0)}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
