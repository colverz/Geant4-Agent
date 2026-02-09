from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

from .dsl import parse_graph_json
from .feasibility import ErrorCode, check_feasibility
from .library import PARAM_SIGNATURE_KEYS, SKELETONS, sample_param_signature


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def run_coverage(dataset_csv: str, outdir: str) -> None:
    _ensure_outdir(outdir)
    total = 0
    expressible = 0
    reason_counts: Counter[str] = Counter()
    checked_lines: List[Dict[str, object]] = []

    with open(dataset_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            dsl_json = (row.get("dsl_json") or "").strip()
            item_id = row.get("id", "")
            scenario_text = row.get("scenario_text", "")
            entry: Dict[str, object] = {
                "id": item_id,
                "scenario_text": scenario_text,
                "dsl_json_present": bool(dsl_json),
            }
            if dsl_json:
                try:
                    graph = parse_graph_json(dsl_json)
                    expressible += 1
                    report = check_feasibility(graph)
                    entry.update(
                        {
                            "feasible": report.ok,
                            "errors": [e.code.value for e in report.errors],
                            "warnings": [w.code.value for w in report.warnings],
                        }
                    )
                except Exception as exc:
                    entry.update({"feasible": False, "errors": [str(exc)], "warnings": []})
            else:
                reason = (row.get("uncovered_reason") or "unknown").strip()
                reason_counts[reason] += 1
                entry.update({"feasible": None, "errors": [], "warnings": []})
            checked_lines.append(entry)

    summary = {
        "total": total,
        "expressible": expressible,
        "ratio": (expressible / total) if total else 0.0,
        "reason_counts": dict(reason_counts),
    }

    summary_path = os.path.join(outdir, "coverage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    checked_path = os.path.join(outdir, "coverage_checked.jsonl")
    with open(checked_path, "w") as f:
        for entry in checked_lines:
            f.write(json.dumps(entry) + "\n")


def run_feasibility_rate(outdir: str, n_samples: int, seed: int) -> None:
    _ensure_outdir(outdir)
    rng = random.Random(seed)
    summary: Dict[str, object] = {}

    for sk in SKELETONS:
        ok_count = 0
        err_counter: Counter[str] = Counter()
        for _ in range(n_samples):
            params = sk.param_sampler(rng)
            graph = sk.build_fn(params)
            report = check_feasibility(graph)
            if report.ok:
                ok_count += 1
            else:
                for e in report.errors:
                    err_counter[e.code.value] += 1
        feasible_rate = ok_count / n_samples if n_samples else 0.0
        top_errors = err_counter.most_common(3)
        summary[sk.name] = {
            "feasible_rate": feasible_rate,
            "n_samples": n_samples,
            "top_errors": top_errors,
        }

    out_path = os.path.join(outdir, "feasibility_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def run_ambiguity(outdir: str, n_param_sets: int, seed: int) -> None:
    _ensure_outdir(outdir)
    rng = random.Random(seed)

    distribution: Counter[int] = Counter()
    examples: List[Dict[str, object]] = []

    for _ in range(n_param_sets):
        params = sample_param_signature(rng)
        feasible_names: List[str] = []
        for sk in SKELETONS:
            graph = sk.build_fn(params)
            report = check_feasibility(graph)
            if report.ok:
                feasible_names.append(sk.name)
        k = len(feasible_names)
        distribution[k] += 1
        if k >= 3:
            examples.append(
                {
                    "params": {k2: params[k2] for k2 in PARAM_SIGNATURE_KEYS if k2 in params},
                    "feasible_skeletons": feasible_names,
                }
            )

    out_path = os.path.join(outdir, "ambiguity_summary.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "distribution_n_feasible_graphs": dict(distribution),
                "examples_k_ge_3": examples[:10],
            },
            f,
            indent=2,
        )
