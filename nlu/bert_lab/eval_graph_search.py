from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from nlu.bert_lab.graph_search import search_candidate_graphs
from nlu.bert_lab.postprocess import merge_params


@dataclass(frozen=True)
class EvalCase:
    cid: str
    text: str
    expected: str


CASES: List[EvalCase] = [
    EvalCase("ring_en_1", "ring with n=12 radius=60 mm module_x=8 mm module_y=10 mm module_z=2 mm clearance=1 mm", "ring"),
    EvalCase("ring_en_2", "structure:ring; n=16; radius=100 mm; module_x=5 mm; module_y=5 mm; module_z=2 mm; clearance=0.5 mm", "ring"),
    EvalCase("grid_en_1", "grid nx=4 ny=5 module_x=6 mm module_y=7 mm module_z=2 mm pitch_x=8 mm pitch_y=9 mm clearance=0.5 mm", "grid"),
    EvalCase("grid_en_2", "structure:grid; nx=10; ny=2; module_x=3 mm; module_y=3 mm; module_z=1 mm; pitch_x=4 mm; pitch_y=4 mm", "grid"),
    EvalCase("nest_en_1", "nest parent_x=100 mm parent_y=80 mm parent_z=60 mm child_rmax=10 mm child_hz=12 mm clearance=1 mm", "nest"),
    EvalCase("stack_en_1", "stack stack_x=30 mm stack_y=20 mm t1=1 mm t2=2 mm t3=1.5 mm stack_clearance=0.2 mm parent_x=40 mm parent_y=30 mm parent_z=15 mm nest_clearance=0.5 mm", "stack"),
    EvalCase("shell_en_1", "shell inner_r=10 mm th1=1 mm th2=1 mm th3=2 mm hz=20 mm child_rmax=12 mm child_hz=10 mm clearance=0.5 mm", "shell"),
    EvalCase("box_en_1", "single_box module_x=1000 mm module_y=1000 mm module_z=1000 mm", "single_box"),
    EvalCase("box_zh_1", "结构是立方体，边长 1m", "single_box"),
    EvalCase("tubs_en_1", "single_tubs child_rmax=25 mm child_hz=40 mm", "single_tubs"),
    EvalCase("ring_zh_1", "环形阵列，n=12，radius=50mm，module_x=8mm，module_y=8mm，module_z=2mm，clearance=1mm", "ring"),
    EvalCase("grid_zh_1", "网格阵列 nx=5 ny=6 module_x=4mm module_y=4mm module_z=2mm pitch_x=6mm pitch_y=6mm", "grid"),
    EvalCase("nest_zh_1", "把圆柱放进盒子：parent_x=80mm parent_y=80mm parent_z=80mm child_rmax=20mm child_hz=30mm clearance=1mm", "nest"),
    EvalCase("ambiguous_1", "geometry may be ring or grid, not fixed yet", "unknown"),
    EvalCase("ambiguous_2", "candidate pattern unresolved, undecided structure", "unknown"),
]


def evaluate(min_confidence: float, seed: int, top_k: int) -> dict[str, Any]:
    rows = []
    for c in CASES:
        params, notes = merge_params(c.text, {})
        gs = search_candidate_graphs(c.text, params, min_confidence=min_confidence, seed=seed, top_k=top_k)
        ranked_labels = [x[0] for x in gs.ranked]
        topk_labels = [x[0] for x in gs.ranked[:top_k]]
        rows.append(
            {
                "cid": c.cid,
                "expected": c.expected,
                "predicted": gs.structure,
                "pred_ok": gs.structure == c.expected,
                "topk_hit": c.expected in topk_labels,
                "best_prob": gs.scores.get("best_prob"),
                "margin": gs.scores.get("margin"),
                "params": params,
                "param_notes": notes,
                "ranked": gs.ranked[:5],
                "graph_candidates": [
                    {
                        "structure": gc.structure,
                        "score": gc.score,
                        "score_breakdown": gc.score_breakdown,
                        "feasible": gc.feasible,
                        "missing_params": gc.missing_params,
                    }
                    for gc in gs.candidates
                ],
            }
        )

    n = len(rows)
    top1_acc = sum(1 for r in rows if r["pred_ok"]) / n
    topk_recall = sum(1 for r in rows if r["topk_hit"]) / n
    unknown_precision_den = sum(1 for r in rows if r["predicted"] == "unknown")
    unknown_precision = (
        sum(1 for r in rows if r["predicted"] == "unknown" and r["expected"] == "unknown") / unknown_precision_den
        if unknown_precision_den
        else 0.0
    )
    return {
        "summary": {
            "n_cases": n,
            "top1_acc": top1_acc,
            "topk_recall": topk_recall,
            "unknown_precision": unknown_precision,
            "min_confidence": min_confidence,
            "seed": seed,
            "top_k": top_k,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate-graph geometry solver.")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--out_json", default="nlu/bert_lab/data/eval/graph_search_eval.json")
    parser.add_argument("--out_md", default="docs/graph_search_eval_report.md")
    args = parser.parse_args()

    report = evaluate(args.min_confidence, args.seed, args.top_k)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    lines = [
        "# 候选图搜索几何求解评测报告",
        "",
        f"- 样本数: {s['n_cases']}",
        f"- Top-1 准确率: {s['top1_acc']:.3f}",
        f"- Top-{s['top_k']} 召回率: {s['topk_recall']:.3f}",
        f"- unknown 精确率: {s['unknown_precision']:.3f}",
        "",
        "## 失败样例",
    ]
    for r in report["results"]:
        if not r["pred_ok"]:
            lines.append(f"- `{r['cid']}`: expected={r['expected']}, predicted={r['predicted']}, ranked={r['ranked'][:3]}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()

