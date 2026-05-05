from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ui.web.server import solve


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    text: str
    expected_structure: str
    expected_feasible: bool | None
    note: str


CASES: list[Case] = [
    Case(
        case_id="ideal_polycone_explicit",
        category="ideal",
        text=(
            "structure: single_polycone; z1=-20 mm; z2=0 mm; z3=20 mm; "
            "r1=8 mm; r2=12 mm; r3=10 mm; material G4_Al; source type point; "
            "particle gamma; physics list FTFP_BERT; output json"
        ),
        expected_structure="single_polycone",
        expected_feasible=True,
        note="Fully specified polycone with simulation fields.",
    ),
    Case(
        case_id="ideal_cuttubs_explicit",
        category="ideal",
        text=(
            "structure: single_cuttubs; rmax=15 mm; hz=20 mm; tilt_x=8; tilt_y=4; "
            "material G4_Si; source type point; particle gamma; physics list FTFP_BERT; output root"
        ),
        expected_structure="single_cuttubs",
        expected_feasible=True,
        note="Fully specified cuttubs with bounded tilts.",
    ),
    Case(
        case_id="ideal_boolean_explicit",
        category="ideal",
        text=(
            "structure: boolean; bool_a_x=30 mm; bool_a_y=20 mm; bool_a_z=10 mm; "
            "bool_b_x=12 mm; bool_b_y=8 mm; bool_b_z=6 mm; material G4_Cu; "
            "source type point; particle gamma; physics list FTFP_BERT; output root"
        ),
        expected_structure="boolean",
        expected_feasible=True,
        note="Boolean placeholder via union of two boxes.",
    ),
    Case(
        case_id="realistic_polycone_natural",
        category="realistic",
        text=(
            "Build a polycone target with z1 -25 mm, z2 0 mm, z3 25 mm and radii "
            "r1 7 mm, r2 11 mm, r3 9 mm. Use G4_WATER and gamma source."
        ),
        expected_structure="single_polycone",
        expected_feasible=True,
        note="Natural phrasing with key-value slots.",
    ),
    Case(
        case_id="realistic_cuttubs_natural",
        category="realistic",
        text=(
            "Please model a cuttubs crystal: rmax 18 mm, hz 22 mm, tilt_x 6, tilt_y 5, "
            "material G4_Si, source point gamma."
        ),
        expected_structure="single_cuttubs",
        expected_feasible=True,
        note="Natural request for single cuttubs.",
    ),
    Case(
        case_id="realistic_boolean_natural",
        category="realistic",
        text=(
            "Create a boolean union of two boxes. box A is 24x16x10 mm and box B is 10x10x8 mm. "
            "Use copper and output root."
        ),
        expected_structure="boolean",
        expected_feasible=True,
        note="Natural request for boolean placeholder path.",
    ),
    Case(
        case_id="stress_ambiguous_layout",
        category="stress",
        text="Geometry may be ring or grid, arrangement remains undecided for now.",
        expected_structure="unknown",
        expected_feasible=None,
        note="Should be rejected as unknown/ambiguous.",
    ),
    Case(
        case_id="stress_no_geometry",
        category="stress",
        text="Use gamma source with 2 MeV and FTFP_BERT, output ROOT only.",
        expected_structure="unknown",
        expected_feasible=None,
        note="No geometry cues.",
    ),
    Case(
        case_id="stress_infeasible_cuttubs",
        category="stress",
        text=(
            "structure: single_cuttubs; rmax=-5 mm; hz=20 mm; tilt_x=12; tilt_y=10; "
            "material G4_Si"
        ),
        expected_structure="single_cuttubs",
        expected_feasible=False,
        note="Negative rmax should fail feasibility.",
    ),
]


def _run_case(
    case: Case,
    *,
    min_conf: float,
    autofix: bool,
    normalize_input: bool,
    llm_fill_missing: bool,
) -> dict[str, Any]:
    out = solve(
        {
            "text": case.text,
            "min_confidence": min_conf,
            "normalize_input": normalize_input,
            "llm_fill_missing": llm_fill_missing,
            "autofix": autofix,
        }
    )

    predicted = str(out.get("structure", "unknown"))
    synthesis = out.get("synthesis") if isinstance(out.get("synthesis"), dict) else {}
    feasible = synthesis.get("feasible") if synthesis else None
    missing = synthesis.get("missing_params", []) if synthesis else []

    return {
        "case_id": case.case_id,
        "category": case.category,
        "note": case.note,
        "text": case.text,
        "expected_structure": case.expected_structure,
        "predicted_structure": predicted,
        "structure_ok": predicted == case.expected_structure,
        "expected_feasible": case.expected_feasible,
        "predicted_feasible": feasible,
        "feasible_ok": (case.expected_feasible == feasible) if case.expected_feasible is not None else None,
        "missing_param_count": len(missing),
        "backend": out.get("inference_backend", ""),
        "top_scores": out.get("scores", {}),
        "errors": synthesis.get("errors", []) if synthesis else [],
        "warnings": synthesis.get("warnings", []) if synthesis else [],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    structure_ok = sum(1 for r in rows if r["structure_ok"])
    feas_rows = [r for r in rows if r["expected_feasible"] is not None]
    feasible_ok = sum(1 for r in feas_rows if r["feasible_ok"] is True)

    by_cat: dict[str, dict[str, Any]] = {}
    for r in rows:
        b = by_cat.setdefault(
            r["category"],
            {"total": 0, "structure_ok": 0, "feasible_eval_total": 0, "feasible_ok": 0},
        )
        b["total"] += 1
        if r["structure_ok"]:
            b["structure_ok"] += 1
        if r["expected_feasible"] is not None:
            b["feasible_eval_total"] += 1
            if r["feasible_ok"] is True:
                b["feasible_ok"] += 1

    for b in by_cat.values():
        b["structure_acc"] = (b["structure_ok"] / b["total"]) if b["total"] else 0.0
        b["feasible_acc"] = (b["feasible_ok"] / b["feasible_eval_total"]) if b["feasible_eval_total"] else None

    return {
        "n_cases": total,
        "structure_acc": (structure_ok / total) if total else 0.0,
        "feasible_acc": (feasible_ok / len(feas_rows)) if feas_rows else None,
        "per_category": by_cat,
    }


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    normalize_input: bool,
    llm_fill_missing: bool,
) -> None:
    lines: list[str] = []
    lines.append("# 三类端到端回归报告")
    lines.append("")
    lines.append(f"- 日期: {date.today().isoformat()}")
    lines.append("- 流程: user text -> semantic parse -> candidate graph search -> feasibility")
    lines.append(
        "- 说明: "
        f"`normalize_input={'true' if normalize_input else 'false'}`, "
        f"`llm_fill_missing={'true' if llm_fill_missing else 'false'}`"
    )
    lines.append("")
    lines.append("## 总体结果")
    lines.append("")
    lines.append(f"- 样例数: {summary['n_cases']}")
    lines.append(f"- 结构识别准确率: {summary['structure_acc']:.3f}")
    if summary.get("feasible_acc") is not None:
        lines.append(f"- 可行性判定准确率: {summary['feasible_acc']:.3f}")
    lines.append("")
    lines.append("## 分类结果")
    lines.append("")
    for cat, v in summary["per_category"].items():
        lines.append(f"### {cat}")
        lines.append(f"- 样例数: {v['total']}")
        lines.append(f"- 结构识别准确率: {v['structure_acc']:.3f}")
        if v.get("feasible_acc") is not None:
            lines.append(f"- 可行性判定准确率: {v['feasible_acc']:.3f}")
        lines.append("")

    lines.append("## 逐例结果")
    lines.append("")
    lines.append("| case_id | category | expected_structure | predicted_structure | structure_ok | expected_feasible | predicted_feasible | feasible_ok |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['case_id']} | {r['category']} | {r['expected_structure']} | {r['predicted_structure']} | {r['structure_ok']} | {r['expected_feasible']} | {r['predicted_feasible']} | {r['feasible_ok']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 3-class end-to-end regression and emit JSON/Markdown reports.")
    ap.add_argument("--out_json", default="nlu/training/bert_lab/data/eval/regression_3class_results.json")
    ap.add_argument("--out_md", default=f"docs/regression_3class_report_{date.today().isoformat()}.md")
    ap.add_argument("--min_confidence", type=float, default=0.6)
    ap.add_argument("--autofix", action="store_true")
    ap.add_argument("--normalize_input", action="store_true")
    ap.add_argument("--llm_fill_missing", action="store_true")
    args = ap.parse_args()

    rows = [
        _run_case(
            c,
            min_conf=args.min_confidence,
            autofix=args.autofix,
            normalize_input=args.normalize_input,
            llm_fill_missing=args.llm_fill_missing,
        )
        for c in CASES
    ]
    summary = _summary(rows)

    payload = {
        "config": {
            "min_confidence": args.min_confidence,
            "autofix": args.autofix,
            "normalize_input": args.normalize_input,
            "llm_fill_missing": args.llm_fill_missing,
        },
        "summary": summary,
        "results": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        Path(args.out_md),
        summary,
        rows,
        normalize_input=args.normalize_input,
        llm_fill_missing=args.llm_fill_missing,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
