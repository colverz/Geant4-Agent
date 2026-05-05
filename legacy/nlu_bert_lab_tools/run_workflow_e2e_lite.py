from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

from ui.web.server import step


@dataclass(frozen=True)
class DialogueCase:
    case_id: str
    category: str
    turns: list[str]
    expect_complete: bool
    note: str


CASES: list[DialogueCase] = [
    DialogueCase(
        case_id="lite_complete_one_turn",
        category="ideal",
        turns=[
            (
                "Build a ring detector with 12 modules, radius 60 mm, module size 8x10x2 mm, clearance 1 mm. "
                "Material G4_Si. Source point gamma 1.25 MeV at (0,0,-80) to (0,0,1). "
                "Physics list FTFP_BERT. Output root to output/ring.root."
            )
        ],
        expect_complete=True,
        note="One-shot complete request.",
    ),
    DialogueCase(
        case_id="lite_two_turn_fill_source",
        category="realistic",
        turns=[
            "I need a 1m x 1m x 1m copper box target. Use FTFP_BERT and output root.",
            "Source point gamma with energy 2 MeV, position (0,0,-100), direction (0,0,1).",
        ],
        expect_complete=True,
        note="First turn lacks source kinematics, second turn fills them.",
    ),
    DialogueCase(
        case_id="lite_no_geometry_should_incomplete",
        category="stress",
        turns=[
            "Use gamma source with 2 MeV and FTFP_BERT, output root.",
        ],
        expect_complete=False,
        note="No geometry provided; should remain incomplete.",
    ),
]


def _run_case(case: DialogueCase, *, min_confidence: float, autofix: bool, lang: str) -> dict[str, Any]:
    session_id = None
    steps: list[dict[str, Any]] = []
    for i, user_text in enumerate(case.turns, start=1):
        payload = {
            "session_id": session_id,
            "text": user_text,
            "llm_router": True,
            "llm_question": True,
            "normalize_input": True,
            "min_confidence": min_confidence,
            "autofix": autofix,
            "lang": lang,
        }
        t0 = perf_counter()
        out = step(payload)
        dt = perf_counter() - t0
        session_id = out.get("session_id", session_id)
        steps.append(
            {
                "turn": i,
                "latency_s": round(dt, 3),
                "user_text": user_text,
                "assistant_message": out.get("assistant_message", ""),
                "phase": out.get("phase"),
                "is_complete": out.get("is_complete"),
                "missing_fields": out.get("missing_fields", []),
                "normalized_text": out.get("normalized_text", ""),
                "inference_backend": out.get("inference_backend", ""),
            }
        )

    final = steps[-1] if steps else {}
    actual_complete = bool(final.get("is_complete", False))
    return {
        "case_id": case.case_id,
        "category": case.category,
        "note": case.note,
        "expect_complete": case.expect_complete,
        "actual_complete": actual_complete,
        "ok": actual_complete == case.expect_complete,
        "turn_count": len(case.turns),
        "total_latency_s": round(sum(s["latency_s"] for s in steps), 3),
        "steps": steps,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r["ok"])
    avg_latency = sum(r["total_latency_s"] for r in results) / max(1, total)
    per_category: dict[str, dict[str, float]] = {}
    for r in results:
        cat = r["category"]
        bucket = per_category.setdefault(cat, {"total": 0, "ok": 0})
        bucket["total"] += 1
        if r["ok"]:
            bucket["ok"] += 1
    for b in per_category.values():
        b["acc"] = b["ok"] / b["total"] if b["total"] else 0.0
    return {
        "n_cases": total,
        "workflow_acc": ok / total if total else 0.0,
        "avg_case_latency_s": round(avg_latency, 3),
        "per_category": per_category,
    }


def _write_markdown(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Workflow E2E Lite Report")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append("- Mode: full workflow (`llm_router=true`, `llm_question=true`, `normalize_input=true`)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases: {summary['n_cases']}")
    lines.append(f"- Workflow accuracy: {summary['workflow_acc']:.3f}")
    lines.append(f"- Average case latency (s): {summary['avg_case_latency_s']:.3f}")
    lines.append("")
    lines.append("## Per Case")
    lines.append("")
    lines.append("| case_id | category | expect_complete | actual_complete | ok | turns | latency_s |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['case_id']} | {r['category']} | {r['expect_complete']} | {r['actual_complete']} | "
            f"{r['ok']} | {r['turn_count']} | {r['total_latency_s']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reduced full-workflow E2E benchmark.")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--autofix", action="store_true")
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument("--out_json", default="nlu/training/bert_lab/data/eval/workflow_e2e_lite_report.json")
    parser.add_argument("--out_md", default=f"docs/workflow_e2e_lite_report_{date.today().isoformat()}.md")
    args = parser.parse_args()

    results = [_run_case(c, min_confidence=args.min_confidence, autofix=args.autofix, lang=args.lang) for c in CASES]
    summary = _summarize(results)

    payload = {
        "config": {
            "min_confidence": args.min_confidence,
            "autofix": args.autofix,
            "lang": args.lang,
            "workflow_flags": {
                "llm_router": True,
                "llm_question": True,
                "normalize_input": True,
            },
        },
        "summary": summary,
        "results": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(Path(args.out_md), summary, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
