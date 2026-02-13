from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ui.web.server import step


@dataclass(frozen=True)
class DialogueCase:
    case_id: str
    category: str
    turns: list[str]
    note: str


CASES: list[DialogueCase] = [
    DialogueCase(
        case_id="ideal_one_turn_complete_en",
        category="ideal",
        turns=[
            (
                "structure: single_box; module_x=1000 mm; module_y=1000 mm; module_z=1000 mm. "
                "material G4_Cu. source point gamma 2 MeV position (0,0,-100) direction (0,0,1). "
                "physics list FTFP_BERT. output format root path output/result.root."
            ),
        ],
        note="All key fields provided in one English turn.",
    ),
    DialogueCase(
        case_id="realistic_two_turn_en",
        category="realistic",
        turns=[
            (
                "Build a ring detector with 12 modules, radius 60 mm, module size 8x10x2 mm, clearance 1 mm. "
                "Use G4_Si for detector."
            ),
            (
                "Use gamma source, point type, energy 1.25 MeV, position (0,0,-80), direction (0,0,1). "
                "Physics list FTFP_BERT. Output json path output/ring_case.json."
            ),
        ],
        note="Natural English split across geometry and simulation settings.",
    ),
    DialogueCase(
        case_id="realistic_three_turn_zh_with_normalization",
        category="realistic",
        turns=[
            "我想做一个1m x 1m x 1m 的铜靶，先搭个最小可运行配置。",
            "源用gamma，点源，2 MeV，position (0,0,-100)，direction (0,0,1)。",
            "物理过程用FTFP_BERT，输出root到 output/zh_case.root。",
        ],
        note="Chinese turns requiring LLM normalization before BERT parsing.",
    ),
    DialogueCase(
        case_id="stress_ambiguous_first_then_clarify",
        category="stress",
        turns=[
            "Geometry may be ring or grid, not fixed yet.",
            (
                "Finalize as grid: nx=4 ny=4 module_x=6 mm module_y=6 mm module_z=2 mm "
                "pitch_x=8 mm pitch_y=8 mm clearance=0.5 mm. Material G4_Al."
            ),
            (
                "source beam proton 150 MeV position (0,0,-300) direction (0,0,1). "
                "physics list QGSP_BERT. output format root path output/stress.root."
            ),
        ],
        note="Starts ambiguous, then resolves with explicit structure and full config.",
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
        out = step(payload)
        session_id = out.get("session_id", session_id)
        steps.append(
            {
                "turn": i,
                "user_text": user_text,
                "assistant_message": out.get("assistant_message", ""),
                "phase": out.get("phase"),
                "phase_title": out.get("phase_title"),
                "is_complete": out.get("is_complete"),
                "missing_fields": out.get("missing_fields", []),
                "inference_backend": out.get("inference_backend"),
                "normalized_text": out.get("normalized_text", ""),
                "normalization": out.get("normalization", {}),
                "normalization_degraded": out.get("normalization_degraded", False),
                "delta_paths": out.get("delta_paths", []),
                "config_min": out.get("config_min", {}),
            }
        )

    final = steps[-1] if steps else {}
    norm_used_any = any(bool(s.get("normalization", {}).get("used", False)) for s in steps)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "note": case.note,
        "turn_count": len(case.turns),
        "final_complete": bool(final.get("is_complete", False)),
        "final_missing_count": len(final.get("missing_fields", [])),
        "final_phase": final.get("phase"),
        "normalization_used_any": norm_used_any,
        "steps": steps,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    complete = sum(1 for r in results if r["final_complete"])
    norm_used = sum(1 for r in results if r["normalization_used_any"])

    per_category: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        b = per_category.setdefault(cat, {"total": 0, "complete": 0})
        b["total"] += 1
        if r["final_complete"]:
            b["complete"] += 1

    return {
        "n_cases": total,
        "n_complete": complete,
        "completion_rate": (complete / total) if total else 0.0,
        "normalization_used_cases": norm_used,
        "per_category": per_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full workflow E2E test through UI step() pipeline.")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--autofix", action="store_true")
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument("--out", default="nlu/bert_lab/data/eval/workflow_e2e_report.json")
    args = parser.parse_args()

    all_results: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    for ridx in range(args.rounds):
        round_results = [
            {"round": ridx + 1, **_run_case(c, min_confidence=args.min_confidence, autofix=args.autofix, lang=args.lang)}
            for c in CASES
        ]
        round_summaries.append({"round": ridx + 1, "summary": _summarize(round_results)})
        all_results.extend(round_results)

    report = {
        "config": {
            "rounds": args.rounds,
            "min_confidence": args.min_confidence,
            "autofix": args.autofix,
            "lang": args.lang,
            "pipeline": "user_text -> ollama router/normalizer/question -> BERT parse -> geometry synth/feasibility -> schema missing",
        },
        "round_summaries": round_summaries,
        "global_summary": _summarize(all_results),
        "results": all_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["global_summary"], indent=2, ensure_ascii=False))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
