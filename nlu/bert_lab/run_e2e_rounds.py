from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from builder.geometry.synthesize import synthesize_from_params
from nlu.bert_lab.infer import extract_params, predict_structure
from nlu.bert_lab.llm_bridge import build_missing_params_prompt
from nlu.bert_lab.ollama_client import chat, extract_json
from nlu.bert_lab.postprocess import merge_params


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"


def _default_structure_model() -> str:
    for name in ["structure_controlled_v4c_e1", "structure_controlled_v3_e1", "structure_controlled_smoke", "structure_opt_v3"]:
        p = MODELS_DIR / name
        if (p / "config.json").exists():
            return str(p)
    return "nlu/bert_lab/models/structure_controlled_v4c_e1"


def _default_ner_model() -> str:
    p = MODELS_DIR / "ner"
    if (p / "config.json").exists():
        return str(p)
    return "nlu/bert_lab/models/ner"


@dataclass(frozen=True)
class Scenario:
    sid: str
    category: str
    text: str
    expected_structure: str
    note: str


SCENARIOS: list[Scenario] = [
    Scenario(
        sid="ideal_ring_complete",
        category="ideal",
        text="structure: ring; n=12; radius=60 mm; module_x=8 mm; module_y=10 mm; module_z=2 mm; clearance=1 mm",
        expected_structure="ring",
        note="Fully specified normalized ring request.",
    ),
    Scenario(
        sid="ideal_grid_complete",
        category="ideal",
        text="structure: grid; nx=6; ny=4; module_x=5 mm; module_y=6 mm; module_z=2 mm; pitch_x=7 mm; pitch_y=8 mm; clearance=0.5 mm",
        expected_structure="grid",
        note="Fully specified normalized grid request.",
    ),
    Scenario(
        sid="ideal_single_box_complete",
        category="ideal",
        text="structure: single_box; module_x=1000 mm; module_y=1000 mm; module_z=1000 mm",
        expected_structure="single_box",
        note="1x1x1 m cube normalized for BERT input.",
    ),
    Scenario(
        sid="real_ring_partial",
        category="realistic",
        text="I need a ring detector with 12 modules at radius 40 mm, each module is 8 by 10 mm, keep 1 mm clearance.",
        expected_structure="ring",
        note="Missing module_z should trigger fill path.",
    ),
    Scenario(
        sid="real_stack_partial",
        category="realistic",
        text="Please build a three-layer stack in a box. stack_x 30 mm stack_y 20 mm t1 1 mm t2 2 mm t3 1.5 mm, parent 40x30x15 mm, clearances 0.5 mm.",
        expected_structure="stack",
        note="Natural text with compact numeric slots.",
    ),
    Scenario(
        sid="real_nest_partial",
        category="realistic",
        text="Put a cylindrical target inside a box: child_rmax 40 mm, child_hz 50 mm, parent 70x70x60 mm, clearance 1 mm.",
        expected_structure="nest",
        note="Potentially infeasible unless autofix grows parent.",
    ),
    Scenario(
        sid="stress_ambiguous",
        category="stress",
        text="Candidate pattern may be ring or grid depending on constraints; arrangement not fixed yet, undecided now.",
        expected_structure="unknown",
        note="Should be rejected as unknown intent.",
    ),
    Scenario(
        sid="stress_non_geometry_text",
        category="stress",
        text="Use gamma source and copper target, run physics list FTFP_BERT and save ROOT output.",
        expected_structure="unknown",
        note="No direct geometry structure cues.",
    ),
]


def _run_one(
    scenario: Scenario,
    seed: int,
    structure_model: str,
    ner_model: str,
    device: str,
    min_confidence: float,
    autofix: bool,
    llm_fill_missing: bool,
    ollama_config: str,
) -> dict[str, Any]:
    structure, scores, ranked = predict_structure(
        scenario.text,
        structure_model,
        device=device,
        min_confidence=min_confidence,
    )
    params = extract_params(scenario.text, ner_model, device=device)
    params, notes = merge_params(scenario.text, params)
    synthesis = None
    synthesis_filled = None
    llm_fill = None
    missing_prompt = ""

    if structure != "unknown":
        synthesis = synthesize_from_params(structure, params, seed, apply_autofix=autofix)
        missing = synthesis.get("missing_params", [])
        missing_prompt = build_missing_params_prompt(structure, missing, fmt="json_schema")
        if llm_fill_missing and missing:
            resp = chat(missing_prompt, config_path=ollama_config, temperature=0.2)
            parsed = extract_json(resp.get("response", ""))
            if isinstance(parsed, dict):
                merged = dict(params)
                merged.update(parsed)
                synthesis_filled = synthesize_from_params(structure, merged, seed, apply_autofix=autofix)
                llm_fill = {"ok": True, "used_params": parsed}
            else:
                llm_fill = {"ok": False, "raw": resp.get("response", "")}

    predicted_ok = structure == scenario.expected_structure
    feasible = None
    filled_feasible = None
    missing_count = None
    if isinstance(synthesis, dict):
        feasible = bool(synthesis.get("feasible"))
        missing_count = len(synthesis.get("missing_params", []))
    if isinstance(synthesis_filled, dict):
        filled_feasible = bool(synthesis_filled.get("feasible"))

    return {
        "scenario_id": scenario.sid,
        "category": scenario.category,
        "text": scenario.text,
        "note": scenario.note,
        "expected_structure": scenario.expected_structure,
        "predicted_structure": structure,
        "predicted_ok": predicted_ok,
        "best_prob": scores.get("best_prob"),
        "unknown_prob": scores.get("unknown_prob"),
        "top3": ranked[:3],
        "params": params,
        "notes": notes,
        "synthesis": synthesis,
        "synthesis_filled": synthesis_filled,
        "llm_fill": llm_fill,
        "missing_prompt": missing_prompt,
        "feasible": feasible,
        "filled_feasible": filled_feasible,
        "missing_count": missing_count,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    structure_ok = sum(1 for r in rows if r["predicted_ok"])
    known = [r for r in rows if r["predicted_structure"] != "unknown"]
    feasible = sum(1 for r in known if r.get("feasible") is True)
    feasible_filled = sum(1 for r in known if (r.get("synthesis_filled") is not None and r.get("filled_feasible") is True))

    per_category: dict[str, dict[str, int]] = {}
    for r in rows:
        cat = r["category"]
        bucket = per_category.setdefault(cat, {"total": 0, "structure_ok": 0, "known": 0, "feasible": 0})
        bucket["total"] += 1
        if r["predicted_ok"]:
            bucket["structure_ok"] += 1
        if r["predicted_structure"] != "unknown":
            bucket["known"] += 1
        if r.get("feasible") is True:
            bucket["feasible"] += 1

    return {
        "n_cases": total,
        "structure_acc": structure_ok / total if total else 0.0,
        "known_cases": len(known),
        "feasible_rate_known": feasible / len(known) if known else 0.0,
        "llm_filled_feasible_cases": feasible_filled,
        "per_category": per_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-round end-to-end benchmark on curated scenarios.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--structure_model", default=_default_structure_model())
    parser.add_argument("--ner_model", default=_default_ner_model())
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--autofix", action="store_true")
    parser.add_argument("--llm_fill_missing", action="store_true")
    parser.add_argument("--ollama_config", default="nlu/bert_lab/configs/ollama_config.json")
    parser.add_argument("--out", default="nlu/bert_lab/data/eval/e2e_rounds_report.json")
    args = parser.parse_args()

    out_rows: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    for ridx in range(args.rounds):
        seed = args.seed + ridx
        round_rows = [
            _run_one(
                scenario=s,
                seed=seed,
                structure_model=args.structure_model,
                ner_model=args.ner_model,
                device=args.device,
                min_confidence=args.min_confidence,
                autofix=args.autofix,
                llm_fill_missing=args.llm_fill_missing,
                ollama_config=args.ollama_config,
            )
            for s in SCENARIOS
        ]
        round_summaries.append({"round": ridx + 1, "seed": seed, "summary": _aggregate(round_rows)})
        out_rows.extend([{"round": ridx + 1, "seed": seed, **row} for row in round_rows])

    report = {
        "config": {
            "rounds": args.rounds,
            "base_seed": args.seed,
            "structure_model": args.structure_model,
            "ner_model": args.ner_model,
            "device": args.device,
            "min_confidence": args.min_confidence,
            "autofix": args.autofix,
            "llm_fill_missing": args.llm_fill_missing,
            "ollama_config": args.ollama_config,
        },
        "round_summaries": round_summaries,
        "global_summary": _aggregate(out_rows),
        "results": out_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["global_summary"], indent=2))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
