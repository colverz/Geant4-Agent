from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlu.bert_lab.graph_search import search_candidate_graphs
from nlu.bert_lab.infer import extract_params
from nlu.bert_lab.llm_bridge import build_missing_params_prompt, build_missing_params_schema
from nlu.bert_lab.ollama_client import chat, extract_json
from nlu.bert_lab.postprocess import merge_params


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"


def _default_structure_model() -> str:
    for name in ["structure_controlled_v4c_e1", "structure_controlled_v3_e1", "structure_controlled_smoke", "structure_opt_v3", "structure_opt_v2", "structure"]:
        p = MODELS_DIR / name
        if (p / "config.json").exists():
            return str(p)
    return "nlu/bert_lab/models/structure_controlled_v4c_e1"


def _default_ner_model() -> str:
    p = MODELS_DIR / "ner"
    if (p / "config.json").exists():
        return str(p)
    return "nlu/bert_lab/models/ner"
from builder.geometry.synthesize import synthesize_from_params


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end demo: text -> structure+params -> DSL")
    parser.add_argument("--text", required=True)
    parser.add_argument("--structure_model", default=_default_structure_model())
    parser.add_argument("--ner_model", default=_default_ner_model())
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--autofix", action="store_true")
    parser.add_argument("--prompt_format", default="text", choices=["text", "json_schema"])
    parser.add_argument("--llm_fill_missing", action="store_true")
    parser.add_argument("--ollama_config", default="nlu/bert_lab/configs/ollama_config.json")
    args = parser.parse_args()

    params = extract_params(args.text, args.ner_model, args.device)
    params, notes = merge_params(args.text, params)
    graph_result = search_candidate_graphs(
        args.text,
        params,
        min_confidence=args.min_confidence,
        seed=args.seed,
        top_k=max(1, args.top_k),
        apply_autofix=args.autofix,
    )
    structure, scores, ranked = graph_result.structure, graph_result.scores, graph_result.ranked
    notes.extend(graph_result.notes)

    top_k = max(1, args.top_k)
    def _fill_missing(structure_name: str, synth_obj: dict, base_params: dict):
        missing = synth_obj.get("missing_params", [])
        if not missing or not args.llm_fill_missing:
            return synth_obj, None
        prompt = build_missing_params_prompt(structure_name, missing, fmt=args.prompt_format)
        resp = chat(prompt, config_path=args.ollama_config, temperature=0.2)
        parsed = extract_json(resp.get("response", ""))
        if not isinstance(parsed, dict):
            return synth_obj, {"error": "failed to parse LLM response", "raw": resp.get("response", "")}
        merged = dict(base_params)
        merged.update(parsed)
        synth2 = synthesize_from_params(structure_name, merged, args.seed, apply_autofix=args.autofix)
        return synth2, {"used_params": parsed}

    candidates = []
    for name, prob in ranked[:top_k]:
        if prob < args.min_confidence:
            continue
        synth = synthesize_from_params(name, params, args.seed, apply_autofix=args.autofix)
        missing = synth.get("missing_params", [])
        prompt = build_missing_params_prompt(name, missing, fmt=args.prompt_format)
        schema = build_missing_params_schema(name, missing) if args.prompt_format == "json_schema" else None
        filled_synth, llm_fill = _fill_missing(name, synth, params)
        candidates.append(
            {
                "structure": name,
                "prob": prob,
                "synthesis": synth,
                "synthesis_filled": filled_synth if llm_fill else None,
                "llm_fill": llm_fill,
                "missing_prompt": prompt,
                "missing_schema": schema,
            }
        )

    if structure == "unknown":
        synth = {"error": "structure confidence below threshold"}
        missing_prompt = ""
        missing_schema = None
        synth_filled = None
        llm_fill_main = None
    else:
        synth = synthesize_from_params(structure, params, args.seed, apply_autofix=args.autofix)
        missing = synth.get("missing_params", [])
        missing_prompt = build_missing_params_prompt(structure, missing, fmt=args.prompt_format)
        missing_schema = build_missing_params_schema(structure, missing) if args.prompt_format == "json_schema" else None
        synth_filled, llm_fill_main = _fill_missing(structure, synth, params)

    out = {
        "graph_program": graph_result.graph_program,
        "chosen_skeleton": graph_result.chosen_skeleton,
        "structure": structure,
        "scores": scores,
        "params": params,
        "notes": notes,
        "synthesis": synth,
        "synthesis_filled": synth_filled if structure != "unknown" else None,
        "llm_fill": llm_fill_main if structure != "unknown" else None,
        "missing_prompt": missing_prompt,
        "missing_schema": missing_schema,
        "candidates": candidates,
        "graph_candidates": [
            {
                "structure": c.structure,
                "score": c.score,
                "score_breakdown": c.score_breakdown,
                "feasible": c.feasible,
                "missing_params": c.missing_params,
                "errors": c.errors,
                "warnings": c.warnings,
                "dsl": c.dsl,
            }
            for c in graph_result.candidates
        ],
    }

    # Rank candidates by feasibility and missing params
    def _cand_score(c):
        synth = c.get("synthesis", {})
        feasible = bool(synth.get("feasible"))
        missing_n = len(synth.get("missing_params", []))
        errors_n = len(synth.get("errors", []))
        return (1 if feasible else 0, -missing_n, -errors_n, c.get("prob", 0.0))

    if candidates:
        best = sorted(candidates, key=_cand_score, reverse=True)[0]
        out["best_candidate"] = {
            "structure": best["structure"],
            "prob": best["prob"],
            "missing_prompt": best.get("missing_prompt", ""),
            "missing_schema": best.get("missing_schema"),
            "synthesis": best["synthesis_filled"] or best["synthesis"],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


