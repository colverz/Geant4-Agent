from __future__ import annotations

import argparse
import json

from bert_lab.infer import extract_params, predict_structure
from bert_lab.postprocess import merge_params
from geometry.synthesize import synthesize_from_params


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end demo: text -> structure+params -> DSL")
    parser.add_argument("--text", required=True)
    parser.add_argument("--structure_model", default="bert_lab/bert_model")
    parser.add_argument("--ner_model", default="bert_lab/bert_ner_model")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    structure, scores = predict_structure(
        args.text,
        args.structure_model,
        args.device,
        args.min_confidence,
    )
    params = extract_params(args.text, args.ner_model, args.device)
    params, notes = merge_params(args.text, params)
    if structure == "unknown":
        synth = {"error": "structure confidence below threshold"}
    else:
        synth = synthesize_from_params(structure, params, args.seed)

    out = {
        "structure": structure,
        "scores": scores,
        "params": params,
        "notes": notes,
        "synthesis": synth,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
