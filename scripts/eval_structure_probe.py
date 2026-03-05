from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from nlu.runtime_components.infer import predict_structure
from nlu.runtime_components.model_preflight import MODELS_DIR, select_default_structure_dir


NOISE_PREFIX = [
    "physics note: attenuation setup;",
    "medical note: phantom planning;",
    "aerospace note: shielding precheck;",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "text" in obj and "structure" in obj:
                rows.append(obj)
    return rows


def _probe_transform(text: str, rng: random.Random) -> str:
    out = text
    if rng.random() < 0.6:
        out = out.replace(";", " / ")
    if rng.random() < 0.5:
        out = out.replace("intent=SET; ", "")
    if rng.random() < 0.4:
        out = out.replace("structure=", "layout=")
    if rng.random() < 0.5:
        out = f"{rng.choice(NOISE_PREFIX)} {out}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe structure-classifier robustness on normalized-style noisy text.")
    parser.add_argument("--data", default="nlu/bert_lab/data/controlled_structure_v2.jsonl")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="docs")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = _load_jsonl(Path(args.data))
    if not rows:
        raise RuntimeError("Probe dataset is empty.")
    rng.shuffle(rows)
    rows = rows[: max(10, min(args.n_samples, len(rows)))]

    model_dir = args.model_dir or str(select_default_structure_dir(MODELS_DIR))
    hits = 0
    total = 0
    by_label_total = Counter()
    by_label_hit = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    for row in rows:
        gold = str(row.get("structure", "unknown"))
        probe_text = _probe_transform(str(row.get("text", "")), rng)
        pred, scores, ranked = predict_structure(
            probe_text,
            model_dir=model_dir,
            device=args.device,
            min_confidence=args.min_confidence,
        )
        total += 1
        by_label_total[gold] += 1
        confusion[gold][pred] += 1
        ok = pred == gold
        if ok:
            hits += 1
            by_label_hit[gold] += 1
        if len(examples) < 15 and not ok:
            examples.append(
                {
                    "gold": gold,
                    "pred": pred,
                    "top3": ranked[:3],
                    "text": probe_text,
                    "scores": {
                        "best_prob": scores.get("best_prob"),
                        "margin": scores.get("margin"),
                        "unknown_prob": scores.get("unknown_prob"),
                    },
                }
            )

    per_label = {
        label: (by_label_hit[label] / by_label_total[label]) if by_label_total[label] else 0.0
        for label in sorted(by_label_total.keys())
    }
    payload = {
        "run_config": {
            "date": date.today().isoformat(),
            "data": args.data,
            "model_dir": model_dir,
            "device": args.device,
            "min_confidence": args.min_confidence,
            "n_samples": len(rows),
            "seed": args.seed,
        },
        "summary": {
            "accuracy": (hits / total) if total else 0.0,
            "total": total,
            "hits": hits,
            "per_label_accuracy": per_label,
        },
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "failure_examples": examples,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"structure_probe_eval_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
