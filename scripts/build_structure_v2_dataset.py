from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


LENGTH_KEYS = {
    "module_x",
    "module_y",
    "module_z",
    "radius",
    "clearance",
    "parent_x",
    "parent_y",
    "parent_z",
    "child_rmax",
    "child_hz",
    "inner_r",
    "th1",
    "th2",
    "th3",
    "hz",
    "stack_x",
    "stack_y",
    "t1",
    "t2",
    "t3",
    "stack_clearance",
    "nest_clearance",
}
INT_KEYS = {"n", "nx", "ny"}

STRUCTURE_INTENT = {
    "ring": "circular_placement",
    "grid": "planar_array",
    "nest": "centered_nesting",
    "stack": "stack_along_z",
    "shell": "coaxial_shells",
    "unknown": "unresolved",
}


def _fmt_num(x: Any) -> str:
    try:
        value = float(x)
    except Exception:
        return str(x)
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _fmt_param(key: str, value: Any) -> str:
    if key in INT_KEYS:
        return f"{key}={int(round(float(value)))}"
    if key in LENGTH_KEYS:
        return f"{key}={_fmt_num(value)} mm"
    return f"{key}={_fmt_num(value)}"


def _render_kv_variant(structure: str, params: dict[str, Any], rng: random.Random) -> str:
    intent = STRUCTURE_INTENT.get(structure, structure)
    pairs = [_fmt_param(k, v) for k, v in sorted(params.items())]
    rng.shuffle(pairs)
    body = "; ".join(pairs) if pairs else "params=none"
    return f"intent=SET; geometry_intent={intent}; structure={structure}; {body};"


def _render_sentence_variant(structure: str, params: dict[str, Any], rng: random.Random) -> str:
    if structure == "unknown":
        candidates = [
            "intent SET; geometry pattern is not fixed yet; keep options open and ask for constraints.",
            "intent SET; arrangement remains ambiguous; choose structure only after extra constraints.",
            "intent SET; candidate patterns may include grid/ring/nest, decision pending more details.",
        ]
        return rng.choice(candidates)
    ordered = [f"{k} {_fmt_num(v)}" for k, v in sorted(params.items())]
    rng.shuffle(ordered)
    return f"set structure {structure}; " + "; ".join(ordered) + ";"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if "text" not in obj or "structure" not in obj:
                continue
            rows.append(obj)
    return rows


def _dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("structure", "")), str(row.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structure dataset v2 focused on LLM-normalized text styles.")
    parser.add_argument("--base", default="nlu/bert_lab/data/controlled_structure.jsonl")
    parser.add_argument("--out", default="nlu/bert_lab/data/controlled_structure_v2.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_original", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    base_rows = _load_jsonl(base_path)
    generated: list[dict[str, Any]] = []

    for row in base_rows:
        structure = str(row.get("structure", "unknown"))
        params = row.get("params", {}) if isinstance(row.get("params"), dict) else {}
        source = str(row.get("source", "base"))
        if args.keep_original:
            generated.append(
                {
                    "text": str(row.get("text", "")),
                    "structure": structure,
                    "params": params,
                    "source": source,
                    "style": "original",
                }
            )
        generated.append(
            {
                "text": _render_kv_variant(structure, params, rng),
                "structure": structure,
                "params": params,
                "source": "strategy_v2_aug",
                "style": "kv_compact",
            }
        )
        generated.append(
            {
                "text": _render_sentence_variant(structure, params, rng),
                "structure": structure,
                "params": params,
                "source": "strategy_v2_aug",
                "style": "sentence_compact",
            }
        )

    generated = _dedup(generated)
    rng.shuffle(generated)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = Counter(str(x.get("structure", "")) for x in generated)
    style_counts = Counter(str(x.get("style", "")) for x in generated)
    summary = {
        "base_path": str(base_path),
        "out_path": str(out_path),
        "seed": args.seed,
        "keep_original": bool(args.keep_original),
        "total_rows": len(generated),
        "label_counts": dict(sorted(label_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
