from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from nlu.bert_lab.bert_lab_data import generate_samples
from nlu.bert_lab.data_multitask import generate_samples as generate_multitask_samples
from nlu.bert_lab.generate_hardcases import generate as generate_hardcases


STRUCT_LABELS = ["nest", "grid", "ring", "stack", "shell"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dedupe(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for r in rows:
        k = (_norm(r.get("text", "")), str(r.get("structure", "")))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _unknown_hard(n: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    tpls = [
        "Need modules with size {mx}x{my}x{mz} mm and clearance {c} mm; layout may be ring or grid depending on constraints.",
        "Container {px}x{py}x{pz} mm with child rmax {r} mm hz {hz} mm; could be nest or shell, pick the practical one.",
        "Use layers {t1},{t2},{t3} mm in z with gap {c} mm; exact arrangement is not fixed yet.",
    ]
    for i in range(n):
        tpl = tpls[i % len(tpls)]
        rows.append(
            {
                "text": tpl.format(
                    mx=round(rng.uniform(5.0, 14.0), 2),
                    my=round(rng.uniform(5.0, 14.0), 2),
                    mz=round(rng.uniform(1.0, 6.0), 2),
                    c=round(rng.uniform(0.0, 1.2), 2),
                    px=round(rng.uniform(50.0, 180.0), 2),
                    py=round(rng.uniform(50.0, 180.0), 2),
                    pz=round(rng.uniform(50.0, 180.0), 2),
                    r=round(rng.uniform(8.0, 45.0), 2),
                    hz=round(rng.uniform(5.0, 40.0), 2),
                    t1=round(rng.uniform(0.2, 2.0), 2),
                    t2=round(rng.uniform(0.2, 2.0), 2),
                    t3=round(rng.uniform(0.2, 2.0), 2),
                ),
                "structure": "unknown",
                "params": {},
            }
        )
    return rows


def _sample_by_target(rows: List[Dict[str, object]], target: Dict[str, int], seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    bucket: Dict[str, List[Dict[str, object]]] = {}
    for lb in target:
        cur = [r for r in rows if str(r.get("structure", "")) == lb]
        cur = _dedupe(cur)
        rng.shuffle(cur)
        bucket[lb] = cur
    out: List[Dict[str, object]] = []
    for lb, n in target.items():
        if len(bucket[lb]) < n:
            raise RuntimeError(f"Not enough rows for label={lb}: got={len(bucket[lb])}, need={n}")
        out.extend(bucket[lb][:n])
    rng.shuffle(out)
    return out


def _target_counts(n: int, labels: List[str]) -> Dict[str, int]:
    per = n // len(labels)
    rem = n % len(labels)
    return {lb: per + (1 if i < rem else 0) for i, lb in enumerate(labels)}


def _remove_leakage(rows: List[Dict[str, object]], train_texts: set[str]) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        if _norm(r.get("text", "")) in train_texts:
            continue
        out.append(r)
    return out


def _stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
    labels = Counter(str(r.get("structure", "")) for r in rows)
    toks = [len(str(r.get("text", "")).split()) for r in rows if str(r.get("text", "")).strip()]
    return {
        "n": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "avg_tokens": round(sum(toks) / max(1, len(toks)), 2),
        "min_tokens": min(toks) if toks else 0,
        "max_tokens": max(toks) if toks else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build structure evaluation suites")
    ap.add_argument("--outdir", default="nlu/bert_lab/data/eval")
    ap.add_argument("--train_structure", default="nlu/bert_lab/data/controlled_structure.jsonl")
    ap.add_argument("--n_in_dist", type=int, default=2400)
    ap.add_argument("--n_hard", type=int, default=2400)
    ap.add_argument("--n_realnorm", type=int, default=1800)
    ap.add_argument("--unknown_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_rows = []
    p = Path(args.train_structure)
    if p.exists():
        train_rows = [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    train_texts = {_norm(r.get("text", "")) for r in train_rows}

    # 1) In-distribution: controlled grammar but seed-disjoint and leakage-removed.
    n_unknown = max(1, int(round(args.n_in_dist * args.unknown_ratio)))
    target_in = _target_counts(args.n_in_dist - n_unknown, STRUCT_LABELS)
    target_in["unknown"] = n_unknown
    pool_in = generate_samples(
        n=max(6000, args.n_in_dist * 4),
        seed=args.seed + 101,
        with_spans=False,
        noise_level="none",
        unknown_rate=args.unknown_ratio,
    )
    pool_in = _remove_leakage(pool_in, train_texts)
    in_dist = _sample_by_target(pool_in, target_in, rng.randint(1, 10_000_000))

    # 2) Hard: distractor-heavy cases + hard unknown.
    n_unknown_h = max(1, int(round(args.n_hard * args.unknown_ratio)))
    n_known_h = args.n_hard - n_unknown_h
    hard_known = generate_hardcases(max(1000, n_known_h * 2), args.seed + 202)
    hard_unknown = _unknown_hard(max(400, n_unknown_h * 2), args.seed + 203)
    hard_pool = _remove_leakage(_dedupe(hard_known + hard_unknown), train_texts)
    target_h = _target_counts(n_known_h, STRUCT_LABELS)
    target_h["unknown"] = n_unknown_h
    hard = _sample_by_target(hard_pool, target_h, rng.randint(1, 10_000_000))

    # 3) RealNorm-like: standardized instruction format close to LLM-normalized outputs.
    n_unknown_r = max(1, int(round(args.n_realnorm * args.unknown_ratio)))
    n_known_r = args.n_realnorm - n_unknown_r
    target_r = _target_counts(n_known_r, STRUCT_LABELS)
    target_r["unknown"] = n_unknown_r
    mt_pool = generate_multitask_samples(max(6000, args.n_realnorm * 5), args.seed + 303)
    real_pool: List[Dict[str, object]] = []
    for r in mt_pool:
        lb = str(r.get("structure", ""))
        if lb not in STRUCT_LABELS:
            continue
        real_pool.append({"text": r.get("text", ""), "structure": lb, "params": r.get("params", {})})
    real_pool.extend(_unknown_hard(max(600, n_unknown_r * 3), args.seed + 304))
    real_pool = _remove_leakage(_dedupe(real_pool), train_texts)
    realnorm = _sample_by_target(real_pool, target_r, rng.randint(1, 10_000_000))

    _write_jsonl(outdir / "structure_eval_in_dist.jsonl", in_dist)
    _write_jsonl(outdir / "structure_eval_hard.jsonl", hard)
    _write_jsonl(outdir / "structure_eval_realnorm.jsonl", realnorm)

    summary = {
        "in_dist": _stats(in_dist),
        "hard": _stats(hard),
        "realnorm": _stats(realnorm),
    }
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

