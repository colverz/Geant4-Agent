from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _basic_stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
    labels = Counter(str(r.get("structure", "")) for r in rows)
    texts = [_norm(str(r.get("text", ""))) for r in rows]
    tok = [len(str(r.get("text", "")).split()) for r in rows if str(r.get("text", "")).strip()]
    unique = len(set(texts))
    return {
        "n": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "unique_text": unique,
        "duplicate_rate": round((1.0 - unique / max(1, len(rows))) * 100.0, 3),
        "avg_tokens": round(sum(tok) / max(1, len(tok)), 3),
        "min_tokens": min(tok) if tok else 0,
        "max_tokens": max(tok) if tok else 0,
    }


def _span_stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
    bad = 0
    missing = 0
    by_key = Counter()
    for r in rows:
        txt = str(r.get("text", ""))
        spans = r.get("spans")
        if not spans:
            missing += 1
            continue
        for sp in spans:
            k = str(sp.get("key", ""))
            st = sp.get("start")
            ed = sp.get("end")
            if not isinstance(st, int) or not isinstance(ed, int) or st < 0 or ed <= st or ed > len(txt):
                bad += 1
                continue
            by_key[k] += 1
    return {
        "rows_missing_spans": missing,
        "bad_span_count": bad,
        "span_key_counts": dict(sorted(by_key.items())),
    }


def _leakage(train_rows: List[Dict[str, object]], eval_rows: List[Dict[str, object]]) -> Dict[str, int]:
    tr_text = {_norm(str(r.get("text", ""))) for r in train_rows}
    ev_text = {_norm(str(r.get("text", ""))) for r in eval_rows}
    tr_pair = {(_norm(str(r.get("text", ""))), str(r.get("structure", ""))) for r in train_rows}
    ev_pair = {(_norm(str(r.get("text", ""))), str(r.get("structure", ""))) for r in eval_rows}
    return {
        "overlap_text": len(tr_text & ev_text),
        "overlap_text_label": len(tr_pair & ev_pair),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit BERT corpora before training")
    ap.add_argument("--train_structure", default="nlu/bert_lab/data/controlled_structure.jsonl")
    ap.add_argument("--train_multitask", default="nlu/bert_lab/data/controlled_multitask.jsonl")
    ap.add_argument("--eval_structure", default="nlu/bert_lab/data/structure_hard_eval.jsonl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train_s = _load_jsonl(Path(args.train_structure))
    train_m = _load_jsonl(Path(args.train_multitask))
    eval_s = _load_jsonl(Path(args.eval_structure))

    report = {
        "train_structure": _basic_stats(train_s),
        "train_multitask": _basic_stats(train_m),
        "train_multitask_spans": _span_stats(train_m),
        "eval_structure": _basic_stats(eval_s),
        "leakage_structure_vs_eval": _leakage(train_s, eval_s),
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

