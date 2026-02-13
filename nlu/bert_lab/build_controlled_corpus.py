from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from nlu.bert_lab.bert_lab_data import generate_samples
from nlu.bert_lab.data_multitask import generate_samples as generate_multitask_samples


STRUCTURE_LABELS = ["nest", "grid", "ring", "stack", "shell"]
MULTITASK_LABELS = ["nest", "grid", "ring", "stack", "shell", "single_box", "single_tubs"]
ALL_LABELS = STRUCTURE_LABELS + ["unknown"]
MATERIAL_POOL = ["G4_Si", "G4_Cu", "G4_Al", "G4_WATER", "G4_AIR"]
PARTICLE_POOL = ["gamma", "e-", "proton", "neutron"]
PHYSICS_POOL = ["FTFP_BERT", "QGSP_BERT", "QBBC"]
SOURCE_POOL = ["point", "beam", "isotropic"]
OUTPUT_POOL = ["root", "csv", "json"]


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _dedupe(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for row in rows:
        key = (_norm_text(str(row.get("text", ""))), str(row.get("structure", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _label_target_counts(n: int, include_unknown: bool, unknown_ratio: float) -> Dict[str, int]:
    if include_unknown:
        n_unknown = max(1, int(round(n * unknown_ratio)))
    else:
        n_unknown = 0
    n_known = max(0, n - n_unknown)
    per = n_known // len(STRUCTURE_LABELS)
    rem = n_known % len(STRUCTURE_LABELS)

    out: Dict[str, int] = {}
    for i, k in enumerate(STRUCTURE_LABELS):
        out[k] = per + (1 if i < rem else 0)
    if include_unknown:
        out["unknown"] = n_unknown
    return out


def _build_hard_unknown_rows(n: int, seed: int) -> List[Dict[str, object]]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    templates = [
        "geometry_intent: unresolved; candidate_pattern: circular_placement | planar_array; module=box({mx},{my},{mz}) mm; count={n}; clearance={c} mm.",
        "geometry_intent: unresolved; candidate_pattern: containment_parent_child | coaxial_shells; parent=box({px},{py},{pz}) mm; child=tubs(rmax={r},hz={hz}) mm.",
        "geometry_intent: unresolved; candidate_pattern: z_layer_sequence | containment_parent_child; layers_thickness_mm=[{t1},{t2},{t3}]; clearance={c} mm.",
        "Ring layout: module x {mx} mm, module y {my} mm, module z {mz} mm, count {n}, radius {r} mm; layout may be ring or grid depending on constraints.",
        "Need parent box {px}x{py}x{pz} mm and an inner cylinder rmax {r} mm hz {hz} mm, but final structure is undecided between nest and shell.",
        "Use layers {t1},{t2},{t3} mm in z with clearance {c} mm; exact arrangement is not fixed yet and may switch to nested containment.",
        "Ambiguous draft: circular_placement with pitch_x={mx} and pitch_y={my}; both ring radius and grid pitch are mentioned without final decision.",
    ]
    for i in range(n):
        tpl = templates[i % len(templates)]
        text = tpl.format(
            mx=round(rng.uniform(5.0, 15.0), 2),
            my=round(rng.uniform(5.0, 15.0), 2),
            mz=round(rng.uniform(1.0, 6.0), 2),
            n=rng.randint(6, 24),
            px=round(rng.uniform(40.0, 150.0), 2),
            py=round(rng.uniform(40.0, 150.0), 2),
            pz=round(rng.uniform(40.0, 150.0), 2),
            r=round(rng.uniform(10.0, 40.0), 2),
            hz=round(rng.uniform(5.0, 30.0), 2),
            t1=round(rng.uniform(0.2, 2.0), 2),
            t2=round(rng.uniform(0.2, 2.0), 2),
            t3=round(rng.uniform(0.2, 2.0), 2),
            c=round(rng.uniform(0.0, 1.0), 2),
        )
        rows.append({"text": text, "structure": "unknown", "params": {}, "source": "unknown_hard"})
    return rows


def _fnum(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}"


def _with_unit_mm_or_cm(v_mm: float, rng: random.Random, *, prefer_mm: float = 0.7) -> str:
    if rng.random() < prefer_mm:
        return f"{_fnum(v_mm)} mm"
    return f"{_fnum(v_mm / 10.0)} cm"


def _with_optional_unit(v_mm: float, rng: random.Random, *, p_mm: float = 0.4, p_cm: float = 0.25) -> str:
    r = rng.random()
    if r < p_mm:
        return f"{_fnum(v_mm)} mm"
    if r < p_mm + p_cm:
        return f"{_fnum(v_mm / 10.0)} cm"
    return _fnum(v_mm)


def _build_context_tail(rng: random.Random) -> str:
    if rng.random() > 0.65:
        return ""
    mat = rng.choice(MATERIAL_POOL)
    particle = rng.choice(PARTICLE_POOL)
    phys = rng.choice(PHYSICS_POOL)
    src = rng.choice(SOURCE_POOL)
    out = rng.choice(OUTPUT_POOL)
    style = rng.choice(
        [
            f" material={mat}; particle={particle}; physics_list={phys}; source_type={src}; output={out}.",
            f" context(material={mat}, particle={particle}, physics={phys}, source={src}, output={out}).",
        ]
    )
    return style


def _maybe_disambiguation_suffix(label: str, rng: random.Random) -> str:
    # Inject "conflicting-term but resolved-intent" notes so the classifier
    # learns to keep the declared structure instead of flipping to unknown.
    if rng.random() > 0.35:
        return ""
    hints = {
        "grid": "ring, circular, arc",
        "ring": "matrix, pitch, nx",
        "nest": "layer, stack, pitch",
        "stack": "contain, inner, outer",
        "shell": "nest, parent-child, containment",
    }
    h = hints.get(label)
    if not h:
        return ""
    return f" Note: terms like {h} may appear in previous designs; keep this request in the current structure."


def _to_layout_style_text(label: str, params: Dict[str, object], rng: random.Random) -> str:
    p = params
    if label == "ring":
        head = (
            "Ring layout: "
            f"module x {_with_optional_unit(float(p['module_x']), rng)}, "
            f"module y {_with_optional_unit(float(p['module_y']), rng)}, "
            f"module z {_with_optional_unit(float(p['module_z']), rng)}, "
            f"count {int(round(float(p['n'])))}, "
            f"radius {_with_optional_unit(float(p['radius']), rng)}, "
            f"clearance {_with_unit_mm_or_cm(float(p['clearance']), rng, prefer_mm=0.9)}."
        )
    elif label == "grid":
        head = (
            "Grid layout: "
            f"module x {_with_optional_unit(float(p['module_x']), rng)}, "
            f"module y {_with_optional_unit(float(p['module_y']), rng)}, "
            f"module z {_with_optional_unit(float(p['module_z']), rng)}, "
            f"nx {int(round(float(p['nx'])))}, "
            f"ny {int(round(float(p['ny'])))}, "
            f"pitch_x {_with_optional_unit(float(p['pitch_x']), rng)}, "
            f"pitch_y {_with_optional_unit(float(p['pitch_y']), rng)}, "
            f"clearance {_with_unit_mm_or_cm(float(p['clearance']), rng, prefer_mm=0.9)}."
        )
    elif label == "nest":
        head = (
            "Nest layout: "
            f"parent x {_with_optional_unit(float(p['parent_x']), rng)}, "
            f"parent y {_with_optional_unit(float(p['parent_y']), rng)}, "
            f"parent z {_with_optional_unit(float(p['parent_z']), rng)}, "
            f"child rmax {_with_optional_unit(float(p['child_rmax']), rng)}, "
            f"child hz {_with_optional_unit(float(p['child_hz']), rng)}, "
            f"clearance {_with_unit_mm_or_cm(float(p['clearance']), rng, prefer_mm=0.9)}."
        )
    elif label == "stack":
        head = (
            "Stack layout: "
            f"stack x {_with_optional_unit(float(p['stack_x']), rng)}, "
            f"stack y {_with_optional_unit(float(p['stack_y']), rng)}, "
            f"t1 {_with_optional_unit(float(p['t1']), rng)}, "
            f"t2 {_with_optional_unit(float(p['t2']), rng)}, "
            f"t3 {_with_optional_unit(float(p['t3']), rng)}, "
            f"stack clearance {_with_optional_unit(float(p['stack_clearance']), rng)}, "
            f"parent x {_with_optional_unit(float(p['parent_x']), rng)}, "
            f"parent y {_with_optional_unit(float(p['parent_y']), rng)}, "
            f"parent z {_with_optional_unit(float(p['parent_z']), rng)}, "
            f"nest clearance {_with_optional_unit(float(p['nest_clearance']), rng)}."
        )
    elif label == "shell":
        head = (
            "Shell layout: "
            f"inner radius {_with_optional_unit(float(p['inner_r']), rng)}, "
            f"thicknesses [{_with_optional_unit(float(p['th1']), rng)}, {_with_optional_unit(float(p['th2']), rng)}, {_with_optional_unit(float(p['th3']), rng)}], "
            f"hz {_with_optional_unit(float(p['hz']), rng)}, "
            f"clearance {_with_unit_mm_or_cm(float(p['clearance']), rng, prefer_mm=0.9)}."
        )
    else:
        head = rng.choice(
            [
                "Need modules with size 10.8x12.1x1.3 mm and clearance 0.5 mm; layout may be ring or grid depending on constraints.",
                "Use layers 1.2,2.0,1.8 mm in z with gap 1.1 mm; exact arrangement is not fixed yet.",
                "Need parent box and inner cylinder, or maybe concentric shells; constraints are incomplete.",
                "Draft says ring count and grid pitch at the same time; final arrangement remains ambiguous.",
                "The request mentions possible nest and shell options, but no decisive structure is selected.",
                "Arrangement is tentative and may change after constraints; do not lock to a single geometry class yet.",
            ]
        )
        return head

    if rng.random() < 0.7:
        mat = rng.choice(MATERIAL_POOL)
        particle = rng.choice(PARTICLE_POOL)
        phys = rng.choice(PHYSICS_POOL)
        src = rng.choice(SOURCE_POOL)
        out = rng.choice(OUTPUT_POOL)
        return head + _maybe_disambiguation_suffix(label, rng) + f" | material {mat}; particle {particle}; physics list {phys}; source type {src}; output {out}"
    return head + _maybe_disambiguation_suffix(label, rng)


def _to_llm_normalized_text(label: str, params: Dict[str, object], rng: random.Random) -> str:
    # Mix controlled-intent style and realnorm-style layout sentences to reduce
    # train/eval distribution gap.
    if rng.random() < 0.42:
        return _to_layout_style_text(label, params, rng)

    p = params
    def keep(prob: float) -> bool:
        return rng.random() < prob

    # Medium-structured normalized text: keep stable intent + key hints,
    # avoid over-specifying every parameter in every sample.
    if label == "ring":
        parts = [
            "geometry_intent: circular_placement",
            f"count={int(round(float(p['n'])))}",
            f"radius_mm={_fnum(float(p['radius']))}",
            f"module=box({_fnum(float(p['module_x']))},{_fnum(float(p['module_y']))},{_fnum(float(p['module_z']))}) mm",
        ]
        if keep(0.6):
            parts.append(f"clearance_mm={_fnum(float(p['clearance']))}")
        head = "; ".join(parts) + "."
    elif label == "grid":
        parts = [
            "geometry_intent: planar_array",
            f"array=({_fnum(float(p['nx']))},{_fnum(float(p['ny']))})",
            f"module=box({_fnum(float(p['module_x']))},{_fnum(float(p['module_y']))},{_fnum(float(p['module_z']))}) mm",
        ]
        if keep(0.75):
            parts.append(f"pitch_mm=({_fnum(float(p['pitch_x']))},{_fnum(float(p['pitch_y']))})")
        if keep(0.55):
            parts.append(f"clearance_mm={_fnum(float(p['clearance']))}")
        head = "; ".join(parts) + "."
    elif label == "nest":
        parts = [
            "geometry_intent: containment_parent_child",
            f"parent=box({_fnum(float(p['parent_x']))},{_fnum(float(p['parent_y']))},{_fnum(float(p['parent_z']))}) mm",
        ]
        parts.append(f"child=tubs(rmax={_fnum(float(p['child_rmax']))},hz={_fnum(float(p['child_hz']))}) mm")
        if keep(0.6):
            parts.append(f"clearance_mm={_fnum(float(p['clearance']))}")
        head = "; ".join(parts) + "."
    elif label == "stack":
        parts = [
            "geometry_intent: z_layer_sequence",
            f"layers_thickness_mm=[{_fnum(float(p['t1']))},{_fnum(float(p['t2']))},{_fnum(float(p['t3']))}]",
            f"layer_footprint_mm=({_fnum(float(p['stack_x']))},{_fnum(float(p['stack_y']))})",
        ]
        if keep(0.6):
            parts.append(f"stack_clearance_mm={_fnum(float(p['stack_clearance']))}")
        if keep(0.65):
            parts.append(f"container=box({_fnum(float(p['parent_x']))},{_fnum(float(p['parent_y']))},{_fnum(float(p['parent_z']))}) mm")
        if keep(0.5):
            parts.append(f"nest_clearance_mm={_fnum(float(p['nest_clearance']))}")
        head = "; ".join(parts) + "."
    elif label == "shell":
        parts = [
            "geometry_intent: coaxial_shells",
            f"inner_radius_mm={_fnum(float(p['inner_r']))}",
            f"thickness_mm=[{_fnum(float(p['th1']))},{_fnum(float(p['th2']))},{_fnum(float(p['th3']))}]",
            f"hz_mm={_fnum(float(p['hz']))}",
        ]
        if keep(0.6):
            parts.append(f"child=tubs(rmax={_fnum(float(p['child_rmax']))},hz={_fnum(float(p['child_hz']))}) mm")
        if keep(0.55):
            parts.append(f"clearance_mm={_fnum(float(p['clearance']))}")
        head = "; ".join(parts) + "."
    else:
        # Unknown soft samples: still normalized, but unresolved between candidates.
        unresolved_templates = [
            "geometry_intent: unresolved; candidate_pattern: circular_placement | planar_array; provide more constraints.",
            "geometry_intent: unresolved; candidate_pattern: containment_parent_child | coaxial_shells; provide more constraints.",
            "geometry_intent: unresolved; candidate_pattern: z_layer_sequence | containment_parent_child; provide more constraints.",
            "geometry_intent: unresolved; candidate_pattern: circular_placement | planar_array; note: ring count and grid pitch are both present.",
            "geometry_intent: unresolved; candidate_pattern: containment_parent_child | coaxial_shells; note: parent-child and shell cues conflict.",
            "geometry_intent: unresolved; candidate_pattern: z_layer_sequence | planar_array; note: layers are defined but arrangement is undecided.",
        ]
        head = rng.choice(unresolved_templates)
    return head + _maybe_disambiguation_suffix(label, rng) + _build_context_tail(rng)


def _collect_label_rows(
    label: str,
    needed: int,
    *,
    seed: int,
    with_spans: bool,
    noise_level: str,
    unknown_rate: float,
) -> List[Dict[str, object]]:
    if needed <= 0:
        return []
    rng = random.Random(seed)
    pool: List[Dict[str, object]] = []
    attempts = 0
    # Re-sample in small batches until each label has enough unique rows.
    while len(pool) < needed and attempts < 80:
        attempts += 1
        batch_n = max(128, needed * 2)
        rows = generate_samples(
            n=batch_n,
            seed=rng.randint(1, 10_000_000),
            with_spans=with_spans,
            noise_level=noise_level,
            unknown_rate=unknown_rate,
        )
        rows = [r for r in rows if str(r.get("structure")) == label]
        if not with_spans:
            # For structure classifier training, mimic LLM-normalized English output
            # instead of free-form human requests.
            for r in rows:
                r["text"] = _to_llm_normalized_text(label, dict(r.get("params", {})), rng)
        pool.extend(rows)
        pool = _dedupe(pool)
    if len(pool) < needed:
        raise RuntimeError(f"Unable to collect enough rows for label={label}, got={len(pool)}, need={needed}")
    rng.shuffle(pool)
    out = pool[:needed]
    for row in out:
        row.setdefault("source", f"synthetic_{noise_level}")
    return out


def _build_balanced_set(
    *,
    n: int,
    seed: int,
    with_spans: bool,
    include_unknown: bool,
    unknown_ratio: float,
    light_ratio: float,
    hard_unknown_ratio: float,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    target = _label_target_counts(n, include_unknown, unknown_ratio)
    out: List[Dict[str, object]] = []

    for label, needed in target.items():
        if needed <= 0:
            continue
        if label == "unknown":
            n_hard = int(round(needed * max(0.0, min(1.0, hard_unknown_ratio))))
            n_soft = max(0, needed - n_hard)
            if n_soft:
                out.extend(
                    _collect_label_rows(
                        "unknown",
                        n_soft,
                        seed=rng.randint(1, 10_000_000),
                        with_spans=False,
                        noise_level="none",
                        unknown_rate=1.0,
                    )
                )
            out.extend(_build_hard_unknown_rows(n_hard, seed=rng.randint(1, 10_000_000)))
            continue

        n_light = int(round(needed * max(0.0, min(0.5, light_ratio))))
        n_none = max(0, needed - n_light)
        if n_none:
            out.extend(
                _collect_label_rows(
                    label,
                    n_none,
                    seed=rng.randint(1, 10_000_000),
                    with_spans=with_spans,
                    noise_level="none",
                    unknown_rate=0.0,
                )
            )
        if n_light:
            out.extend(
                _collect_label_rows(
                    label,
                    n_light,
                    seed=rng.randint(1, 10_000_000),
                    with_spans=with_spans,
                    noise_level="light",
                    unknown_rate=0.0,
                )
            )

    out = _dedupe(out)
    if len(out) < n:
        # Top-up by per-label deficits after dedupe collisions.
        cur_counts = Counter(str(r.get("structure", "")) for r in out)
        add_rows: List[Dict[str, object]] = []
        for label, target_n in target.items():
            deficit = max(0, target_n - cur_counts.get(label, 0))
            if deficit <= 0:
                continue
            if label == "unknown":
                add_rows.extend(
                    _collect_label_rows(
                        "unknown",
                        deficit,
                        seed=rng.randint(1, 10_000_000),
                        with_spans=False,
                        noise_level="none",
                        unknown_rate=1.0,
                    )
                )
            else:
                add_rows.extend(
                    _collect_label_rows(
                        label,
                        deficit,
                        seed=rng.randint(1, 10_000_000),
                        with_spans=with_spans,
                        noise_level="light",
                        unknown_rate=0.0,
                    )
                )
        out.extend(add_rows)
        out = _dedupe(out)
    rng.shuffle(out)
    if len(out) < n:
        raise RuntimeError(f"Dataset smaller than target after dedupe: got={len(out)}, target={n}")
    return out[:n]


def _report(rows: List[Dict[str, object]]) -> Dict[str, object]:
    counts = Counter(str(r.get("structure", "")) for r in rows)
    source = Counter(str(r.get("source", "unknown")) for r in rows)
    lengths = [len(str(r.get("text", "")).split()) for r in rows if str(r.get("text", "")).strip()]
    return {
        "n": len(rows),
        "label_counts": dict(sorted(counts.items(), key=lambda kv: kv[0])),
        "source_counts": dict(source),
        "avg_tokens": round(sum(lengths) / max(1, len(lengths)), 2),
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }


def _label_target_counts_from_labels(n: int, labels: List[str]) -> Dict[str, int]:
    per = n // len(labels)
    rem = n % len(labels)
    return {lb: per + (1 if i < rem else 0) for i, lb in enumerate(labels)}


def _build_multitask_set(n: int, seed: int) -> List[Dict[str, object]]:
    target = _label_target_counts_from_labels(n, MULTITASK_LABELS)
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    attempts = 0
    while attempts < 40:
        attempts += 1
        batch = generate_multitask_samples(max(256, n), seed=rng.randint(1, 10_000_000))
        for row in batch:
            lb = str(row.get("structure", ""))
            if lb not in target:
                continue
            # Keep multitask samples only when spans exist for token supervision.
            if not row.get("spans"):
                continue
            row.setdefault("source", "synthetic_multitask")
            by_label[lb].append(row)
        for lb in target:
            by_label[lb] = _dedupe(by_label[lb])
        if all(len(by_label[lb]) >= target[lb] for lb in target):
            break
    if not all(len(by_label[lb]) >= target[lb] for lb in target):
        got = {lb: len(v) for lb, v in by_label.items()}
        raise RuntimeError(f"Unable to build multitask set to target counts. got={got}, target={target}")

    rows: List[Dict[str, object]] = []
    for lb in MULTITASK_LABELS:
        pool = by_label[lb][:]
        rng.shuffle(pool)
        rows.extend(pool[: target[lb]])
    rows = _dedupe(rows)
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build controlled-English corpora for BERT training.")
    ap.add_argument("--outdir", default="nlu/bert_lab/data", help="Output directory")
    ap.add_argument("--n_structure", type=int, default=4000)
    ap.add_argument("--n_ner", type=int, default=4000)
    ap.add_argument("--n_multitask", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--unknown_ratio", type=float, default=0.12)
    ap.add_argument("--light_ratio", type=float, default=0.15, help="Light-variant ratio, max 0.5")
    ap.add_argument("--hard_unknown_ratio", type=float, default=0.35, help="Fraction of unknowns that are hard negatives")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    structure_rows = _build_balanced_set(
        n=args.n_structure,
        seed=args.seed,
        with_spans=False,
        include_unknown=True,
        unknown_ratio=args.unknown_ratio,
        light_ratio=args.light_ratio,
        hard_unknown_ratio=args.hard_unknown_ratio,
    )
    ner_rows = _build_balanced_set(
        n=args.n_ner,
        seed=args.seed + 11,
        with_spans=True,
        include_unknown=False,
        unknown_ratio=0.0,
        light_ratio=args.light_ratio,
        hard_unknown_ratio=0.0,
    )
    multitask_rows = _build_multitask_set(args.n_multitask, args.seed + 23)

    _write_jsonl(outdir / "controlled_structure.jsonl", structure_rows)
    _write_jsonl(outdir / "controlled_ner.jsonl", ner_rows)
    _write_jsonl(outdir / "controlled_multitask.jsonl", multitask_rows)

    summary = {
        "outdir": str(outdir),
        "controlled_structure": _report(structure_rows),
        "controlled_ner": _report(ner_rows),
        "controlled_multitask": _report(multitask_rows),
    }
    (outdir / "controlled_corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
