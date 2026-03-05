from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "nlu" / "bert_lab" / "models"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_acc(payload: dict[str, Any]) -> float | None:
    for key in ("top1_accuracy", "accuracy", "eval_accuracy"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_training_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        out: dict[str, Any] = {}
        for key in (
            "learning_rate",
            "per_device_train_batch_size",
            "num_train_epochs",
            "warmup_ratio",
            "weight_decay",
            "max_steps",
            "seed",
            "gradient_accumulation_steps",
        ):
            if hasattr(obj, key):
                out[key] = getattr(obj, key)
        return out
    return {}


def _weighted_score(metrics: dict[str, float | None], weights: dict[str, float]) -> float | None:
    available = [(k, v) for k, v in metrics.items() if isinstance(v, float)]
    if not available:
        return None
    total_weight = sum(weights.get(k, 0.0) for k, _ in available)
    if total_weight <= 0:
        return None
    return sum(float(v) * (weights.get(k, 0.0) / total_weight) for k, v in available)


@dataclass
class ModelAudit:
    name: str
    in_dist_acc: float | None
    hard_acc: float | None
    realnorm_acc: float | None
    eval_acc: float | None
    weighted_score: float | None
    robustness: float | None
    generalization_gap: float | None
    training_args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "in_dist_acc": self.in_dist_acc,
            "hard_acc": self.hard_acc,
            "realnorm_acc": self.realnorm_acc,
            "eval_acc": self.eval_acc,
            "weighted_score": self.weighted_score,
            "robustness": self.robustness,
            "generalization_gap": self.generalization_gap,
            "training_args": self.training_args,
        }


def _collect_model_audits(models_dir: Path) -> list[ModelAudit]:
    audits: list[ModelAudit] = []
    for model_dir in sorted(models_dir.glob("structure_controlled*")):
        if not model_dir.is_dir():
            continue
        in_dist = _extract_acc(_read_json(model_dir / "eval_in_dist.json"))
        hard = _extract_acc(_read_json(model_dir / "eval_hard.json"))
        realnorm = _extract_acc(_read_json(model_dir / "eval_realnorm.json"))
        eval_acc = _extract_acc(_read_json(model_dir / "eval_metrics.json"))
        training_args = _load_training_args(model_dir / "training_args.bin")

        weights = {"in_dist_acc": 0.1, "hard_acc": 0.45, "realnorm_acc": 0.45}
        metric_map = {"in_dist_acc": in_dist, "hard_acc": hard, "realnorm_acc": realnorm}
        weighted = _weighted_score(metric_map, weights)

        robust_candidates = [x for x in (hard, realnorm, in_dist) if isinstance(x, float)]
        robustness = min(robust_candidates) if robust_candidates else None
        generalization_gap = None
        if isinstance(in_dist, float):
            out_of_dist = [x for x in (hard, realnorm) if isinstance(x, float)]
            if out_of_dist:
                generalization_gap = in_dist - sum(out_of_dist) / len(out_of_dist)

        audits.append(
            ModelAudit(
                name=model_dir.name,
                in_dist_acc=in_dist,
                hard_acc=hard,
                realnorm_acc=realnorm,
                eval_acc=eval_acc,
                weighted_score=weighted,
                robustness=robustness,
                generalization_gap=generalization_gap,
                training_args=training_args,
            )
        )
    audits.sort(
        key=lambda x: (
            x.weighted_score if isinstance(x.weighted_score, float) else -1.0,
            x.robustness if isinstance(x.robustness, float) else -1.0,
        ),
        reverse=True,
    )
    return audits


def _assessment(best: ModelAudit | None, probe_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    if best is None:
        return {"status": "insufficient_data", "reason": "No evaluated structure_controlled models found."}
    weighted = best.weighted_score if isinstance(best.weighted_score, float) else 0.0
    robust = best.robustness if isinstance(best.robustness, float) else 0.0
    gap = best.generalization_gap if isinstance(best.generalization_gap, float) else 1.0

    probe_acc = None
    if isinstance(probe_summary, dict):
        probe_acc = _safe_float(probe_summary.get("accuracy"))

    near_optimal = weighted >= 0.95 and robust >= 0.90 and gap <= 0.08
    if isinstance(probe_acc, float):
        near_optimal = near_optimal and probe_acc >= 0.95
    status = "near_optimal" if near_optimal else "not_optimal"
    recommendations: list[str] = []
    if gap > 0.08:
        recommendations.append("Increase hard-case proportion and split by template family to reduce overfitting.")
    if robust < 0.90:
        recommendations.append("Raise unknown/ambiguous sample ratio and keep strict class balancing.")
    if weighted < 0.95:
        recommendations.append("Tune LR/batch/label smoothing with short sweeps and select by hard+realnorm score.")
    if isinstance(probe_acc, float) and probe_acc < 0.95:
        recommendations.append(
            "Probe accuracy on noisy normalized text is below 0.95; add slot-style perturbation augmentation for nest/shell."
        )
    recommendations.append("Keep LLM-first normalization; train BERT on normalized-text styles instead of free-form text.")

    return {
        "status": status,
        "best_model": best.name,
        "weighted_score": weighted,
        "robustness": robust,
        "generalization_gap": gap,
        "probe_accuracy": probe_acc,
        "recommendations": recommendations,
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    run_cfg = payload["run_config"]
    audits = payload["audits"]
    assessment = payload["assessment"]
    probe_summary = payload.get("probe_summary") or {}
    lines: list[str] = []
    lines.append("# BERT Training Strategy Audit")
    lines.append("")
    lines.append(f"- Date: `{run_cfg['date']}`")
    lines.append(f"- Models dir: `{run_cfg['models_dir']}`")
    lines.append(f"- Audited models: `{len(audits)}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(f"- Status: `{assessment['status']}`")
    if assessment.get("best_model"):
        lines.append(f"- Best model: `{assessment['best_model']}`")
        lines.append(f"- Weighted score: `{assessment['weighted_score']:.4f}`")
        lines.append(f"- Robustness(min metric): `{assessment['robustness']:.4f}`")
        lines.append(f"- Generalization gap: `{assessment['generalization_gap']:.4f}`")
    if isinstance(assessment.get("probe_accuracy"), float):
        lines.append(f"- Probe accuracy (noisy normalized text): `{assessment['probe_accuracy']:.4f}`")
    lines.append("")
    if probe_summary:
        lines.append("## Probe Summary")
        lines.append(f"- Probe total: `{probe_summary.get('total')}`")
        lines.append(f"- Probe hits: `{probe_summary.get('hits')}`")
        if isinstance(probe_summary.get("accuracy"), float):
            lines.append(f"- Probe accuracy: `{probe_summary.get('accuracy'):.4f}`")
        per_label = probe_summary.get("per_label_accuracy", {})
        if isinstance(per_label, dict) and per_label:
            lines.append("- Per-label:")
            for key, value in sorted(per_label.items(), key=lambda x: x[0]):
                if isinstance(value, (int, float)):
                    lines.append(f"  - `{key}`: `{float(value):.4f}`")
                else:
                    lines.append(f"  - `{key}`: `{value}`")
    lines.append("")
    lines.append("## Ranked Models")
    lines.append("")
    lines.append("| model | in_dist | hard | realnorm | weighted | robustness | gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in audits:
        lines.append(
            "| {name} | {in_dist} | {hard} | {realnorm} | {weighted} | {robust} | {gap} |".format(
                name=row["name"],
                in_dist="-" if row["in_dist_acc"] is None else f"{row['in_dist_acc']:.4f}",
                hard="-" if row["hard_acc"] is None else f"{row['hard_acc']:.4f}",
                realnorm="-" if row["realnorm_acc"] is None else f"{row['realnorm_acc']:.4f}",
                weighted="-" if row["weighted_score"] is None else f"{row['weighted_score']:.4f}",
                robust="-" if row["robustness"] is None else f"{row['robustness']:.4f}",
                gap="-" if row["generalization_gap"] is None else f"{row['generalization_gap']:.4f}",
            )
        )
    lines.append("")
    lines.append("## Recommendations")
    for item in assessment.get("recommendations", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Suggested Next Run")
    lines.append("- Build normalized-style v2 dataset:")
    lines.append("  - `python scripts/build_structure_v2_dataset.py --keep_original`")
    lines.append("- Train structure model with v2 profile:")
    lines.append(
        "  - `python scripts/train_structure_v2.py --data nlu/bert_lab/data/controlled_structure_v2.jsonl "
        "--outdir nlu/bert_lab/models/structure_controlled_v5_e2 --config nlu/bert_lab/configs/structure_train_v2.json`"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BERT structure-training strategy from existing model artifacts.")
    parser.add_argument("--models_dir", default=str(MODELS_DIR))
    parser.add_argument("--outdir", default="docs")
    parser.add_argument(
        "--probe_json",
        default="docs/structure_probe_eval_2026-03-05.json",
        help="Optional probe evaluation JSON produced by scripts/eval_structure_probe.py",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    audits = _collect_model_audits(models_dir)
    best = audits[0] if audits else None
    probe_payload = _read_json(Path(args.probe_json)) if args.probe_json else {}
    probe_summary = probe_payload.get("summary") if isinstance(probe_payload, dict) else None
    assessment = _assessment(best, probe_summary=probe_summary if isinstance(probe_summary, dict) else None)

    payload = {
        "run_config": {
            "date": stamp,
            "models_dir": str(models_dir),
            "probe_json": args.probe_json,
        },
        "assessment": assessment,
        "audits": [x.to_dict() for x in audits],
        "probe_summary": probe_summary,
    }

    json_path = outdir / f"bert_strategy_audit_{stamp}.json"
    md_path = outdir / f"bert_strategy_audit_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")
    print(json.dumps(assessment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
