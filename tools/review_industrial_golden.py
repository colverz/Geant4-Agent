from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.create_industrial_golden import DEFAULT_INDUSTRIAL_GOLDEN_DIR, INDUSTRIAL_GOLDEN_SCHEMA_VERSION


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metrics_hash(metrics: dict[str, Any]) -> str:
    encoded = json.dumps(metrics, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _golden_path(*, path: Path | None, case_id: str | None, golden_dir: Path) -> Path:
    if path is not None:
        return path
    if not case_id:
        raise ValueError("case_id_or_path_required")
    return golden_dir / f"{case_id}.golden.json"


def validate_industrial_golden_for_review(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != INDUSTRIAL_GOLDEN_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if not str(payload.get("case_id") or "").strip():
        errors.append("missing_case_id")

    fingerprint = payload.get("runtime_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint:
        errors.append("missing_runtime_fingerprint")
    else:
        for key in ("runtime_payload_hash", "physics_list", "seed", "events", "threads"):
            if fingerprint.get(key) in (None, ""):
                errors.append(f"missing_runtime_fingerprint:{key}")
        if fingerprint.get("geant4_version") in (None, ""):
            warnings.append("missing_runtime_fingerprint:geant4_version")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("missing_metrics")
    elif isinstance(metrics, dict):
        for metric_name, metric in sorted(metrics.items()):
            if not isinstance(metric, dict):
                errors.append(f"metric_not_object:{metric_name}")
                continue
            if not _is_number(metric.get("expected")):
                errors.append(f"metric_expected_not_numeric:{metric_name}")
            if not _is_number(metric.get("tolerance")):
                errors.append(f"metric_tolerance_not_numeric:{metric_name}")

    review = payload.get("review")
    if review is not None and not isinstance(review, dict):
        errors.append("review_not_object")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def review_industrial_golden(
    *,
    path: Path | None = None,
    case_id: str | None = None,
    golden_dir: Path = DEFAULT_INDUSTRIAL_GOLDEN_DIR,
    reviewer: str,
    notes: str = "",
    evidence: str = "",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    reviewer = reviewer.strip()
    if not reviewer:
        return {
            "ok": False,
            "status": "failed",
            "failure_category": "missing_reviewer",
            "errors": ["reviewer_required"],
        }

    try:
        golden_file = _golden_path(path=path, case_id=case_id, golden_dir=golden_dir)
    except ValueError as exc:
        return {
            "ok": False,
            "status": "failed",
            "failure_category": str(exc),
            "errors": [str(exc)],
        }
    if not golden_file.exists():
        return {
            "ok": False,
            "status": "failed",
            "failure_category": "golden_missing",
            "golden_file": str(golden_file),
            "errors": ["golden_file_missing"],
        }

    try:
        payload = _load_json(golden_file)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "failed",
            "failure_category": "golden_unreadable",
            "golden_file": str(golden_file),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "status": "failed",
            "failure_category": "golden_not_object",
            "golden_file": str(golden_file),
            "errors": ["golden_payload_not_object"],
        }

    validation = validate_industrial_golden_for_review(payload)
    if not validation["ok"]:
        return {
            "ok": False,
            "status": "failed",
            "failure_category": "golden_validation_failed",
            "golden_file": str(golden_file),
            "validation": validation,
            "errors": validation["errors"],
        }

    current_review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    current_status = str(current_review.get("status") or "").strip().lower()
    if current_status == "reviewed" and not force:
        return {
            "ok": False,
            "status": "blocked",
            "failure_category": "already_reviewed",
            "golden_file": str(golden_file),
            "errors": ["golden_already_reviewed_use_force_to_update_review_metadata"],
        }

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics_hash = _metrics_hash(metrics)
    if dry_run:
        return {
            "ok": True,
            "status": "would_review",
            "failure_category": None,
            "case_id": payload.get("case_id"),
            "golden_file": str(golden_file),
            "current_review_status": current_status or None,
            "metrics_hash": metrics_hash,
            "warnings": validation["warnings"],
        }

    payload["review"] = {
        "status": "reviewed",
        "reviewer": reviewer,
        "reviewed_at_utc": _utc_now(),
        "notes": notes.strip(),
        "evidence": evidence.strip(),
        "previous_status": current_status or None,
        "metrics_hash": metrics_hash,
    }
    golden_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "status": "reviewed",
        "failure_category": None,
        "case_id": payload.get("case_id"),
        "golden_file": str(golden_file),
        "metrics_hash": payload["review"]["metrics_hash"],
        "warnings": validation["warnings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark an industrial runtime golden file as reviewed.")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_INDUSTRIAL_GOLDEN_DIR)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = review_industrial_golden(
        path=args.path,
        case_id=args.case_id or None,
        golden_dir=args.golden_dir,
        reviewer=args.reviewer,
        notes=args.notes,
        evidence=args.evidence,
        force=args.force,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps({"ok": report["ok"], "report": report}, ensure_ascii=False, indent=2))
    else:
        print(f"industrial_golden_review: status={report['status']} file={report.get('golden_file', '')}")
        for error in report.get("errors") or []:
            print(f"  error: {error}")
        for warning in report.get("warnings") or []:
            print(f"  warning: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
