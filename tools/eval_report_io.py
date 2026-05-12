from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

EVAL_REPORT_SCHEMA_VERSION = "geant4_agent_eval_report_record.v1"
DEFAULT_EVAL_REPORT_DIR = Path("docs/reports/eval")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return cleaned or "eval"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_metadata(cwd: Path | None = None) -> dict[str, Any]:
    root = cwd or Path.cwd()

    def run_git(args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]) or None,
        "branch": run_git(["branch", "--show-current"]) or None,
        "dirty": bool(status),
        "changed_line_count": len([line for line in status.splitlines() if line.strip()]),
    }


def build_eval_run_id(tool: str, created_at_utc: str | None = None) -> str:
    timestamp = (created_at_utc or _utc_now()).replace(":", "").replace("-", "")
    timestamp = timestamp.replace("Z", "").replace("+0000", "")
    return f"{timestamp}__{_slug(tool)}"


def save_eval_output(
    output: dict[str, Any],
    *,
    outdir: Path | str | None,
    tool: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    if outdir is None:
        return output
    target_dir = Path(outdir)
    target_dir.mkdir(parents=True, exist_ok=True)

    created_at = _utc_now()
    normalized_tool = _slug(tool)
    normalized_run_id = _slug(run_id or build_eval_run_id(normalized_tool, created_at))
    report_path = target_dir / f"{normalized_run_id}.json"
    latest_path = target_dir / f"{normalized_tool}.latest.json"

    record = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "run_id": normalized_run_id,
        "tool": normalized_tool,
        "created_at_utc": created_at,
        "report_path": str(report_path),
        "latest_path": str(latest_path),
        "git": _git_metadata(),
    }
    saved = deepcopy(output)
    saved["eval_record"] = record
    report_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "run_id": normalized_run_id,
        "tool": normalized_tool,
        "created_at_utc": created_at,
        "ok": bool(saved.get("ok")),
        "report_path": str(report_path),
    }
    latest_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved
