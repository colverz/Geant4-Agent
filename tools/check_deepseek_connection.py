from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid
from urllib.parse import urlparse

from nlu.llm_support.ollama_client import chat, load_config


DEEPSEEK_CONNECTION_CHECK_SCHEMA_VERSION = "geant4_agent_deepseek_connection_check.v1"
DEFAULT_CONFIG_PATH = Path("nlu/llm_support/configs/deepseek_api.local.json")


def validate_deepseek_response(payload: dict[str, Any], nonce: str) -> dict[str, Any]:
    raw = payload.get("provider_raw") if isinstance(payload.get("provider_raw"), dict) else {}
    response = str(payload.get("response") or "").strip()
    request_id = str(raw.get("id") or "").strip()
    model = str(raw.get("model") or "").strip()
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    errors: list[str] = []
    if nonce not in response:
        errors.append("nonce_not_returned")
    if not request_id:
        errors.append("provider_request_id_missing")
    if not model:
        errors.append("provider_model_missing")
    if not usage:
        errors.append("provider_usage_missing")
    return {
        "ok": not errors,
        "request_id": request_id or None,
        "model": model or None,
        "usage": usage,
        "nonce_verified": nonce in response,
        "errors": errors,
    }


def _safe_config_summary(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    parsed = urlparse(config.base_url)
    return {
        "provider": config.provider,
        "endpoint_host": parsed.hostname,
        "model": config.model,
        "api_key_configured": bool(config.api_key or config.api_key_env),
    }


def _worker(config_path: Path, nonce: str, timeout_s: int) -> int:
    os.environ["GEANT4_LLM_TIMEOUT_S"] = str(timeout_s)
    try:
        response = chat(
            f"Return exactly this token and nothing else: {nonce}",
            config_path=config_path,
            temperature=0.0,
            max_tokens=40,
        )
        result = validate_deepseek_response(response, nonce)
    except Exception as exc:
        result = {
            "ok": False,
            "request_id": None,
            "model": None,
            "usage": {},
            "nonce_verified": False,
            "errors": [f"request_failed:{type(exc).__name__}"],
        }
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result["ok"] else 1


def _run_worker_process(command: list[str], *, total_timeout_s: float) -> tuple[str, bool]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.monotonic() + total_timeout_s
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        process.kill()
        try:
            stdout, _ = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout = ""
        return stdout, True
    stdout, _ = process.communicate()
    return stdout, False


def check_deepseek_connection(config_path: Path, *, timeout_s: int = 45) -> dict[str, Any]:
    summary = _safe_config_summary(config_path)
    config_errors: list[str] = []
    if summary["provider"] not in {"deepseek", "openai_compatible"}:
        config_errors.append("provider_is_not_deepseek_compatible")
    if summary["endpoint_host"] != "api.deepseek.com":
        config_errors.append("endpoint_is_not_official_deepseek_api")
    if not summary["api_key_configured"]:
        config_errors.append("api_key_not_configured")
    if config_errors:
        return {
            "schema_version": DEEPSEEK_CONNECTION_CHECK_SCHEMA_VERSION,
            "ok": False,
            "config": summary,
            "errors": config_errors,
        }

    nonce = f"geant4-agent-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--nonce",
        nonce,
        "--timeout",
        str(timeout_s),
    ]
    stdout, timed_out = _run_worker_process(command, total_timeout_s=max(timeout_s + 5, 10))
    if timed_out:
        result = {
            "ok": False,
            "request_id": None,
            "model": None,
            "usage": {},
            "nonce_verified": False,
            "errors": ["total_deadline_exceeded"],
        }
    else:
        try:
            result = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            result = {
                "ok": False,
                "request_id": None,
                "model": None,
                "usage": {},
                "nonce_verified": False,
                "errors": ["worker_result_missing"],
            }
    return {
        "schema_version": DEEPSEEK_CONNECTION_CHECK_SCHEMA_VERSION,
        "ok": bool(result.get("ok")),
        "config": summary,
        "verification": result,
        "total_timeout_s": timeout_s + 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a real DeepSeek API response without exposing credentials.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nonce", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    timeout_s = max(1, args.timeout)
    if args.worker:
        return _worker(args.config, args.nonce, timeout_s)
    report = check_deepseek_connection(args.config, timeout_s=timeout_s)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
