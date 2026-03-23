from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


OPENAI_COMPAT_PROVIDERS = {"openai", "openai_compatible", "deepseek", "siliconflow"}


@dataclass
class OllamaConfig:
    provider: str
    base_url: str
    model: str
    timeout_s: int = 60
    headers: Dict[str, str] | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    chat_path: str | None = None


def load_config(path: str | Path) -> OllamaConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    provider = str(payload.get("provider", "ollama")).strip().lower()
    if not provider:
        provider = "ollama"
    return OllamaConfig(
        provider=provider,
        base_url=str(payload.get("base_url", "http://localhost:11434")),
        model=str(payload.get("model", "llama3")),
        timeout_s=int(payload.get("timeout_s", 60)),
        headers=dict(payload.get("headers", {"Content-Type": "application/json"})),
        api_key=str(payload.get("api_key", "")).strip() or None,
        api_key_env=str(payload.get("api_key_env", "")).strip() or None,
        chat_path=str(payload.get("chat_path", "")).strip() or None,
    )


def _resolve_auth_token(cfg: OllamaConfig) -> str | None:
    if cfg.api_key:
        return cfg.api_key
    if cfg.api_key_env:
        token = os.getenv(cfg.api_key_env, "").strip()
        return token or None
    for env_key in ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        token = os.getenv(env_key, "").strip()
        if token:
            return token
    return None


def _final_headers(cfg: OllamaConfig) -> Dict[str, str]:
    headers = dict(cfg.headers or {})
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    token = _resolve_auth_token(cfg)
    if token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_s: int) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_ollama(prompt: str, cfg: OllamaConfig, options: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    url = cfg.base_url.rstrip("/") + (cfg.chat_path or "/api/generate")
    return _post_json(url, payload, _final_headers(cfg), cfg.timeout_s)


def _chat_openai_compatible(prompt: str, cfg: OllamaConfig, options: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(options or {})
    if "temperature" not in payload:
        payload["temperature"] = 0.0
    url = cfg.base_url.rstrip("/") + (cfg.chat_path or "/v1/chat/completions")
    raw = _post_json(url, payload, _final_headers(cfg), cfg.timeout_s)
    content = ""
    try:
        content = str(raw.get("choices", [{}])[0].get("message", {}).get("content", ""))
    except Exception:
        content = ""
    return {"response": content, "provider_raw": raw}


def chat(
    prompt: str,
    config_path: str | Path = "nlu/llm_support/configs/ollama_config.json",
    **options: Any,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    if cfg.provider == "ollama":
        return _chat_ollama(prompt, cfg, dict(options))
    if cfg.provider in OPENAI_COMPAT_PROVIDERS:
        return _chat_openai_compatible(prompt, cfg, dict(options))
    raise RuntimeError(
        f"Unsupported provider '{cfg.provider}'. "
        "Use provider=ollama or provider=openai_compatible."
    )


def extract_json(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


