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
    proxy_url: str | None = None
    thinking: Dict[str, Any] | None = None


def load_config(path: str | Path) -> OllamaConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    provider = str(payload.get("provider", "ollama")).strip().lower()
    if not provider:
        provider = "ollama"
    model = str(payload.get("model", "llama3"))
    model_override = os.getenv("GEANT4_LLM_MODEL_OVERRIDE", "").strip()
    if model_override:
        model = model_override
    timeout_override = os.getenv("GEANT4_LLM_TIMEOUT_S", "").strip()
    configured_timeout = payload.get("timeout_s", payload.get("timeout_seconds", 60))
    timeout_s = int(timeout_override) if timeout_override else int(configured_timeout)
    proxy_url = os.getenv("GEANT4_LLM_PROXY_URL", "").strip() or str(payload.get("proxy_url", "")).strip()
    return OllamaConfig(
        provider=provider,
        base_url=str(payload.get("base_url", "http://localhost:11434")),
        model=model,
        timeout_s=timeout_s,
        headers=dict(payload.get("headers", {"Content-Type": "application/json"})),
        api_key=str(payload.get("api_key", "")).strip() or None,
        api_key_env=str(payload.get("api_key_env", "")).strip() or None,
        chat_path=str(payload.get("chat_path", "")).strip() or None,
        proxy_url=proxy_url or None,
        thinking=dict(payload.get("thinking")) if isinstance(payload.get("thinking"), dict) else None,
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


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: int,
    proxy_url: str | None = None,
) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    if proxy_url:
        proxy_handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        response = urllib.request.build_opener(proxy_handler).open(req, timeout=timeout_s)
    else:
        response = urllib.request.urlopen(req, timeout=timeout_s)
    with response as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_ollama(prompt: str, cfg: OllamaConfig, options: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    url = cfg.base_url.rstrip("/") + (cfg.chat_path or "/api/generate")
    return _post_json(url, payload, _final_headers(cfg), cfg.timeout_s, cfg.proxy_url)


def _chat_openai_compatible(prompt: str, cfg: OllamaConfig, options: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(options or {})
    if cfg.thinking and "thinking" not in payload:
        payload["thinking"] = dict(cfg.thinking)
    if "temperature" not in payload:
        payload["temperature"] = 0.0
    url = cfg.base_url.rstrip("/") + (cfg.chat_path or "/v1/chat/completions")
    raw = _post_json(url, payload, _final_headers(cfg), cfg.timeout_s, cfg.proxy_url)
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


