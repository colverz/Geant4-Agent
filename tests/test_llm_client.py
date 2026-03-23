from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nlu.llm_support import ollama_client


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LlmClientTest(unittest.TestCase):
    def test_load_config_defaults_to_ollama_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.json"
            p.write_text(json.dumps({"base_url": "http://x", "model": "m"}), encoding="utf-8")
            cfg = ollama_client.load_config(p)
        self.assertEqual(cfg.provider, "ollama")
        self.assertEqual(cfg.base_url, "http://x")
        self.assertEqual(cfg.model, "m")

    def test_chat_ollama_payload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.json"
            p.write_text(
                json.dumps(
                    {
                        "provider": "ollama",
                        "base_url": "http://localhost:11434",
                        "model": "qwen3:14b",
                        "headers": {"Content-Type": "application/json"},
                    }
                ),
                encoding="utf-8",
            )
            captured: dict = {}

            def _fake_urlopen(req, timeout=0):
                captured["url"] = req.full_url
                captured["body"] = json.loads(req.data.decode("utf-8"))
                return _FakeResponse({"response": "ok"})

            with mock.patch("nlu.llm_support.ollama_client.urllib.request.urlopen", side_effect=_fake_urlopen):
                out = ollama_client.chat("hello", config_path=p, temperature=0.1)

        self.assertEqual(out["response"], "ok")
        self.assertEqual(captured["url"], "http://localhost:11434/api/generate")
        self.assertEqual(captured["body"]["prompt"], "hello")
        self.assertEqual(captured["body"]["options"]["temperature"], 0.1)

    def test_chat_openai_compatible_payload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.json"
            p.write_text(
                json.dumps(
                    {
                        "provider": "openai_compatible",
                        "base_url": "https://api.openai.com",
                        "chat_path": "/v1/chat/completions",
                        "model": "gpt-4.1-mini",
                        "api_key": "dummy",
                    }
                ),
                encoding="utf-8",
            )
            captured: dict = {}

            def _fake_urlopen(req, timeout=0):
                captured["url"] = req.full_url
                captured["body"] = json.loads(req.data.decode("utf-8"))
                captured["auth"] = req.headers.get("Authorization", "")
                return _FakeResponse({"choices": [{"message": {"content": "{\"ok\":true}"}}]})

            with mock.patch("nlu.llm_support.ollama_client.urllib.request.urlopen", side_effect=_fake_urlopen):
                out = ollama_client.chat("hello", config_path=p, temperature=0.0)

        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["body"]["messages"][0]["content"], "hello")
        self.assertEqual(captured["body"]["temperature"], 0.0)
        self.assertTrue(captured["auth"].startswith("Bearer "))
        self.assertEqual(out["response"], "{\"ok\":true}")


if __name__ == "__main__":
    unittest.main()


