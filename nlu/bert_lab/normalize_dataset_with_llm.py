from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from nlu.bert_lab.llm_bridge import build_normalization_prompt
from nlu.bert_lab.ollama_client import chat, extract_json


def normalize_text(text: str, config_path: str, temperature: float = 0.0) -> Dict[str, Any]:
    prompt = build_normalization_prompt(text)
    resp = chat(prompt, config_path=config_path, temperature=temperature)
    payload = extract_json(resp.get("response", ""))
    if not isinstance(payload, dict):
        return {"ok": False, "normalized_text": text, "raw": resp.get("response", "")}
    normalized = str(payload.get("normalized_text", "")).strip() or text
    return {
        "ok": True,
        "normalized_text": normalized,
        "language_detected": payload.get("language_detected", ""),
        "structure_hint": payload.get("structure_hint", ""),
        "raw": payload,
    }


def run(
    input_path: Path,
    output_path: Path,
    config_path: str,
    limit: int | None = None,
    sleep_ms: int = 0,
) -> Dict[str, Any]:
    total = 0
    ok = 0
    fallback = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if limit is not None and total >= limit:
                break
            total += 1
            sample = json.loads(line)
            text = str(sample.get("text", "")).strip()
            if not text:
                sample["normalize_meta"] = {"ok": False, "error": "empty_text"}
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fallback += 1
                continue

            try:
                result = normalize_text(text, config_path=config_path)
            except Exception as ex:
                result = {"ok": False, "normalized_text": text, "error": str(ex)}

            sample["text_raw"] = text
            sample["text"] = result.get("normalized_text", text)
            sample["normalize_meta"] = {
                "ok": bool(result.get("ok", False)),
                "language_detected": result.get("language_detected", ""),
                "structure_hint": result.get("structure_hint", ""),
                "error": result.get("error", ""),
            }
            if sample["normalize_meta"]["ok"]:
                ok += 1
            else:
                fallback += 1

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    return {"total": total, "ok": ok, "fallback": fallback}


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize JSONL dataset text into controlled English via Ollama.")
    ap.add_argument("--input", required=True, help="Input JSONL with at least key: text")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--config", default="nlu/bert_lab/configs/ollama_config.json", help="Ollama config path")
    ap.add_argument("--limit", type=int, default=None, help="Optional max samples")
    ap.add_argument("--sleep_ms", type=int, default=0, help="Sleep milliseconds between requests")
    args = ap.parse_args()

    stats = run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        config_path=args.config,
        limit=args.limit,
        sleep_ms=args.sleep_ms,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
