from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from nlu.llm_support.ollama_client import chat, extract_json


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(user_text: str, schema_text: str, system_text: str) -> str:
    return (
        system_text.strip()
        + "\n\nUser request:\n"
        + user_text.strip()
        + "\n\nJSON schema:\n"
        + schema_text.strip()
    )


def generate_min_config(
    user_text: str,
    schema_path: str = "core/schema/geant4_min_config.schema.json",
    system_path: str = "core/prompts/min_config_system.txt",
    ollama_config: str = "nlu/llm_support/configs/ollama_config.json",
) -> Dict[str, Any]:
    schema_text = load_text(schema_path)
    system_text = load_text(system_path)
    prompt = build_prompt(user_text, schema_text, system_text)
    resp = chat(prompt, config_path=ollama_config, temperature=0.2)
    obj = extract_json(resp.get("response", ""))
    return {"raw": resp.get("response", ""), "config": obj}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a minimal Geant4 config via Ollama")
    parser.add_argument("--text", required=True)
    parser.add_argument("--schema", default="core/schema/geant4_min_config.schema.json")
    parser.add_argument("--system", default="core/prompts/min_config_system.txt")
    parser.add_argument("--ollama_config", default="nlu/llm_support/configs/ollama_config.json")
    args = parser.parse_args()

    out = generate_min_config(args.text, args.schema, args.system, args.ollama_config)
    print(json.dumps(out, indent=2))


