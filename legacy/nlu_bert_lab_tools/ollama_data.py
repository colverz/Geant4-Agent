from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from nlu.llm_support.ollama_client import chat, extract_json


PROMPT_TEMPLATE = """You are generating training samples for a geometry parameter extraction task.
Return a single JSON object with keys: text, structure, params.
- text: natural language request
- structure: one of [nest, grid, ring, stack, shell]
- params: numeric values required by the structure (use mm for lengths)

Constraints:
- Keep text concise, structured, and unambiguous.
- Include all required parameters in the text.
- Use integers for counts.

Generate one sample for structure: {structure}.
"""


STRUCTURES = ["nest", "grid", "ring", "stack", "shell"]


def _validate_sample(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("structure") not in STRUCTURES:
        return False
    if not isinstance(obj.get("text"), str):
        return False
    if not isinstance(obj.get("params"), dict):
        return False
    return True


def _generate_one(
    structure: str,
    config: str,
    seed: int,
    max_retries: int,
    num_predict: int,
    num_ctx: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    prompt = PROMPT_TEMPLATE.format(structure=structure)
    for _ in range(max_retries):
        resp = chat(
            prompt,
            config_path=config,
            temperature=0.2,
            num_predict=num_predict,
            num_ctx=num_ctx,
        )
        text = resp.get("response", "")
        obj = extract_json(text)
        if obj and _validate_sample(obj):
            return obj
    # Fallback: return minimal stub
    return {"text": "", "structure": structure, "params": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM-based samples via Ollama")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--n", type=int, default=50, help="Number of samples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="nlu/llm_support/configs/ollama_config.json")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--num_predict", type=int, default=200)
    parser.add_argument("--num_ctx", type=int, default=2048)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples: List[Dict[str, Any]] = []
    for i in range(args.n):
        structure = STRUCTURES[i % len(STRUCTURES)]
        seed = rng.randint(0, 10_000_000)
        obj = _generate_one(structure, args.config, seed, args.max_retries, args.num_predict, args.num_ctx)
        samples.append(obj)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


