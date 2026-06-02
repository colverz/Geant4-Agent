from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class V3Intent(str, Enum):
    DESIGN_NEW = "design_new"
    MODIFY_AND_RUN = "modify_and_run"
    RESULT_QUESTION = "result_question"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class IntentResult:
    intent: V3Intent
    confidence: float = 1.0
    source: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


def classify_intent_llm(text: str, *, config_path: str) -> IntentResult:
    """Use LLM to classify user intent. Falls back to UNKNOWN on failure."""
    try:
        prompt = (
            "Classify the user's intent into exactly one category. Return JSON only.\n\n"
            f'User text: "{text}"\n\n'
            "Categories:\n"
            '- design_new: wants to create or discuss a new simulation design\n'
            '- modify_and_run: wants to change parameters and possibly run\n'
            '- result_question: asking about a previous simulation result or metric\n'
            '- confirm: confirming or approving a proposed action\n'
            '- cancel: canceling or refusing a proposed action\n'
            '- help: asking what the system can do\n'
            '- unknown: does not clearly fit any category\n\n'
            'Return: {"intent": "<category>", "confidence": 0.95, "reason": "one short sentence"}\n'
        )
        from nlu.llm_support.ollama_client import chat, extract_json
        response = chat(prompt, config_path=config_path)
        raw = str(response.get("response") or "")
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            return IntentResult(intent=V3Intent.UNKNOWN, source="llm")
        intent_str = str(parsed.get("intent") or "").strip().lower()
        try:
            intent = V3Intent(intent_str)
        except ValueError:
            return IntentResult(intent=V3Intent.UNKNOWN, source="llm")
        confidence = float(parsed.get("confidence", 0.8))
        return IntentResult(
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            source="llm",
            metadata={"reason": str(parsed.get("reason", "")), "raw_response": raw[:500]},
        )
    except Exception:
        return IntentResult(intent=V3Intent.UNKNOWN, source="llm")


__all__ = ["V3Intent", "IntentResult", "classify_intent_llm"]
