from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


V3_RESPONSE_NATURALIZATION_SCHEMA_VERSION = "geant4_agent_v3_response_naturalization.v1"

_RAW_INTERNAL_MARKERS = (
    "raw trace",
    "V3TraceEvent",
    "turn.metadata",
    "state.metadata",
    "raw observation",
    "runtime_payload",
    "large_raw_payload",
    "pending_action",
    "commit_gate",
    "proposal_critic",
)


@dataclass(slots=True)
class V3ResponseNaturalizationResult:
    ok: bool
    used_llm: bool
    display_message: str = ""
    fallback_reason: str = ""
    prompt_profile_id: str = "geant4_agent_v3_response_naturalizer.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V3_RESPONSE_NATURALIZATION_SCHEMA_VERSION,
            "ok": self.ok,
            "used_llm": self.used_llm,
            "display_message": self.display_message,
            "fallback_reason": self.fallback_reason,
            "fallback_category": _fallback_category(self.fallback_reason),
            "prompt_profile_id": self.prompt_profile_id,
        }


class V3ResponseNaturalizer:
    """Optional display-message naturalizer with a strict context boundary."""

    def __init__(self, llm_config_path: str) -> None:
        self._llm_config_path = str(llm_config_path or "").strip()

    def naturalize(self, response: dict[str, Any], *, locale: str = "zh-CN") -> V3ResponseNaturalizationResult:
        original = str(response.get("display_message") or "")
        if not self._llm_config_path:
            return V3ResponseNaturalizationResult(ok=False, used_llm=False, display_message=original, fallback_reason="missing_llm_config_path")
        try:
            prompt = self.build_prompt(response, locale=locale)
            raw = self._call_llm(prompt)
            parsed = self._parse_llm_response(raw)
            message = str(parsed.get("display_message") or parsed.get("message") or "").strip() if isinstance(parsed, dict) else ""
            if not message:
                return V3ResponseNaturalizationResult(ok=False, used_llm=True, display_message=original, fallback_reason="empty_llm_message")
            ok, reason = _message_is_safe(message, response)
            if not ok:
                return V3ResponseNaturalizationResult(ok=False, used_llm=True, display_message=original, fallback_reason=reason)
            return V3ResponseNaturalizationResult(ok=True, used_llm=True, display_message=message)
        except Exception as exc:
            return V3ResponseNaturalizationResult(
                ok=False,
                used_llm=True,
                display_message=original,
                fallback_reason=f"naturalizer_error:{exc.__class__.__name__}",
            )

    def build_prompt(self, response: dict[str, Any], *, locale: str = "zh-CN") -> str:
        safe_payload = {
            "locale": locale,
            "dialogue_act": str(response.get("dialogue_act") or ""),
            "display_message": str(response.get("display_message") or ""),
            "answer_parts": _safe_answer_parts(response.get("answer_parts") or _dict(response.get("dialogue")).get("answer_parts")),
            "evidence_used": _safe_evidence(response.get("evidence_used") or _dict(response.get("dialogue")).get("evidence_used")),
            "context": _safe_context_pack(response.get("context")),
            "summary": _safe_summary(response.get("summary")),
        }
        payload_json = json.dumps(safe_payload, ensure_ascii=False, indent=2)
        language = "Chinese" if str(locale).lower().startswith("zh") else "English"
        return f"""You are the response naturalization layer for a Geant4 v3 agent.

Rewrite the display message to sound helpful and collaborative in {language}.
Use only the safe payload below. Do not add new physics facts, numbers, tools,
materials, particles, or run results. Keep the same next-step intent.

Safe payload:
{payload_json}

Rules:
- Preserve all concrete facts from answer_parts/context, especially material,
  particle, source energy, event counts, and runtime metrics.
- Do not mention raw trace, metadata, hidden state, prompt details, or internal
  tool names.
- Do not substitute examples such as lead/gamma when the context says water or
  proton.
- Keep it concise: one short paragraph, no markdown.

Return JSON only:
{{"display_message": "..."}}
"""

    def _call_llm(self, prompt: str) -> str:
        from nlu.llm_support.ollama_client import chat

        result = chat(prompt, config_path=self._llm_config_path)
        return str(result.get("response") or "") if result else ""

    def _parse_llm_response(self, raw: str) -> dict[str, Any] | None:
        from nlu.llm_support.ollama_client import extract_json

        parsed = extract_json(raw)
        return parsed if isinstance(parsed, dict) else None


def _safe_answer_parts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        safe: dict[str, Any] = {
            "kind": str(item.get("kind") or ""),
            "title": str(item.get("title") or ""),
        }
        if item.get("text") is not None:
            safe["text"] = str(item.get("text") or "")[:1000]
        if isinstance(item.get("items"), list):
            safe["items"] = [_safe_small_dict(entry) for entry in item["items"][:8] if isinstance(entry, dict)]
        out.append(safe)
    return out


def _safe_evidence(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    return [
        {"source": str(item.get("source") or ""), "status": str(item.get("status") or "")}
        for item in raw[:12]
        if isinstance(item, dict)
    ]


def _safe_context_pack(raw: Any) -> dict[str, Any]:
    context = _dict(raw)
    allowed = {
        "schema_version",
        "session_id",
        "goal",
        "phase",
        "latest_design",
        "latest_payload",
        "latest_runtime_facts",
        "pending_action",
        "open_questions",
        "assumptions",
        "suggested_next_actions",
        "last_user_turn",
    }
    return {key: _json_safe(context.get(key)) for key in allowed if key in context}


def _safe_summary(raw: Any) -> dict[str, Any]:
    summary = _dict(raw)
    allowed = {
        "phase",
        "runtime_ready",
        "runtime_ready_reason",
        "needs_confirmation",
        "has_design",
        "has_payload",
        "has_preflight",
        "preflight_status",
        "has_runtime_result",
        "next_action",
        "risk_level",
        "evidence_sources",
    }
    return {key: _json_safe(summary.get(key)) for key in allowed if key in summary}


def _safe_small_dict(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(value)
        for key, value in raw.items()
        if key in {"text", "prefill", "kind", "source", "status", "title"}
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _message_is_safe(message: str, response: dict[str, Any]) -> tuple[bool, str]:
    lowered = message.lower()
    if any(marker.lower() in lowered for marker in _RAW_INTERNAL_MARKERS):
        return False, "raw_internal_marker_detected"
    facts = _runtime_or_payload_facts(response)
    material = str(facts.get("material") or "")
    particle = str(facts.get("particle") or "").lower()
    energy = facts.get("source_energy_mev")
    if material == "G4_WATER" and any(token in lowered for token in ("g4_pb", "lead", "pb shielding")):
        return False, "material_conflict"
    if material == "G4_Pb" and any(token in lowered for token in ("g4_water", "water phantom")):
        return False, "material_conflict"
    if particle and particle != "gamma" and any(token in lowered for token in ("gamma primary", "photon primary", "gamma beam")):
        return False, "particle_conflict"
    if particle == "gamma" and any(token in lowered for token in ("proton beam", "proton primary")):
        return False, "particle_conflict"
    if energy is not None:
        try:
            expected = float(energy)
        except (TypeError, ValueError):
            expected = None
        if expected is not None:
            for mentioned in _mentioned_mev_values(message):
                if abs(mentioned - expected) > max(0.05, abs(expected) * 0.02):
                    return False, "source_energy_conflict"
    return True, ""


def _runtime_or_payload_facts(response: dict[str, Any]) -> dict[str, Any]:
    context = _dict(response.get("context"))
    runtime = _dict(context.get("latest_runtime_facts"))
    payload = _dict(context.get("latest_payload"))
    return {
        "material": _first_present(runtime.get("material"), payload.get("material")),
        "particle": _first_present(runtime.get("particle"), payload.get("particle")),
        "source_energy_mev": _first_present(runtime.get("source_energy_mev"), payload.get("source_energy_mev")),
    }


def _mentioned_mev_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*MeV", text, flags=re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    return values


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _fallback_category(reason: str) -> str:
    value = str(reason or "")
    if not value:
        return ""
    if value.startswith("naturalizer_error:"):
        if any(token in value for token in ("URLError", "TimeoutError", "ConnectionError", "HTTPError")):
            return "network_error"
        return "llm_error"
    if value in {"empty_llm_message", "missing_llm_config_path"}:
        return "model_unavailable"
    if value in {"material_conflict", "particle_conflict", "source_energy_conflict", "raw_internal_marker_detected"}:
        return "safety_rejected"
    return "other"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "V3_RESPONSE_NATURALIZATION_SCHEMA_VERSION",
    "V3ResponseNaturalizationResult",
    "V3ResponseNaturalizer",
]
