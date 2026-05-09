from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", flags=re.IGNORECASE)
_UNIT_PATTERN = re.compile(r"\b(mm|cm|m|mev|kev|gev|ev|deg|degree|degrees)\b", flags=re.IGNORECASE)
_INTERNAL_PATH_SEGMENTS = {"internal", "debug", "runtime", "prompt", "api_key", "secret", "subprocess", "command"}
_NUMERIC_UNIT_PATH_HINTS = ("_mm", "_mev", "_deg")
_NUMERIC_UNIT_PATHS = {"source.energy_mev", "scoring.plane_z_mm"}


@dataclass(frozen=True)
class EvidenceGroundingContext:
    user_text: str = ""
    context_summary: str = ""
    stable_context_text: str = ""
    allowed_paths: frozenset[str] = frozenset()
    allowed_evidence_sources: frozenset[str] = frozenset({"user", "context", "capability_kb"})
    unsupported_terms: frozenset[str] = frozenset()
    deprecated_terms: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any] | None,
        *,
        allowed_paths: Iterable[str] = (),
        allowed_evidence_sources: Iterable[str] = ("user", "context", "capability_kb"),
    ) -> "EvidenceGroundingContext":
        data = mapping or {}
        return cls(
            user_text=str(data.get("user_text", "") or ""),
            context_summary=str(data.get("context_summary", "") or ""),
            stable_context_text=str(data.get("stable_context_text", "") or ""),
            allowed_paths=frozenset(str(path) for path in allowed_paths if str(path)),
            allowed_evidence_sources=frozenset(str(source) for source in allowed_evidence_sources if str(source)),
            unsupported_terms=_coerce_term_set(data.get("unsupported_terms") or data.get("unsupported_capabilities")),
            deprecated_terms=_coerce_term_set(data.get("deprecated_terms") or data.get("deprecated_capabilities")),
        )

    @property
    def grounding_text(self) -> str:
        return " ".join(part for part in (self.user_text, self.context_summary, self.stable_context_text) if part)


@dataclass(frozen=True)
class EvidenceGroundingResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _coerce_term_set(value: Any) -> frozenset[str]:
    terms: set[str] = set()
    if isinstance(value, dict):
        items: Iterable[Any] = value.values()
    else:
        items = value if isinstance(value, (list, tuple, set)) else []
    for item in items:
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                text = str(nested).strip().lower()
                if text:
                    terms.add(text)
        else:
            text = str(item).strip().lower()
            if text:
                terms.add(text)
    return frozenset(terms)


def _canonical_numeric_token(token: str) -> str:
    try:
        value = Decimal(str(token).lower())
    except (InvalidOperation, ValueError):
        return str(token).lower()
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def numeric_tokens(value: Any) -> set[str]:
    return {_canonical_numeric_token(token) for token in _NUMBER_PATTERN.findall(str(value or ""))}


def numeric_tokens_from_value(value: Any) -> set[str]:
    return numeric_tokens(json.dumps(value, ensure_ascii=False))


def _path_has_internal_segment(path: str) -> bool:
    segments = {segment.strip().lower() for segment in re.split(r"[.\[\]/\\_-]+", path) if segment.strip()}
    return bool(segments & _INTERNAL_PATH_SEGMENTS)


def _path_requires_explicit_unit(path: str) -> bool:
    return path in _NUMERIC_UNIT_PATHS or path.endswith(_NUMERIC_UNIT_PATH_HINTS)


def _evidence_text_contains_known_bad_term(text: str, terms: frozenset[str]) -> bool:
    lowered = text.lower()
    return any(term and term in lowered for term in terms)


def _contains_text_span(haystack: str, needle: str) -> bool:
    compact_haystack = re.sub(r"\s+", " ", haystack or "").strip().lower()
    compact_needle = re.sub(r"\s+", " ", needle or "").strip().lower()
    return bool(compact_needle and compact_needle in compact_haystack)


def check_candidate_update_grounding(
    update: dict[str, Any],
    *,
    context: EvidenceGroundingContext,
    prefix: str = "candidate_updates[0].",
) -> EvidenceGroundingResult:
    errors: list[str] = []
    path = str(update.get("path", "") or "")
    op = str(update.get("op", "") or "")
    value = update.get("value")

    if path and context.allowed_paths and path not in context.allowed_paths:
        errors.append(f"value_not_allowed:{prefix}path")
    if path and _path_has_internal_segment(path):
        errors.append(f"internal_path:{prefix}path")

    evidence = update.get("evidence")
    if op in {"set", "remove"} and not evidence:
        errors.append(f"missing_evidence:{prefix}evidence")
    if evidence is not None and not isinstance(evidence, list):
        errors.append(f"json_key_not_array:{prefix}evidence")
        return EvidenceGroundingResult(ok=False, errors=errors)

    evidence_items = evidence if isinstance(evidence, list) else []
    sources: set[str] = set()
    for ev_idx, ev in enumerate(evidence_items):
        ev_prefix = f"{prefix}evidence[{ev_idx}]."
        if not isinstance(ev, dict):
            errors.append(f"json_key_not_object:{ev_prefix.rstrip('.')}")
            continue
        source = str(ev.get("source", "") or "")
        text = str(ev.get("text", "") or "").strip()
        if source:
            sources.add(source)
            if source not in context.allowed_evidence_sources:
                errors.append(f"value_not_allowed:{ev_prefix}source")
        if not text:
            errors.append(f"missing_evidence_text:{ev_prefix}text")
        else:
            if source == "user" and context.user_text and not _contains_text_span(context.user_text, text):
                errors.append(f"evidence_text_not_found:{ev_prefix}text")
            if source == "context":
                context_text = " ".join(part for part in (context.context_summary, context.stable_context_text) if part)
                if context_text and not _contains_text_span(context_text, text):
                    errors.append(f"evidence_text_not_found:{ev_prefix}text")
            if source == "capability_kb" and _evidence_text_contains_known_bad_term(text, context.unsupported_terms):
                errors.append(f"unsupported_kb_grounding:{ev_prefix}text")
            if source == "capability_kb" and _evidence_text_contains_known_bad_term(text, context.deprecated_terms):
                errors.append(f"deprecated_kb_grounding:{ev_prefix}text")

    value_numbers = numeric_tokens_from_value(value)
    if value_numbers:
        grounded_numbers = numeric_tokens(context.grounding_text)
        invented = value_numbers - grounded_numbers
        if invented:
            errors.append(f"ungrounded_numeric_value:{prefix}value")
        if sources and sources <= {"capability_kb"}:
            errors.append(f"capability_kb_cannot_ground_numeric_value:{prefix}value")
        if _path_requires_explicit_unit(path):
            user_evidence_text = " ".join(
                str(ev.get("text", "") or "") for ev in evidence_items if isinstance(ev, dict) and ev.get("source") == "user"
            )
            if user_evidence_text and numeric_tokens(user_evidence_text) and not _UNIT_PATTERN.search(user_evidence_text):
                errors.append(f"missing_unit_for_numeric_value:{prefix}value")

    return EvidenceGroundingResult(ok=not errors, errors=errors)


__all__ = [
    "EvidenceGroundingContext",
    "EvidenceGroundingResult",
    "check_candidate_update_grounding",
    "numeric_tokens",
    "numeric_tokens_from_value",
]
