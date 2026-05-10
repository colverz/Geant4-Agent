from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_MUTATION_PATTERN = re.compile(
    r"\b(change|set|modify|update|use|add|remove|delete|configure|build|create)\b|"
    r"(\u4fee\u6539|\u8bbe\u7f6e|\u6539\u6210|\u66f4\u65b0|\u6dfb\u52a0|\u5220\u9664|"
    r"(?<!\u5f53\u524d)\u914d\u7f6e(?!\u91cc|\u7684|\u4e2d|\u4e0a|\u662f)|"
    r"\u6784\u5efa|\u5efa\u7acb)",
    flags=re.IGNORECASE,
)
_NEGATED_MUTATION_PHRASE_PATTERN = re.compile(
    r"\b(?:do\s+not|don't|dont|without|never)\s+"
    r"(?:change|modify|update|edit|alter|delete|remove|configure|set|apply)\b(?:\s+\w+){0,6}|"
    r"(?:\u4e0d\u8981|\u522b|\u4e0d\u9700\u8981|\u65e0\u9700|\u4e0d\u7528).{0,8}"
    r"(?:\u4fee\u6539|\u66f4\u6539|\u6539\u53d8|\u5220\u9664|\u914d\u7f6e|\u8bbe\u7f6e|\u5e94\u7528)",
    flags=re.IGNORECASE,
)
_RUNTIME_PATTERN = re.compile(
    r"\b(run|rerun|execute|start)\b.*\b(events?|simulation|geant4|beam)\b|"
    r"\brun\s+\d+\s+events?\b|"
    r"\brun\s+it\b|"
    r"(\u8fd0\u884c|\u91cd\u8dd1|\u6267\u884c|\u5f00\u59cb).*(event|\u4e8b\u4ef6|geant4|\u6a21\u62df|\u7c92\u5b50\u675f)",
    flags=re.IGNORECASE,
)
_VIEWER_PATTERN = re.compile(
    r"\b(open|launch|show)\b.*\b(viewer|visuali[sz]ation|geometry window)\b|"
    r"(\u6253\u5f00|\u542f\u52a8|\u663e\u793a).*(viewer|\u53ef\u89c6\u5316|\u51e0\u4f55\u7a97\u53e3)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CompositeIntent:
    has_config_mutation: bool
    has_runtime_request: bool
    has_viewer_request: bool
    requires_staged_runtime_guard: bool

    def to_dict(self) -> dict:
        return asdict(self)


def detect_composite_intent(text: str) -> CompositeIntent:
    raw = str(text or "")
    mutation_scan_text = _NEGATED_MUTATION_PHRASE_PATTERN.sub(" ", raw)
    has_config_mutation = bool(_MUTATION_PATTERN.search(mutation_scan_text))
    has_runtime_request = bool(_RUNTIME_PATTERN.search(raw))
    has_viewer_request = bool(_VIEWER_PATTERN.search(raw))
    return CompositeIntent(
        has_config_mutation=has_config_mutation,
        has_runtime_request=has_runtime_request,
        has_viewer_request=has_viewer_request,
        requires_staged_runtime_guard=has_config_mutation and (has_runtime_request or has_viewer_request),
    )


__all__ = ["CompositeIntent", "detect_composite_intent"]
