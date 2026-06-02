from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DESIGN_ADVICE_SCHEMA_VERSION = "geant4_agent_design_advice.v1"


@dataclass(frozen=True)
class DesignOption:
    option_id: str
    title: str
    purpose: str
    setup: dict[str, Any] = field(default_factory=dict)
    tradeoffs: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    requires_user_approval: bool = False
    runnable_with_current_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "title": self.title,
            "purpose": self.purpose,
            "setup": dict(self.setup),
            "tradeoffs": list(self.tradeoffs),
            "assumptions": list(self.assumptions),
            "requires_user_approval": self.requires_user_approval,
            "runnable_with_current_runtime": self.runnable_with_current_runtime,
        }


@dataclass(frozen=True)
class DesignAdvice:
    schema_version: str = DESIGN_ADVICE_SCHEMA_VERSION
    goal: str = ""
    status: str = "needs_more_information"
    primary_option: dict[str, Any] = field(default_factory=dict)
    alternative_options: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    user_decisions_required: list[str] = field(default_factory=list)
    unsupported_capabilities: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    user_visible_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "goal": self.goal,
            "status": self.status,
            "primary_option": dict(self.primary_option),
            "alternative_options": [dict(item) for item in self.alternative_options],
            "assumptions": list(self.assumptions),
            "user_decisions_required": list(self.user_decisions_required),
            "unsupported_capabilities": list(self.unsupported_capabilities),
            "recommended_next_steps": list(self.recommended_next_steps),
            "user_visible_summary": self.user_visible_summary,
        }


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _setup_summary(setup: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry": setup.get("geometry"),
        "material": setup.get("material"),
        "source": setup.get("source"),
        "detector": setup.get("detector"),
        "scoring": list(setup.get("scoring") or []) if isinstance(setup.get("scoring"), list) else setup.get("scoring"),
        "dimensions_mm": setup.get("dimensions_mm"),
        "source_particle": setup.get("source_particle"),
        "source_energy_mev": setup.get("source_energy_mev"),
        "design_rationale": setup.get("design_rationale"),
    }


def _status(candidate: dict[str, Any], recommended_config: dict[str, Any] | None) -> str:
    next_action = str(candidate.get("next_action") or "")
    if next_action == "unsupported_capability":
        return "unsupported"
    if next_action == "ask_user_to_choose_approximation":
        return "needs_user_decision"
    if isinstance(recommended_config, dict) and recommended_config:
        return "candidate_config_ready"
    if next_action == "build_candidate_config":
        return "design_ready"
    return "needs_more_information"


def _summary_text(*, lang: str, status: str, goal: str, option_title: str, decisions: list[str], unsupported: list[str]) -> str:
    decision_text = "; ".join(decisions[:2]) or goal
    unsupported_text = ", ".join(unsupported[:3]) or goal
    if lang == "zh":
        if status == "candidate_config_ready":
            return f"已经形成可提交的仿真方案：{option_title}。建议先确认假设，再进入运行前检查。"
        if status == "needs_user_decision":
            return f"已经形成方案草稿，但需要你决定近似或取舍：{decision_text}。"
        if status == "unsupported":
            return f"目标中包含当前运行时暂不支持的能力：{unsupported_text}。我给出可支持的替代方向。"
        return f"已经整理仿真目标：{goal or option_title}，还需要补充关键信息。"
    if status == "candidate_config_ready":
        return f"A runnable candidate design is ready: {option_title}. Review the assumptions before runtime preflight."
    if status == "needs_user_decision":
        return f"A design draft is ready, but it needs your decision: {'; '.join(decisions[:2]) or goal}."
    if status == "unsupported":
        return f"The goal includes unsupported runtime capabilities: {', '.join(unsupported[:3]) or goal}. I included a supported alternative direction."
    return f"I organized the simulation goal: {goal or option_title}, but key details are still needed."


def _recommended_next_steps(status: str, *, has_config: bool) -> list[str]:
    if status == "candidate_config_ready":
        return [
            "Review assumptions and user decisions.",
            "Accept the recommended config if the design matches the intended physics approximation.",
            "Run runtime preflight before launching Geant4.",
        ]
    if status == "needs_user_decision":
        return [
            "Choose whether to approve the proposed approximation.",
            "If approved, accept the recommended simplified config.",
            "If not approved, refine the geometry or defer until the runtime supports it.",
        ]
    if status == "unsupported":
        return [
            "Select a supported approximation or reduce the scope.",
            "Keep unsupported capabilities in the roadmap instead of silently simplifying them.",
        ]
    if has_config:
        return ["Review and accept the candidate config.", "Run runtime preflight."]
    return ["Provide missing geometry, source, material, physics, or scoring details."]


def build_design_advice(
    simulation_design: dict[str, Any] | None,
    *,
    recommended_config: dict[str, Any] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    candidate = simulation_design if isinstance(simulation_design, dict) else {}
    setup = candidate.get("recommended_setup") if isinstance(candidate.get("recommended_setup"), dict) else {}
    goal = str(candidate.get("goal") or "").strip()
    assumptions = _strings(candidate.get("assumptions"))
    simplifications = _strings(candidate.get("simplifications"))
    decisions = _strings(candidate.get("user_decisions_required"))
    unsupported = _strings(candidate.get("unsupported_capabilities"))
    capability_check = candidate.get("capability_check") if isinstance(candidate.get("capability_check"), dict) else {}
    status = _status(candidate, recommended_config)
    runnable = status in {"candidate_config_ready", "design_ready"} and not unsupported and not capability_check.get("requires_user_approval")
    title = str(setup.get("design_rationale") or "").strip()
    if not title:
        geometry = str(setup.get("geometry") or "simulation")
        material = str(setup.get("material") or "material")
        source = str(setup.get("source") or "source")
        title = f"{geometry} / {material} / {source}"
    tradeoffs = []
    if simplifications:
        tradeoffs.extend(simplifications)
    if unsupported:
        tradeoffs.append("Some requested capabilities are not supported by the current runtime.")
    if not tradeoffs:
        tradeoffs.append("Keeps the first pass narrow enough to compile and run before adding model complexity.")

    primary = DesignOption(
        option_id="primary",
        title=title,
        purpose="Produce a bounded Geant4 setup that can be reviewed and moved toward runtime execution.",
        setup=_setup_summary(setup),
        tradeoffs=tradeoffs,
        assumptions=assumptions,
        requires_user_approval=bool(decisions or capability_check.get("requires_user_approval")),
        runnable_with_current_runtime=bool(runnable and recommended_config),
    ).to_dict()

    alternatives: list[dict[str, Any]] = []
    for index, item in enumerate(_strings(setup.get("alternatives_considered")), start=1):
        alternatives.append(
            DesignOption(
                option_id=f"alternative_{index}",
                title=f"Alternative considered {index}",
                purpose=item,
                setup={},
                tradeoffs=["Rejected or deferred by the current design rationale."],
                assumptions=[],
                requires_user_approval=False,
                runnable_with_current_runtime=False,
            ).to_dict()
        )
    if unsupported and not alternatives:
        alternatives.append(
            DesignOption(
                option_id="supported_approximation",
                title="Supported approximation",
                purpose="Reduce the request to supported geometry, source, and scoring primitives.",
                setup={},
                tradeoffs=["Less physically detailed than the original request, but keeps the runtime path executable."],
                assumptions=[],
                requires_user_approval=True,
                runnable_with_current_runtime=False,
            ).to_dict()
        )

    advice = DesignAdvice(
        goal=goal,
        status=status,
        primary_option=primary,
        alternative_options=alternatives,
        assumptions=assumptions,
        user_decisions_required=decisions,
        unsupported_capabilities=unsupported,
        recommended_next_steps=_recommended_next_steps(status, has_config=bool(recommended_config)),
        user_visible_summary=_summary_text(
            lang=lang,
            status=status,
            goal=goal,
            option_title=title,
            decisions=decisions,
            unsupported=unsupported,
        ),
    )
    return advice.to_dict()


__all__ = [
    "DESIGN_ADVICE_SCHEMA_VERSION",
    "DesignAdvice",
    "DesignOption",
    "build_design_advice",
]
