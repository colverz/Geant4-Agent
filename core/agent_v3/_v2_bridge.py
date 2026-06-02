from __future__ import annotations

from typing import Any

from core.agent.simulation_design import (
    build_simulation_design_candidate,
    build_simulation_design_reference_pack,
    default_runtime_capabilities,
)
from core.agent.simulation_design_llm import build_llm_simulation_design_candidate


def bridge_design_candidate(goal: str, *, runtime_capabilities: dict[str, Any] | None = None) -> Any:
    """v3 → v2 bridge: build a SimulationDesignCandidate from a user goal."""
    return build_simulation_design_candidate(goal, runtime_capabilities=runtime_capabilities)


def bridge_design_reference_pack(goal: str, *, runtime_capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """v3 → v2 bridge: build a reference knowledge pack for simulation design."""
    return build_simulation_design_reference_pack(goal, runtime_capabilities=runtime_capabilities)


def bridge_runtime_capabilities() -> dict[str, Any]:
    """v3 → v2 bridge: return the default runtime capabilities payload."""
    return default_runtime_capabilities()


def bridge_llm_design_candidate(
    goal: str,
    *,
    config_path: str,
    lang: str = "zh-CN",
    runtime_capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """v3 → v2 bridge: build a SimulationDesign candidate via LLM."""
    return build_llm_simulation_design_candidate(
        goal,
        config_path=config_path,
        lang=lang,
        runtime_capabilities=runtime_capabilities,
    )
