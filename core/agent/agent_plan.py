from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


AGENT_PLAN_SCHEMA_VERSION = "agent_plan.v1"
AGENT_STATE_SCHEMA_VERSION = "agent_state.v1"


@dataclass(frozen=True)
class AgentPlan:
    schema_version: str = AGENT_PLAN_SCHEMA_VERSION
    goal: str = ""
    candidate_set: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate_id: str = ""
    required_runtime_capabilities: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    user_decisions_required: list[str] = field(default_factory=list)
    next_action: str = "needs_more_information"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentState:
    schema_version: str = AGENT_STATE_SCHEMA_VERSION
    goal_understood: bool = False
    plan_proposed: bool = False
    candidate_compiled: bool = False
    config_committed: bool = False
    runtime_ready: bool = False
    result_available: bool = False
    needs_revision: bool = False
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_agent_plan(
    simulation_design: dict[str, Any] | None,
    *,
    recommended_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    design = simulation_design if isinstance(simulation_design, dict) else {}
    candidate_id = "candidate_1"
    observables = [str(item) for item in design.get("observables") or [] if str(item)]
    unsupported = [str(item) for item in design.get("unsupported_capabilities") or [] if str(item)]
    simplifications = [str(item) for item in design.get("simplifications") or [] if str(item)]
    capability_check = design.get("capability_check") if isinstance(design.get("capability_check"), dict) else {}
    required = []
    for item in observables:
        required.append(f"scoring:{item}")
    setup = design.get("recommended_setup") if isinstance(design.get("recommended_setup"), dict) else {}
    if setup.get("geometry"):
        required.append(f"geometry:{setup['geometry']}")
    if setup.get("source"):
        required.append(f"source:{setup['source']}")
    for item in unsupported:
        required.append(f"unsupported:{item}")

    candidate = {
        "candidate_id": candidate_id,
        "simulation_design": design,
        "recommended_config": recommended_config or {},
        "compile_status": "compiled" if isinstance(recommended_config, dict) and recommended_config else "design_only",
        "capability_check": capability_check,
    }
    plan = AgentPlan(
        goal=str(design.get("goal") or ""),
        candidate_set=[candidate],
        selected_candidate_id=candidate_id,
        required_runtime_capabilities=list(dict.fromkeys(required)),
        success_metrics=observables,
        assumptions=[str(item) for item in design.get("assumptions") or [] if str(item)],
        user_decisions_required=[str(item) for item in design.get("user_decisions_required") or [] if str(item)],
        next_action=str(design.get("next_action") or "needs_more_information"),
    )
    if simplifications and not plan.user_decisions_required:
        plan = AgentPlan(
            goal=plan.goal,
            candidate_set=plan.candidate_set,
            selected_candidate_id=plan.selected_candidate_id,
            required_runtime_capabilities=plan.required_runtime_capabilities,
            success_metrics=plan.success_metrics,
            assumptions=plan.assumptions,
            user_decisions_required=[f"Approve simplification: {item}" for item in simplifications],
            next_action=plan.next_action,
        )
    return plan.to_dict()


def build_agent_state(
    *,
    plan: dict[str, Any] | None = None,
    candidate_status: dict[str, Any] | None = None,
    runtime_ready: bool = False,
    result_available: bool = False,
    needs_revision: bool = False,
) -> dict[str, Any]:
    has_plan = isinstance(plan, dict) and bool(plan)
    status = candidate_status if isinstance(candidate_status, dict) else {}
    committed = status.get("status") == "committed"
    compiled = False
    if has_plan:
        for candidate in plan.get("candidate_set") or []:
            if isinstance(candidate, dict) and candidate.get("compile_status") == "compiled":
                compiled = True
                break
    state = AgentState(
        goal_understood=has_plan and bool(plan.get("goal")),
        plan_proposed=has_plan,
        candidate_compiled=compiled,
        config_committed=committed,
        runtime_ready=runtime_ready,
        result_available=result_available,
        needs_revision=needs_revision,
        done=bool(result_available and not needs_revision),
    )
    return state.to_dict()


__all__ = [
    "AGENT_PLAN_SCHEMA_VERSION",
    "AGENT_STATE_SCHEMA_VERSION",
    "AgentPlan",
    "AgentState",
    "build_agent_plan",
    "build_agent_state",
]
