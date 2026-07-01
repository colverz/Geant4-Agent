from __future__ import annotations

import json

from .context import build_v3_context_pack
from .contracts import (
    V3ActionKind,
    V3ActionProposal,
    V3AgentState,
    V3Observation,
    V3ObservationStatus,
    V3ToolCall,
    V3ToolRiskLevel,
    V3TurnInput,
)
from .patches import (
    build_patches_from_config_overrides,
    build_patches_from_requested_changes,
    config_overrides_from_llm_parameters,
)
from .pending_action import runtime_authorization_source, runtime_execution_authorized
from .result_explainer import build_v3_runtime_result_answer, looks_like_result_question
from .runtime_policy import runtime_tool_arguments
from .tools.geant4_tools import (
    GEANT4_CAPABILITY_TOOL,
    GEANT4_DESIGN_TEMPLATE_TOOL,
    GEANT4_LLM_DESIGN_TOOL,
    GEANT4_PAYLOAD_BUILDER_TOOL,
    GEANT4_RUNTIME_PREFLIGHT_TOOL,
    GEANT4_RUNTIME_TOOL,
)


class BasicGeant4Reasoner:
    """Small deterministic v3 reasoner used for smoke tests before live LLM wiring."""

    def propose(self, turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
        if not state.goal:
            state.goal = turn.user_text
        if _state_patch_needs_clarification(turn):
            return _invalid_patch_proposal(turn, state)
        runtime_observation = _latest_observation_data(state, GEANT4_RUNTIME_TOOL)
        runtime_status = _latest_observation_status(state, GEANT4_RUNTIME_TOOL)
        if _looks_like_current_configuration_question(turn.user_text):
            design = _latest_design(state)
            payload = _latest_payload(state)
            evidence = []
            if design:
                evidence.append({"source": _latest_design_source(state), "role": "design"})
            if payload:
                evidence.append({"source": GEANT4_PAYLOAD_BUILDER_TOOL, "role": "runtime_payload"})
            return V3ActionProposal(
                kind=V3ActionKind.FINAL_ANSWER,
                intent="present_current_geant4_configuration",
                arguments={"message": _answer_for_current_configuration(payload, design, turn.locale)},
                evidence=evidence,
            )
        if looks_like_result_question(turn.user_text):
            explanation = build_v3_runtime_result_answer(
                turn.user_text,
                runtime_observation,
                locale=turn.locale,
                use_llm=bool(turn.metadata.get("llm_result_enabled")),
                llm_config_path=str(turn.metadata.get("llm_config_path") or ""),
            )
            evidence = [
                {
                    "source": GEANT4_RUNTIME_TOOL,
                    "role": "runtime_observation",
                    "status": runtime_status,
                    "result_explanation": {
                        "source": explanation.get("source"),
                        "prompt_profile_id": explanation.get("prompt_profile_id"),
                        "fallback_reason": explanation.get("fallback_reason"),
                    },
                }
            ] if runtime_observation else []
            return V3ActionProposal(
                kind=V3ActionKind.FINAL_ANSWER,
                intent="answer_geant4_runtime_result_question",
                arguments={
                    "message": explanation["message"]
                    or ("当前还没有可解释的 Geant4 运行结果。" if str(turn.locale).lower().startswith("zh") else "No Geant4 runtime result is available yet."),
                    "result_explanation": explanation,
                },
                evidence=evidence,
            )
        if not _has_observation(state, GEANT4_CAPABILITY_TOOL):
            return V3ActionProposal(
                kind=V3ActionKind.CREATE_DESIGN,
                intent="inspect_geant4_capability",
                tool_call=V3ToolCall(
                    tool_name=GEANT4_CAPABILITY_TOOL,
                    arguments={"goal": turn.user_text},
                    risk_level=V3ToolRiskLevel.READ_ONLY,
                ),
                expected_observation="Geant4 capability report",
            )
        if not _has_design_observation(state):
            runtime_capabilities = _latest_runtime_capabilities(state)
            if _llm_design_enabled(turn):
                return V3ActionProposal(
                    kind=V3ActionKind.CREATE_DESIGN,
                    intent="draft_geant4_design_with_llm",
                    arguments={"artifact_id": "geant4_design_draft"},
                    tool_call=V3ToolCall(
                        tool_name=GEANT4_LLM_DESIGN_TOOL,
                        arguments={
                            "goal": turn.user_text,
                            "artifact_id": "geant4_design_draft",
                            "runtime_capabilities": runtime_capabilities,
                            "llm_config_path": str(turn.metadata.get("llm_config_path") or ""),
                            "lang": turn.locale,
                        },
                        risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                    ),
                    expected_observation="LLM-assisted SimulationDesign draft",
                )
            return V3ActionProposal(
                kind=V3ActionKind.CREATE_DESIGN,
                intent="draft_geant4_design",
                arguments={"artifact_id": "geant4_design_draft"},
                tool_call=V3ToolCall(
                    tool_name=GEANT4_DESIGN_TEMPLATE_TOOL,
                    arguments={
                        "goal": turn.user_text,
                        "artifact_id": "geant4_design_draft",
                        "runtime_capabilities": runtime_capabilities,
                    },
                    risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                ),
                expected_observation="SimulationDesign draft",
            )
        design = _latest_design(state)
        user_explicitly_accepted = _accept_defaults(turn)
        # Design needs user input and user hasn't accepted → ask
        if design and _needs_user_choice(design) and not user_explicitly_accepted:
            question = _question_for_design(design)
            return V3ActionProposal(
                kind=V3ActionKind.ASK_USER,
                intent="confirm_geant4_design_assumption",
                arguments={
                    "question": question,
                    "options": ["接受默认近似并生成配置", "调整几何/材料/源项", "先只查看方案说明"],
                },
                evidence=[{"source": _latest_design_source(state), "role": "design"}],
            )
        # User accepted + design has unresolved questions: re-design first.
        # If the same turn already carries concrete edits or a run request, the user has
        # effectively chosen to proceed from the current design, so build payload instead
        # of spending another LLM design pass.
        should_build_from_current_design = _has_config_overrides(turn) or _run_requested(turn)
        if (
            user_explicitly_accepted
            and design
            and _needs_user_choice(design)
            and not should_build_from_current_design
            and not _has_observation(state, GEANT4_PAYLOAD_BUILDER_TOOL)
        ):
            # Count LLM design attempts to prevent infinite loop
            design_attempts = sum(1 for o in state.observations if o.source == GEANT4_LLM_DESIGN_TOOL)
            if design_attempts <= 2:
                return V3ActionProposal(
                    kind=V3ActionKind.CREATE_DESIGN,
                    intent="draft_geant4_design_with_llm",
                    arguments={"artifact_id": "geant4_design_draft"},
                    tool_call=V3ToolCall(
                        tool_name=GEANT4_LLM_DESIGN_TOOL,
                        arguments={
                            "goal": str(state.goal or "") + " | User answers: " + turn.user_text,
                            "artifact_id": "geant4_design_draft",
                            "runtime_capabilities": _latest_runtime_capabilities(state),
                            "llm_config_path": str(turn.metadata.get("llm_config_path") or ""),
                            "lang": turn.locale,
                        },
                        risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                    ),
                    expected_observation="LLM-assisted SimulationDesign draft",
                )
            # After 2 design attempts, skip to payload regardless
            return V3ActionProposal(
                kind=V3ActionKind.DRAFT_SPEC,
                intent="draft_geant4_runtime_payload",
                arguments={"artifact_id": "geant4_runtime_payload_draft"},
                tool_call=V3ToolCall(
                    tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                    arguments={
                        "design": design,
                        "events": turn.metadata.get("events", 1000),
                        "config_overrides": _config_overrides(turn),
                        "accept_defaults": True,
                        "artifact_id": "geant4_runtime_payload_draft",
                    },
                    risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                ),
                expected_observation="RuntimePayload draft",
            )
        # Build payload: user accepted, run requested, or config overrides present
        if (user_explicitly_accepted or _run_requested(turn) or _has_config_overrides(turn)) and design and not _has_observation(state, GEANT4_PAYLOAD_BUILDER_TOOL):
            return V3ActionProposal(
                kind=V3ActionKind.DRAFT_SPEC,
                intent="draft_geant4_runtime_payload",
                arguments={"artifact_id": "geant4_runtime_payload_draft"},
                tool_call=V3ToolCall(
                    tool_name=GEANT4_PAYLOAD_BUILDER_TOOL,
                    arguments={
                        "design": design,
                        "events": turn.metadata.get("events", 1000),
                        "config_overrides": _config_overrides(turn),
                        "accept_defaults": _accept_defaults(turn),
                        "artifact_id": "geant4_runtime_payload_draft",
                    },
                    risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                ),
                expected_observation="RuntimePayload draft",
            )
        payload = _latest_payload(state)
        if _run_requested(turn) and payload and not _has_observation(state, GEANT4_RUNTIME_PREFLIGHT_TOOL):
            return V3ActionProposal(
                kind=V3ActionKind.DRAFT_SPEC,
                intent="preflight_geant4_runtime",
                tool_call=V3ToolCall(
                    tool_name=GEANT4_RUNTIME_PREFLIGHT_TOOL,
                    arguments={
                        "payload_builder_observation": payload,
                        "events": turn.metadata.get("events", 1000),
                        **runtime_tool_arguments(turn),
                    },
                    risk_level=V3ToolRiskLevel.DRAFT_ONLY,
                ),
                expected_observation="Runtime preflight observation",
            )
        preflight = _latest_observation_data(state, GEANT4_RUNTIME_PREFLIGHT_TOOL)
        preflight_status = _latest_observation_status(state, GEANT4_RUNTIME_PREFLIGHT_TOOL)
        if _run_requested(turn) and payload and preflight_status == "ok" and not _has_observation(state, GEANT4_RUNTIME_TOOL):
            authorized = runtime_execution_authorized(turn.metadata)
            return V3ActionProposal(
                kind=V3ActionKind.RUN_SIMULATION,
                intent="run_geant4_runtime",
                arguments={"authorization_source": runtime_authorization_source(turn.metadata)},
                tool_call=V3ToolCall(
                    tool_name=GEANT4_RUNTIME_TOOL,
                    arguments={
                        "payload_builder_observation": payload,
                        "events": turn.metadata.get("events", 1000),
                        **runtime_tool_arguments(turn),
                    },
                    risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                    idempotency_key=str(turn.metadata.get("run_id") or f"{turn.session_id}:run"),
                ),
                risk_level=V3ToolRiskLevel.RUNTIME_EXECUTION,
                confirmed=authorized,
                expected_observation="Geant4 runtime observation",
            )
        if _run_requested(turn) and runtime_observation:
            explanation = build_v3_runtime_result_answer(
                turn.user_text,
                runtime_observation,
                locale=turn.locale,
                use_llm=bool(turn.metadata.get("llm_result_enabled")),
                llm_config_path=str(turn.metadata.get("llm_config_path") or ""),
            )
            return V3ActionProposal(
                kind=V3ActionKind.FINAL_ANSWER,
                intent="present_geant4_runtime_observation",
                arguments={"message": explanation["message"], "result_explanation": explanation},
                evidence=[
                    {
                        "source": GEANT4_RUNTIME_TOOL,
                        "role": "runtime_observation",
                        "status": runtime_status,
                        "result_explanation": {
                            "source": explanation.get("source"),
                            "prompt_profile_id": explanation.get("prompt_profile_id"),
                            "fallback_reason": explanation.get("fallback_reason"),
                        },
                    }
                ],
            )
        if _run_requested(turn) and payload and preflight:
            return V3ActionProposal(
                kind=V3ActionKind.FINAL_ANSWER,
                intent="present_geant4_preflight_result",
                arguments={"message": _answer_for_preflight(preflight, preflight_status, turn.user_text)},
                evidence=[{"source": GEANT4_RUNTIME_PREFLIGHT_TOOL, "role": "runtime_preflight"}],
            )
        if payload:
            return V3ActionProposal(
                kind=V3ActionKind.FINAL_ANSWER,
                intent="present_geant4_payload_draft",
                arguments={"message": _answer_for_payload(payload, design, turn.user_text)},
                evidence=[
                    {"source": _latest_design_source(state), "role": "design"},
                    {"source": GEANT4_PAYLOAD_BUILDER_TOOL, "role": "runtime_payload"},
                ],
            )
        return V3ActionProposal(
            kind=V3ActionKind.FINAL_ANSWER,
            intent="present_geant4_design",
            arguments={"message": _answer_for_design(design, turn.user_text)},
            evidence=[{"source": _latest_design_source(state), "role": "design"}] if design else [],
        )


def _has_observation(state: V3AgentState, source: str) -> bool:
    return any(observation.source == source for observation in state.observations)


def _has_design_observation(state: V3AgentState) -> bool:
    return any(observation.source in {GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_LLM_DESIGN_TOOL} for observation in state.observations)


def _llm_design_enabled(turn: V3TurnInput) -> bool:
    return bool(turn.metadata.get("llm_design_enabled")) and bool(str(turn.metadata.get("llm_config_path") or "").strip())


def _has_config_overrides(turn: V3TurnInput) -> bool:
    return bool(_config_overrides(turn))


def _config_overrides(turn: V3TurnInput) -> dict:
    value = turn.metadata.get("config_overrides")
    return value if isinstance(value, dict) else {}


def _latest_runtime_capabilities(state: V3AgentState) -> dict:
    for observation in reversed(state.observations):
        if observation.source == GEANT4_CAPABILITY_TOOL:
            data = observation.data if isinstance(observation.data, dict) else {}
            caps = data.get("design_capabilities")
            return caps if isinstance(caps, dict) else {}
    return {}


def _latest_design(state: V3AgentState) -> dict:
    for observation in reversed(state.observations):
        if observation.source in {GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_LLM_DESIGN_TOOL}:
            data = observation.data if isinstance(observation.data, dict) else {}
            design = data.get("design")
            return design if isinstance(design, dict) else {}
    return {}


def _latest_design_source(state: V3AgentState) -> str:
    for observation in reversed(state.observations):
        if observation.source in {GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_LLM_DESIGN_TOOL}:
            return observation.source
    return GEANT4_DESIGN_TEMPLATE_TOOL


def _latest_payload(state: V3AgentState) -> dict:
    for observation in reversed(state.observations):
        if observation.source == GEANT4_PAYLOAD_BUILDER_TOOL:
            data = observation.data if isinstance(observation.data, dict) else {}
            return data if data else {}
    return {}


def _latest_observation_data(state: V3AgentState, source: str) -> dict:
    for observation in reversed(state.observations):
        if observation.source == source:
            return observation.data if isinstance(observation.data, dict) else {}
    return {}


def _latest_observation_status(state: V3AgentState, source: str) -> str:
    for observation in reversed(state.observations):
        if observation.source == source:
            return observation.status.value
    return ""


def _looks_like_current_configuration_question(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    configuration_focus = (
        "current configured",
        "currently configured",
        "current config",
        "current configuration",
        "what is configured",
        "what is currently configured",
        "configured source",
        "configured material",
        "configured energy",
        "runtime payload",
        "payload config",
        "当前配置",
        "现在配置",
        "目前配置",
        "配置里面",
        "配置里",
    )
    result_focus = (
        "result",
        "results",
        "runtime result",
        "what happened",
        "how did it go",
        "target_edep",
        "detector_crossing",
        "plane_crossing",
        "edep",
        "dose",
        "hit",
        "命中",
        "结果",
        "计分",
        "沉积",
    )
    return any(token in lowered for token in configuration_focus) and not any(token in lowered for token in result_focus)


def _accept_defaults(turn: V3TurnInput) -> bool:
    if bool(turn.metadata.get("accept_defaults")):
        return True
    text = str(turn.user_text or "")
    return any(token in text.lower() for token in ("accept defaults", "use defaults")) or any(
        token in text for token in ("接受默认", "按默认", "用默认")
    )


def _run_requested(turn: V3TurnInput) -> bool:
    if bool(turn.metadata.get("suppress_run")):
        return False
    if bool(turn.metadata.get("run")):
        return True
    text = str(turn.user_text or "")
    lowered = text.lower()
    if any(token in lowered for token in ("do not run", "don't run", "no run", "without running")):
        return False
    if any(token in text for token in ("取消运行", "不要运行", "不运行", "先不要运行", "先不要跑", "不要跑", "只给方案", "只保留方案", "只设计")):
        return False
    return any(token in lowered for token in ("run", "execute", "simulate now")) or any(
        token in text for token in ("运行", "执行", "跑一下", "再跑", "开始模拟")
    )


def _needs_user_choice(design: dict) -> bool:
    na = design.get("next_action") or ""
    if na in ("ask_user_to_choose_approximation", "unsupported_capability", "needs_more_information"):
        return True
    return bool(design.get("user_decisions_required"))


def _question_for_design(design: dict) -> str:
    decisions = [str(item) for item in design.get("user_decisions_required") or []]
    if decisions:
        return "我已经形成 Geant4 草案，但需要你确认：" + "；".join(decisions)
    return "我已经形成 Geant4 草案。是否接受默认近似并继续生成配置？"


def _answer_for_design(design: dict, fallback_goal: str) -> str:
    if not design:
        return 'I have not formed a Geant4 design draft yet.'
    setup = design.get('recommended_setup') if isinstance(design.get('recommended_setup'), dict) else {}
    obs_list = [str(item) for item in design.get('observables') or []]
    observables = ', '.join(obs_list) or 'TBD'
    assumption_list = design.get('assumptions') or []
    assumptions = '; '.join(_localize_assumption(str(item)) for item in assumption_list) or 'using default assumptions'
    geometry = setup.get('geometry') or 'single_box'
    material = setup.get('material') or 'G4_Cu'
    source = setup.get('source') or 'beam'
    next_action = design.get('next_action') or 'needs_more_information'
    goal = _clean_goal(str(design.get('goal') or fallback_goal))
    return (
        'Geant4 design draft for [' + goal + ']: '
        + 'geometry=' + geometry + ', material=' + material
        + ', source=' + source + ', observables=' + observables + '. '
        + 'Assumptions: ' + assumptions + '. Next: ' + next_action + '.'
    )


def _answer_for_payload(payload: dict, design: dict, fallback_goal: str) -> str:
    spec = payload.get("simulation_spec") if isinstance(payload.get("simulation_spec"), dict) else {}
    runtime_payload = payload.get("runtime_payload") if isinstance(payload.get("runtime_payload"), dict) else {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    run = spec.get("run") if isinstance(spec.get("run"), dict) else {}
    scoring = spec.get("scoring") if isinstance(spec.get("scoring"), dict) else {}
    goal = _clean_goal(str((design or {}).get("goal") or fallback_goal))
    metrics = runtime_payload.get("scoring") if isinstance(runtime_payload.get("scoring"), dict) else {}
    scoring_names = ", ".join(str(item) for item in scoring.get("volume_names") or []) or "Target"
    detector = "已启用" if spec.get("detector_enabled") else "未启用"
    return (
        f"已按默认假设把'{goal}'推进为 Geant4 runtime payload 草案："
        f"几何 {geometry.get('structure')}，材料 {geometry.get('material')}，"
        f"源项 {source.get('particle')} {source.get('energy_mev')} MeV，"
        f"事件数 {run.get('events')}，探测器{detector}。"
        f"scoring 目标体包括 {scoring_names}；payload schema 为 {runtime_payload.get('schema_version')}。"
        f"当前还没有执行 Geant4，下一步需要 runtime preflight 后再运行。"
        f"启用的 scoring 标志：target_edep={metrics.get('target_edep')}, detector_crossings={metrics.get('detector_crossings')}。"
    )


def _answer_for_current_configuration(payload: dict, design: dict, locale: str) -> str:
    english = str(locale).lower().startswith("en")
    if payload:
        spec = payload.get("simulation_spec") if isinstance(payload.get("simulation_spec"), dict) else {}
        geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
        source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
        run = spec.get("run") if isinstance(spec.get("run"), dict) else {}
        material = geometry.get("material") or "TBD"
        structure = geometry.get("structure") or "TBD"
        particle = source.get("particle") or "TBD"
        energy = source.get("energy_mev")
        events = run.get("events") or "TBD"
        if english:
            return (
                "Current runtime payload: "
                f"geometry={structure}, material={material}, source={particle}"
                + (f" {energy} MeV" if energy is not None else "")
                + f", events={events}. It has not run yet unless a runtime result is already shown."
            )
        return (
            "当前运行配置是："
            f"几何={structure}，材料={material}，源粒子={particle}"
            + (f" {energy} MeV" if energy is not None else "")
            + f"，事件数={events}。除非界面已有运行结果，否则这只是配置，还没有执行 Geant4。"
        )
    if design:
        setup = design.get("recommended_setup") if isinstance(design.get("recommended_setup"), dict) else {}
        material = setup.get("material") or "TBD"
        geometry = setup.get("geometry") or "TBD"
        source = setup.get("source") or "beam"
        particle = setup.get("source_particle") or "TBD"
        energy = setup.get("source_energy_mev")
        observables = ", ".join(str(item) for item in design.get("observables") or []) or "TBD"
        if english:
            return (
                "Current design draft: "
                f"geometry={geometry}, material={material}, source={source}, particle={particle}"
                + (f", energy={energy} MeV" if energy is not None else "")
                + f", observables={observables}. No runtime payload has been generated yet."
            )
        return (
            "当前只有方案草稿："
            f"几何={geometry}，材料={material}，源={source}，粒子={particle}"
            + (f"，能量={energy} MeV" if energy is not None else "")
            + f"，关注量={observables}。还没有生成 runtime payload。"
        )
    return (
        "No Geant4 design or runtime payload is configured yet."
        if english
        else "当前还没有 Geant4 方案或运行配置。"
    )


def _answer_for_preflight(preflight: dict, status: str, fallback_goal: str) -> str:
    validation = preflight.get("validation") if isinstance(preflight.get("validation"), dict) else {}
    adapter = preflight.get("adapter") or "unknown"
    if status == "not_evaluable":
        reason = preflight.get("not_evaluable_reason") or "local_process_runtime_required"
        return (
            f"已完成 Geant4 运行前检查，配置结构可检查，但当前不能报告真实模拟结果：{reason}。"
            f"当前 runtime adapter 是 {adapter}；需要配置真实 local_process Geant4 runtime 后才能运行。"
        )
    if status == "blocked":
        missing = []
        payload = validation.get("payload") if isinstance(validation.get("payload"), dict) else {}
        if isinstance(payload.get("missing_paths"), list):
            missing = [str(item) for item in payload["missing_paths"]]
        return f"Geant4 运行前检查未通过，缺失或无效字段：{', '.join(missing) or 'unknown'}。"
    return f"Geant4 运行前检查通过，runtime adapter={adapter}。"


def _answer_for_runtime(observation: dict, status: str, fallback_goal: str) -> str:
    if status == "not_evaluable":
        reason = observation.get("not_evaluable_reason") or "local_process_runtime_required"
        return f"尚未执行真实 Geant4 模拟：{reason}。我不会把未运行的结果当作模拟结论。"
    if status != "ok":
        return "Geant4 运行未成功，已将运行观察和日志信息记录到 trace。"
    result_summary = observation.get("result_summary") if isinstance(observation.get("result_summary"), dict) else {}
    run = result_summary.get("run") if isinstance(result_summary.get("run"), dict) else {}
    scoring = result_summary.get("scoring") if isinstance(result_summary.get("scoring"), dict) else {}
    return (
        "Geant4 运行完成。"
        f"事件数：{run.get('events_completed', run.get('events_requested', 'unknown'))}；"
        f"scoring 摘要：{scoring if scoring else '已生成 run summary'}。"
    )


def _clean_goal(goal: str) -> str:
    marker = "\n\nDesign hints:"
    if marker in goal:
        return goal.split(marker, 1)[0].strip()
    return goal.strip()


def _localize_assumption(value: str) -> str:
    mapping = {
        "Use current single-thread deterministic runtime defaults unless the user specifies otherwise.": "除非用户另行指定，先使用当前单线程确定性运行默认值",
        "Gamma energy is not explicit and must be confirmed before final configuration.": "gamma 能量尚未明确，生成最终配置前需要确认",
        "Detector is treated as downstream along the source direction when direction is not otherwise specified.": "未指定方向时，探测器默认放在源方向下游",
    }
    return mapping.get(value, value)


class LLMGeant4Reasoner:
    """LLM-native v3 reasoner with plan cache and deep Geant4 expertise.

    Calls the LLM once per turn to form a comprehensive understanding and plan.
    Subsequent controller steps execute deterministically from the cached plan.
    After a runtime observation, the LLM analyzes results and suggests next steps."""

    def __init__(self, llm_config_path: str = "", *, lang: str = "zh-CN") -> None:
        self._llm_config_path = llm_config_path
        self._lang = lang
        self._fallback = BasicGeant4Reasoner()

    def propose(self, turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
        if not self._llm_config_path or not self._llm_config_path.strip():
            return self._fallback.propose(turn, state)
        try:
            result = self._llm_propose(turn, state)
            if result is not None:
                return result
        except Exception:
            pass
        return self._fallback.propose(turn, state)

    def _llm_propose(self, turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
        if not state.goal:
            state.goal = turn.user_text
        if _state_patch_needs_clarification(turn):
            return _invalid_patch_proposal(turn, state)

        # Always check for result analysis first (regardless of step)
        has_runtime = _has_observation(state, "geant4_runtime_tool")
        if has_runtime:
            # Check if user is asking about results
            text_lower = str(turn.user_text or "").lower()
            analysis_triggers = ("分析", "解释", "说明", "为什么", "合理", "怎么样", "如何",
                                "找到", "优化", "最优", "最小", "最大", "刚好", "恰好", "调整",
                                "analyze", "explain", "why", "reasonable", "how", "what does",
                                "find", "optimize", "optimal", "minimum", "maximum", "thickness")
            is_analysis_q = any(t in text_lower for t in analysis_triggers)
            # Also run analysis if auto_sweep is pending
            has_auto_sweep = isinstance(turn.metadata.get("auto_sweep"), dict)
            if is_analysis_q or has_auto_sweep:
                try:
                    analysis = self._call_llm_analyze_result(turn, state)
                except Exception:
                    analysis = None
                if analysis is not None:
                    if _analysis_conflicts_with_runtime(analysis, _latest_observation_data(state, "geant4_runtime_tool")):
                        return self._hard_fallback_result(turn, state)
                    sweep = analysis.get("sweep_suggestion")
                    auto = analysis.get("auto_sweep")
                    if isinstance(sweep, dict) and auto:
                        turn.metadata["auto_sweep"] = sweep
                    opt = analysis.get("optimization")
                    if isinstance(opt, dict) and not opt.get("goal_met") and opt.get("next_value") is not None:
                        turn.metadata["optimization"] = opt
                    suggestions = analysis.get("suggestions")
                    if isinstance(suggestions, list) and suggestions:
                        turn.metadata["suggestions"] = suggestions
                    return self._build_proposal(analysis, turn) or self._fallback.propose(turn, state)
                return self._hard_fallback_result(turn, state)

        # LLM comprehension only on first controller step
        is_first_step = len(state.observations) == 0 or all(
            obs.source == "commit_gate" for obs in state.observations
        )

        if is_first_step:
            parsed = self._call_llm_once(turn, state)
            if parsed is None:
                return self._fallback.propose(turn, state)
            _apply_llm_change_patches(parsed, turn, state)
            if _state_patch_needs_clarification(turn):
                return _invalid_patch_proposal(turn, state)
            return self._build_proposal(parsed, turn) or self._fallback.propose(turn, state)

        # Subsequent steps: deterministic BasicGeant4Reasoner
        return self._fallback.propose(turn, state)

    # ── LLM calls ─────────────────────────────────────────────────

    def _call_llm_once(self, turn: V3TurnInput, state: V3AgentState) -> dict | None:
        prompt = self._comprehension_prompt(turn, state)
        raw = self._call_llm(prompt)
        parsed = self._parse_llm_response(raw)
        return parsed

    def _call_llm_analyze_result(self, turn: V3TurnInput, state: V3AgentState) -> dict | None:
        runtime_data = _latest_observation_data(state, "geant4_runtime_tool")
        prompt = self._result_analysis_prompt(turn, state, runtime_data)
        raw = self._call_llm(prompt)
        parsed = self._parse_llm_response(raw)
        return parsed

    # ── Prompts ───────────────────────────────────────────────────

    def _comprehension_prompt(self, turn: V3TurnInput, state: V3AgentState) -> str:
        catalog = self._knowledge_catalog()
        context_pack = build_v3_context_pack(state, last_user_turn=turn.user_text)
        context_json = json.dumps(context_pack, ensure_ascii=False, indent=2)
        lang = self._lang
        return f"""You are an expert Geant4 simulation physicist and agent planner.

Your job: understand the user's goal, extract physics parameters, validate physical
reasonableness, and produce an execution plan.

{self._geant4_expertise()}

Knowledge catalog (use these IDs):
{catalog}

User said: {turn.user_text}
Accept defaults: {bool(turn.metadata.get('accept_defaults'))}
Run requested: {bool(turn.metadata.get('run'))}
V3 context pack (safe, summarized, authoritative for prior state):
{context_json}

Return JSON:
"understanding": one sentence describing the user's physics goal, in natural language.
"parameters": extracted physics parameters (use catalog IDs, include ONLY what the user specified):
  energy_mev (number), particle (string from catalog), source_type (beam/point/isotropic),
  material (G4_ ID), environment_material (G4_ ID), events (integer), observables (list),
  physics_list (optional). Use scenario-appropriate defaults when not specified:
  - Medical proton: 150 MeV, proton, G4_WATER, depth_bins
  - Gamma shielding: 1 MeV, gamma, G4_Pb, detector_crossing_count
  - NDT: 0.5 MeV, gamma, G4_Al, region_contrast
  - Space radiation: isotropic source, G4_Galactic environment, proton 100 MeV
  - Neutron: 2 MeV, neutron, G4_POLYETHYLENE, Shielding physics list
  geometry: structured object:
    {{"volumes": [{{"name": "descriptive", "shape": "box|sphere|tubs|cons",
      "material": "G4_ ID", "dimensions": {{shape-appropriate keys}} }}],
     "environment": {{"material": "G4_ ID"}} }}
  For boxes: size_x/y/z_mm. For spheres: radius_mm. For tubs: radius_mm, half_length_mm.
  Always set environment_material=G4_Galactic for vacuum/space scenarios.
  If the user describes multi-layer, use multiple volumes with appropriate positions.
"requested_changes": optional list of user-requested edits to an existing design/payload.
  Use this only when the user is revising an existing setup.
  Each item: {{"field": "source_energy_mev|run_events|target_material|target_thickness_mm|geometry_dimensions_mm|enable_downstream_scoring", "value": ..., "evidence": "short reason"}}.
"physics_check": list any physics concerns. Check:
  - Environment material appropriate? (vacuum/space → must use G4_Galactic, NOT G4_AIR)
  - Energy range reasonable for the material? (e.g. 1 keV gamma won't penetrate lead)
  - Detector placed downstream of target? (not inside or upstream)
  - Geometry dimensions positive and physically plausible?
  - Multi-layer: are layers in correct order (e.g. moderator before absorber)?
  Empty list if all checks pass.
"action_kind": "create_design" (new scenarios) or "final_answer" (questions).
"tool_name": "geant4_llm_design_tool" for designs, "" for answers.
"message": conversational 1-2 sentence response in the user's language.
  Be specific about the physics choices and why they make sense.
Return JSON only. No markdown."""

    def _result_analysis_prompt(self, turn: V3TurnInput, state: V3AgentState, runtime_data: dict) -> str:
        lang = self._lang
        context_state = state
        if runtime_data and not _latest_observation_data(state, "geant4_runtime_tool"):
            context_state = V3AgentState(
                session_id=state.session_id,
                goal=state.goal,
                assumptions=list(state.assumptions),
                active_plan=list(state.active_plan),
                observations=[
                    V3Observation(source="geant4_runtime_tool", status=V3ObservationStatus.OK, data=runtime_data),
                ],
                open_questions=list(state.open_questions),
            )
        context_pack = build_v3_context_pack(context_state, last_user_turn=turn.user_text)
        context_json = json.dumps(context_pack, ensure_ascii=False, indent=2)
        runtime_facts = context_pack.get("latest_runtime_facts") if isinstance(context_pack.get("latest_runtime_facts"), dict) else {}
        edep = runtime_facts.get("target_edep_total_mev", "N/A")
        crossings = runtime_facts.get("detector_crossing_count", "N/A")
        events_done = runtime_facts.get("events_completed", "?")
        events_req = runtime_facts.get("events_requested", "?")
        material = _first_present(runtime_facts.get("material"), "unknown")
        particle = _first_present(runtime_facts.get("particle"), "unknown")
        source_type = _first_present(runtime_facts.get("source_type"), "unknown")
        source_energy = _first_present(runtime_facts.get("source_energy_mev"), "unknown")
        physics_list = _first_present(runtime_facts.get("physics_list"), "unknown")
        geometry = _first_present(runtime_facts.get("geometry"), "unknown")
        physics_notes = _runtime_physics_notes(
            {
                "material": material,
                "particle": particle,
                "source_energy_mev": source_energy,
            }
        )
        source_info = (
            "\n\nAuthoritative runtime configuration (use this, not Session goal, for physics interpretation):"
            f"\n  geometry: {geometry}"
            f"\n  material: {material}"
            f"\n  source_type: {source_type}"
            f"\n  particle: {particle}"
            f"\n  source_energy_mev: {source_energy}"
            f"\n  physics_list: {physics_list}"
            "\nRules: do not mention lead/Pb unless material is G4_Pb; do not call primaries photons/gamma unless particle is gamma; "
            "do not mention a source energy other than source_energy_mev."
        )

        return f"""You are an expert Geant4 physicist analyzing simulation results. Think like a physicist, not a reporter.

Runtime-specific physics notes. Use only notes consistent with the authoritative runtime configuration:
{physics_notes}
- target_edep: energy deposited in the target volume.
- detector_crossing_count: particles reaching the detector behind the target.
- plane_crossing_count: particles crossing the scoring plane.
- Low event counts are statistically limited. For quantitative conclusions, suggest 1000+ events.

Runtime results:{source_info}
  events: {events_done}/{events_req}
  target_edep: {edep} MeV
  detector_crossings: {crossings}
  plane_crossings: {runtime_facts.get("plane_crossing_count", "N/A")}

User said: {turn.user_text}
V3 context pack (goal is non-authoritative; latest_runtime_facts is authoritative):
{context_json}

Your task:
1. Check whether these results are physically reasonable for the authoritative runtime configuration.
2. Explain what the numbers mean physically. Do not just restate them; explain using the runtime-specific notes above.
3. Identify the dominant interaction mechanism only when it is supported by the runtime configuration.
4. Suggest ONE concrete next step that would add insight, such as a different energy, thickness, material, or event count.
5. If a parameter sweep would reveal a trend, include a specific sweep suggestion with exact values.
6. Never invent numbers not present above.

Return JSON:
"action_kind": "final_answer"
"tool_name": ""
"message": natural physics analysis in {("Chinese" if lang.startswith("zh") else "English")}. Like a senior physicist explaining to a colleague. Use runtime facts. Name the dominant mechanism only if supported. Suggest next step.
"sweep_suggestion": {{"parameter": "...", "values": [...], "label": "..."}} if a sweep adds insight. Omit if not useful.
"auto_sweep": true if the sweep is immediately valuable. false if just a suggestion.
"optimization": {{"goal_met": bool, "parameter": "...", "current_value": N, "next_value": N, "reasoning": "..."}} for iterative optimization. Omit if not applicable.
"suggestions": REQUIRED. Always include 2-3 concrete next actions:
  [{{"text": "short suggestion", "prefill": "exact text to execute this"}}]

Return JSON only. No markdown."""

    # ── Geant4 expertise ──────────────────────────────────────────

    def _geant4_expertise(self) -> str:
        return """Geant4 physics expertise:
- FTFP_BERT: default general-purpose (hadronic + EM). Good for most shielding and detector scenarios.
- Shielding: specialized for neutron/gamma shielding, low-energy neutron transport.
- QGSP_BERT_HP: high-precision neutron (HP = High Precision neutron model < 20 MeV).
- FTFP_BERT_HP: FTFP_BERT with HP neutrons. Good for mixed neutron/gamma shielding.
- QGSP_BIC: alternative intra-nuclear cascade. Lighter nuclides.
- For medical proton/ion: FTFP_BERT is standard. For low-energy medical EM: emstandard_opt3 or Penelope.
- For space radiation: FTFP_BERT with appropriate generators. Shielding if neutron-dominated.
- Choose based on the dominant physics. If user mentions neutrons, prefer HP variants.
  If user mentions medical/dose, FTFP_BERT is fine. If unsure, default to FTFP_BERT.
- Always set environment_material=G4_Galactic for space/vacuum/太空/宇宙/真空.
  G4_AIR is ONLY for actual air gaps, never as vacuum substitute.
- Water (G4_WATER) is standard tissue-equivalent for medical/radiation protection.
  Soft tissue (G4_TISSUE_SOFT_ICRP) or bone (G4_BONE_COMPACT_ICRU) for detailed dosimetry.
- For shielding: G4_Pb (lead, high-Z gamma), G4_POLYETHYLENE (neutron moderator),
  concrete (G4_CONCRETE), multi-layer combinations for mixed fields.
- For detectors: G4_Si (silicon), G4_PLASTIC_SC_VINYLTOLUENE (scintillator).
- Isotropic source is correct for space radiation, internal contamination, or diffuse fields.
  Beam is correct for collimated lab sources. Point for simple check sources.
- Energy ranges: gamma shielding typically 0.1-10 MeV. Medical protons 50-250 MeV.
  Space radiation: broad spectrum, protons 10-1000 MeV. NDT: 0.1-1 MeV gamma.
- target_edep measures energy deposited in target volume. detector_crossing_count counts
  particles reaching detector. transmission_factor = detector/total for attenuation.
  depth_bins is needed for depth-dose curves. Use plane_crossing_count for exit dosimetry.
- Common mistakes: forgetting G4_Galactic for vacuum, using single_box when sphere needed,
  not setting detector position downstream of target, omitting depth_bins for dose profiles.
- After a simulation runs, examine results. If target_edep ≈ source_energy × events,
  the target fully absorbed the beam. If detector_crossing_count ≈ 0, the shield is
  effective. Suggest parameter sweeps for optimization."""

    # ── Helpers ───────────────────────────────────────────────────

    def _hard_fallback_result(self, turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
        data = _latest_observation_data(state, "geant4_runtime_tool")
        explanation = build_v3_runtime_result_answer(
            turn.user_text,
            data,
            locale=turn.locale,
            use_llm=False,
        )
        return V3ActionProposal(
            kind=V3ActionKind.FINAL_ANSWER,
            intent="present_raw_runtime_data",
            arguments={"message": explanation.get("message") or "Geant4 run completed. See trace for details."},
            evidence=[{"source": "geant4_runtime_tool", "role": "runtime_observation"}],
        )

    def _knowledge_catalog(self) -> str:
        try:
            import json
            from pathlib import Path
            lines = []

            # Load annotated materials (curated subset with use cases)
            anno_path = Path("knowledge/data/simulation_design_annotations.json")
            if anno_path.exists():
                data = json.loads(anno_path.read_text(encoding="utf-8"))
                mats = data.get("materials") or {}
                if mats:
                    lines.append("Common materials:")
                    for name, info in sorted(mats.items()):
                        if isinstance(info, dict):
                            use = ", ".join(info.get("use_cases") or [])
                            lines.append(f"  {name}: {use}" if use else f"  {name}")

            # Load physics lists
            phys_path = Path("knowledge/data/physics_lists.json")
            if phys_path.exists():
                pdata = json.loads(phys_path.read_text(encoding="utf-8"))
                items = pdata.get("items", []) if isinstance(pdata, dict) else []
                if items:
                    lines.append("\nPhysics lists (choose based on scenario):")
                    lines.append("  FTFP_BERT: default general-purpose")
                    lines.append("  Shielding: neutron/gamma shielding")
                    lines.append("  QGSP_BERT_HP: high-precision neutron")
                    lines.append("  FTFP_BERT_HP: FTFP_BERT + HP neutrons")
                    lines.append("  QGSP_BIC: alternative cascade")
                    lines.append(f"  Full list: {', '.join(items[:10])}...")

            # Scoring guidance
            lines.append("\nScoring options (choose based on physics scenario):")
            lines.append("  target_edep: energy deposited in target volume — always enable")
            lines.append("  detector_crossing_count: particles reaching detector — use when detector exists")
            lines.append("  detector_edep: energy in detector — use for detector response studies")
            lines.append("  plane_crossing_count: particles through scoring plane — use for transmission measurements")
            lines.append("  depth_bins: dose/fluence vs depth — use for medical dose profiles, shielding profiles")
            lines.append("  region_contrast: contrast between regions — use for NDT void/inclusion detection")
            lines.append("  transmission_factor: derived from detector/plane crossings — use when user asks about transmission")

            # Geometry guidance
            lines.append("\nGeometry guidance:")
            lines.append("  For slabs/plates/blocks: use box shape with size_x/y/z_mm.")
            lines.append("  For spherical phantoms/targets: use sphere shape with radius_mm.")
            lines.append("  For cylindrical targets/pipes: use tubs/cylinder shape with radius_mm, half_length_mm.")
            lines.append("  For multi-layer shields: use multiple box volumes stacked along z-axis.")
            lines.append("  For space/vacuum environments: always set environment material to G4_Galactic.")
            lines.append("  Always provide appropriate dimensions. Prefer mm units.")

            return "\n".join(lines) if lines else "(catalog unavailable)"
        except Exception:
            return "(catalog unavailable)"

    def _previous_turn_summary(self, state: V3AgentState) -> str:
        """Summarize what happened in previous turns for context."""
        parts = []
        has_design = _has_observation(state, "geant4_llm_design_tool") or _has_observation(state, "geant4_design_template_tool")
        has_payload = _has_observation(state, "geant4_payload_builder_tool")
        has_runtime = _has_observation(state, "geant4_runtime_tool")
        if has_runtime:
            data = _latest_observation_data(state, "geant4_runtime_tool")
            summary = data.get("result_summary", {}) if data else {}
            run = summary.get("run", {})
            sc = summary.get("scoring", {})
            parts.append(f"A simulation completed: {run.get('events_completed', '?')} events, "
                        f"edep={sc.get('target', {}).get('target_edep_total_mev', 'N/A')} MeV")
        elif has_payload:
            parts.append("A payload was drafted and is ready to run.")
        elif has_design:
            parts.append("A simulation design exists. Waiting for user to accept defaults or modify parameters.")
        else:
            parts.append("No prior design or simulation. User is starting a new request.")
        return " ".join(parts) if parts else "No prior context."

    def _describe_state(self, state: V3AgentState) -> str:
        lines = []
        for obs in state.observations[-8:]:
            lines.append(f"  [{obs.source}] status={obs.status.value}")
        if state.metadata.get("pending_action"):
            pending = state.metadata["pending_action"]
            if isinstance(pending, dict):
                lines.append(
                    "  pending_action: "
                    f"kind={pending.get('kind', 'unknown')}, "
                    f"requires_confirmation={bool(pending.get('requires_confirmation'))}"
                )
            else:
                lines.append("  pending_action: present")
        if state.artifacts:
            lines.append(f"  artifacts: {', '.join(state.artifacts)}")
        return "\n".join(lines) if lines else "  (empty)"

    def _call_llm(self, prompt: str) -> str:
        from nlu.llm_support.ollama_client import chat, load_config
        try:
            cfg = load_config(self._llm_config_path)
            timeout = min(cfg.timeout_s, 30) if cfg.timeout_s > 0 else 30
        except Exception:
            timeout = 30
        result = chat(prompt, config_path=self._llm_config_path)
        return str(result.get("response") or "") if result else ""

    def _parse_llm_response(self, raw: str) -> dict | None:
        from nlu.llm_support.ollama_client import extract_json
        parsed = extract_json(raw)
        return parsed if isinstance(parsed, dict) else None

    def _build_proposal(self, parsed: dict, turn: V3TurnInput) -> V3ActionProposal | None:
        from typing import Any
        kind_str = str(parsed.get("action_kind") or parsed.get("kind") or "").strip().lower()
        kind_map = {
            "create_design": V3ActionKind.CREATE_DESIGN,
            "draft_spec": V3ActionKind.DRAFT_SPEC,
            "run_simulation": V3ActionKind.RUN_SIMULATION,
            "ask_user": V3ActionKind.ASK_USER,
            "final_answer": V3ActionKind.FINAL_ANSWER,
        }
        if kind_str not in kind_map:
            return None
        kind = kind_map[kind_str]
        intent = str(parsed.get("action_intent") or parsed.get("intent") or kind_str)
        message = str(parsed.get("message") or "")
        reasoning = str(parsed.get("reasoning") or parsed.get("understanding") or "")
        tool_name = str(parsed.get("tool_name") or "").strip()
        tool_call = None
        if tool_name:
            from .tools.geant4_tools import (
                GEANT4_CAPABILITY_TOOL, GEANT4_DESIGN_TEMPLATE_TOOL, GEANT4_LLM_DESIGN_TOOL,
                GEANT4_PAYLOAD_BUILDER_TOOL, GEANT4_RUNTIME_PREFLIGHT_TOOL, GEANT4_RUNTIME_TOOL,
            )
            risk_map = {
                GEANT4_CAPABILITY_TOOL: V3ToolRiskLevel.READ_ONLY,
                GEANT4_DESIGN_TEMPLATE_TOOL: V3ToolRiskLevel.DRAFT_ONLY,
                GEANT4_LLM_DESIGN_TOOL: V3ToolRiskLevel.DRAFT_ONLY,
                GEANT4_PAYLOAD_BUILDER_TOOL: V3ToolRiskLevel.DRAFT_ONLY,
                GEANT4_RUNTIME_PREFLIGHT_TOOL: V3ToolRiskLevel.DRAFT_ONLY,
                GEANT4_RUNTIME_TOOL: V3ToolRiskLevel.RUNTIME_EXECUTION,
            }
            if tool_name not in risk_map:
                tool_name = ""  # Unknown tool → skip
            if tool_name:
                risk = risk_map[tool_name]
                tool_args: dict[str, Any] = {"goal": turn.user_text}
                if tool_name in (GEANT4_LLM_DESIGN_TOOL,):
                    tool_args["llm_config_path"] = self._llm_config_path
                    tool_args["lang"] = turn.locale
                tool_call = V3ToolCall(
                    tool_name=tool_name,
                    arguments=tool_args,
                    risk_level=risk,
                )
        requires_confirmation = bool(parsed.get("requires_confirmation"))
        authorized = runtime_execution_authorized(turn.metadata)
        if kind == V3ActionKind.RUN_SIMULATION and not authorized:
            requires_confirmation = True
        return V3ActionProposal(
            kind=kind,
            intent=intent,
            arguments={"message": message or intent, "reasoning": reasoning},
            tool_call=tool_call,
            risk_level=tool_call.risk_level if tool_call else V3ToolRiskLevel.READ_ONLY,
            requires_confirmation=requires_confirmation,
            confirmed=authorized,
        )


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _apply_llm_change_patches(parsed: dict, turn: V3TurnInput, state: V3AgentState) -> None:
    overrides: dict[str, object] = {}
    existing = turn.metadata.get("config_overrides")
    if isinstance(existing, dict):
        overrides.update(existing)
    overrides.update(config_overrides_from_llm_parameters(parsed.get("parameters")))

    patch_results = []
    if overrides:
        patch_results.append(build_patches_from_config_overrides(overrides, evidence="llm_parameters"))
    if isinstance(parsed.get("requested_changes"), list):
        patch_results.append(build_patches_from_requested_changes(parsed.get("requested_changes"), evidence="llm_requested_changes"))
    if not patch_results:
        return

    merged_overrides: dict[str, object] = {}
    merged_patches: list[dict[str, object]] = []
    errors: list[str] = []
    for result in patch_results:
        merged_overrides.update(result.config_overrides)
        merged_patches.extend(patch.to_dict() for patch in result.patches)
        errors.extend(result.errors)

    apply_to_payload = _patch_should_affect_payload(turn, state, parsed, bool(merged_overrides))
    turn.metadata["state_patch"] = {
        "schema_version": "geant4_agent_v3_patch.v1",
        "ok": not errors,
        "patches": merged_patches,
        "config_overrides": dict(merged_overrides),
        "applied_to_payload": bool(apply_to_payload and merged_overrides),
        "errors": list(errors),
    }
    turn.metadata["config_overrides"] = merged_overrides if apply_to_payload else {}
    if merged_overrides and apply_to_payload:
        turn.metadata["accept_defaults"] = True
        if "run_events" in merged_overrides:
            turn.metadata["events"] = merged_overrides["run_events"]


def _patch_should_affect_payload(turn: V3TurnInput, state: V3AgentState, parsed: dict, has_overrides: bool) -> bool:
    if not has_overrides:
        return False
    if _has_design_observation(state) or _has_observation(state, GEANT4_PAYLOAD_BUILDER_TOOL):
        return True
    kind = str(parsed.get("action_kind") or parsed.get("kind") or "").strip().lower()
    if kind in {"draft_spec", "run_simulation"}:
        return True
    text = str(turn.user_text or "").lower()
    explicit_no_payload = (
        "design only",
        "only design",
        "do not generate payload",
        "don't generate payload",
        "do not run",
        "don't run",
        "without running",
        "只生成方案",
        "只设计",
        "暂时不要运行",
        "不要运行",
        "先不要运行",
    )
    if any(token in text for token in explicit_no_payload):
        return False
    explicit_payload = (
        "payload",
        "runtime config",
        "generate config",
        "accept defaults",
        "run",
        "rerun",
        "运行配置",
        "生成配置",
        "生成 payload",
        "接受默认",
        "运行",
        "再跑",
    )
    return any(token in text for token in explicit_payload)


def _state_patch_needs_clarification(turn: V3TurnInput) -> bool:
    patch = turn.metadata.get("state_patch")
    if not isinstance(patch, dict):
        return False
    errors = patch.get("errors")
    overrides = patch.get("config_overrides")
    return patch.get("ok") is False and isinstance(errors, list) and bool(errors) and not bool(overrides)


def _invalid_patch_proposal(turn: V3TurnInput, state: V3AgentState) -> V3ActionProposal:
    state.metadata.pop("pending_action", None)
    state.metadata.pop("suggested_next_actions", None)
    patch = turn.metadata.get("state_patch") if isinstance(turn.metadata.get("state_patch"), dict) else {}
    errors = [str(item) for item in patch.get("errors") or []]
    return V3ActionProposal(
        kind=V3ActionKind.ASK_USER,
        intent="clarify_invalid_state_patch",
        arguments={
            "question": _invalid_patch_question(errors, turn.locale),
            "options": _invalid_patch_options(errors, turn.locale),
        },
        evidence=[{"source": "state_patch", "role": "patch_validation", "status": "failed", "errors": errors}],
    )


def _invalid_patch_question(errors: list[str], locale: str) -> str:
    details = "; ".join(_patch_error_hint(error, locale) for error in errors[:4])
    if str(locale).lower().startswith("en"):
        return (
            "I could not apply that parameter change safely. "
            + (details or "One or more requested values are outside the v3 editable parameter schema.")
            + " Please restate the change with a valid Geant4 material id, positive energy/thickness/event count, or use the confirmation button for runtime approval."
        )
    return (
        "我不能安全应用这次参数修改。"
        + (details or "有一个或多个参数不在 v3 可编辑配置范围内。")
        + " 请用合法的 Geant4 材料名、正数能量/厚度/事件数重新说明；如果是确认运行，请使用确认按钮或明确说确认运行。"
    )


def _invalid_patch_options(errors: list[str], locale: str) -> list[str]:
    if str(locale).lower().startswith("en"):
        return [
            "change source_energy_mev to 2.0",
            "change target_material to G4_WATER",
            "confirm run",
        ]
    return [
        "把源能量改成 2 MeV",
        "把靶材料改成 G4_WATER",
        "确认运行",
    ]


def _patch_error_hint(error: str, locale: str) -> str:
    english = str(locale).lower().startswith("en")
    if error.startswith("unsupported_patch_field:"):
        field = error.split(":", 1)[1]
        if english:
            return f"`{field}` is not user-editable through state patches"
        return f"`{field}` 不是可由用户 patch 直接修改的字段"
    if error.startswith("invalid_patch_value:"):
        parts = error.split(":")
        field = parts[1] if len(parts) > 1 else "parameter"
        reason = parts[2] if len(parts) > 2 else "invalid"
        if reason == "must_be_positive":
            return f"`{field}` must be positive" if english else f"`{field}` 必须是正数"
        if reason == "must_be_geant4_material_id":
            return f"`{field}` must look like `G4_WATER` or `G4_Pb`" if english else f"`{field}` 需要写成 `G4_WATER`、`G4_Pb` 这类 Geant4 材料名"
        if reason == "must_be_three_item_list":
            return f"`{field}` must be three dimensions in mm" if english else f"`{field}` 需要是 3 个 mm 尺寸"
        return f"`{field}` has invalid value ({reason})" if english else f"`{field}` 的值不合法（{reason}）"
    return error


def _runtime_physics_notes(facts: dict[str, object]) -> str:
    material = str(facts.get("material") or "unknown")
    particle = str(facts.get("particle") or "unknown").lower()
    energy = facts.get("source_energy_mev")
    lines = [
        f"- Authoritative material is {material}. Do not substitute another material from examples or session memory.",
        f"- Authoritative primary particle is {particle or 'unknown'}. Do not substitute photons/gamma/protons from examples.",
        f"- Authoritative source energy is {energy} MeV. Do not substitute another source energy from examples or session memory.",
    ]
    if material == "G4_WATER":
        lines.append("- G4_WATER is tissue-equivalent; for charged particles, dose/energy loss is dominated by electromagnetic ionization and slowing down.")
    if material == "G4_Pb":
        lines.append("- G4_Pb is high-Z shielding; gamma attenuation may involve photoelectric effect, Compton scattering, and pair production depending on energy.")
    if particle == "proton":
        lines.append("- Proton transport is charged-particle transport; discuss ionization energy loss, scattering, range, and possible Bragg-curve behavior when depth information is available.")
    elif particle in {"gamma", "photon"}:
        lines.append("- Gamma transport is neutral-photon transport; discuss attenuation, scattering, and secondary electron production when supported by material and energy.")
    elif particle == "neutron":
        lines.append("- Neutron transport is dominated by elastic/inelastic scattering and capture; moderation depends strongly on hydrogen-rich materials.")
    return "\n".join(lines)


def _analysis_conflicts_with_runtime(parsed: dict, runtime_data: dict) -> bool:
    message = str(parsed.get("message") or "")
    if not message:
        return False
    facts = _runtime_fact_summary(runtime_data)
    material = str(facts.get("material") or "")
    particle = str(facts.get("particle") or "")
    energy = facts.get("source_energy_mev")
    lowered = message.lower()

    wrong_material_tokens = {
        "G4_WATER": ("铅", "lead", "pb", "g4_pb"),
        "G4_Pb": ("水", "water", "g4_water"),
        "G4_Cu": ("铅", "lead", "pb", "水", "water", "g4_water"),
        "G4_Al": ("铅", "lead", "pb", "水", "water", "g4_water"),
    }
    if any(token in lowered or token in message for token in wrong_material_tokens.get(material, ())):
        return True

    if particle and particle.lower() != "gamma" and any(token in lowered or token in message for token in ("光子", "伽马", "gamma", "photon")):
        return True
    if particle.lower() == "gamma" and any(token in lowered or token in message for token in ("质子", "proton")):
        return True

    try:
        energy_value = float(energy)
    except (TypeError, ValueError):
        return False
    for value in _mentioned_mev_values(message):
        if abs(value - energy_value) > max(0.05, abs(energy_value) * 0.02):
            # Ignore obvious deposited-energy metrics; reject only source-energy-like prose.
            before = lowered[max(0, lowered.find(str(int(value)) if value.is_integer() else str(value)) - 16):]
            if "source" in before or "源" in before or "入射" in before or value in (1.0, 100.0):
                return True
    return False


def _runtime_fact_summary(runtime_data: dict) -> dict[str, object]:
    summary = runtime_data.get("result_summary", {}) if isinstance(runtime_data, dict) else {}
    configuration = summary.get("configuration", {}) if isinstance(summary.get("configuration"), dict) else {}
    runtime_payload = runtime_data.get("runtime_payload", {}) if isinstance(runtime_data.get("runtime_payload"), dict) else {}
    source = runtime_payload.get("source", {}) if isinstance(runtime_payload.get("source"), dict) else {}
    geometry = runtime_payload.get("geometry", {}) if isinstance(runtime_payload.get("geometry"), dict) else {}
    return {
        "material": _first_present(configuration.get("material"), geometry.get("material"), runtime_payload.get("material")),
        "particle": _first_present(configuration.get("particle"), source.get("particle"), runtime_payload.get("particle")),
        "source_energy_mev": _first_present(source.get("energy_mev"), runtime_payload.get("energy")),
    }


def _mentioned_mev_values(text: str) -> list[float]:
    import re

    values: list[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*MeV", text, flags=re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    return values


__all__ = ["BasicGeant4Reasoner", "LLMGeant4Reasoner"]
