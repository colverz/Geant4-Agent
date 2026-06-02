from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _is_en(locale: str) -> bool:
    return str(locale).lower().startswith("en")


def _msg(cn: str, en: str, locale: str) -> str:
    return en if _is_en(locale) else cn


def _localized_suggestions(cn_items: list[str], en_items: list[str], locale: str) -> list[dict[str, Any]]:
    items = en_items if _is_en(locale) else cn_items
    return [{"text": item, "prefill": item} for item in items]


@dataclass(slots=True)
class V3DialogueMessage:
    display_message: str
    raw_message: str
    dialogue_act: str
    evidence_used: list[dict[str, str]] = field(default_factory=list)
    next_suggestions: list[dict[str, Any]] = field(default_factory=list)
    answer_parts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        suggestions = [_suggestion_to_dict(item) for item in self.next_suggestions]
        parts = self.answer_parts or _default_answer_parts(
            self.display_message,
            self.dialogue_act,
            self.evidence_used,
            suggestions,
        )
        return {
            "display_message": self.display_message,
            "raw_message": self.raw_message,
            "dialogue_act": self.dialogue_act,
            "evidence_used": list(self.evidence_used),
            "next_suggestions": suggestions,
            "answer_parts": parts,
        }


def compose_v3_dialogue(response: dict[str, Any], *, locale: str = "zh-CN") -> V3DialogueMessage:
    raw_message = _raw_answer_message(response)
    if isinstance(response.get("cancelled_pending_action"), dict):
        return V3DialogueMessage(
            display_message=_msg(
                "已取消运行。我保留当前方案草案，不会启动 Geant4。",
                "Run cancelled. I kept the current design draft and will not start Geant4.",
                locale,
            ),
            raw_message=raw_message,
            dialogue_act="action_cancelled",
            evidence_used=_evidence(response, "pending_action", "geant4_payload_builder_tool"),
            next_suggestions=_localized_suggestions(
                ["修改方案", "确认运行", "查看当前 payload"],
                ["Modify design", "Confirm run", "View current payload"],
                locale,
            ),
        )

    reason = str(response.get("terminated_reason") or "")
    if reason == "waiting_confirmation":
        return _compose_confirmation_message(response, raw_message, locale)
    if reason == "observed":
        return _compose_runtime_message(response, raw_message, locale)
    if reason == "blocked":
        return _compose_blocked_message(response, raw_message, locale)
    if reason == "waiting_user":
        if _latest_design_observation(response) and not _has_invalid_state_patch(response):
            return _compose_design_waiting_user_message(response, raw_message, locale)
        return V3DialogueMessage(
            display_message=raw_message,
            raw_message=raw_message,
            dialogue_act="needs_user_input",
            evidence_used=_evidence(response),
            next_suggestions=_answer_options(response),
        )
    if _answer_has_runtime_evidence(response):
        return _compose_runtime_answer_message(response, raw_message, locale)
    if _looks_like_no_runtime_result(raw_message):
        return _compose_no_runtime_result_message(response, raw_message, locale)

    current_observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    if raw_message and len(raw_message) > 80 and not current_observations and _latest_observation(response, "geant4_runtime_tool"):
        return _compose_runtime_answer_message(response, raw_message, locale)
    if _looks_like_current_configuration_answer(raw_message):
        return _compose_current_configuration_message(response, raw_message, locale)
    if _latest_observation(response, "geant4_payload_builder_tool"):
        return _compose_payload_message(response, raw_message, locale)
    if _latest_design_observation(response):
        return _compose_design_message(response, raw_message, locale)

    return V3DialogueMessage(
        display_message=raw_message,
        raw_message=raw_message,
        dialogue_act=reason or "final_answer",
        evidence_used=_evidence(response),
        next_suggestions=_answer_options(response),
    )


def _suggestion_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("label") or item.get("title") or item.get("prefill") or "").strip()
        prefill = str(item.get("prefill") or item.get("text") or item.get("label") or item.get("title") or "").strip()
        out: dict[str, Any] = {"text": text, "prefill": prefill}
        if isinstance(item.get("confirmation_event"), dict):
            out["confirmation_event"] = dict(item["confirmation_event"])
        if item.get("kind"):
            out["kind"] = str(item["kind"])
        return out
    text = str(item or "").strip()
    return {"text": text, "prefill": text}


def _default_answer_parts(
    display_message: str,
    dialogue_act: str,
    evidence: list[dict[str, str]],
    suggestions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"kind": "summary", "title": "What happened", "text": display_message}]
    if evidence:
        parts.append(
            {
                "kind": "evidence",
                "title": "Evidence used",
                "items": [
                    {"source": str(item.get("source") or ""), "status": str(item.get("status") or "")}
                    for item in evidence
                    if isinstance(item, dict)
                ],
            }
        )
    if suggestions:
        parts.append({"kind": "next_step", "title": "Next step", "items": [dict(item) for item in suggestions if isinstance(item, dict)]})
    if dialogue_act:
        parts.append({"kind": "dialogue_act", "title": "Dialogue act", "text": dialogue_act})
    return parts


def _compose_confirmation_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    payload = _payload_data(response)
    spec = _dict(payload.get("simulation_spec"))
    source = _dict(spec.get("source"))
    geometry = _dict(spec.get("geometry"))
    run = _dict(spec.get("run"))
    pending = _dict(response.get("pending_action"))
    arguments = _dict(_dict(pending.get("tool_call")).get("arguments"))
    events = arguments.get("events") or run.get("events") or _preflight_data(response).get("events")
    parts = []
    if source.get("energy_mev") is not None:
        parts.append(f"{_format_number(source.get('energy_mev'))} MeV")
    if geometry.get("material"):
        parts.append(str(geometry.get("material")))
    if events:
        parts.append(f"{events} events")
    summary = ", ".join(parts) if parts else _msg("当前草案", "current draft", locale)
    return V3DialogueMessage(
        display_message=_msg(
            f"方案已到运行确认点，将按 {summary} 调用 Geant4。回复“确认运行”后我再执行；也可以继续改参数。",
            f"I have advanced the design to the run-confirmation point with {summary}. Reply 'confirm run' to execute, or keep editing parameters.",
            locale,
        ),
        raw_message=raw_message,
        dialogue_act="action_needs_confirmation",
        evidence_used=_evidence(response, "geant4_payload_builder_tool", "geant4_runtime_preflight_tool", "commit_gate"),
        next_suggestions=_localized_suggestions(["确认运行", "修改方案", "取消运行"], ["Confirm run", "Modify design", "Cancel run"], locale),
    )


def _compose_runtime_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    runtime = _runtime_data(response)
    result = _dict(runtime.get("result_summary"))
    run = _dict(result.get("run"))
    scoring = _dict(result.get("scoring"))
    metrics = _compact_metrics(scoring)
    if runtime.get("not_evaluable_reason"):
        display = _msg(
            f"这次还没有真实 Geant4 结果：{runtime.get('not_evaluable_reason')}。",
            f"No real Geant4 result this time: {runtime.get('not_evaluable_reason')}.",
            locale,
        )
        act = "runtime_not_evaluable"
    else:
        events = run.get("events_completed") or run.get("events_requested") or "unknown"
        display = _msg(
            f"Geant4 已运行完成，完成事件数是 {events}。" + (f" 关键计分：{metrics}。" if metrics else ""),
            f"Geant4 run completed. Events completed: {events}." + (f" Key metrics: {metrics}." if metrics else ""),
            locale,
        )
        act = "runtime_observed"
    return V3DialogueMessage(
        display_message=display,
        raw_message=raw_message,
        dialogue_act=act,
        evidence_used=_evidence(response, "geant4_runtime_tool"),
        next_suggestions=_localized_suggestions(
            ["解释结果", "修改参数再运行", "查看 runtime observation"],
            ["Explain result", "Modify and rerun", "View runtime observation"],
            locale,
        ),
    )


def _compose_runtime_answer_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    return V3DialogueMessage(
        display_message=raw_message or _msg("当前还没有可解释的 Geant4 运行结果。", "No explainable Geant4 runtime result is available yet.", locale),
        raw_message=raw_message,
        dialogue_act="runtime_result_answered",
        evidence_used=_evidence(response, "geant4_runtime_tool"),
        next_suggestions=_localized_suggestions(
            ["继续追问结果", "增加事件数再运行", "修改方案"],
            ["Ask more about result", "Increase events and rerun", "Modify design"],
            locale,
        ),
    )


def _compose_no_runtime_result_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    message = raw_message or _msg("当前还没有可解释的运行结果。", "No runtime result is available yet.", locale)
    return V3DialogueMessage(
        display_message=message,
        raw_message=raw_message,
        dialogue_act="final_answer",
        evidence_used=_evidence(response, "geant4_payload_builder_tool", "geant4_runtime_preflight_tool", "geant4_runtime_tool"),
        next_suggestions=_localized_suggestions(
            ["生成运行配置", "确认运行", "修改方案"],
            ["Generate payload", "Confirm run", "Modify design"],
            locale,
        ),
    )


def _compose_current_configuration_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    return V3DialogueMessage(
        display_message=raw_message,
        raw_message=raw_message,
        dialogue_act="configuration_answered",
        evidence_used=_evidence(response, "geant4_payload_builder_tool", "geant4_llm_design_tool", "geant4_design_template_tool"),
        next_suggestions=_localized_suggestions(
            ["生成运行配置", "修改方案", "运行前检查"],
            ["Generate payload", "Modify design", "Run preflight"],
            locale,
        ),
    )


def _compose_payload_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    payload = _payload_data(response)
    spec = _dict(payload.get("simulation_spec"))
    source = _dict(spec.get("source"))
    geometry = _dict(spec.get("geometry"))
    run = _dict(spec.get("run"))
    overrides = _dict(payload.get("applied_overrides"))
    material = geometry.get("material") or "material TBD"
    energy = source.get("energy_mev")
    events = run.get("events")
    display = _msg(
        f"运行配置已就绪：{material} 靶材，{_format_number(energy) if energy is not None else 'TBD'} MeV，{events or 'TBD'} 个事件。",
        f"Runtime config ready: {material} target, {_format_number(energy) if energy is not None else 'TBD'} MeV, {events or 'TBD'} events.",
        locale,
    )
    change_text = _override_summary(overrides)
    if change_text:
        display += _msg(f" 本轮应用的修改：{change_text}。", f" Changes: {change_text}.", locale)
    display += _msg(" 还没有实际运行。回复“确认运行”来启动 Geant4。", " Not yet executed. Reply 'confirm run' to start Geant4.", locale)
    return V3DialogueMessage(
        display_message=display,
        raw_message=raw_message,
        dialogue_act="payload_draft_presented",
        evidence_used=_evidence(response, "geant4_payload_builder_tool"),
        next_suggestions=_localized_suggestions(["确认运行", "继续修改", "查看 payload"], ["Confirm run", "Continue editing", "View payload"], locale),
    )


def _compose_design_waiting_user_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    design_message = _compose_design_message(response, raw_message, locale)
    followup = _msg(
        " 我先停在这里：还没有生成 payload，也没有运行 Geant4。",
        " I will stop here: no payload has been generated and Geant4 has not run.",
        locale,
    )
    return V3DialogueMessage(
        display_message=design_message.display_message + followup,
        raw_message=raw_message,
        dialogue_act="needs_user_input",
        evidence_used=design_message.evidence_used or _evidence(response),
        next_suggestions=_answer_options(response) or design_message.next_suggestions,
    )


def _compose_design_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    observation = _latest_design_observation(response) or {}
    design = _dict(_dict(observation.get("data")).get("design"))
    setup = _dict(design.get("recommended_setup"))
    geometry = _localize_geometry(setup.get("geometry") or "", locale)
    material = setup.get("material") or _msg("待定材料", "material TBD", locale)
    source = _source_summary(setup, locale)
    observables = _join_list(
        [_localize_observable(str(item), locale) for item in design.get("observables")]
        if isinstance(design.get("observables"), list)
        else design.get("observables"),
        separator=", ",
    )
    assumptions = _join_list(
        [_localize_assumption(str(item)) for item in design.get("assumptions")]
        if isinstance(design.get("assumptions"), list)
        else design.get("assumptions"),
        limit=3,
        separator="; ",
    )
    explanation = _clean_sentence(setup.get("user_explanation") or setup.get("design_rationale") or "")
    message = _msg(
        f"我根据你的需求整理了一个 Geant4 方案：用 {geometry} 作为主体结构，材料是 {material}，采用 {source} 源项。重点关注的物理量：{observables or '待确认'}。",
        f"Here is the simulation design: a {geometry} structure with {material} material, using a {source} source. Key observables: {observables or 'TBD'}.",
        locale,
    )
    if explanation:
        message += " " + explanation + ("." if _is_en(locale) else "。")
    if assumptions:
        message += _msg(f" 假设：{assumptions}。", f" Assumptions: {assumptions}.", locale)
    message += _msg(
        " 目前只是方案阶段，还没有开始运行。你想调整参数，还是生成运行配置？",
        " This is still a design draft. Want me to adjust anything, or proceed to generating a runnable payload?",
        locale,
    )
    return V3DialogueMessage(
        display_message=message,
        raw_message=raw_message,
        dialogue_act="design_presented",
        evidence_used=_evidence(response, "geant4_llm_design_tool", "geant4_design_template_tool"),
        next_suggestions=_localized_suggestions(["生成 payload", "修改设计", "运行前检查"], ["Generate payload", "Modify design", "Run preflight"], locale),
    )


def _compose_blocked_message(response: dict[str, Any], raw_message: str, locale: str) -> V3DialogueMessage:
    review = _latest_observation(response, "proposal_critic") or _latest_observation(response, "commit_gate") or {}
    data = _dict(review.get("data"))
    reason = str(data.get("reason") or "")
    if reason == "runtime_without_payload":
        display = _msg(
            "我没有启动 Geant4，因为当前还没有可运行的 payload draft。下一步是生成 runtime payload 并做 preflight。",
            "I did not start Geant4 because there is no runnable payload draft yet. The next step is to generate the runtime payload and run preflight.",
            locale,
        )
        suggestions = _localized_suggestions(["生成运行配置"], ["Generate payload"], locale)
        suggestions[0]["prefill"] = "accept the current design and generate runtime payload"
    elif reason == "runtime_without_preflight":
        display = _msg(
            "我没有启动 Geant4，因为运行前检查还没有通过。",
            "I did not start Geant4 because runtime preflight has not passed.",
            locale,
        )
        suggestions = _localized_suggestions(["重新做运行前检查"], ["Run preflight again"], locale)
        suggestions[0]["prefill"] = "run runtime preflight again for the current payload"
    elif reason == "tool_schema_invalid":
        repair_text = _repair_text(data)
        display = _msg(
            "我没有调用这个工具，因为 proposal 参数和已注册 schema 不一致。",
            "I did not call the tool because the proposal arguments do not match the registered schema.",
            locale,
        )
        if repair_text:
            display += _msg(f" 修复建议：{repair_text}", f" Repair hint: {repair_text}", locale)
        suggestions = _localized_suggestions(["重新生成工具参数"], ["Rebuild tool arguments"], locale)
        suggestions[0]["prefill"] = "rebuild tool arguments from current context"
    elif reason == "context_fact_not_grounded":
        repair_text = _repair_text(data)
        display = _msg(
            "我没有继续执行这个 proposal，因为它没有被当前 session 的 design 或 payload 事实支撑。",
            "I did not continue with this proposal because its design or payload is not grounded in the current session context. Rebuild from the latest context.",
            locale,
        )
        if repair_text:
            display += _msg(f" 修复建议：{repair_text}", f" Repair hint: {repair_text}", locale)
        suggestions = _localized_suggestions(["重新生成 payload"], ["Rebuild payload"], locale)
        suggestions[0]["prefill"] = "rebuild the payload from the latest design"
    elif reason in {"unknown_tool", "internal_argument_not_allowed"}:
        display = _msg(
            "我没有执行这个 proposal，因为它不符合 v3 已注册工具契约。",
            "I did not execute this proposal because it does not fit the registered v3 tool contract.",
            locale,
        )
        suggestions = _localized_suggestions(["回到当前方案"], ["Return to current design"], locale)
        suggestions[0]["prefill"] = "return to the current design"
    else:
        display = _msg(
            "这一步被 v3 安全检查拦下了，所以我没有执行高风险动作。",
            "This step was blocked by the v3 safety check, so I did not execute the high-risk action.",
            locale,
        )
        suggestions = _localized_suggestions(["调整方案", "继续说明需求"], ["Adjust design", "Clarify goal"], locale)
    return V3DialogueMessage(
        display_message=display,
        raw_message=raw_message,
        dialogue_act="blocked",
        evidence_used=_evidence(response, "proposal_critic", "commit_gate"),
        next_suggestions=suggestions,
    )


def _repair_text(data: dict[str, Any]) -> str:
    repairs = data.get("repair_suggestions") if isinstance(data.get("repair_suggestions"), list) else []
    return "; ".join(str(item) for item in repairs[:2] if str(item).strip())


def _latest_design_observation(response: dict[str, Any]) -> dict[str, Any] | None:
    return _latest_observation(response, "geant4_llm_design_tool") or _latest_observation(response, "geant4_design_template_tool")


def _latest_observation(response: dict[str, Any], source: str) -> dict[str, Any] | None:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    state = _dict(response.get("state"))
    state_observations = state.get("observations") if isinstance(state.get("observations"), list) else []
    for observation in reversed([*state_observations, *observations]):
        if isinstance(observation, dict) and observation.get("source") == source:
            return observation
    return None


def _payload_data(response: dict[str, Any]) -> dict[str, Any]:
    return _dict(_dict(_latest_observation(response, "geant4_payload_builder_tool")).get("data"))


def _preflight_data(response: dict[str, Any]) -> dict[str, Any]:
    return _dict(_dict(_latest_observation(response, "geant4_runtime_preflight_tool")).get("data"))


def _runtime_data(response: dict[str, Any]) -> dict[str, Any]:
    return _dict(_dict(_latest_observation(response, "geant4_runtime_tool")).get("data"))


def _raw_answer_message(response: dict[str, Any]) -> str:
    return str(_dict(response.get("answer")).get("message") or "")


def _answer_options(response: dict[str, Any]) -> list[dict[str, Any]]:
    options = _dict(response.get("answer")).get("next_options")
    if not isinstance(options, list):
        return []
    out: list[dict[str, Any]] = []
    for item in options:
        text = str(item.get("text") or item.get("label") or item.get("prefill") if isinstance(item, dict) else item or "").strip()
        if text:
            out.append({"text": text, "prefill": text})
    return out


def _answer_has_runtime_evidence(response: dict[str, Any]) -> bool:
    evidence = _dict(response.get("answer")).get("evidence")
    return isinstance(evidence, list) and any(isinstance(item, dict) and item.get("source") == "geant4_runtime_tool" for item in evidence)


def _has_invalid_state_patch(response: dict[str, Any]) -> bool:
    patch = _dict(_dict(_dict(response.get("state")).get("metadata")).get("last_state_patch"))
    if patch.get("ok") is False and isinstance(patch.get("errors"), list) and patch.get("errors"):
        return True
    evidence = _dict(response.get("answer")).get("evidence")
    return isinstance(evidence, list) and any(isinstance(item, dict) and item.get("source") == "state_patch" and item.get("status") == "failed" for item in evidence)


def _evidence(response: dict[str, Any], *preferred_sources: str) -> list[dict[str, str]]:
    observations = response.get("observations") if isinstance(response.get("observations"), list) else []
    state = _dict(response.get("state"))
    state_observations = state.get("observations") if isinstance(state.get("observations"), list) else []
    observations = [*observations, *state_observations]
    preferred = set(preferred_sources)
    out = []
    if response.get("cancelled_pending_action") and (not preferred or "pending_action" in preferred):
        out.append({"source": "pending_action", "status": "cancelled"})
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        source = str(observation.get("source") or "")
        if preferred and source not in preferred:
            continue
        out.append({"source": source, "status": str(observation.get("status") or "")})
    if out:
        return out
    return [{"source": str(item.get("source") or ""), "status": str(item.get("status") or "")} for item in observations if isinstance(item, dict)]


def _looks_like_no_runtime_result(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(
        token in lowered
        for token in (
            "no runtime result",
            "no explainable geant4 runtime result",
            "没有可解释的运行结果",
            "还没有可解释的运行结果",
            "还没有运行结果",
        )
    )


def _looks_like_current_configuration_answer(message: str) -> bool:
    lowered = str(message or "").lower()
    return lowered.startswith(("current design draft:", "current runtime payload:")) or lowered.startswith(
        "no geant4 design or runtime payload is configured yet."
    ) or str(message or "").startswith(("当前只有方案草稿：", "当前运行配置是：", "当前还没有 Geant4 方案或运行配置。"))


def _override_summary(overrides: dict[str, Any]) -> str:
    labels = {
        "source_energy_mev": "source_energy_mev",
        "run_events": "run_events",
        "target_material": "target_material",
        "target_thickness_mm": "target_thickness_mm",
        "geometry_dimensions_mm": "geometry_dimensions_mm",
    }
    parts = []
    for key, value in overrides.items():
        value_text = " x ".join(_format_number(item) for item in value) if isinstance(value, list) else _format_number(value)
        parts.append(f"{labels.get(key, key)}={value_text}")
    return ", ".join(parts)


def _compact_metrics(scoring: dict[str, Any]) -> str:
    metrics = []
    target = _dict(scoring.get("target"))
    detector = _dict(scoring.get("detector_crossing"))
    plane = _dict(scoring.get("plane_crossing"))
    if target.get("target_edep_total_mev") is not None:
        metrics.append(f"target_edep_total_mev={target.get('target_edep_total_mev')}")
    if detector.get("detector_crossing_count") is not None:
        metrics.append(f"detector_crossing_count={detector.get('detector_crossing_count')}")
    if plane.get("plane_crossing_count") is not None:
        metrics.append(f"plane_crossing_count={plane.get('plane_crossing_count')}")
    return ", ".join(metrics)


def _join_list(value: Any, *, limit: int = 4, separator: str = ", ") -> str:
    if not isinstance(value, list):
        return ""
    return separator.join(_clean_sentence(str(item)) for item in value[:limit] if str(item).strip())


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _clean_sentence(value: Any) -> str:
    text = str(value or "").strip()
    while text and text[-1] in "。.!?;；":
        text = text[:-1].strip()
    return text


def _localize_assumption(value: str) -> str:
    return value


def _localize_geometry(value: Any, locale: str) -> str:
    if isinstance(value, dict):
        volumes = value.get("volumes")
        if isinstance(volumes, list) and volumes:
            first = volumes[0] if isinstance(volumes[0], dict) else {}
            return _shape_label(str(first.get("shape") or ""), locale)
        return "custom geometry"
    return _shape_label(str(value or ""), locale)


def _source_summary(setup: dict[str, Any], locale: str) -> str:
    source = _localize_source(str(setup.get("source") or ""), locale)
    parts = [source]
    if setup.get("source_particle"):
        parts.append(str(setup.get("source_particle")))
    if setup.get("source_energy_mev") is not None:
        parts.append(f"{_format_number(setup.get('source_energy_mev'))} MeV")
    return " ".join(parts)


def _localize_source(value: str, locale: str) -> str:
    return {"beam": "beam", "point": "point source", "isotropic": "isotropic source"}.get(value, value or "beam")


def _shape_label(shape: str, locale: str) -> str:
    if _is_en(locale):
        return {
            "single_box": "box target",
            "box": "box target",
            "sphere": "sphere",
            "tubs": "cylinder",
            "cylinder": "cylinder",
            "multi_layer": "multi-layer stack",
            "multi_layer_stack": "multi-layer stack",
        }.get(shape, shape or "custom geometry")
    return {"single_box": "single_box", "box": "box"}.get(shape, shape or "custom geometry")


def _localize_observable(value: str, locale: str) -> str:
    labels = {
        "target_edep": "target energy deposition",
        "detector_crossing_count": "detector crossing count",
        "detector_edep": "detector energy deposition",
        "plane_crossing_count": "plane crossing count",
        "region_contrast": "region contrast",
        "depth_bins": "depth bins",
        "transmission_factor": "transmission factor",
    }
    return labels.get(value, value)


__all__ = ["V3DialogueMessage", "compose_v3_dialogue"]
