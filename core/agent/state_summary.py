from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AGENT_STATE_SUMMARY_SCHEMA_VERSION = "agent_state_summary.v1"


@dataclass(frozen=True)
class AgentStateSummary:
    schema_version: str = AGENT_STATE_SUMMARY_SCHEMA_VERSION
    status: str = "needs_information"
    intent: str = "normal_chat"
    safety_class: str = "read_only"
    terminal_state: str = "read_only_answer"
    understood: list[str] = field(default_factory=list)
    applied_paths: list[str] = field(default_factory=list)
    rejected_paths: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    missing_fields_friendly: list[str] = field(default_factory=list)
    asked_fields: list[str] = field(default_factory=list)
    asked_fields_friendly: list[str] = field(default_factory=list)
    pending_confirmation: bool = False
    pending_confirmation_paths: list[str] = field(default_factory=list)
    runtime_ready: bool = False
    runtime_action_guarded: bool = False
    guarded_runtime_intent_pending: bool = False
    tool_calls_blocked: list[str] = field(default_factory=list)
    llm: dict[str, Any] = field(default_factory=dict)
    next_action: str = "answer_user"
    user_visible_summary: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "intent": self.intent,
            "safety_class": self.safety_class,
            "terminal_state": self.terminal_state,
            "understood": list(self.understood),
            "applied_paths": list(self.applied_paths),
            "rejected_paths": list(self.rejected_paths),
            "missing_fields": list(self.missing_fields),
            "missing_fields_friendly": list(self.missing_fields_friendly),
            "asked_fields": list(self.asked_fields),
            "asked_fields_friendly": list(self.asked_fields_friendly),
            "pending_confirmation": self.pending_confirmation,
            "pending_confirmation_paths": list(self.pending_confirmation_paths),
            "runtime_ready": self.runtime_ready,
            "runtime_action_guarded": self.runtime_action_guarded,
            "guarded_runtime_intent_pending": self.guarded_runtime_intent_pending,
            "tool_calls_blocked": list(self.tool_calls_blocked),
            "llm": dict(self.llm),
            "next_action": self.next_action,
            "user_visible_summary": self.user_visible_summary,
            "notes": list(self.notes),
        }


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item)))


def _status_from_inputs(
    *,
    terminal_state: str,
    missing_fields: list[str],
    pending_confirmation: bool,
    runtime_action_guarded: bool,
    runtime_ready: bool,
    rejected_paths: list[str],
    llm_degraded: bool,
) -> str:
    if terminal_state == "unsupported":
        return "unsupported"
    if runtime_action_guarded:
        return "runtime_action_guarded"
    if pending_confirmation:
        return "waiting_confirmation"
    if rejected_paths and terminal_state in {"rejected", "error"}:
        return "rejected"
    if missing_fields:
        return "needs_information"
    if runtime_ready:
        return "ready_to_run"
    if llm_degraded:
        return "degraded"
    return "answered"


def _next_action_from_status(status: str) -> str:
    return {
        "unsupported": "offer_supported_alternative",
        "runtime_action_guarded": "ask_user_to_confirm_runtime_action",
        "waiting_confirmation": "ask_user_to_confirm_patch",
        "rejected": "explain_rejection_and_ask_revision",
        "needs_information": "ask_clarifying_question",
        "ready_to_run": "offer_runtime_execution",
        "degraded": "explain_degraded_mode_and_continue",
    }.get(status, "answer_user")


def _summary_text(
    *,
    lang: str,
    status: str,
    understood: list[str],
    missing_friendly: list[str],
    pending_paths: list[str],
    runtime_ready: bool,
    llm_degraded: bool,
) -> str:
    understood_text = ", ".join(understood[:4])
    missing_text = ", ".join(missing_friendly[:4])
    pending_text = ", ".join(pending_paths[:4]) or "若干配置项"
    understood_fallback = understood_text or "部分需求"
    missing_fallback = missing_text or "关键仿真信息"
    ready_suffix = "，但本轮 LLM 已降级" if llm_degraded else ""
    done_suffix = "，配置已具备运行条件" if runtime_ready else ""
    if lang == "zh":
        if status == "unsupported":
            return "当前请求包含暂不支持或被安全门拒绝的部分，我会先说明边界并给出可支持的替代方案。"
        if status == "runtime_action_guarded":
            return "我识别到你想运行或打开运行时工具，这类操作需要经过运行保护，当前不会自动执行。"
        if status == "waiting_confirmation":
            return f"我识别到需要确认的配置变更：{pending_text}，确认后才会写入。"
        if status == "needs_information":
            return f"我已经理解：{understood_fallback}；还缺少：{missing_fallback}。"
        if status == "ready_to_run":
            return f"配置已经足够进入运行前检查{ready_suffix}。"
        if status == "degraded":
            return "本轮 LLM 能力不可用或输出未通过校验，系统已降级到确定性规则路径。"
        return f"我已经处理本轮请求{done_suffix}。"
    if status == "unsupported":
        return "This request includes unsupported or rejected parts; I will explain the boundary and offer a supported alternative."
    if status == "runtime_action_guarded":
        return "I detected a runtime action request; it is guarded and will not execute automatically."
    if status == "waiting_confirmation":
        return f"I found configuration changes that need confirmation: {', '.join(pending_paths[:4]) or 'several fields'}."
    if status == "needs_information":
        return f"I understood: {understood_text or 'part of the request'}; still missing: {missing_text or 'key simulation details'}."
    if status == "ready_to_run":
        return f"The configuration is ready for runtime preflight{' but this turn used degraded LLM handling' if llm_degraded else ''}."
    if status == "degraded":
        return "The LLM was unavailable or rejected, so this turn used deterministic fallback behavior."
    return f"I handled this turn{' and the configuration is runtime-ready' if runtime_ready else ''}."


def build_agent_state_summary(
    *,
    lang: str,
    intent: str,
    safety_class: str,
    terminal_state: str,
    understood: list[str] | None = None,
    applied_paths: list[str] | None = None,
    rejected_paths: list[str] | None = None,
    missing_fields: list[str] | None = None,
    missing_fields_friendly: list[str] | None = None,
    asked_fields: list[str] | None = None,
    asked_fields_friendly: list[str] | None = None,
    pending_confirmation_paths: list[str] | None = None,
    runtime_ready: bool = False,
    runtime_action_guarded: bool = False,
    guarded_runtime_intent_pending: bool = False,
    tool_calls_blocked: list[str] | None = None,
    llm_used: bool = False,
    fallback_reason: str | None = None,
    llm_stage_failures: list[str] | None = None,
    llm_schema_errors: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    applied = _dedupe(applied_paths or [])
    rejected = _dedupe(rejected_paths or [])
    missing = _dedupe(missing_fields or [])
    missing_friendly = _dedupe(missing_fields_friendly or [])
    asked = _dedupe(asked_fields or [])
    asked_friendly = _dedupe(asked_fields_friendly or [])
    pending_paths = _dedupe(pending_confirmation_paths or [])
    blocked = _dedupe(tool_calls_blocked or [])
    failures = _dedupe(llm_stage_failures or [])
    schema_errors = _dedupe(llm_schema_errors or [])
    llm_degraded = bool(fallback_reason or failures or schema_errors or not llm_used)
    pending_confirmation = bool(pending_paths)
    runtime_guarded = bool(runtime_action_guarded or terminal_state == "runtime_action_guarded")
    status = _status_from_inputs(
        terminal_state=terminal_state,
        missing_fields=missing,
        pending_confirmation=pending_confirmation,
        runtime_action_guarded=runtime_guarded,
        runtime_ready=runtime_ready,
        rejected_paths=rejected,
        llm_degraded=llm_degraded,
    )
    understood_items = _dedupe(understood or applied or asked_friendly or missing_friendly)
    summary = AgentStateSummary(
        status=status,
        intent=str(intent or "normal_chat"),
        safety_class=str(safety_class or "read_only"),
        terminal_state=str(terminal_state or "read_only_answer"),
        understood=understood_items,
        applied_paths=applied,
        rejected_paths=rejected,
        missing_fields=missing,
        missing_fields_friendly=missing_friendly,
        asked_fields=asked,
        asked_fields_friendly=asked_friendly,
        pending_confirmation=pending_confirmation,
        pending_confirmation_paths=pending_paths,
        runtime_ready=runtime_ready,
        runtime_action_guarded=runtime_guarded,
        guarded_runtime_intent_pending=guarded_runtime_intent_pending,
        tool_calls_blocked=blocked,
        llm={
            "used": bool(llm_used),
            "degraded": llm_degraded,
            "fallback_reason": fallback_reason,
            "stage_failures": failures,
            "schema_errors": schema_errors,
        },
        next_action=_next_action_from_status(status),
        user_visible_summary=_summary_text(
            lang=lang,
            status=status,
            understood=understood_items,
            missing_friendly=missing_friendly,
            pending_paths=pending_paths,
            runtime_ready=runtime_ready,
            llm_degraded=llm_degraded,
        ),
        notes=_dedupe(notes or []),
    )
    return summary.to_dict()


__all__ = [
    "AGENT_STATE_SUMMARY_SCHEMA_VERSION",
    "AgentStateSummary",
    "build_agent_state_summary",
]
