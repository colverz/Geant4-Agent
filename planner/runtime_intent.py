from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from core.config.prompt_profiles import PromptTask, build_prompt, validate_prompt_output
from core.runtime.types import ActionSafetyClass


class RuntimeIntent(str, Enum):
    READ_SUMMARY = "read_summary"
    READ_CONFIG = "read_config"
    RUN_REQUESTED = "run_requested"
    VIEWER_REQUESTED = "viewer_requested"
    NORMAL_CHAT = "normal_chat"


@dataclass(frozen=True)
class RuntimeIntentClassification:
    intent: RuntimeIntent
    action_safety_class: ActionSafetyClass
    prompt_profile_id: str
    prompt_validation: dict


def _lang_key(lang: str) -> str:
    return "zh" if str(lang).lower() == "zh" else "en"


def _safety_for_intent(intent: RuntimeIntent) -> ActionSafetyClass:
    if intent in {RuntimeIntent.READ_SUMMARY, RuntimeIntent.READ_CONFIG}:
        return ActionSafetyClass.READ_ONLY
    if intent in {RuntimeIntent.RUN_REQUESTED, RuntimeIntent.VIEWER_REQUESTED}:
        return ActionSafetyClass.EXPENSIVE_RUNTIME
    return ActionSafetyClass.READ_ONLY


def _classify_rule(text: str, lang: str) -> RuntimeIntent:
    raw = str(text or "").strip().lower()
    if not raw:
        if re.search(r"(配置|设置|当前|已经|还缺|缺少|几何|材料|源|物理|输出|config|setup)", raw) and re.search(
            r"(什么|多少|如何|怎么|状态|摘要|还缺|缺少|current|what|missing|summary|status)", raw
        ):
            return RuntimeIntent.READ_CONFIG
        return RuntimeIntent.NORMAL_CHAT

    if _lang_key(lang) == "zh":
        if re.search(r"(打开|显示|启动).*(viewer|视窗|窗口|几何窗口|可视化)", raw):
            return RuntimeIntent.VIEWER_REQUESTED
        if re.search(r"(运行|再跑|开始跑|执行).*(geant4|模拟|仿真|事件|event)|跑\s*\d+\s*(个)?\s*(事件|event)", raw):
            return RuntimeIntent.RUN_REQUESTED
        if re.search(r"(刚才|上次|最近|当前).*(结果|模拟|运行|计分|得分)|结果怎么样|模拟怎么样|运行怎么样|剂量多少|沉积能量|hit|crossing", raw):
            return RuntimeIntent.READ_SUMMARY
        return RuntimeIntent.NORMAL_CHAT

    if re.search(r"\b(open|launch|show)\b.*\b(viewer|visuali[sz]ation|geometry window)\b", raw):
        return RuntimeIntent.VIEWER_REQUESTED
    if re.search(r"\b(run|rerun|execute|start)\b.*\b(geant4|simulation|events?|beam)\b|\brun\s+\d+\s+events?\b", raw):
        return RuntimeIntent.RUN_REQUESTED
    if re.search(
        r"\b(last|latest|previous|current)\b.*\b(result|simulation|run|scoring|score|edep|hit|crossing)\b|\bwhat happened\b.*\b(run|simulation)\b|\bhow did\b.*\b(run|simulation)\b",
        raw,
    ):
        return RuntimeIntent.READ_SUMMARY
    if re.search(
        r"\b(current|existing|configured|configuration|config|setup)\b.*\b(config|configuration|setup|geometry|material|source|physics|output|missing|need|status|summary)\b|\bwhat(?:'s| is)\b.*\b(configured|missing|left|current setup)\b|\bwhat do we still need\b",
        raw,
    ):
        return RuntimeIntent.READ_CONFIG
    return RuntimeIntent.NORMAL_CHAT


def classify_user_runtime_intent(text: str, lang: str = "zh") -> RuntimeIntentClassification:
    prompt_build = build_prompt(
        PromptTask.RESULT_QUESTION_ROUTE,
        lang,
        {"user_text": str(text or "").strip()},
    )
    intent = _classify_rule(text, lang)
    validation = validate_prompt_output(PromptTask.RESULT_QUESTION_ROUTE, lang, intent.value)
    return RuntimeIntentClassification(
        intent=intent,
        action_safety_class=_safety_for_intent(intent),
        prompt_profile_id=prompt_build.profile_id,
        prompt_validation=validation.__dict__,
    )
