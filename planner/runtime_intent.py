from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from core.config.prompt_profiles import PromptTask, build_prompt, validate_prompt_output
from core.runtime.types import ActionSafetyClass


class RuntimeIntent(str, Enum):
    READ_SUMMARY = "read_summary"
    READ_CONFIG = "read_config"
    CONFIG_MUTATION = "config_mutation"
    RUN_REQUESTED = "run_requested"
    VIEWER_REQUESTED = "viewer_requested"
    NORMAL_CHAT = "normal_chat"


@dataclass(frozen=True)
class RuntimeIntentClassification:
    intent: RuntimeIntent
    action_safety_class: ActionSafetyClass
    prompt_profile_id: str
    prompt_validation: dict


class RuntimeIntentAction(str, Enum):
    CONFIG_SUMMARY = "config_summary"
    RUNTIME_SUMMARY = "runtime_summary"
    STEP_ASYNC = "step_async"
    GUARDED_RUNTIME_UI = "guarded_runtime_ui"
    READ_ONLY_CHAT = "read_only_chat"


_CONFIG_DOMAIN_PATTERN = re.compile(
    r"\b(geometry|material|materials|source|particle|energy|physics|output|detector|scoring|target|beam|gamma|proton|neutron|electron|box|cylinder|sphere|config|configuration|setup)\b",
    flags=re.IGNORECASE,
)
_CONFIG_MUTATION_PATTERN = re.compile(
    r"\b(create|build|make|set|change|modify|update|use|add|remove|delete|clear|unset|switch|configure)\b",
    flags=re.IGNORECASE,
)
_CONFIG_READ_PATTERN = re.compile(
    r"\b(current|existing|configured|configuration|config|setup)\b.*\b(config|configuration|setup|geometry|material|source|physics|output|missing|need|status|summary)\b|\bwhat(?:'s| is)\b.*\b(configured|missing|left|current setup)\b|\bwhat do we still need\b",
    flags=re.IGNORECASE,
)
_RESULT_READ_PATTERN = re.compile(
    r"\b(last|latest|previous|current)\b.*\b(result|simulation|run|scoring|score|edep|hit|crossing)\b|\bwhat happened\b.*\b(run|simulation)\b|\bhow did\b.*\b(run|simulation)\b",
    flags=re.IGNORECASE,
)
_RUN_REQUEST_PATTERN = re.compile(
    r"\b(run|rerun|execute|start)\b.*\b(geant4|simulation|events?|beam)\b|\brun\s+\d+\s+events?\b",
    flags=re.IGNORECASE,
)
_VIEWER_REQUEST_PATTERN = re.compile(
    r"\b(open|launch|show)\b.*\b(viewer|visuali[sz]ation|geometry window)\b",
    flags=re.IGNORECASE,
)

_ZH_CONFIG_DOMAIN_PATTERN = re.compile(
    r"(配置|设置|几何|材料|源|粒子|能量|物理|输出|探测器|计分|靶|束流|点源|盒|圆柱|球|geometry|material|source|particle|energy|physics|output|detector|scoring)"
)
_ZH_CONFIG_MUTATION_PATTERN = re.compile(r"(创建|建立|设置|设为|改成|修改|更新|使用|加入|添加|删除|移除|清除|切换|配置)")
_ZH_CONFIG_READ_PATTERN = re.compile(r"(什么|多少|如何|怎么|状态|摘要|还缺|缺少|当前|已经)")
_ZH_RESULT_READ_PATTERN = re.compile(
    r"(刚才|上次|最近|当前).*(结果|模拟|运行|计分|得分)|结果怎么样|模拟怎么样|运行怎么样|剂量多少|沉积能量|hit|crossing",
    flags=re.IGNORECASE,
)
_ZH_RUN_REQUEST_PATTERN = re.compile(r"(运行|再跑|开始跑|执行).*(geant4|模拟|仿真|事件|event)|跑\s*\d+\s*(个)?\s*(事件|event)", flags=re.IGNORECASE)
_ZH_VIEWER_REQUEST_PATTERN = re.compile(r"(打开|显示|启动).*(viewer|视窗|窗口|几何窗口|可视化)", flags=re.IGNORECASE)


def _lang_key(lang: str) -> str:
    return "zh" if str(lang).lower() == "zh" else "en"


def _safety_for_intent(intent: RuntimeIntent) -> ActionSafetyClass:
    if intent in {RuntimeIntent.READ_SUMMARY, RuntimeIntent.READ_CONFIG}:
        return ActionSafetyClass.READ_ONLY
    if intent == RuntimeIntent.CONFIG_MUTATION:
        return ActionSafetyClass.CONFIG_MUTATION
    if intent in {RuntimeIntent.RUN_REQUESTED, RuntimeIntent.VIEWER_REQUESTED}:
        return ActionSafetyClass.EXPENSIVE_RUNTIME
    return ActionSafetyClass.READ_ONLY


def action_for_runtime_intent(intent: RuntimeIntent | str) -> RuntimeIntentAction:
    intent_key = RuntimeIntent(intent)
    if intent_key == RuntimeIntent.READ_CONFIG:
        return RuntimeIntentAction.CONFIG_SUMMARY
    if intent_key == RuntimeIntent.READ_SUMMARY:
        return RuntimeIntentAction.RUNTIME_SUMMARY
    if intent_key == RuntimeIntent.CONFIG_MUTATION:
        return RuntimeIntentAction.STEP_ASYNC
    if intent_key in {RuntimeIntent.RUN_REQUESTED, RuntimeIntent.VIEWER_REQUESTED}:
        return RuntimeIntentAction.GUARDED_RUNTIME_UI
    return RuntimeIntentAction.READ_ONLY_CHAT


def _classify_rule(text: str, lang: str) -> RuntimeIntent:
    raw = str(text or "").strip().lower()
    if not raw:
        return RuntimeIntent.NORMAL_CHAT

    if _lang_key(lang) == "zh":
        if _ZH_VIEWER_REQUEST_PATTERN.search(raw):
            return RuntimeIntent.VIEWER_REQUESTED
        if _ZH_RUN_REQUEST_PATTERN.search(raw):
            return RuntimeIntent.RUN_REQUESTED
        if _ZH_RESULT_READ_PATTERN.search(raw):
            return RuntimeIntent.READ_SUMMARY
        if _ZH_CONFIG_DOMAIN_PATTERN.search(raw) and _ZH_CONFIG_READ_PATTERN.search(raw):
            return RuntimeIntent.READ_CONFIG
        if _ZH_CONFIG_DOMAIN_PATTERN.search(raw) and _ZH_CONFIG_MUTATION_PATTERN.search(raw):
            return RuntimeIntent.CONFIG_MUTATION
        # Chinese UI can still receive English or mixed-language technical commands.
        return _classify_english_rule(raw)
    return _classify_english_rule(raw)


def _classify_english_rule(raw: str) -> RuntimeIntent:
    if _VIEWER_REQUEST_PATTERN.search(raw):
        return RuntimeIntent.VIEWER_REQUESTED
    if _RUN_REQUEST_PATTERN.search(raw):
        return RuntimeIntent.RUN_REQUESTED
    if _RESULT_READ_PATTERN.search(raw):
        return RuntimeIntent.READ_SUMMARY
    if _CONFIG_READ_PATTERN.search(raw):
        return RuntimeIntent.READ_CONFIG
    if _CONFIG_DOMAIN_PATTERN.search(raw) and _CONFIG_MUTATION_PATTERN.search(raw):
        return RuntimeIntent.CONFIG_MUTATION
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
