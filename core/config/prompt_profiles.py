from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from string import Template
from typing import Any


class PromptTask(str, Enum):
    SLOT_EXTRACT = "slot_extract"
    SEMANTIC_EXTRACT = "semantic_extract"
    CLARIFICATION = "clarification"
    RESPONSE_NATURALIZE = "response_naturalize"
    RUNTIME_RESULT_EXPLAIN = "runtime_result_explain"
    RESULT_QUESTION_ROUTE = "result_question_route"


class PromptOutputContract(str, Enum):
    JSON_ONLY = "json_only"
    FREE_TEXT = "free_text"
    GROUNDED_REWRITE = "grounded_rewrite"
    QUESTION_ONLY = "question_only"
    ROUTE_LABEL = "route_label"


@dataclass(frozen=True)
class PromptProfile:
    id: str
    task: PromptTask
    lang: str
    version: str
    template: str
    output_contract: PromptOutputContract
    temperature: float
    validator_name: str


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    profile_id: str
    validator_name: str
    output_contract: str
    temperature: float


@dataclass(frozen=True)
class PromptValidationResult:
    ok: bool
    validator_name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_INTERNAL_FIELD_PATTERN = re.compile(r"\b[a-z]+(?:\.[a-z_]+)+\b")
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", flags=re.IGNORECASE)


def _lang_key(lang: str) -> str:
    return "zh" if str(lang).lower() == "zh" else "en"


_PROFILES: dict[tuple[PromptTask, str], PromptProfile] = {
    (PromptTask.CLARIFICATION, "zh"): PromptProfile(
        id="clarification_zh_v1",
        task=PromptTask.CLARIFICATION,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.QUESTION_ONLY,
        temperature=1.0,
        validator_name="question_no_internal_fields_lang_match",
        template=(
            "你是 Geant4-Agent 的对话助手。目标：基于当前上下文，用自然、不机械的语气发起追问。"
            "硬约束：1) 不要列出内部字段名；2) 不要新增需求；3) 一轮最多问 1~2 个关键缺失点；"
            "4) 输出一段最终问句，不要解释。\n"
            "用户最近输入：$recent_user_text\n"
            "已确认信息：$confirmed_items\n"
            "本轮待补充：$missing_items\n"
            "追问："
        ),
    ),
    (PromptTask.CLARIFICATION, "en"): PromptProfile(
        id="clarification_en_v1",
        task=PromptTask.CLARIFICATION,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.QUESTION_ONLY,
        temperature=1.0,
        validator_name="question_no_internal_fields_lang_match",
        template=(
            "You are the Geant4-Agent dialogue assistant. Write a natural, human clarification question using current context. "
            "Constraints: 1) do not expose internal field names; 2) do not introduce new requirements; "
            "3) ask at most 1-2 missing items in this turn; 4) return one final question only.\n"
            "Latest user input: $recent_user_text\n"
            "Confirmed context: $confirmed_items\n"
            "Missing items this turn: $missing_items\n"
            "Question:"
        ),
    ),
    (PromptTask.RUNTIME_RESULT_EXPLAIN, "zh"): PromptProfile(
        id="runtime_result_explain_zh_v1",
        task=PromptTask.RUNTIME_RESULT_EXPLAIN,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=0.2,
        validator_name="grounded_rewrite_no_new_numbers_lang_match",
        template=(
            "你是 Geant4 模拟结果解释层。请把 base_message 改写得更自然，但必须严格受 report 约束。"
            "不得新增任何数值、物理结论、过程解释或 report 中不存在的事实。"
            "如果字段缺失，必须保留缺失含义。只输出最终中文回复。\n\n"
            "Input JSON:\n$payload_json\n\nRewrite now."
        ),
    ),
    (PromptTask.RUNTIME_RESULT_EXPLAIN, "en"): PromptProfile(
        id="runtime_result_explain_en_v1",
        task=PromptTask.RUNTIME_RESULT_EXPLAIN,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=0.2,
        validator_name="grounded_rewrite_no_new_numbers_lang_match",
        template=(
            "You are the Geant4 simulation-result explanation layer. Rewrite base_message naturally, "
            "but stay strictly grounded in report. Do not add any new numbers, physics conclusions, process explanations, "
            "or facts not present in the report. If a field is missing, preserve that meaning. "
            "Return only the final English answer.\n\nInput JSON:\n$payload_json\n\nRewrite now."
        ),
    ),
    (PromptTask.RESULT_QUESTION_ROUTE, "zh"): PromptProfile(
        id="result_question_route_zh_v1",
        task=PromptTask.RESULT_QUESTION_ROUTE,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.ROUTE_LABEL,
        temperature=0.0,
        validator_name="route_label_known_values",
        template=(
            "判断用户是否在询问最近一次 Geant4 运行结果，或是否明确要求运行/打开 viewer。"
            "只输出 read_summary、run_requested、viewer_requested、normal_chat 之一。\n用户：$user_text\nRoute:"
        ),
    ),
    (PromptTask.RESULT_QUESTION_ROUTE, "en"): PromptProfile(
        id="result_question_route_en_v1",
        task=PromptTask.RESULT_QUESTION_ROUTE,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.ROUTE_LABEL,
        temperature=0.0,
        validator_name="route_label_known_values",
        template=(
            "Classify whether the user asks about the latest Geant4 runtime result or explicitly asks to run/open viewer. "
            "Return exactly one label: read_summary, run_requested, viewer_requested, normal_chat.\nUser: $user_text\nRoute:"
        ),
    ),
    (PromptTask.RESPONSE_NATURALIZE, "zh"): PromptProfile(
        id="response_naturalize_zh_v1",
        task=PromptTask.RESPONSE_NATURALIZE,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=1.0,
        validator_name="grounded_rewrite_no_internal_fields_lang_match",
        template="把 base_message 改写成自然中文，不新增事实。\nContext JSON:\n$payload_json\n\nRewrite now.",
    ),
    (PromptTask.RESPONSE_NATURALIZE, "en"): PromptProfile(
        id="response_naturalize_en_v1",
        task=PromptTask.RESPONSE_NATURALIZE,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=1.0,
        validator_name="grounded_rewrite_no_internal_fields_lang_match",
        template="Rewrite base_message into natural English without adding facts.\nContext JSON:\n$payload_json\n\nRewrite now.",
    ),
    (PromptTask.SLOT_EXTRACT, "zh"): PromptProfile(
        id="slot_extract_zh_v1",
        task=PromptTask.SLOT_EXTRACT,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="将用户请求转换为 Geant4 slot frame。只输出 JSON。\nContext: $context_summary\nUser: $user_text\nJSON:",
    ),
    (PromptTask.SLOT_EXTRACT, "en"): PromptProfile(
        id="slot_extract_en_v1",
        task=PromptTask.SLOT_EXTRACT,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="Convert the user request into a Geant4 slot frame. Return JSON only.\nContext: $context_summary\nUser: $user_text\nJSON:",
    ),
    (PromptTask.SEMANTIC_EXTRACT, "zh"): PromptProfile(
        id="semantic_extract_zh_v1",
        task=PromptTask.SEMANTIC_EXTRACT,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="将用户请求转换为 strict semantic frame。只输出 JSON。\nContext: $context_summary\nUser: $user_text\nJSON:",
    ),
    (PromptTask.SEMANTIC_EXTRACT, "en"): PromptProfile(
        id="semantic_extract_en_v1",
        task=PromptTask.SEMANTIC_EXTRACT,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="Convert the user request into a strict semantic frame. Return JSON only.\nContext: $context_summary\nUser: $user_text\nJSON:",
    ),
}


def get_prompt_profile(task: PromptTask | str, lang: str) -> PromptProfile:
    task_key = PromptTask(task)
    key = (task_key, _lang_key(lang))
    return _PROFILES[key]


def list_prompt_profiles() -> list[PromptProfile]:
    return list(_PROFILES.values())


def build_prompt(task: PromptTask | str, lang: str, context: dict[str, Any]) -> PromptBuildResult:
    profile = get_prompt_profile(task, lang)
    values = {key: str(value) for key, value in context.items()}
    if "payload" in context and "payload_json" not in values:
        values["payload_json"] = json.dumps(context["payload"], ensure_ascii=False)
    prompt = Template(profile.template).safe_substitute(values)
    return PromptBuildResult(
        prompt=prompt,
        profile_id=profile.id,
        validator_name=profile.validator_name,
        output_contract=profile.output_contract.value,
        temperature=profile.temperature,
    )


def _looks_language_mismatched(text: str, lang: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    has_cjk = bool(_CJK_PATTERN.search(compact))
    has_ascii_words = bool(re.search(r"[A-Za-z]{3,}", compact))
    if _lang_key(lang) == "zh":
        return has_ascii_words and not has_cjk
    return has_cjk and not has_ascii_words


def _numeric_tokens(text: str) -> set[str]:
    return {token.lower() for token in _NUMBER_PATTERN.findall(str(text or ""))}


def validate_prompt_output(
    task: PromptTask | str,
    lang: str,
    text_or_json: Any,
    source_context: dict[str, Any] | None = None,
) -> PromptValidationResult:
    profile = get_prompt_profile(task, lang)
    text = text_or_json if isinstance(text_or_json, str) else json.dumps(text_or_json, ensure_ascii=False)
    errors: list[str] = []
    context = source_context or {}

    if profile.output_contract == PromptOutputContract.JSON_ONLY:
        try:
            json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append("not_json")
    if profile.output_contract == PromptOutputContract.ROUTE_LABEL:
        if text.strip() not in {"read_summary", "run_requested", "viewer_requested", "normal_chat"}:
            errors.append("unknown_route_label")
    if profile.output_contract in {PromptOutputContract.QUESTION_ONLY, PromptOutputContract.GROUNDED_REWRITE}:
        if _INTERNAL_FIELD_PATTERN.search(text):
            errors.append("internal_field_leak")
        if _looks_language_mismatched(text, lang):
            errors.append("language_mismatch")
    if profile.validator_name == "grounded_rewrite_no_new_numbers_lang_match":
        base_message = str(context.get("base_message", ""))
        if _numeric_tokens(text) - _numeric_tokens(base_message):
            errors.append("new_numeric_value")

    return PromptValidationResult(ok=not errors, validator_name=profile.validator_name, errors=errors)
