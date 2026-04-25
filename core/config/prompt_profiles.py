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
_SLOT_TOP_LEVEL_KEYS = {"intent", "confidence", "normalized_text", "target_slots", "slots", "candidates"}
_SLOT_SLOT_SECTIONS = {"geometry", "materials", "source", "physics", "output"}
_SLOT_FIELDS = {
    "geometry": {
        "kind",
        "size_triplet_mm",
        "radius_mm",
        "half_length_mm",
        "radius1_mm",
        "radius2_mm",
        "x1_mm",
        "x2_mm",
        "y1_mm",
        "y2_mm",
        "z_mm",
        "z_planes_mm",
        "radii_mm",
        "polyhedra_sides",
        "trap_x1_mm",
        "trap_x2_mm",
        "trap_x3_mm",
        "trap_x4_mm",
        "trap_y1_mm",
        "trap_y2_mm",
        "trap_z_mm",
        "para_x_mm",
        "para_y_mm",
        "para_z_mm",
        "para_alpha_deg",
        "para_theta_deg",
        "para_phi_deg",
        "torus_major_radius_mm",
        "torus_minor_radius_mm",
        "ellipsoid_ax_mm",
        "ellipsoid_by_mm",
        "ellipsoid_cz_mm",
        "elltube_ax_mm",
        "elltube_by_mm",
        "elltube_hz_mm",
        "tilt_x_deg",
        "tilt_y_deg",
    },
    "materials": {"primary"},
    "source": {"kind", "particle", "energy_mev", "position_mm", "direction_vec"},
    "physics": {"explicit_list", "recommendation_intent"},
    "output": {"format", "path"},
}
_SLOT_CANDIDATE_SECTIONS = {"geometry", "source"}
_SLOT_CANDIDATE_FIELDS = {
    "geometry": {
        "kind_candidate",
        "side_length_mm",
        "radius_mm",
        "diameter_mm",
        "half_length_mm",
        "full_length_mm",
        "thickness_mm",
        "plate_size_xy_mm",
    },
    "source": {"relation", "offset_mm", "axis", "direction_mode", "direction_relation"},
}
_SEMANTIC_TOP_LEVEL_KEYS = {
    "intent",
    "target_paths",
    "normalized_text",
    "structure_hint",
    "confidence",
    "updates",
}
_SEMANTIC_UPDATE_KEYS = {"path", "op", "value"}


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
            "只输出 read_summary、read_config、run_requested、viewer_requested、normal_chat 之一。\n用户：$user_text\nRoute:"
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
            "Return exactly one label: read_summary, read_config, run_requested, viewer_requested, normal_chat.\nUser: $user_text\nRoute:"
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
        id="slot_extract_zh_strict_slot_v2",
        task=PromptTask.SLOT_EXTRACT,
        lang="zh",
        version="strict_slot_v2",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="__STRICT_SLOT_PROMPT__",
    ),
    (PromptTask.SLOT_EXTRACT, "en"): PromptProfile(
        id="slot_extract_en_strict_slot_v2",
        task=PromptTask.SLOT_EXTRACT,
        lang="en",
        version="strict_slot_v2",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="__STRICT_SLOT_PROMPT__",
    ),
    (PromptTask.SEMANTIC_EXTRACT, "zh"): PromptProfile(
        id="semantic_extract_zh_strict_semantic_v1",
        task=PromptTask.SEMANTIC_EXTRACT,
        lang="zh",
        version="strict_semantic_v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="__STRICT_SEMANTIC_PROMPT__",
    ),
    (PromptTask.SEMANTIC_EXTRACT, "en"): PromptProfile(
        id="semantic_extract_en_strict_semantic_v1",
        task=PromptTask.SEMANTIC_EXTRACT,
        lang="en",
        version="strict_semantic_v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="json_only",
        template="__STRICT_SEMANTIC_PROMPT__",
    ),
}


def get_prompt_profile(task: PromptTask | str, lang: str) -> PromptProfile:
    task_key = PromptTask(task)
    key = (task_key, _lang_key(lang))
    return _PROFILES[key]


def list_prompt_profiles() -> list[PromptProfile]:
    return list(_PROFILES.values())


def build_prompt(task: PromptTask | str, lang: str, context: dict[str, Any]) -> PromptBuildResult:
    task_key = PromptTask(task)
    profile = get_prompt_profile(task_key, lang)
    if task_key in {PromptTask.SLOT_EXTRACT, PromptTask.SEMANTIC_EXTRACT}:
        user_text = str(context.get("user_text", ""))
        context_summary = str(context.get("context_summary", ""))
        if task_key == PromptTask.SLOT_EXTRACT:
            from core.config.llm_prompt_registry import build_strict_slot_prompt

            prompt = build_strict_slot_prompt(user_text, context_summary)
        else:
            from core.config.llm_prompt_registry import build_strict_semantic_prompt

            prompt = build_strict_semantic_prompt(user_text, context_summary)
        return PromptBuildResult(
            prompt=prompt,
            profile_id=profile.id,
            validator_name=profile.validator_name,
            output_contract=profile.output_contract.value,
            temperature=profile.temperature,
        )

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


def _append_unknown_keys(errors: list[str], data: dict[str, Any], allowed: set[str], prefix: str = "") -> None:
    for key in data:
        key_text = str(key)
        if key_text not in allowed:
            errors.append(f"unknown_json_key:{prefix}{key_text}")


def _validate_slot_json_object(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _append_unknown_keys(errors, payload, _SLOT_TOP_LEVEL_KEYS)
    slots = payload.get("slots")
    if isinstance(slots, dict):
        _append_unknown_keys(errors, slots, _SLOT_SLOT_SECTIONS, "slots.")
        for section, allowed_fields in _SLOT_FIELDS.items():
            section_payload = slots.get(section)
            if isinstance(section_payload, dict):
                _append_unknown_keys(errors, section_payload, allowed_fields, f"slots.{section}.")
    candidates = payload.get("candidates")
    if isinstance(candidates, dict):
        _append_unknown_keys(errors, candidates, _SLOT_CANDIDATE_SECTIONS, "candidates.")
        for section, allowed_fields in _SLOT_CANDIDATE_FIELDS.items():
            section_payload = candidates.get(section)
            if isinstance(section_payload, dict):
                _append_unknown_keys(errors, section_payload, allowed_fields, f"candidates.{section}.")
    return errors


def _validate_semantic_json_object(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _append_unknown_keys(errors, payload, _SEMANTIC_TOP_LEVEL_KEYS)
    updates = payload.get("updates")
    if isinstance(updates, list):
        for idx, item in enumerate(updates):
            if isinstance(item, dict):
                _append_unknown_keys(errors, item, _SEMANTIC_UPDATE_KEYS, f"updates[{idx}].")
    return errors


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
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append("not_json")
            parsed = None
        if isinstance(parsed, dict):
            if profile.task == PromptTask.SLOT_EXTRACT:
                errors.extend(_validate_slot_json_object(parsed))
            if profile.task == PromptTask.SEMANTIC_EXTRACT:
                errors.extend(_validate_semantic_json_object(parsed))
    if profile.output_contract == PromptOutputContract.ROUTE_LABEL:
        if text.strip() not in {"read_summary", "read_config", "run_requested", "viewer_requested", "normal_chat"}:
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
