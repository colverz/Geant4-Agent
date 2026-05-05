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
    RUNTIME_RESULT_QA = "runtime_result_qa"
    RESULT_QUESTION_ROUTE = "result_question_route"
    PHYSICS_RECOMMEND = "physics_recommend"
    NORMALIZE_USER_TURN = "normalize_user_turn"


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
_SLOT_SLOT_SECTIONS = {"geometry", "materials", "source", "detector", "scoring", "physics", "output"}
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
    "source": {
        "kind",
        "particle",
        "energy_mev",
        "position_mm",
        "direction_vec",
        "spot_radius_mm",
        "spot_profile",
        "spot_sigma_mm",
        "divergence_half_angle_deg",
        "divergence_profile",
        "divergence_sigma_deg",
    },
    "detector": {"enabled", "name", "material", "position_mm", "size_triplet_mm"},
    "scoring": {"target_edep", "detector_crossings", "plane_crossings", "plane_name", "plane_z_mm"},
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
_PHYSICS_RECOMMEND_KEYS = {
    "physics_list",
    "backup_physics_list",
    "reasons",
    "covered_processes",
    "confidence",
}
_NORMALIZE_TOP_LEVEL_KEYS = {"normalized_text", "language_detected", "structure_hint"}
_NORMALIZE_STRUCTURE_HINTS = {
    "ring",
    "grid",
    "nest",
    "stack",
    "shell",
    "single_box",
    "single_tubs",
    "single_sphere",
    "single_cons",
    "single_trd",
    "single_polycone",
    "single_cuttubs",
    "boolean",
    "unknown",
}
_NORMALIZE_CANONICAL_KEYS = [
    "geometry_intent",
    "structure",
    "n",
    "nx",
    "ny",
    "module_x",
    "module_y",
    "module_z",
    "pitch_x",
    "pitch_y",
    "radius",
    "clearance",
    "parent_x",
    "parent_y",
    "parent_z",
    "child_rmax",
    "child_hz",
    "rmax1",
    "rmax2",
    "x1",
    "x2",
    "y1",
    "y2",
    "z1",
    "z2",
    "z3",
    "r1",
    "r2",
    "r3",
    "tilt_x",
    "tilt_y",
    "bool_a_x",
    "bool_a_y",
    "bool_a_z",
    "bool_b_x",
    "bool_b_y",
    "bool_b_z",
    "stack_x",
    "stack_y",
    "t1",
    "t2",
    "t3",
    "stack_clearance",
    "nest_clearance",
    "inner_r",
    "th1",
    "th2",
    "th3",
    "hz",
    "particle",
    "source_type",
    "energy",
    "position",
    "direction",
    "material",
    "physics_list",
    "output_format",
    "output_path",
]
_NORMALIZE_BANNED_ALIASES = [
    "num_elements",
    "element_size",
    "module_size",
    "dimensions",
    "element_radius",
    "element_clearance",
    "source_position",
    "source_direction",
]
_NORMALIZE_GEOMETRY_INTENTS = (
    "circular_placement|planar_array|containment_parent_child|z_layer_sequence|"
    "coaxial_shells|single_box|single_tubs|single_sphere|single_cons|single_trd|"
    "single_polycone|single_cuttubs|boolean|unresolved"
)


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
    (PromptTask.RUNTIME_RESULT_QA, "zh"): PromptProfile(
        id="runtime_result_qa_zh_v1",
        task=PromptTask.RUNTIME_RESULT_QA,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=0.2,
        validator_name="grounded_rewrite_no_new_numbers_lang_match",
        template=(
            "你是 Geant4 模拟结果问答层。请只基于 report 和 base_message 回答用户问题。"
            "不得新增任何数值、因果解释、物理结论或 report 中不存在的事实。"
            "如果 report 不足以回答，必须明确说无法从当前结果确认。只输出最终中文回复。\n\n"
            "用户问题：$user_question\n"
            "Input JSON:\n$payload_json\n\nAnswer now."
        ),
    ),
    (PromptTask.RUNTIME_RESULT_QA, "en"): PromptProfile(
        id="runtime_result_qa_en_v1",
        task=PromptTask.RUNTIME_RESULT_QA,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=0.2,
        validator_name="grounded_rewrite_no_new_numbers_lang_match",
        template=(
            "You are the Geant4 simulation-result Q&A layer. Answer the user question using only report and base_message. "
            "Do not add new numbers, causal explanations, physics conclusions, or facts not present in the report. "
            "If the report is insufficient, say that the current result cannot confirm it. Return only the final English answer.\n\n"
            "User question: $user_question\n"
            "Input JSON:\n$payload_json\n\nAnswer now."
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
            "判断用户是在询问配置、询问最近一次 Geant4 运行结果、修改配置、明确要求运行、打开 viewer，还是普通聊天。"
            "只输出 read_summary、read_config、config_mutation、run_requested、viewer_requested、normal_chat 之一。\n用户：$user_text\nRoute:"
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
            "Classify whether the user asks about current config, asks about the latest Geant4 runtime result, mutates config, explicitly asks to run/open viewer, or is normal chat. "
            "Return exactly one label: read_summary, read_config, config_mutation, run_requested, viewer_requested, normal_chat.\nUser: $user_text\nRoute:"
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
        template=(
            "你是 Geant4 配置助手的用户层改写器。任务：把 base_message 改写成自然、简洁、友好的中文。\n"
            "硬约束：1) 不得新增事实、参数、字段或结论；2) 不得删除关键约束，尤其是覆盖确认提示；"
            "3) 不输出推理过程；4) 不暴露内部字段名；5) 只输出最终给用户的一段文本。\n\n"
            "Context JSON:\n$payload_json\n\nRewrite now."
        ),
    ),
    (PromptTask.RESPONSE_NATURALIZE, "en"): PromptProfile(
        id="response_naturalize_en_v1",
        task=PromptTask.RESPONSE_NATURALIZE,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.GROUNDED_REWRITE,
        temperature=1.0,
        validator_name="grounded_rewrite_no_internal_fields_lang_match",
        template=(
            "You are the user-facing rewrite layer for a Geant4 configuration assistant. "
            "Rewrite base_message into natural, concise English.\n"
            "Hard constraints: 1) do not add facts, parameters, fields, or conclusions; "
            "2) do not remove critical constraints, especially overwrite confirmation prompts; "
            "3) do not output reasoning; 4) do not expose internal field names; "
            "5) return only the final user-facing message.\n\n"
            "Context JSON:\n$payload_json\n\nRewrite now."
        ),
    ),
    (PromptTask.PHYSICS_RECOMMEND, "zh"): PromptProfile(
        id="physics_recommend_zh_v1",
        task=PromptTask.PHYSICS_RECOMMEND,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="physics_recommend_json_allowed_values",
        template=(
            "你是 Geant4 physics list 推荐层。只根据请求和上下文，从 allowed list 中选择 physics_list 和 backup_physics_list。\n"
            "输出 JSON only，允许 keys: physics_list, backup_physics_list, reasons, covered_processes, confidence。\n"
            "硬约束：physics_list 和 backup_physics_list 必须来自 allowed list；不要输出工具调用、配置路径、API key 或额外字段；reasons 保持简短。\n"
            "Allowed: $allowed_lists_csv\n"
            "Context: $context_summary\n"
            "Request: $request_text\n"
            "JSON:"
        ),
    ),
    (PromptTask.PHYSICS_RECOMMEND, "en"): PromptProfile(
        id="physics_recommend_en_v1",
        task=PromptTask.PHYSICS_RECOMMEND,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="physics_recommend_json_allowed_values",
        template=(
            "You are a Geant4 physics-list recommendation layer. Select physics_list and backup_physics_list only from the allowed list.\n"
            "Return JSON only with keys: physics_list, backup_physics_list, reasons, covered_processes, confidence.\n"
            "Hard constraints: physics_list and backup_physics_list must come from the allowed list; do not output tool calls, config paths, API keys, or extra fields; keep reasons concise.\n"
            "Allowed: $allowed_lists_csv\n"
            "Context: $context_summary\n"
            "Request: $request_text\n"
            "JSON:"
        ),
    ),
    (PromptTask.NORMALIZE_USER_TURN, "zh"): PromptProfile(
        id="normalize_user_turn_zh_v1",
        task=PromptTask.NORMALIZE_USER_TURN,
        lang="zh",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="normalization_json_contract",
        template="__NORMALIZE_USER_TURN_PROMPT__",
    ),
    (PromptTask.NORMALIZE_USER_TURN, "en"): PromptProfile(
        id="normalize_user_turn_en_v1",
        task=PromptTask.NORMALIZE_USER_TURN,
        lang="en",
        version="v1",
        output_contract=PromptOutputContract.JSON_ONLY,
        temperature=0.0,
        validator_name="normalization_json_contract",
        template="__NORMALIZE_USER_TURN_PROMPT__",
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


def build_normalization_user_turn_prompt(user_text: str, context_summary: str = "") -> str:
    ctx_block = ""
    if context_summary.strip():
        ctx_block = (
            "Session context (persistent facts from previous turns; keep unless user explicitly changes them):\n"
            f"{context_summary}\n"
        )
    examples = (
        "Examples of valid normalized_text:\n"
        "- User: Set up a copper target box that is 10 by 20 by 30 millimeters.\n"
        "  normalized_text: geometry_intent:single_box; structure:single_box; module_x:10 mm; module_y:20 mm; module_z:30 mm; material:G4_Cu\n"
        "- User: gamma point source 1 MeV at (0,0,-20) mm along +z.\n"
        "  normalized_text: source_type:point; particle:gamma; energy:1 MeV; position:(0,0,-20) mm; direction:+z\n"
        "- User: water cylinder radius 40 mm half length 80 mm; proton beam 150 MeV from (0,0,-120) mm along +z.\n"
        "  normalized_text: geometry_intent:single_tubs; structure:single_tubs; child_rmax:40 mm; child_hz:80 mm; material:G4_WATER; source_type:beam; particle:proton; energy:150 MeV; position:(0,0,-120) mm; direction:+z\n"
        "- User: 请配置一个10 mm x 20 mm x 30 mm的铜盒靶，1 MeV伽马点源放在(0,0,-20) mm，沿+z方向入射。\n"
        "  normalized_text: geometry_intent:single_box; structure:single_box; module_x:10 mm; module_y:20 mm; module_z:30 mm; material:G4_Cu; source_type:point; particle:gamma; energy:1 MeV; position:(0,0,-20) mm; direction:+z\n"
        "Invalid normalized_text examples:\n"
        "- set geometry to copper box with size 10 by 20 by 30 millimeters\n"
        "- set source energy to 1 MeV; set source position to (0,0,-20) mm\n"
    )
    return (
        "Rewrite the user request into controlled English for downstream BERT parsing.\n"
        "Output JSON only with keys:\n"
        "- normalized_text: string\n"
        "- language_detected: string\n"
        "- structure_hint: one of [ring, grid, nest, stack, shell, single_box, single_tubs, single_sphere, single_cons, single_trd, single_polycone, single_cuttubs, boolean, unknown]\n"
        "Normalization rules:\n"
        "- Preserve all numeric values and units exactly (do not convert or round).\n"
        "- normalized_text must be semicolon-separated key:value clauses (no narrative sentence).\n"
        "- normalized_text MUST NOT contain phrases like 'set ... to ...'. Use only key:value clauses.\n"
        f"- geometry_intent must be one of: {_NORMALIZE_GEOMETRY_INTENTS}.\n"
        "- If user text does not explicitly mention geometry shape/layout, geometry_intent must be unresolved.\n"
        "- Use only these canonical keys in normalized_text (plus geometry_intent):\n"
        f"  {', '.join(_NORMALIZE_CANONICAL_KEYS)}\n"
        "- Do NOT output alias keys such as:\n"
        f"  {', '.join(_NORMALIZE_BANNED_ALIASES)}\n"
        "- For 3D size, always emit module_x/module_y/module_z instead of any packed form. Convert '10 by 20 by 30 millimeters' into module_x:10 mm; module_y:20 mm; module_z:30 mm.\n"
        "- For a box/cuboid target, emit geometry_intent:single_box and structure:single_box.\n"
        "- For a cylinder/tube target, emit geometry_intent:single_tubs and structure:single_tubs.\n"
        "- For source vectors, always emit position and direction.\n"
        "- For point source / 点源, emit source_type:point. For beam / 束流, emit source_type:beam.\n"
        "- For gamma / 伽马, emit particle:gamma. For proton / 质子, emit particle:proton.\n"
        "- Normalize common materials to Geant4 names when explicit: copper/铜 -> G4_Cu; water/水 -> G4_WATER; air/空气 -> G4_AIR; silicon/硅 -> G4_Si; lead/铅 -> G4_Pb.\n"
        "- If geometry is ambiguous, use:\n"
        "  geometry_intent: unresolved; structure: unknown; ...\n"
        "- If current turn omits fields but context already contains stable values, keep those values.\n"
        "- Only overwrite a context value when user explicitly requests a change.\n"
        "- Keep text concise and field-like (semicolon-separated clauses), no narrative sentences.\n"
        "- Include only information present in user text; do not hallucinate values.\n"
        "- No explanation or markdown.\n"
        + examples
        + ctx_block
        + f"User text: {user_text}\n"
        + "JSON:"
    )


def build_prompt(task: PromptTask | str, lang: str, context: dict[str, Any]) -> PromptBuildResult:
    task_key = PromptTask(task)
    profile = get_prompt_profile(task_key, lang)
    if task_key == PromptTask.NORMALIZE_USER_TURN:
        prompt = build_normalization_user_turn_prompt(
            str(context.get("user_text", "")),
            str(context.get("context_summary", "")),
        )
        return PromptBuildResult(
            prompt=prompt,
            profile_id=profile.id,
            validator_name=profile.validator_name,
            output_contract=profile.output_contract.value,
            temperature=profile.temperature,
        )
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


def _validate_physics_recommend_json_object(payload: dict[str, Any], allowed_lists: list[str]) -> list[str]:
    errors: list[str] = []
    _append_unknown_keys(errors, payload, _PHYSICS_RECOMMEND_KEYS)
    allowed = {str(item) for item in allowed_lists if str(item)}
    for key in ("physics_list", "backup_physics_list"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        if str(value) not in allowed:
            errors.append(f"value_not_allowed:{key}")
    return errors


def _validate_normalization_json_object(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _append_unknown_keys(errors, payload, _NORMALIZE_TOP_LEVEL_KEYS)
    structure_hint = payload.get("structure_hint")
    if structure_hint not in (None, "") and str(structure_hint) not in _NORMALIZE_STRUCTURE_HINTS:
        errors.append("value_not_allowed:structure_hint")
    normalized_text = str(payload.get("normalized_text", "") or "")
    for banned in _NORMALIZE_BANNED_ALIASES:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(banned)}(?![A-Za-z0-9_])", normalized_text):
            errors.append(f"banned_normalized_key:{banned}")
    if re.search(r"\bset\s+.+\s+to\b", normalized_text, flags=re.IGNORECASE):
        errors.append("narrative_set_to_phrase")
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
            if profile.task == PromptTask.PHYSICS_RECOMMEND:
                allowed_lists = context.get("allowed_lists", [])
                if not isinstance(allowed_lists, list):
                    allowed_lists = []
                errors.extend(_validate_physics_recommend_json_object(parsed, allowed_lists))
            if profile.task == PromptTask.NORMALIZE_USER_TURN:
                errors.extend(_validate_normalization_json_object(parsed))
    if profile.output_contract == PromptOutputContract.ROUTE_LABEL:
        if text.strip() not in {"read_summary", "read_config", "config_mutation", "run_requested", "viewer_requested", "normal_chat"}:
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
