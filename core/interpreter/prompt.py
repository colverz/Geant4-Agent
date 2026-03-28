from __future__ import annotations


def detect_prompt_language(user_text: str) -> str:
    text = user_text or ""
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    has_ascii_alpha = any(("a" <= ch.lower() <= "z") for ch in text)
    if has_cjk and has_ascii_alpha:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def _build_interpreter_prompt_en(user_text: str, context_summary: str) -> str:
    return (
        "Interpret the user request for a Geant4 configuration session.\n"
        "Your job is to explain what the user appears to mean, not to write final config paths.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "turn_summary": {\n'
        '    "intent": "set|modify|confirm|reject|question|other",\n'
        '    "focus": "geometry|source|physics|output|mixed",\n'
        '    "scope": "full_request|partial_update|clarification",\n'
        '    "user_goal": "brief summary of the user goal",\n'
        '    "explicit_domains": ["geometry"],\n'
        '    "uncertain_domains": ["source"]\n'
        "  },\n"
        '  "geometry_candidate": {\n'
        '    "kind_candidate": "box|cylinder|sphere|orb|cons|trd|slab|plate|null",\n'
        '    "material_candidate": "G4_Cu|null",\n'
        '    "dimension_hints": {\n'
        '      "size_triplet_mm": [null,null,null],\n'
        '      "side_length_mm": null,\n'
        '      "radius_mm": null,\n'
        '      "diameter_mm": null,\n'
        '      "half_length_mm": null,\n'
        '      "full_length_mm": null,\n'
        '      "thickness_mm": null\n'
        "    },\n"
        '    "placement_relation": null,\n'
        '    "confidence": 0.0,\n'
        '    "ambiguities": ["what is unclear"],\n'
        '    "evidence_spans": [{"text":"10 mm x 20 mm x 30 mm","role":"dimensions"}]\n'
        "  },\n"
        '  "source_candidate": {\n'
        '    "source_type_candidate": "point|beam|plane|isotropic|null",\n'
        '    "particle_candidate": "gamma|e-|proton|neutron|null",\n'
        '    "energy_candidate_mev": null,\n'
        '    "position_mode": "absolute|relative_to_target_center|relative_to_target_face|null",\n'
        '    "position_hint": {\n'
        '      "position_mm": [null,null,null],\n'
        '      "offset_mm": null,\n'
        '      "axis": "+x|-x|+y|-y|+z|-z|null"\n'
        "    },\n"
        '    "direction_mode": "explicit_vector|toward_target_center|toward_target_face|toward_target_face_normal|normal_to_target_face|unknown|null",\n'
        '    "direction_hint": {\n'
        '      "direction_vec": [null,null,null],\n'
        '      "axis": "+x|-x|+y|-y|+z|-z|null"\n'
        "    },\n"
        '    "confidence": 0.0,\n'
        '    "ambiguities": ["what is unclear"],\n'
        '    "evidence_spans": [{"text":"at (0,0,-20) mm","role":"position"}]\n'
        "  }\n"
        "}\n"
        "Hard rules:\n"
        "- Do not output final config paths.\n"
        "- Do not invent missing values.\n"
        "- If the user is unclear, keep the corresponding field null and explain the ambiguity.\n"
        "- Stay inside the schema. Do not add new keys.\n"
        "- Prefer a short, faithful interpretation over a clever one.\n"
        "- Use evidence_spans to point to the exact wording that supports your interpretation.\n"
        "- If the request only changes one area, keep unrelated domains out of focus.\n"
        "- Geometry and source are interpreted candidates only; final execution decisions happen later.\n"
        "Examples:\n"
        '- User: "10 mm x 20 mm x 30 mm copper box target"\n'
        '  geometry_candidate.kind_candidate = "box"\n'
        '  geometry_candidate.dimension_hints.size_triplet_mm = [10,20,30]\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '- User: "gamma point source 1 MeV at (0,0,-20) mm along +z"\n'
        '  source_candidate.source_type_candidate = "point"\n'
        '  source_candidate.particle_candidate = "gamma"\n'
        '  source_candidate.energy_candidate_mev = 1.0\n'
        '  source_candidate.position_mode = "absolute"\n'
        '  source_candidate.direction_mode = "explicit_vector"\n'
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )


def _build_interpreter_prompt_zh(user_text: str, context_summary: str) -> str:
    return (
        "请解释这轮 Geant4 配置请求真正表达的意思。\n"
        "你的任务是把用户的话整理成受控候选含义，而不是直接写最终配置路径。\n"
        "只返回 JSON，结构如下：\n"
        "{\n"
        '  "turn_summary": {\n'
        '    "intent": "set|modify|confirm|reject|question|other",\n'
        '    "focus": "geometry|source|physics|output|mixed",\n'
        '    "scope": "full_request|partial_update|clarification",\n'
        '    "user_goal": "一句话概括用户目标",\n'
        '    "explicit_domains": ["geometry"],\n'
        '    "uncertain_domains": ["source"]\n'
        "  },\n"
        '  "geometry_candidate": {\n'
        '    "kind_candidate": "box|cylinder|sphere|orb|cons|trd|slab|plate|null",\n'
        '    "material_candidate": "G4_Cu|null",\n'
        '    "dimension_hints": {\n'
        '      "size_triplet_mm": [null,null,null],\n'
        '      "side_length_mm": null,\n'
        '      "radius_mm": null,\n'
        '      "diameter_mm": null,\n'
        '      "half_length_mm": null,\n'
        '      "full_length_mm": null,\n'
        '      "thickness_mm": null\n'
        "    },\n"
        '    "placement_relation": null,\n'
        '    "confidence": 0.0,\n'
        '    "ambiguities": ["哪里不确定"],\n'
        '    "evidence_spans": [{"text":"10 mm 见方","role":"dimensions"}]\n'
        "  },\n"
        '  "source_candidate": {\n'
        '    "source_type_candidate": "point|beam|plane|isotropic|null",\n'
        '    "particle_candidate": "gamma|e-|proton|neutron|null",\n'
        '    "energy_candidate_mev": null,\n'
        '    "position_mode": "absolute|relative_to_target_center|relative_to_target_face|null",\n'
        '    "position_hint": {\n'
        '      "position_mm": [null,null,null],\n'
        '      "offset_mm": null,\n'
        '      "axis": "+x|-x|+y|-y|+z|-z|null"\n'
        "    },\n"
        '    "direction_mode": "explicit_vector|toward_target_center|toward_target_face|toward_target_face_normal|normal_to_target_face|unknown|null",\n'
        '    "direction_hint": {\n'
        '      "direction_vec": [null,null,null],\n'
        '      "axis": "+x|-x|+y|-y|+z|-z|null"\n'
        "    },\n"
        '    "confidence": 0.0,\n'
        '    "ambiguities": ["哪里不确定"],\n'
        '    "evidence_spans": [{"text":"位于 (0,0,-20) mm","role":"position"}]\n'
        "  }\n"
        "}\n"
        "硬规则：\n"
        "- 不要输出最终 config path。\n"
        "- 不要脑补缺失值。\n"
        "- 如果用户没说清楚，就把对应字段留空，并把不确定点写进 ambiguities。\n"
        "- 严格遵守上面的 schema，不要新增字段。\n"
        "- 候选解释要忠实、克制，不要自作聪明。\n"
        "- evidence_spans 要尽量指向原句中的明确证据。\n"
        "- 如果这一轮只改一个领域，就不要把无关领域写进 focus。\n"
        "- geometry 和 source 这里只是候选解释，最终是否可执行由后续层决定。\n"
        "示例：\n"
        '- 用户: "10 mm x 20 mm x 30 mm 铜盒靶"\n'
        '  geometry_candidate.kind_candidate = "box"\n'
        '  geometry_candidate.dimension_hints.size_triplet_mm = [10,20,30]\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '- 用户: "gamma 点源 1 MeV，位于 (0,0,-20) mm，沿 +z 方向"\n'
        '  source_candidate.source_type_candidate = "point"\n'
        '  source_candidate.particle_candidate = "gamma"\n'
        '  source_candidate.energy_candidate_mev = 1.0\n'
        '  source_candidate.position_mode = "absolute"\n'
        '  source_candidate.direction_mode = "explicit_vector"\n'
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )


def build_interpreter_prompt(user_text: str, context_summary: str) -> str:
    language = detect_prompt_language(user_text)
    if language in {"zh", "mixed"}:
        return _build_interpreter_prompt_zh(user_text, context_summary)
    return _build_interpreter_prompt_en(user_text, context_summary)
