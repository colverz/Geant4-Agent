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
        "- Bind geometry_candidate to the target/object being built.\n"
        "- Bind source_candidate to the emitter/beam/source phrase.\n"
        "- If a sentence mentions both target and source, do not mix their properties.\n"
        "- Relative phrases such as 'in front of target', 'outside target center', or 'toward target center' belong to source placement or source direction, not target size.\n"
        "- If the user says '10 mm square target' or '10 mm cube', prefer box plus side_length_mm rather than slab thickness.\n"
        "- If the user says a material together with target/object words such as 'copper target', 'lead slab', or 'tungsten box', treat that material as geometry_candidate.material_candidate unless the sentence clearly assigns it to source or something else.\n"
        "- Words like 'target', '靶', '靶体', 'box target', or 'target box' usually describe the geometry object, not the source.\n"
        "Examples:\n"
        '- User: "10 mm x 20 mm x 30 mm copper box target"\n'
        '  geometry_candidate.kind_candidate = "box"\n'
        '  geometry_candidate.dimension_hints.size_triplet_mm = [10,20,30]\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '- User: "10 mm square copper target"\n'
        '  geometry_candidate.kind_candidate = "box"\n'
        '  geometry_candidate.dimension_hints.side_length_mm = 10\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '- User: "copper target"\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '  geometry_candidate.kind_candidate should stay null if the shape is not actually specified\n'
        '- User: "gamma point source 1 MeV at (0,0,-20) mm along +z"\n'
        '  source_candidate.source_type_candidate = "point"\n'
        '  source_candidate.particle_candidate = "gamma"\n'
        '  source_candidate.energy_candidate_mev = 1.0\n'
        '  source_candidate.position_mode = "absolute"\n'
        '  source_candidate.direction_mode = "explicit_vector"\n'
        '- User: "place a gamma point source 5 mm in front of the target, toward target center"\n'
        '  source_candidate.position_mode = "relative_to_target_face"\n'
        '  source_candidate.direction_mode = "toward_target_center"\n'
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )


def _build_interpreter_prompt_zh(user_text: str, context_summary: str) -> str:
    return (
        "\u8bf7\u89e3\u91ca\u8fd9\u8f6e Geant4 \u914d\u7f6e\u8bf7\u6c42\u771f\u6b63\u8868\u8fbe\u7684\u610f\u601d\u3002\n"
        "\u4f60\u7684\u4efb\u52a1\u662f\u628a\u7528\u6237\u7684\u8bdd\u6574\u7406\u6210\u53d7\u63a7\u5019\u9009\u542b\u4e49\uff0c\u800c\u4e0d\u662f\u76f4\u63a5\u5199\u6700\u7ec8\u914d\u7f6e\u8def\u5f84\u3002\n"
        "\u53ea\u8fd4\u56de JSON\uff0c\u7ed3\u6784\u5982\u4e0b\uff1a\n"
        "{\n"
        '  "turn_summary": {\n'
        '    "intent": "set|modify|confirm|reject|question|other",\n'
        '    "focus": "geometry|source|physics|output|mixed",\n'
        '    "scope": "full_request|partial_update|clarification",\n'
        '    "user_goal": "\u4e00\u53e5\u8bdd\u6982\u62ec\u7528\u6237\u76ee\u6807",\n'
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
        '    "ambiguities": ["\u54ea\u91cc\u4e0d\u786e\u5b9a"],\n'
        '    "evidence_spans": [{"text":"10 mm \u89c1\u65b9","role":"dimensions"}]\n'
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
        '    "ambiguities": ["\u54ea\u91cc\u4e0d\u786e\u5b9a"],\n'
        '    "evidence_spans": [{"text":"\u4f4d\u4e8e (0,0,-20) mm","role":"position"}]\n'
        "  }\n"
        "}\n"
        "\u786c\u89c4\u5219\uff1a\n"
        "- \u4e0d\u8981\u8f93\u51fa\u6700\u7ec8 config path\u3002\n"
        "- \u4e0d\u8981\u8111\u8865\u7f3a\u5931\u503c\u3002\n"
        "- \u5982\u679c\u7528\u6237\u6ca1\u8bf4\u6e05\u695a\uff0c\u5c31\u628a\u5bf9\u5e94\u5b57\u6bb5\u7559\u7a7a\uff0c\u5e76\u628a\u4e0d\u786e\u5b9a\u70b9\u5199\u8fdb ambiguities\u3002\n"
        "- \u4e25\u683c\u9075\u5b88\u4e0a\u9762\u7684 schema\uff0c\u4e0d\u8981\u65b0\u589e\u5b57\u6bb5\u3002\n"
        "- \u5019\u9009\u89e3\u91ca\u8981\u5fe0\u5b9e\u3001\u514b\u5236\uff0c\u4e0d\u8981\u81ea\u4f5c\u806a\u660e\u3002\n"
        "- evidence_spans \u8981\u5c3d\u91cf\u6307\u5411\u539f\u53e5\u4e2d\u7684\u660e\u786e\u8bc1\u636e\u3002\n"
        "- \u5982\u679c\u8fd9\u4e00\u8f6e\u53ea\u6539\u4e00\u4e2a\u9886\u57df\uff0c\u5c31\u4e0d\u8981\u628a\u65e0\u5173\u9886\u57df\u5199\u8fdb focus\u3002\n"
        "- geometry \u548c source \u8fd9\u91cc\u53ea\u662f\u5019\u9009\u89e3\u91ca\uff0c\u6700\u7ec8\u662f\u5426\u53ef\u6267\u884c\u7531\u540e\u7eed\u5c42\u51b3\u5b9a\u3002\n"
        "- geometry_candidate \u53ea\u7ed1\u5b9a\u201c\u9776\u3001\u51e0\u4f55\u4f53\u3001\u76ee\u6807\u7269\u4f53\u201d\u8fd9\u4e00\u7c7b\u63cf\u8ff0\u3002\n"
        "- source_candidate \u53ea\u7ed1\u5b9a\u201c\u6e90\u3001\u675f\u6d41\u3001\u5165\u5c04\u7c92\u5b50\u201d\u8fd9\u4e00\u7c7b\u63cf\u8ff0\u3002\n"
        "- \u5982\u679c\u540c\u4e00\u53e5\u91cc\u65e2\u6709\u9776\u53c8\u6709\u6e90\uff0c\u4e0d\u8981\u628a\u6e90\u7684\u5c5e\u6027\u89e3\u91ca\u6210\u51e0\u4f55\uff0c\u4e5f\u4e0d\u8981\u628a\u51e0\u4f55\u5c3a\u5bf8\u89e3\u91ca\u6210 source \u3002\n"
        "- \u201c\u8ddd\u9776\u524d\u8868\u9762\u5916 5 mm\u201d\u3001\u201c\u5728\u9776\u524d\u65b9\u201d\u3001\u201c\u671d\u9776\u5fc3\u201d\u3001\u201c\u671d\u9776\u9762\u6cd5\u7ebf\u65b9\u5411\u201d\u8fd9\u7c7b\u8868\u8fbe\u9ed8\u8ba4\u5c5e\u4e8e source \u7684\u4f4d\u7f6e\u6216\u65b9\u5411\uff0c\u4e0d\u662f geometry \u7684\u5c3a\u5bf8\u3002\n"
        "- \u201c10 mm \u89c1\u65b9\u9776\u201d\u6216\u201c10 mm \u7acb\u65b9\u4f53\u201d\u66f4\u503e\u5411\u4e8e box \u7684 side_length_mm\uff0c\u4e0d\u662f slab \u539a\u5ea6\u3002\n"
        "- \u5982\u679c\u201c\u94dc\u9776\u201d\u3001\u201c\u94c5\u677f\u201d\u3001\u201c\u94a8\u76d2\u9776\u201d\u8fd9\u79cd\u201c\u6750\u6599 + \u9776/\u677f/\u9776\u4f53/\u76ee\u6807\u7269\u4f53\u201d\u7684\u8bf4\u6cd5\u4e00\u8d77\u51fa\u73b0\uff0c\u9ed8\u8ba4\u628a\u6750\u6599\u7406\u89e3\u6210 geometry_candidate.material_candidate\uff0c\u4e0d\u8981\u628a\u6750\u6599\u5f52\u5230 source \u3002\n"
        "- \u201c\u9776\u201d\u3001\u201c\u9776\u4f53\u201d\u3001\u201c\u76d2\u9776\u201d\u3001\u201c\u9776\u6750\u201d\u8fd9\u7c7b\u8bcd\u9ed8\u8ba4\u63cf\u8ff0 geometry \u5bf9\u8c61\uff0c\u800c\u4e0d\u662f source\u3002\n"
        "\u793a\u4f8b\uff1a\n"
        '- \u7528\u6237: "10 mm x 20 mm x 30 mm \u94dc\u76d2\u9776"\n'
        '  geometry_candidate.kind_candidate = "box"\n'
        '  geometry_candidate.dimension_hints.size_triplet_mm = [10,20,30]\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '- \u7528\u6237: "10 mm \u89c1\u65b9\u94dc\u9776"\n'
        '  geometry_candidate.kind_candidate = "box"\n'
        '  geometry_candidate.dimension_hints.side_length_mm = 10\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '- \u7528\u6237: "\u94dc\u9776"\n'
        '  geometry_candidate.material_candidate = "G4_Cu"\n'
        '  geometry_candidate.kind_candidate \u53ef\u4ee5\u4e3a null\uff0c\u56e0\u4e3a\u53ea\u8bf4\u4e86\u6750\u6599\u548c\u9776\uff0c\u6ca1\u6709\u771f\u6b63\u8bf4\u6e05\u51e0\u4f55\u5f62\u72b6\n'
        '- \u7528\u6237: "gamma \u70b9\u6e90 1 MeV\uff0c\u4f4d\u4e8e (0,0,-20) mm\uff0c\u6cbf +z \u65b9\u5411"\n'
        '  source_candidate.source_type_candidate = "point"\n'
        '  source_candidate.particle_candidate = "gamma"\n'
        '  source_candidate.energy_candidate_mev = 1.0\n'
        '  source_candidate.position_mode = "absolute"\n'
        '  source_candidate.direction_mode = "explicit_vector"\n'
        '- \u7528\u6237: "\u5728\u9776\u524d\u8868\u9762\u5916 5 mm \u653e\u4e00\u4e2a gamma \u70b9\u6e90\uff0c\u671d\u9776\u5fc3\u5165\u5c04"\n'
        '  source_candidate.position_mode = "relative_to_target_face"\n'
        '  source_candidate.direction_mode = "toward_target_center"\n'
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )


def build_interpreter_prompt(user_text: str, context_summary: str) -> str:
    language = detect_prompt_language(user_text)
    if language in {"zh", "mixed"}:
        return _build_interpreter_prompt_zh(user_text, context_summary)
    return _build_interpreter_prompt_en(user_text, context_summary)
