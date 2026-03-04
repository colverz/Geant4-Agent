from __future__ import annotations


STRICT_SLOT_PROMPT_PROFILE = "strict_slot_v2"
STRICT_SEMANTIC_PROMPT_PROFILE = "strict_semantic_v1"


def _build_strict_slot_prompt_v1(user_text: str, context_summary: str) -> str:
    return (
        "Convert the user request into a deterministic slot frame for a Geant4 configuration session.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "intent": "SET|MODIFY|REMOVE|CONFIRM|QUESTION|OTHER",\n'
        '  "confidence": 0.0,\n'
        '  "normalized_text": "semicolon-separated canonical clauses",\n'
        '  "target_slots": ["geometry.kind", "materials.primary"],\n'
        '  "slots": {\n'
        '    "geometry": {"kind": "box|cylinder|sphere|null", "size_triplet_mm": [1000,1000,1000], "radius_mm": null, "half_length_mm": null},\n'
        '    "materials": {"primary": "G4_Cu|null"},\n'
        '    "source": {"kind": "point|beam|plane|isotropic|null", "particle": "gamma|e-|proton|neutron|null", "energy_mev": 1.0, "position_mm": [0,0,-100], "direction_vec": [0,0,1]},\n'
        '    "physics": {"explicit_list": "FTFP_BERT|null", "recommendation_intent": "gamma_attenuation|null"},\n'
        '    "output": {"format": "csv|hdf5|root|xml|json|null", "path": null}\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Use slots, not config paths.\n"
        "- If the user gives dimensions like 1 m x 1 m x 1 m, convert them to size_triplet_mm.\n"
        "- For cylinders, use radius_mm and half_length_mm (Geant4 half-length semantics), not full height.\n"
        "- If the user gives vectors, convert them to numeric arrays.\n"
        "- If the user names a material informally (e.g. copper), normalize it to Geant4 NIST material if known.\n"
        "- If the user explicitly names a physics list, put it in physics.explicit_list.\n"
        "- If the user asks for a best-fit physics list without naming one, set physics.recommendation_intent instead of inventing a list.\n"
        "- Prefer official Geant4 analysis file types (csv, hdf5, root, xml); use json only for the project-local export mode.\n"
        "- Internally canonicalize any input language into English clauses.\n"
        "- Never output placeholder strings like 'null', 'none', or 'unknown'; use JSON null when the value is absent.\n"
        "- Keep normalized_text concise and in English canonical clauses.\n"
        "Examples:\n"
        '- User: "copper cylinder, radius 30 mm, half-length 50 mm"\n'
        '  JSON slots.geometry = {"kind":"cylinder","radius_mm":30,"half_length_mm":50}\n'
        '- User: "\\u4e00\\u7c73\\u89c1\\u65b9\\u7684\\u94dc\\u7acb\\u65b9\\u4f53"\n'
        '  JSON slots.geometry = {"kind":"box","size_triplet_mm":[1000,1000,1000]}\n'
        '  JSON slots.materials = {"primary":"G4_Cu"}\n'
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )


def _build_strict_slot_prompt_v2(user_text: str, context_summary: str) -> str:
    return (
        "Convert the user request into a deterministic slot frame for a Geant4 configuration session.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "intent": "SET|MODIFY|REMOVE|CONFIRM|QUESTION|OTHER",\n'
        '  "confidence": 0.0,\n'
        '  "normalized_text": "semicolon-separated canonical clauses",\n'
        '  "target_slots": ["geometry.kind", "materials.primary"],\n'
        '  "slots": {\n'
        '    "geometry": {"kind": "box|cylinder|sphere|null", "size_triplet_mm": [1000,1000,1000], "radius_mm": null, "half_length_mm": null},\n'
        '    "materials": {"primary": "G4_Cu|null"},\n'
        '    "source": {"kind": "point|beam|plane|isotropic|null", "particle": "gamma|e-|proton|neutron|null", "energy_mev": 1.0, "position_mm": [0,0,-100], "direction_vec": [0,0,1]},\n'
        '    "physics": {"explicit_list": "FTFP_BERT|null", "recommendation_intent": "gamma_attenuation|null"},\n'
        '    "output": {"format": "csv|hdf5|root|xml|json|null", "path": null}\n'
        "  }\n"
        "}\n"
        "Hard rules:\n"
        "- Use slots, not config paths.\n"
        "- Fill only what the user explicitly states or explicitly changes in this turn.\n"
        "- Do not restate unrelated slots from previous turns; leave them null when absent in the current turn.\n"
        "- If the turn is a narrow update (for example only output format, only one material change, or only source energy), keep other slot groups empty.\n"
        "- If the user asks for an explanation or recommendation reason, keep the turn as QUESTION unless they also explicitly change a value.\n"
        "- If the user confirms a pending overwrite, set intent=CONFIRM and leave slots empty.\n"
        "- Never output placeholder strings like 'null', 'none', or 'unknown'; use JSON null when the value is absent.\n"
        "Domain rules:\n"
        "- If the user gives dimensions like 1 m x 1 m x 1 m, convert them to size_triplet_mm.\n"
        "- For cylinders, use radius_mm and half_length_mm (Geant4 half-length semantics), not full height.\n"
        "- If the user gives vectors, convert them to numeric arrays.\n"
        "- If the user names a material informally (e.g. copper), normalize it to Geant4 NIST material if known.\n"
        "- If the user explicitly names a physics list, put it in physics.explicit_list.\n"
        "- If the user asks for a best-fit physics list without naming one, set physics.recommendation_intent instead of inventing a list.\n"
        "- Prefer official Geant4 analysis file types (csv, hdf5, root, xml); use json only for the project-local export mode.\n"
        "- Internally canonicalize any input language into English clauses.\n"
        "- Keep normalized_text concise and in English canonical clauses.\n"
        "Examples:\n"
        '- User: "copper cylinder, radius 30 mm, half-length 50 mm"\n'
        '  JSON slots.geometry = {"kind":"cylinder","radius_mm":30,"half_length_mm":50}\n'
        '- User: "\\u4e00\\u7c73\\u89c1\\u65b9\\u7684\\u94dc\\u7acb\\u65b9\\u4f53"\n'
        '  JSON slots.geometry = {"kind":"box","size_triplet_mm":[1000,1000,1000]}\n'
        '  JSON slots.materials = {"primary":"G4_Cu"}\n'
        '- User: "Output json."\n'
        '  JSON intent = "SET"; JSON slots.output = {"format":"json"}; all other slot groups stay null\n'
        '- User: "change material to G4_Al"\n'
        '  JSON intent = "MODIFY"; JSON slots.materials = {"primary":"G4_Al"}; all unrelated slot groups stay null\n'
        '- User: "why did you choose QBBC?"\n'
        '  JSON intent = "QUESTION"; JSON slots stay null unless the user also changes a value\n'
        '- User: "\\u786e\\u8ba4"\n'
        '  JSON intent = "CONFIRM"; JSON slots stay null\n'
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )


def build_strict_slot_prompt(user_text: str, context_summary: str) -> str:
    if STRICT_SLOT_PROMPT_PROFILE == "strict_slot_v1":
        return _build_strict_slot_prompt_v1(user_text, context_summary)
    return _build_strict_slot_prompt_v2(user_text, context_summary)


def build_strict_semantic_prompt(user_text: str, context_summary: str) -> str:
    return (
        "Convert user request into a strict semantic frame for a Geant4 config session.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "intent": "SET|MODIFY|REMOVE|CONFIRM|QUESTION|OTHER",\n'
        '  "target_paths": ["json.path", "..."],\n'
        '  "normalized_text": "semicolon-separated key:value clauses",\n'
        '  "structure_hint": "ring|grid|nest|stack|shell|single_box|single_tubs|single_sphere|single_cons|single_trd|single_polycone|single_cuttubs|boolean|unknown",\n'
        '  "confidence": 0.0,\n'
        '  "updates": [\n'
        '    {"path":"source.type","op":"set","value":"point"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- path must start with geometry., materials., source., physics., or output.\n"
        "- include updates only when user provided or explicitly changed information.\n"
        "- do not invent values.\n"
        "- keep normalized_text concise.\n"
        f"Context: {context_summary}\n"
        f"User: {user_text}\n"
        "JSON:"
    )
