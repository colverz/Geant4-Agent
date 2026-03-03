from __future__ import annotations


STRICT_SLOT_PROMPT_PROFILE = "strict_slot_v1"
STRICT_SEMANTIC_PROMPT_PROFILE = "strict_semantic_v1"


def build_strict_slot_prompt(user_text: str, context_summary: str) -> str:
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
        '    "output": {"format": "root|json|csv|null", "path": null}\n'
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
