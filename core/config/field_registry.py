from __future__ import annotations


_FRIENDLY = {
    "en": {
        "geometry.structure": "geometry structure",
        "materials.volume_material_map": "volume-to-material binding",
        "source.type": "source type",
        "source.particle": "particle type",
        "source.energy": "source energy",
        "source.position": "source position",
        "source.direction": "source direction",
        "physics.physics_list": "physics list",
        "output.format": "output format",
        "output.path": "output path",
    },
    "zh": {
        "geometry.structure": "几何结构类型",
        "materials.volume_material_map": "体积与材料绑定",
        "source.type": "源类型",
        "source.particle": "粒子类型",
        "source.energy": "源能量",
        "source.position": "源位置",
        "source.direction": "源方向",
        "physics.physics_list": "物理列表",
        "output.format": "输出格式",
        "output.path": "输出路径",
    },
}

_QUESTION = {
    "en": {
        "physics.physics_list": "Please provide the physics list name (for example FTFP_BERT or QBBC).",
        "source.energy": "Please provide the source energy in MeV.",
        "source.position": "Please provide the source position vector as (x, y, z).",
        "source.direction": "Please provide the source direction vector as (dx, dy, dz).",
        "materials.volume_material_map": "Please confirm the volume-to-material binding (volume -> material).",
        "geometry.structure": "Please confirm the geometry structure type (for example single_box).",
    },
    "zh": {
        "physics.physics_list": "请提供物理列表名称（例如 FTFP_BERT 或 QBBC）。",
        "source.energy": "请提供源能量（单位：MeV）。",
        "source.position": "请提供源位置向量（x, y, z）。",
        "source.direction": "请提供源方向向量（dx, dy, dz）。",
        "materials.volume_material_map": "请确认体积到材料的绑定关系（volume -> material）。",
        "geometry.structure": "请确认几何结构类型（例如 single_box）。",
    },
}

_GEOMETRY_PARAM = {
    "en": "geometry parameter",
    "zh": "几何参数",
}

_CANONICAL_ALIASES = {
    "source.energy_MeV": "source.energy",
    "physics_list.name": "physics.physics_list",
}


def normalize_lang(lang: str) -> str:
    return "zh" if lang == "zh" else "en"


def canonical_field_path(path: str) -> str:
    return _CANONICAL_ALIASES.get(path, path)


def friendly_label(path: str, lang: str) -> str:
    path = canonical_field_path(path)
    lang_key = normalize_lang(lang)
    if path.startswith("geometry.params."):
        key = path.split(".", 2)[-1]
        return f"{_GEOMETRY_PARAM[lang_key]} {key}"
    return _FRIENDLY[lang_key].get(path, path)


def friendly_labels(paths: list[str], lang: str) -> list[str]:
    return [friendly_label(path, lang) for path in paths]


def missing_field_question(path: str, lang: str) -> str:
    path = canonical_field_path(path)
    lang_key = normalize_lang(lang)
    if not path:
        return "配置已完成。" if lang_key == "zh" else "Configuration complete."
    if path in _QUESTION[lang_key]:
        return _QUESTION[lang_key][path]
    if lang_key == "zh":
        return f"请补充字段：{path}"
    return f"Please provide: {path}"
