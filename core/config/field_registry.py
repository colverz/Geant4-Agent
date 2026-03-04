from __future__ import annotations

from core.config.path_registry import canonical_field_path
from core.config.prompt_registry import completion_message, single_field_request


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
        "geometry.structure": "\u51e0\u4f55\u7ed3\u6784\u7c7b\u578b",
        "materials.volume_material_map": "\u4f53\u79ef\u4e0e\u6750\u6599\u7ed1\u5b9a",
        "source.type": "\u6e90\u7c7b\u578b",
        "source.particle": "\u7c92\u5b50\u7c7b\u578b",
        "source.energy": "\u6e90\u80fd\u91cf",
        "source.position": "\u6e90\u4f4d\u7f6e",
        "source.direction": "\u6e90\u65b9\u5411",
        "physics.physics_list": "\u7269\u7406\u5217\u8868",
        "output.format": "\u8f93\u51fa\u683c\u5f0f",
        "output.path": "\u8f93\u51fa\u8def\u5f84",
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
        "physics.physics_list": "\u8bf7\u63d0\u4f9b\u7269\u7406\u5217\u8868\u540d\u79f0\uff08\u4f8b\u5982 FTFP_BERT \u6216 QBBC\uff09\u3002",
        "source.energy": "\u8bf7\u63d0\u4f9b\u6e90\u80fd\u91cf\uff08\u5355\u4f4d\uff1aMeV\uff09\u3002",
        "source.position": "\u8bf7\u63d0\u4f9b\u6e90\u4f4d\u7f6e\u5411\u91cf\uff08x, y, z\uff09\u3002",
        "source.direction": "\u8bf7\u63d0\u4f9b\u6e90\u65b9\u5411\u5411\u91cf\uff08dx, dy, dz\uff09\u3002",
        "materials.volume_material_map": "\u8bf7\u786e\u8ba4\u4f53\u79ef\u5230\u6750\u6599\u7684\u7ed1\u5b9a\u5173\u7cfb\uff08volume -> material\uff09\u3002",
        "geometry.structure": "\u8bf7\u786e\u8ba4\u51e0\u4f55\u7ed3\u6784\u7c7b\u578b\uff08\u4f8b\u5982 single_box\uff09\u3002",
    },
}

_GEOMETRY_PARAM = {
    "en": "geometry parameter",
    "zh": "\u51e0\u4f55\u53c2\u6570",
}

_CLARIFICATION = {
    "en": {
        "geometry.structure": "geometry structure (for example single_box or single_tubs)",
        "materials.selected_materials": "materials (for example G4_WATER / G4_Al / G4_Si)",
        "materials.volume_material_map": "volume-to-material mapping",
        "source.particle": "particle type (gamma / e- / proton)",
        "source.type": "source type (point / beam / isotropic)",
        "source.energy": "source energy (MeV)",
        "source.position": "source position (x, y, z)",
        "source.direction": "source direction (dx, dy, dz)",
        "physics.physics_list": "physics list (for example FTFP_BERT)",
        "output.format": "output format (root / csv / hdf5 / xml / json)",
        "output.path": "output path",
    },
    "zh": {
        "geometry.structure": "\u51e0\u4f55\u7ed3\u6784\u7c7b\u578b\uff08\u4f8b\u5982 single_box \u6216 single_tubs\uff09",
        "materials.selected_materials": "\u6750\u6599\uff08\u4f8b\u5982 G4_WATER / G4_Al / G4_Si\uff09",
        "materials.volume_material_map": "\u4f53\u79ef\u5230\u6750\u6599\u7684\u6620\u5c04\u5173\u7cfb",
        "source.particle": "\u7c92\u5b50\u7c7b\u578b\uff08gamma / e- / proton\uff09",
        "source.type": "\u6e90\u7c7b\u578b\uff08point / beam / isotropic\uff09",
        "source.energy": "\u6e90\u80fd\u91cf\uff08MeV\uff09",
        "source.position": "\u6e90\u4f4d\u7f6e\uff08x, y, z\uff09",
        "source.direction": "\u6e90\u65b9\u5411\uff08dx, dy, dz\uff09",
        "physics.physics_list": "\u7269\u7406\u5217\u8868\uff08\u4f8b\u5982 FTFP_BERT\uff09",
        "output.format": "\u8f93\u51fa\u683c\u5f0f\uff08root / csv / hdf5 / xml / json\uff09",
        "output.path": "\u8f93\u51fa\u8def\u5f84",
    },
}


def normalize_lang(lang: str) -> str:
    return "zh" if lang == "zh" else "en"


def friendly_label(path: str, lang: str) -> str:
    path = canonical_field_path(path)
    lang_key = normalize_lang(lang)
    if path.startswith("geometry.params."):
        key = path.split(".", 2)[-1]
        return f"{_GEOMETRY_PARAM[lang_key]} {key}"
    return _FRIENDLY[lang_key].get(path, path)


def friendly_labels(paths: list[str], lang: str) -> list[str]:
    return [friendly_label(path, lang) for path in paths]


def clarification_item(path: str, lang: str) -> str:
    path = canonical_field_path(path)
    lang_key = normalize_lang(lang)
    if path.startswith("geometry.params."):
        key = path.split(".", 2)[-1]
        return f"{_GEOMETRY_PARAM[lang_key]} {key}"
    return _CLARIFICATION[lang_key].get(path, friendly_label(path, lang_key))


def clarification_items(paths: list[str], lang: str) -> list[str]:
    return [clarification_item(path, lang) for path in paths]


def missing_field_question(path: str, lang: str) -> str:
    path = canonical_field_path(path)
    lang_key = normalize_lang(lang)
    if not path:
        return completion_message(lang_key)
    if path in _QUESTION[lang_key]:
        return _QUESTION[lang_key][path]
    return single_field_request(path, lang_key)
