from __future__ import annotations

from typing import Dict, List

from nlu.bert_lab.ollama_client import chat


def _friendly_missing(missing: List[str], lang: str) -> List[str]:
    if lang == "en":
        mapping = {
            "geometry.structure": "geometry structure (box / ring / grid / stack / shell)",
            "materials.selected_materials": "materials (e.g., G4_WATER / G4_Al / G4_Si)",
            "source.particle": "particle type (gamma / e- / proton)",
            "source.type": "source type (point / beam / isotropic)",
            "physics.physics_list": "physics list (e.g., FTFP_BERT)",
            "output.format": "output format (root / csv / json)",
        }
    else:
        mapping = {
            "geometry.structure": "几何结构（如 box / ring / grid / stack / shell）",
            "materials.selected_materials": "材料（如 G4_WATER / G4_Al / G4_Si）",
            "source.particle": "粒子类型（如 gamma / e- / proton）",
            "source.type": "源类型（如 point / beam / isotropic）",
            "physics.physics_list": "物理过程列表（如 FTFP_BERT）",
            "output.format": "输出格式（如 root / csv / json）",
        }
    friendly = []
    for item in missing:
        if item.startswith("geometry.params."):
            key = item.split("geometry.params.", 1)[1]
            friendly.append(f"geometry parameter {key}" if lang == "en" else f"几何参数 {key}")
        else:
            friendly.append(mapping.get(item, item))
    return friendly


def ask_missing(
    missing: List[str],
    lang: str,
    ollama_config: str = "nlu/bert_lab/configs/ollama_config.json",
    temperature: float = 0.2,
) -> str:
    if not missing:
        return ""
    friendly = _friendly_missing(missing, lang)
    if lang == "en":
        prompt = (
            "You are a research assistant. Ask one concise and friendly clarification question in English. "
            "Do not expose internal field names.\n"
            f"Missing items: {friendly}\n"
            "Question:"
        )
    else:
        prompt = (
            "你是科研软件助手。请用自然、友好的中文，合并成一句简短追问，向用户补齐缺失信息。"
            "不要暴露内部字段名。\n"
            f"缺失内容: {friendly}\n"
            "问题:"
        )
    try:
        resp = chat(prompt, config_path=ollama_config, temperature=temperature)
        text = resp.get("response", "").strip()
        if text:
            return text
    except Exception:
        pass
    return f"Please provide: {', '.join(friendly)}" if lang == "en" else f"请补充：{', '.join(friendly)}"
