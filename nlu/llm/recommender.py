from __future__ import annotations

import re

from core.orchestrator.types import CandidateUpdate, Intent, Producer, UpdateOp
from nlu.llm_support.ollama_client import chat, extract_json


def _clean_text(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    return text


def _pick_known(name: str, allowed: list[str]) -> str:
    val = str(name or "").strip()
    for item in allowed:
        if item.lower() == val.lower():
            return item
    return ""


def recommend_physics_list(
    user_text: str,
    normalized_text: str,
    context_summary: str,
    allowed_lists: list[str],
    *,
    turn_id: int,
    config_path: str,
) -> CandidateUpdate | None:
    if not allowed_lists:
        return None
    merged_text = f"{user_text} ; {normalized_text}"
    low = merged_text.lower()
    has_physics_domain = any(
        k in low
        for k in [
            "physics",
            "physics list",
            "鐗╃悊",
            "鍒楄〃",
            "geant4",
            "qgsp",
            "ftfp",
            "qbbc",
            "shielding",
        ]
    )
    has_decision_intent = any(
        k in low
        for k in [
            "recommend",
            "choose",
            "select",
            "pick",
            "best",
            "most suitable",
            "鎺ㄨ崘",
            "閫夋嫨",
            "鏈€鍚堥€?,
            "缁欏嚭鍚嶇О",
            "澶囬€?,
        ]
    )
    trigger = has_physics_domain and has_decision_intent
    if not trigger:
        return None

    prompt = (
        "You are a Geant4 physics-list recommender.\n"
        "Return JSON only with keys: physics_list, backup_physics_list, reasons, covered_processes, confidence.\n"
        "- physics_list and backup_physics_list must be selected from allowed list.\n"
        "- Keep reasons concise.\n"
        f"Allowed: {', '.join(allowed_lists)}\n"
        f"Context: {context_summary}\n"
        f"Request: {merged_text}\n"
        "JSON:"
    )
    parsed: dict = {}
    try:
        resp = chat(prompt, config_path=config_path, temperature=0.0)
        maybe = extract_json(_clean_text(str(resp.get("response", "")))) or {}
        if isinstance(maybe, dict):
            parsed = maybe
    except Exception:
        parsed = {}

    main = _pick_known(parsed.get("physics_list", ""), allowed_lists)
    backup = _pick_known(parsed.get("backup_physics_list", ""), allowed_lists)
    low = merged_text.lower()
    em_like = any(k in low for k in ["gamma", "photon", "electromagnetic", "搴锋櫘椤?, "鍏夌數", "瀵逛骇鐢?])
    no_hadron = any(k in low for k in ["涓嶆秹鍙婂己瀛?, "no hadron", "without hadron", "no hadrons"])
    if em_like and no_hadron and "FTFP_BERT" in allowed_lists:
        main = "FTFP_BERT"
        if "QBBC" in allowed_lists:
            backup = "QBBC"
    if not main:
        if em_like and no_hadron and "FTFP_BERT" in allowed_lists:
            main = "FTFP_BERT"
            if "QBBC" in allowed_lists:
                backup = "QBBC"
        elif "FTFP_BERT" in allowed_lists:
            main = "FTFP_BERT"
        elif allowed_lists:
            main = allowed_lists[0]
    if not main:
        return None
    reasons = parsed.get("reasons", [])
    covered = parsed.get("covered_processes", [])
    conf = parsed.get("confidence", 0.7)
    if not isinstance(conf, (int, float)):
        conf = 0.7
    if not isinstance(reasons, list):
        reasons = []
    if not isinstance(covered, list):
        covered = []
    if not reasons:
        reasons = ["Selected by rule-backed fallback recommender for current request semantics."]
    if not covered:
        covered = ["photoelectric effect", "Compton scattering", "pair production"]

    updates = [
        UpdateOp(
            path="physics.physics_list",
            op="set",
            value=main,
            producer=Producer.LLM_RECOMMENDER,
            confidence=float(conf),
            turn_id=turn_id,
        ),
        UpdateOp(
            path="physics.backup_physics_list",
            op="set",
            value=backup or None,
            producer=Producer.LLM_RECOMMENDER,
            confidence=float(conf),
            turn_id=turn_id,
        ),
        UpdateOp(
            path="physics.selection_reasons",
            op="set",
            value=[str(x).strip() for x in reasons if str(x).strip()][:3],
            producer=Producer.LLM_RECOMMENDER,
            confidence=float(conf),
            turn_id=turn_id,
        ),
        UpdateOp(
            path="physics.covered_processes",
            op="set",
            value=[str(x).strip() for x in covered if str(x).strip()][:8],
            producer=Producer.LLM_RECOMMENDER,
            confidence=float(conf),
            turn_id=turn_id,
        ),
        UpdateOp(
            path="physics.selection_source",
            op="set",
            value="llm_recommender",
            producer=Producer.LLM_RECOMMENDER,
            confidence=float(conf),
            turn_id=turn_id,
        ),
    ]

    return CandidateUpdate(
        producer=Producer.LLM_RECOMMENDER,
        intent=Intent.SET,
        target_paths=["physics.physics_list"],
        updates=updates,
        confidence=float(conf),
        rationale="llm_recommender",
    )

