from __future__ import annotations

import json
import logging
import re
from typing import Any

from nlu.llm_support.ollama_client import chat


logger = logging.getLogger(__name__)

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", flags=re.IGNORECASE)


def _clean_chat_text(raw: Any) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.IGNORECASE | re.DOTALL).strip()
    return re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text, flags=re.DOTALL).strip()


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "not provided"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != 0.0 and abs(numeric) < 0.001:
            return f"{numeric:.3e}"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6g}"
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _completion_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not provided"
    return f"{float(value) * 100:.1f}%"


def build_runtime_result_message(report: dict[str, Any] | None, *, lang: str = "zh") -> str:
    report = _as_dict(report)
    if not report:
        return "当前还没有可解释的运行结果。" if lang == "zh" else "No runtime result is available yet."

    config = _as_dict(report.get("configuration"))
    metrics = _as_dict(report.get("key_metrics"))
    ok = bool(report.get("ok"))
    events_requested = report.get("events_requested")
    events_completed = report.get("events_completed")
    completion_fraction = report.get("completion_fraction")
    run_summary_path = report.get("run_summary_path") or "not provided"

    if lang == "zh":
        status = "模拟运行已完成。" if ok else "模拟运行没有成功完成。"
        return "\n".join(
            [
                status,
                f"事件完成情况：{_fmt(events_completed)} / {_fmt(events_requested)}，完成率 {_completion_percent(completion_fraction)}。",
                (
                    "配置摘要："
                    f"几何={_fmt(config.get('geometry_structure'))}，"
                    f"材料={_fmt(config.get('material'))}，"
                    f"粒子={_fmt(config.get('particle'))}，"
                    f"物理列表={_fmt(config.get('physics_list'))}。"
                ),
                (
                    "关键计分："
                    f"target_edep_total_mev={_fmt(metrics.get('target_edep_total_mev'))}，"
                    f"target_hit_events={_fmt(metrics.get('target_hit_events'))}，"
                    f"detector_crossing_count={_fmt(metrics.get('detector_crossing_count'))}，"
                    f"plane_crossing_count={_fmt(metrics.get('plane_crossing_count'))}。"
                ),
                f"结果文件：{run_summary_path}",
                "下一步可以基于这些已观测指标决定是否增加事件数、调整源或检查探测器计分设置。",
            ]
        )

    status = "The simulation run completed." if ok else "The simulation run did not complete successfully."
    return "\n".join(
        [
            status,
            f"Events: {_fmt(events_completed)} / {_fmt(events_requested)}, completion {_completion_percent(completion_fraction)}.",
            (
                "Configuration: "
                f"geometry={_fmt(config.get('geometry_structure'))}, "
                f"material={_fmt(config.get('material'))}, "
                f"particle={_fmt(config.get('particle'))}, "
                f"physics={_fmt(config.get('physics_list'))}."
            ),
            (
                "Key scoring: "
                f"target_edep_total_mev={_fmt(metrics.get('target_edep_total_mev'))}, "
                f"target_hit_events={_fmt(metrics.get('target_hit_events'))}, "
                f"detector_crossing_count={_fmt(metrics.get('detector_crossing_count'))}, "
                f"plane_crossing_count={_fmt(metrics.get('plane_crossing_count'))}."
            ),
            f"Result file: {run_summary_path}",
            "Next, use these observed metrics to decide whether to increase events, adjust the source, or inspect detector scoring.",
        ]
    )


def _looks_language_mismatched(text: str, lang: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    has_cjk = bool(_CJK_PATTERN.search(compact))
    has_ascii_words = bool(re.search(r"[A-Za-z]{3,}", compact))
    if lang == "zh":
        return has_ascii_words and not has_cjk
    if lang == "en":
        return has_cjk and not has_ascii_words
    return False


def _numeric_tokens(text: str) -> set[str]:
    return {token.lower() for token in _NUMBER_PATTERN.findall(str(text or ""))}


def _invalid_llm_result_message(text: str, *, base_message: str, lang: str) -> bool:
    if not text.strip():
        return True
    if _looks_language_mismatched(text, lang):
        return True
    return bool(_numeric_tokens(text) - _numeric_tokens(base_message))


def naturalize_runtime_result_message(
    report: dict[str, Any] | None,
    *,
    lang: str = "zh",
    use_llm: bool = False,
    ollama_config: str = "nlu/llm_support/configs/ollama_config.json",
    temperature: float = 0.2,
) -> dict[str, Any]:
    base_message = build_runtime_result_message(report, lang=lang)
    if not use_llm:
        return {"message": base_message, "source": "deterministic", "fallback_reason": None}

    if lang == "zh":
        rules = (
            "你是 Geant4 模拟结果解释层。请把 base_message 改写得更自然，但必须严格受 report 约束。"
            "不得新增任何数值、物理结论、过程解释或 report 中不存在的事实。"
            "如果字段缺失，必须保留缺失含义。只输出最终中文回复。"
        )
    else:
        rules = (
            "You are the Geant4 simulation-result explanation layer. Rewrite base_message naturally, "
            "but stay strictly grounded in report. Do not add any new numbers, physics conclusions, "
            "process explanations, or facts not present in the report. If a field is missing, preserve that meaning. "
            "Return only the final English answer."
        )

    prompt = (
        f"{rules}\n\n"
        f"Input JSON:\n{json.dumps({'lang': lang, 'report': report or {}, 'base_message': base_message}, ensure_ascii=False)}\n\n"
        "Rewrite now."
    )
    try:
        resp = chat(prompt, config_path=ollama_config, temperature=temperature)
        text = _clean_chat_text(resp.get("response", ""))
        if not _invalid_llm_result_message(text, base_message=base_message, lang=lang):
            return {"message": text, "source": "llm", "fallback_reason": None}
        logger.warning("LLM runtime-result explanation was rejected by grounded validation; using deterministic fallback.")
        return {"message": base_message, "source": "deterministic", "fallback_reason": "invalid_llm_output"}
    except Exception:
        logger.warning("LLM runtime-result explanation failed; using deterministic fallback.", exc_info=True)
        return {"message": base_message, "source": "deterministic", "fallback_reason": "llm_failed"}
