from __future__ import annotations

from typing import Any

from planner.runtime_result import naturalize_runtime_result_message, naturalize_runtime_result_question_answer


def looks_like_result_question(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    # If the user is clearly making a design/simulation request, don't treat as result question
    design_intent_tokens = (
        "设计", "模拟", "方案", "配置", "创建", "搭建", "建立一个",
        "design", "simulate", "create a", "build a", "setup", "set up",
        "我想", "我要", "帮我", "请", "i want", "i need", "can you",
    )
    if any(token in lowered for token in design_intent_tokens):
        return False
    question_tokens = (
        "解释", "分析", "建议", "什么意思", "什么含义", "怎么样", "怎么看", "如何", "说明一下",
        "explain", "what does", "what is", "how many", "how much",
        "why is", "tell me about", "interpret", "analyze", "recommend",
    )
    metric_tokens = (
        "detector crossing", "detector_crossing", "target_edep",
        "plane_crossing", "edep", "dose", "did it hit",
        "结果", "命中", "剂量", "能量沉积", "计分",
        "result", "results", "what happened", "how did it go", "summary",
    )
    has_question = any(token in lowered for token in question_tokens)
    has_metric = any(token in lowered for token in metric_tokens)
    return has_question or (has_metric and any(c in text for c in ("?", "？", "吗", "多少")))


def build_v3_runtime_result_answer(
    question: str,
    runtime_observation: dict[str, Any] | None,
    *,
    locale: str = "zh-CN",
    use_llm: bool = False,
    llm_config_path: str = "",
) -> dict[str, Any]:
    lang = "zh" if str(locale).lower().startswith("zh") else "en"
    report = _runtime_observation_to_report(runtime_observation)
    if looks_like_result_question(question):
        result = naturalize_runtime_result_question_answer(
            question,
            report,
            lang=lang,
            use_llm=use_llm and bool(llm_config_path),
            ollama_config=llm_config_path or "nlu/llm_support/configs/ollama_config.json",
        )
        mode = "qa"
    else:
        result = naturalize_runtime_result_message(
            report,
            lang=lang,
            use_llm=use_llm and bool(llm_config_path),
            ollama_config=llm_config_path or "nlu/llm_support/configs/ollama_config.json",
        )
        mode = "summary"
    return {
        "message": result.get("message", ""),
        "mode": mode,
        "report": report,
        "source": result.get("source", "deterministic"),
        "fallback_reason": result.get("fallback_reason"),
        "prompt_profile_id": result.get("prompt_profile_id", ""),
        "prompt_validation": result.get("prompt_validation", {}),
    }


def _runtime_observation_to_report(runtime_observation: dict[str, Any] | None) -> dict[str, Any]:
    data = runtime_observation if isinstance(runtime_observation, dict) else {}
    result_summary = data.get("result_summary") if isinstance(data.get("result_summary"), dict) else {}
    if not result_summary:
        return {}
    if "key_metrics" in result_summary:
        return result_summary

    run = _dict(result_summary.get("run"))
    configuration = _configuration_summary(result_summary, data)
    scoring = _dict(result_summary.get("scoring"))
    target = _dict(scoring.get("target"))
    detector = _dict(scoring.get("detector_crossing"))
    plane = _dict(scoring.get("plane_crossing"))
    summary_payload = _dict(_dict(data.get("summary")).get("payload"))

    events_requested = _first_present(run.get("events_requested"), data.get("events"))
    events_completed = _first_present(run.get("events_completed"), run.get("events_requested"), data.get("events"))
    completion_fraction = _first_present(run.get("completion_fraction"), _completion_fraction(events_completed, events_requested))

    return {
        "ok": run.get("ok", True),
        "events_requested": events_requested,
        "events_completed": events_completed,
        "completion_fraction": completion_fraction,
        "configuration": configuration,
        "key_metrics": {
            "target_edep_total_mev": target.get("target_edep_total_mev"),
            "target_hit_events": target.get("target_hit_events"),
            "detector_crossing_count": detector.get("detector_crossing_count"),
            "plane_crossing_count": plane.get("plane_crossing_count"),
        },
        "artifact_dir": summary_payload.get("artifact_dir"),
        "run_summary_path": summary_payload.get("run_summary_path"),
        "result_summary": result_summary,
    }


def _configuration_summary(result_summary: dict[str, Any], runtime_data: dict[str, Any]) -> dict[str, Any]:
    config = _dict(result_summary.get("configuration"))
    manifest = _dict(result_summary.get("run_manifest"))
    source = _dict(result_summary.get("source"))
    geometry = _dict(result_summary.get("geometry"))
    runtime_payload = _dict(runtime_data.get("runtime_payload"))
    payload = _dict(_dict(runtime_data.get("run")).get("payload"))
    source_payload = _dict(runtime_payload.get("source"))
    geometry_payload = _dict(runtime_payload.get("geometry"))
    if config:
        out = dict(config)
        out["source_energy_mev"] = _first_present(
            out.get("source_energy_mev"),
            source_payload.get("energy_mev"),
            runtime_payload.get("energy"),
            payload.get("energy"),
        )
        return out
    return {
        "geometry_structure": _first_present(geometry.get("structure"), geometry_payload.get("structure"), manifest.get("geometry_root_volume")),
        "material": _first_present(geometry.get("material"), geometry_payload.get("material"), runtime_payload.get("material"), payload.get("material")),
        "source_type": _first_present(source.get("type"), source_payload.get("type"), runtime_payload.get("source_type")),
        "particle": _first_present(source.get("particle"), source_payload.get("particle"), runtime_payload.get("particle"), payload.get("particle")),
        "source_energy_mev": _first_present(source_payload.get("energy_mev"), runtime_payload.get("energy"), payload.get("energy")),
        "physics_list": _first_present(result_summary.get("physics_list"), runtime_payload.get("physics_list"), payload.get("physics_list")),
    }


def _completion_fraction(completed: Any, requested: Any) -> float | None:
    try:
        requested_value = float(requested)
        if requested_value <= 0:
            return None
        return float(completed) / requested_value
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["build_v3_runtime_result_answer", "looks_like_result_question"]
