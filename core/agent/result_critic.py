from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CRITIC_REPORT_SCHEMA_VERSION = "critic_report.v1"


METRIC_ALIASES = {
    "target_edep": ("target_edep_total_mev", "target_edep_mean_mev_per_event"),
    "detector_crossing_count": ("detector_crossing_count",),
    "detector_edep": ("detector_edep_total_mev", "detector_edep_mean_mev_per_event"),
    "plane_crossing_count": ("plane_crossing_count",),
    "region_contrast": ("contrast_ratio", "region_a_count", "region_b_count"),
    "depth_bins": ("depth_bins", "peak_depth_mm", "peak_bin"),
    "transmission_factor": ("transmission_ratio", "transmission_factor", "plane_crossing_count", "detector_crossing_count"),
}


@dataclass(frozen=True)
class CriticReport:
    schema_version: str = CRITIC_REPORT_SCHEMA_VERSION
    satisfied: bool = False
    answerable: bool = False
    missing_metrics: list[str] = field(default_factory=list)
    runtime_limitations: list[str] = field(default_factory=list)
    recommended_next_action: str = "revise_plan"
    user_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            metrics[str(key)] = item
            metrics.update(_flatten_metrics(item, name))
    return metrics


def _metric_available(metric_name: str, metrics: dict[str, Any]) -> bool:
    aliases = METRIC_ALIASES.get(metric_name, (metric_name,))
    for alias in aliases:
        if alias in metrics and metrics[alias] not in (None, "", []):
            return True
    return False


def build_critic_report(
    runtime_smoke_report: dict[str, Any] | None,
    agent_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    report = runtime_smoke_report if isinstance(runtime_smoke_report, dict) else {}
    plan = agent_plan if isinstance(agent_plan, dict) else {}
    key_metrics = report.get("key_metrics") if isinstance(report.get("key_metrics"), dict) else {}
    result_summary = report.get("result_summary") if isinstance(report.get("result_summary"), dict) else {}
    metrics = _flatten_metrics(key_metrics)
    metrics.update(_flatten_metrics(result_summary.get("scoring") if isinstance(result_summary, dict) else {}))
    success_metrics = [str(item) for item in plan.get("success_metrics") or [] if str(item)]
    if not success_metrics:
        success_metrics = ["target_edep"]
    missing = [metric for metric in success_metrics if not _metric_available(metric, metrics)]
    limitations = []
    for capability in plan.get("required_runtime_capabilities") or []:
        text = str(capability)
        if text.startswith("unsupported:"):
            limitations.append(text.removeprefix("unsupported:"))
    answerable = bool(report.get("ok", False)) and not missing
    satisfied = answerable and not limitations
    if satisfied:
        next_action = "explain_result"
        message = "Runtime produced the metrics required by the selected agent plan."
    elif limitations:
        next_action = "revise_plan"
        message = "Runtime completed, but the plan still contains unsupported capabilities that require revision."
    else:
        next_action = "request_additional_scoring"
        message = "Runtime completed, but required metrics are missing from the structured result."
    return CriticReport(
        satisfied=satisfied,
        answerable=answerable,
        missing_metrics=missing,
        runtime_limitations=limitations,
        recommended_next_action=next_action,
        user_message=message,
    ).to_dict()


__all__ = ["CRITIC_REPORT_SCHEMA_VERSION", "CriticReport", "build_critic_report"]
