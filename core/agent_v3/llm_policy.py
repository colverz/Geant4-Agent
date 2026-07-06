from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import V3TurnInput


LLM_POLICY_SCHEMA_VERSION = "geant4_agent_v3_llm_policy.v1"


@dataclass(slots=True)
class V3LlmPolicy:
    understanding_enabled: bool = True
    planning_enabled: bool = True
    design_enabled: bool = False
    result_enabled: bool = False
    naturalize_enabled: bool = False
    schema_version: str = LLM_POLICY_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "V3LlmPolicy":
        raw = payload.get("llm_policy") if isinstance(payload.get("llm_policy"), dict) else {}
        return cls(
            understanding_enabled=bool(raw.get("understanding_enabled", payload.get("llm_understanding_enabled", True))),
            planning_enabled=bool(raw.get("planning_enabled", payload.get("llm_planning_enabled", True))),
            design_enabled=bool(raw.get("design_enabled", payload.get("llm_design_enabled", False))),
            result_enabled=bool(raw.get("result_enabled", payload.get("llm_result_enabled", False))),
            naturalize_enabled=bool(raw.get("naturalize_enabled", payload.get("llm_naturalize_enabled", False))),
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "V3LlmPolicy":
        raw = metadata.get("llm_policy") if isinstance(metadata.get("llm_policy"), dict) else {}
        return cls(
            understanding_enabled=bool(raw.get("understanding_enabled", metadata.get("llm_understanding_enabled", True))),
            planning_enabled=bool(raw.get("planning_enabled", metadata.get("llm_planning_enabled", True))),
            design_enabled=bool(raw.get("design_enabled", metadata.get("llm_design_enabled", False))),
            result_enabled=bool(raw.get("result_enabled", metadata.get("llm_result_enabled", False))),
            naturalize_enabled=bool(raw.get("naturalize_enabled", metadata.get("llm_naturalize_enabled", False))),
            schema_version=str(raw.get("schema_version") or LLM_POLICY_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "understanding_enabled": self.understanding_enabled,
            "planning_enabled": self.planning_enabled,
            "design_enabled": self.design_enabled,
            "result_enabled": self.result_enabled,
            "naturalize_enabled": self.naturalize_enabled,
        }


def set_llm_policy(metadata: dict[str, Any], policy: V3LlmPolicy) -> None:
    metadata["llm_policy"] = policy.to_dict()
    metadata["llm_understanding_enabled"] = policy.understanding_enabled
    metadata["llm_planning_enabled"] = policy.planning_enabled
    metadata["llm_design_enabled"] = policy.design_enabled
    metadata["llm_result_enabled"] = policy.result_enabled
    metadata["llm_naturalize_enabled"] = policy.naturalize_enabled


def llm_policy_from_turn(turn: V3TurnInput) -> V3LlmPolicy:
    policy = V3LlmPolicy.from_metadata(turn.metadata)
    set_llm_policy(turn.metadata, policy)
    return policy


__all__ = ["LLM_POLICY_SCHEMA_VERSION", "V3LlmPolicy", "llm_policy_from_turn", "set_llm_policy"]
