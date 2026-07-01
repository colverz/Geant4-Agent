from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import V3TurnInput


RUNTIME_POLICY_SCHEMA_VERSION = "geant4_agent_v3_runtime_policy.v1"


@dataclass(slots=True)
class V3RuntimePolicy:
    allow_in_memory: bool = False
    env: dict[str, Any] = field(default_factory=dict)
    backend_preference: str = "local_process"
    source: str = "request"
    schema_version: str = RUNTIME_POLICY_SCHEMA_VERSION

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], *, source: str = "metadata") -> "V3RuntimePolicy":
        raw = metadata.get("runtime_policy") if isinstance(metadata.get("runtime_policy"), dict) else {}
        raw_env = raw.get("env") if isinstance(raw.get("env"), dict) else metadata.get("runtime_env")
        env = dict(raw_env) if isinstance(raw_env, dict) else {}
        allow_in_memory = raw.get("allow_in_memory") if "allow_in_memory" in raw else metadata.get("allow_in_memory")
        backend_preference = str(raw.get("backend_preference") or metadata.get("runtime_backend") or "local_process")
        return cls(
            allow_in_memory=bool(allow_in_memory),
            env=env,
            backend_preference=backend_preference or "local_process",
            source=str(raw.get("source") or source),
            schema_version=str(raw.get("schema_version") or RUNTIME_POLICY_SCHEMA_VERSION),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, runtime_env: dict[str, Any] | None = None) -> "V3RuntimePolicy":
        raw = payload.get("runtime_policy") if isinstance(payload.get("runtime_policy"), dict) else {}
        env = raw.get("env") if isinstance(raw.get("env"), dict) else runtime_env
        allow_in_memory = raw.get("allow_in_memory") if "allow_in_memory" in raw else payload.get("allow_in_memory")
        backend_preference = str(raw.get("backend_preference") or payload.get("runtime_backend") or "local_process")
        return cls(
            allow_in_memory=bool(allow_in_memory),
            env=dict(env) if isinstance(env, dict) else {},
            backend_preference=backend_preference or "local_process",
            source=str(raw.get("source") or "request"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allow_in_memory": self.allow_in_memory,
            "env": dict(self.env),
            "backend_preference": self.backend_preference,
            "source": self.source,
        }

    def tool_arguments(self) -> dict[str, Any]:
        return {
            "allow_in_memory": self.allow_in_memory,
            "env": dict(self.env),
        }


def set_runtime_policy(metadata: dict[str, Any], policy: V3RuntimePolicy) -> None:
    metadata["runtime_policy"] = policy.to_dict()
    metadata["allow_in_memory"] = policy.allow_in_memory
    metadata["runtime_env"] = dict(policy.env)
    metadata["runtime_backend"] = policy.backend_preference


def runtime_policy_from_turn(turn: V3TurnInput) -> V3RuntimePolicy:
    policy = V3RuntimePolicy.from_metadata(turn.metadata, source="turn")
    set_runtime_policy(turn.metadata, policy)
    return policy


def runtime_tool_arguments(turn: V3TurnInput) -> dict[str, Any]:
    return runtime_policy_from_turn(turn).tool_arguments()
