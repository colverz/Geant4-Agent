from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Geant4RuntimePhase(str, Enum):
    DETACHED = "detached"
    IDLE = "idle"
    CONFIGURED = "configured"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FAILED = "failed"


class RuntimeActionStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionSafetyClass(str, Enum):
    READ_ONLY = "read_only"
    CONFIG_MUTATION = "config_mutation"
    EXPENSIVE_RUNTIME = "expensive_runtime"


@dataclass(frozen=True)
class RuntimeAction:
    name: str
    description: str
    required_phase: Geant4RuntimePhase | None = None
    mutates_state: bool = True


@dataclass
class ExecutionObservation:
    status: RuntimeActionStatus
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_phase: Geant4RuntimePhase = Geant4RuntimePhase.DETACHED


@dataclass
class RuntimeStateSnapshot:
    connected: bool = False
    runtime_phase: Geant4RuntimePhase = Geant4RuntimePhase.DETACHED
    session_id: str = ""
    geometry_ready: bool = False
    source_ready: bool = False
    physics_ready: bool = False
    last_action: str = ""
    last_error: str = ""
    available_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCallRequest:
    tool_name: str
    arguments: dict[str, Any]
