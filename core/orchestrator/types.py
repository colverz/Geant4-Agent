from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Phase(str, Enum):
    GEOMETRY = "geometry"
    MATERIALS = "materials"
    SOURCE = "source"
    PHYSICS = "physics"
    OUTPUT = "output"
    FINALIZE = "finalize"


class Intent(str, Enum):
    SET = "SET"
    MODIFY = "MODIFY"
    REMOVE = "REMOVE"
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    QUESTION = "QUESTION"
    OTHER = "OTHER"


class Producer(str, Enum):
    USER_EXPLICIT = "user_explicit"
    USER_NORMALIZER = "user_normalizer"
    SLOT_MAPPER = "slot_mapper"
    LLM_SEMANTIC_FRAME = "llm_semantic_frame"
    BERT_EXTRACTOR = "bert_extractor"
    LLM_RECOMMENDER = "llm_recommender"
    RULE_DEFAULT = "rule_default"


class Scope(str, Enum):
    EXACT = "exact"
    SUBTREE = "subtree"


class LockReason(str, Enum):
    USER_CONFIRMED = "USER_CONFIRMED"
    USER_EXPLICIT = "USER_EXPLICIT"
    SYSTEM_LOCK_ON_COMPLETE = "SYSTEM_LOCK_ON_COMPLETE"
    TEST_FIXTURE = "TEST_FIXTURE"


@dataclass(frozen=True)
class UpdateOp:
    path: str
    op: Literal["set", "remove"]
    value: Any
    producer: Producer
    confidence: float
    turn_id: int


@dataclass(frozen=True)
class CandidateUpdate:
    producer: Producer
    intent: Intent
    target_paths: list[str]
    updates: list[UpdateOp]
    confidence: float
    rationale: str


@dataclass
class ConstraintItem:
    path: str
    value: Any
    locked: bool
    reason_code: LockReason
    scope: Scope
    source: str
    turn_id: int


@dataclass
class ValidationReport:
    ok: bool
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    missing_required_paths: list[str] = field(default_factory=list)
    first_error: dict | None = None
    candidate_rejection_reason_codes: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: str
    phase: Phase
    turn_id: int
    config: dict
    constraint_ledger: list[ConstraintItem] = field(default_factory=list)
    audit_trail: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    field_sources: dict[str, str] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    last_asked_paths: list[str] = field(default_factory=list)
    question_attempts: dict[str, int] = field(default_factory=dict)
    last_dialogue_action: str = ""
    dialogue_summary: dict[str, Any] = field(default_factory=dict)
    confirmed_fact_paths: list[str] = field(default_factory=list)
    dialogue_memory: list[dict[str, Any]] = field(default_factory=list)
    pending_overwrite: list[dict[str, Any]] = field(default_factory=list)
    semantic_missing_paths: list[str] = field(default_factory=list)
    slot_memory: dict[str, Any] = field(default_factory=dict)
