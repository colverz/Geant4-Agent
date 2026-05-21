from __future__ import annotations

from collections import defaultdict

from core.orchestrator.constraint_ledger import find_lock
from core.orchestrator.intent_guard import can_override_locked_field
from core.orchestrator.types import CandidateUpdate, Producer, SessionState, UpdateOp
from core.validation.error_codes import E_LOCKED_FIELD_OVERRIDE, E_SAME_PRIORITY_CONFLICT


DEFAULT_PRODUCER_RANK = {
    Producer.USER_EXPLICIT: 100,
    Producer.SLOT_MAPPER: 90,
    Producer.LLM_SEMANTIC_FRAME: 85,
    Producer.RUNTIME_SEMANTIC: 80,
    Producer.USER_NORMALIZER: 70,
    Producer.BERT_EXTRACTOR: 50,
    Producer.LLM_RECOMMENDER: 60,
    Producer.RULE_DEFAULT: 10,
}


def _priority(p: Producer) -> int:
    return {
        Producer.USER_EXPLICIT: 5,
        Producer.SLOT_MAPPER: 4,
        Producer.LLM_SEMANTIC_FRAME: 4,
        Producer.RUNTIME_SEMANTIC: 3,
        Producer.USER_NORMALIZER: 3,
        Producer.BERT_EXTRACTOR: 2,
        Producer.LLM_RECOMMENDER: 2,
        Producer.RULE_DEFAULT: 1,
    }[p]


def _is_locked_conflict(state: SessionState, candidate: CandidateUpdate, update: UpdateOp) -> tuple[bool, str]:
    lock = find_lock(state.constraint_ledger, update.path)
    if lock is None:
        return False, ""
    if candidate.producer == Producer.LLM_RECOMMENDER:
        return True, E_LOCKED_FIELD_OVERRIDE
    allowed, code = can_override_locked_field(candidate, state.constraint_ledger, update.path)
    if not allowed:
        return True, code or E_LOCKED_FIELD_OVERRIDE
    return False, ""


def arbitrate_candidates(
    state: SessionState,
    candidates: list[CandidateUpdate],
    producer_rank: dict[Producer, int] | None = None,
) -> tuple[list[UpdateOp], list[dict], list[dict]]:
    rank = producer_rank or DEFAULT_PRODUCER_RANK
    by_path: dict[str, list[tuple[CandidateUpdate, UpdateOp]]] = defaultdict(list)
    rejected: list[dict] = []
    applied_rules: list[dict] = []

    for cand in candidates:
        for upd in cand.updates:
            locked_conflict, code = _is_locked_conflict(state, cand, upd)
            if locked_conflict:
                rejected.append(
                    {
                        "path": upd.path,
                        "producer": cand.producer.value,
                        "reason_code": code,
                        "detail": "attempt to modify locked field",
                    }
                )
                continue
            by_path[upd.path].append((cand, upd))

    accepted: list[UpdateOp] = []
    for path, items in by_path.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                _priority(x[0].producer),
                float(x[0].confidence),
                rank.get(x[0].producer, 0),
                int(x[1].turn_id),
            ),
            reverse=True,
        )
        if len(items_sorted) > 1:
            top = items_sorted[0]
            second = items_sorted[1]
            tied = (
                _priority(top[0].producer) == _priority(second[0].producer)
                and float(top[0].confidence) == float(second[0].confidence)
                and rank.get(top[0].producer, 0) == rank.get(second[0].producer, 0)
                and top[1].value != second[1].value
            )
            if tied:
                rejected.append(
                    {
                        "path": path,
                        "producer": "multiple",
                        "reason_code": E_SAME_PRIORITY_CONFLICT,
                        "detail": "same priority and confidence conflict",
                    }
                )
                continue
        winner_cand, winner_upd = items_sorted[0]
        accepted.append(winner_upd)
        applied_rules.append(
            {
                "path": path,
                "rule": "winner_selected",
                "producer": winner_cand.producer.value,
                "confidence": float(winner_cand.confidence),
            }
        )
        for loser_cand, _ in items_sorted[1:]:
            rejected.append(
                {
                    "path": path,
                    "producer": loser_cand.producer.value,
                    "reason_code": "E_LOWER_PRIORITY",
                    "detail": "superseded by winner",
                }
            )

    return accepted, rejected, applied_rules
