from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.agent.staged_patch import StagedPatch, apply_staged_patch, verify_staged_patch_hash
from core.orchestrator.confirmation_policy import build_candidate_from_pending_confirmation
from core.orchestrator.path_ops import deep_copy, remove_path, set_path
from core.orchestrator.types import CandidateUpdate


@dataclass(frozen=True)
class StagedPatchCompatibilityPreview:
    candidate: CandidateUpdate
    staged_config: dict[str, Any]
    candidate_config: dict[str, Any]
    equivalent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "candidate": {
                "producer": self.candidate.producer.value,
                "intent": self.candidate.intent.value,
                "target_paths": list(self.candidate.target_paths),
                "update_count": len(self.candidate.updates),
            },
            "staged_config": self.staged_config,
            "candidate_config": self.candidate_config,
        }


def staged_patch_to_confirmation_candidate(patch: StagedPatch, *, turn_id: int) -> CandidateUpdate:
    if not verify_staged_patch_hash(patch):
        raise ValueError("staged_patch_hash_mismatch")
    return build_candidate_from_pending_confirmation([dict(item) for item in patch.pending_items], turn_id=turn_id)


def apply_confirmation_candidate(config: dict[str, Any], candidate: CandidateUpdate) -> dict[str, Any]:
    working = deep_copy(config)
    for update in candidate.updates:
        if update.op == "remove":
            remove_path(working, update.path)
        else:
            set_path(working, update.path, update.value)
    return working


def preview_staged_patch_compatibility(
    config: dict[str, Any],
    patch: StagedPatch,
    *,
    turn_id: int,
) -> StagedPatchCompatibilityPreview:
    candidate = staged_patch_to_confirmation_candidate(patch, turn_id=turn_id)
    staged_config = apply_staged_patch(config, patch)
    candidate_config = apply_confirmation_candidate(config, candidate)
    return StagedPatchCompatibilityPreview(
        candidate=candidate,
        staged_config=staged_config,
        candidate_config=candidate_config,
        equivalent=staged_config == candidate_config,
    )


__all__ = [
    "StagedPatchCompatibilityPreview",
    "apply_confirmation_candidate",
    "preview_staged_patch_compatibility",
    "staged_patch_to_confirmation_candidate",
]
