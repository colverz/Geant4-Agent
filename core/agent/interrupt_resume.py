from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.agent.staged_patch import (
    StagedPatch,
    StagedPatchStatus,
    StagedPatchStore,
    apply_staged_patch,
    verify_staged_patch_hash,
)
from core.agent.turn_trace import stable_hash


class InterruptResumeStatus:
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale_confirmation"
    NOT_FOUND = "confirmation_not_found"
    HASH_MISMATCH = "patch_hash_mismatch"
    ALREADY_RESOLVED = "already_resolved"


@dataclass(frozen=True)
class InterruptResumeResult:
    status: str
    patch: StagedPatch | None = None
    config: dict[str, Any] | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {InterruptResumeStatus.APPROVED, InterruptResumeStatus.REJECTED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "patch": None if self.patch is None else self.patch.to_dict(),
            "config": self.config,
        }


class InterruptResumeController:
    def __init__(self, store: StagedPatchStore | None = None) -> None:
        self.store = store or StagedPatchStore()

    def stage_confirmation(
        self,
        *,
        session_id: str,
        base_turn_id: int,
        base_config: dict[str, Any],
        pending_items: list[dict[str, Any]],
    ) -> StagedPatch:
        return self.store.stage(
            session_id=session_id,
            base_turn_id=base_turn_id,
            base_config=base_config,
            pending_items=pending_items,
        )

    def approve(
        self,
        confirmation_id: str,
        *,
        current_turn_id: int,
        current_config: dict[str, Any],
    ) -> InterruptResumeResult:
        patch = self.store.get(confirmation_id)
        if patch is None:
            return InterruptResumeResult(status=InterruptResumeStatus.NOT_FOUND, error="confirmation_not_found")
        stale = self._stale_reason(patch, current_turn_id=current_turn_id, current_config=current_config)
        if stale:
            return InterruptResumeResult(status=InterruptResumeStatus.STALE, patch=patch, error=stale)
        if patch.status != StagedPatchStatus.PENDING:
            return InterruptResumeResult(status=InterruptResumeStatus.ALREADY_RESOLVED, patch=patch, error=patch.status)
        if not verify_staged_patch_hash(patch):
            return InterruptResumeResult(status=InterruptResumeStatus.HASH_MISMATCH, patch=patch, error="patch_hash_mismatch")
        updated_config = apply_staged_patch(current_config, patch)
        resolved = self.store.update_status(confirmation_id, StagedPatchStatus.APPROVED) or patch
        return InterruptResumeResult(status=InterruptResumeStatus.APPROVED, patch=resolved, config=updated_config)

    def reject(self, confirmation_id: str) -> InterruptResumeResult:
        patch = self.store.get(confirmation_id)
        if patch is None:
            return InterruptResumeResult(status=InterruptResumeStatus.NOT_FOUND, error="confirmation_not_found")
        if patch.status != StagedPatchStatus.PENDING:
            return InterruptResumeResult(status=InterruptResumeStatus.ALREADY_RESOLVED, patch=patch, error=patch.status)
        resolved = self.store.update_status(confirmation_id, StagedPatchStatus.REJECTED) or patch
        return InterruptResumeResult(status=InterruptResumeStatus.REJECTED, patch=resolved)

    @staticmethod
    def _stale_reason(patch: StagedPatch, *, current_turn_id: int, current_config: dict[str, Any]) -> str:
        if int(current_turn_id) != patch.base_turn_id:
            return "turn_id_changed"
        if stable_hash(current_config) != patch.base_config_hash:
            return "config_changed"
        return ""


__all__ = [
    "InterruptResumeController",
    "InterruptResumeResult",
    "InterruptResumeStatus",
]
