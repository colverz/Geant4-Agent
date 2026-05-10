from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from core.agent.turn_trace import stable_hash
from core.orchestrator.path_ops import deep_copy, remove_path, set_path


STAGED_PATCH_SCHEMA_VERSION = "staged_patch.v1"


class StagedPatchStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class StagedPatch:
    confirmation_id: str
    patch_hash: str
    session_id: str
    base_turn_id: int
    base_config_hash: str
    pending_items: tuple[dict[str, Any], ...]
    status: str = StagedPatchStatus.PENDING
    schema_version: str = STAGED_PATCH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "confirmation_id": self.confirmation_id,
            "patch_hash": self.patch_hash,
            "session_id": self.session_id,
            "base_turn_id": self.base_turn_id,
            "base_config_hash": self.base_config_hash,
            "pending_items": [dict(item) for item in self.pending_items],
            "status": self.status,
        }


def build_staged_patch(
    *,
    session_id: str,
    base_turn_id: int,
    base_config: dict[str, Any],
    pending_items: list[dict[str, Any]],
    confirmation_id: str | None = None,
) -> StagedPatch:
    normalized_items = tuple(_normalize_pending_item(item) for item in pending_items)
    cid = confirmation_id or uuid.uuid4().hex
    base_config_hash = stable_hash(base_config)
    patch_hash = _patch_hash(
        session_id=session_id,
        base_turn_id=base_turn_id,
        base_config_hash=base_config_hash,
        pending_items=normalized_items,
    )
    return StagedPatch(
        confirmation_id=cid,
        patch_hash=patch_hash,
        session_id=session_id,
        base_turn_id=int(base_turn_id),
        base_config_hash=base_config_hash,
        pending_items=normalized_items,
    )


def verify_staged_patch_hash(patch: StagedPatch) -> bool:
    return patch.patch_hash == _patch_hash(
        session_id=patch.session_id,
        base_turn_id=patch.base_turn_id,
        base_config_hash=patch.base_config_hash,
        pending_items=patch.pending_items,
    )


def apply_staged_patch(config: dict[str, Any], patch: StagedPatch) -> dict[str, Any]:
    if not verify_staged_patch_hash(patch):
        raise ValueError("staged_patch_hash_mismatch")
    working = deep_copy(config)
    for item in patch.pending_items:
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        if str(item.get("op") or "").strip() == "remove":
            remove_path(working, path)
        else:
            set_path(working, path, item.get("new"))
    return working


def _normalize_pending_item(item: dict[str, Any]) -> dict[str, Any]:
    path = str(item.get("path") or "").strip()
    op = "remove" if str(item.get("op") or "").strip() == "remove" else "set"
    return {
        "path": path,
        "op": op,
        "old": item.get("old"),
        "new": item.get("new"),
        "producer": str(item.get("producer") or ""),
        "reason": str(item.get("reason") or ""),
    }


def _patch_hash(
    *,
    session_id: str,
    base_turn_id: int,
    base_config_hash: str,
    pending_items: tuple[dict[str, Any], ...],
) -> str:
    return stable_hash(
        {
            "schema_version": STAGED_PATCH_SCHEMA_VERSION,
            "session_id": session_id,
            "base_turn_id": int(base_turn_id),
            "base_config_hash": base_config_hash,
            "pending_items": [dict(item) for item in pending_items],
        }
    )


class StagedPatchStore:
    def __init__(self) -> None:
        self._patches: dict[str, StagedPatch] = {}

    def stage(
        self,
        *,
        session_id: str,
        base_turn_id: int,
        base_config: dict[str, Any],
        pending_items: list[dict[str, Any]],
    ) -> StagedPatch:
        patch = build_staged_patch(
            session_id=session_id,
            base_turn_id=base_turn_id,
            base_config=base_config,
            pending_items=pending_items,
        )
        self._patches[patch.confirmation_id] = patch
        return patch

    def get(self, confirmation_id: str) -> StagedPatch | None:
        return self._patches.get(str(confirmation_id or "").strip())

    def update_status(self, confirmation_id: str, status: str) -> StagedPatch | None:
        existing = self.get(confirmation_id)
        if existing is None:
            return None
        updated = StagedPatch(
            confirmation_id=existing.confirmation_id,
            patch_hash=existing.patch_hash,
            session_id=existing.session_id,
            base_turn_id=existing.base_turn_id,
            base_config_hash=existing.base_config_hash,
            pending_items=existing.pending_items,
            status=status,
            schema_version=existing.schema_version,
        )
        self._patches[updated.confirmation_id] = updated
        return updated


__all__ = [
    "STAGED_PATCH_SCHEMA_VERSION",
    "StagedPatch",
    "StagedPatchStatus",
    "StagedPatchStore",
    "apply_staged_patch",
    "build_staged_patch",
    "verify_staged_patch_hash",
]
