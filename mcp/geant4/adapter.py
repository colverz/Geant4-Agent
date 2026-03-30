from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import tempfile
from typing import Any

from core.runtime.types import (
    ExecutionObservation,
    Geant4RuntimePhase,
    RuntimeActionStatus,
    RuntimeStateSnapshot,
)
from core.simulation import derive_role_stats, load_simulation_result
from mcp.geant4.runtime_payload import build_runtime_payload


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _extract_artifact_dir(lines: list[str]) -> Path | None:
    for line in reversed(lines):
        if not line.startswith("artifact_dir="):
            continue
        raw_value = line.split("=", 1)[1].strip()
        if not raw_value:
            return None
        return Path(raw_value)
    return None


def _load_run_summary_payload(
    stdout_lines: list[str],
    stderr_lines: list[str],
    runtime_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    artifact_dir = _extract_artifact_dir(stdout_lines) or _extract_artifact_dir(stderr_lines)
    if artifact_dir is None:
        return None
    summary_path = artifact_dir / "run_summary.json"
    if not summary_path.exists():
        return {"artifact_dir": str(artifact_dir)}
    try:
        result = load_simulation_result(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "artifact_dir": str(artifact_dir),
            "run_summary_path": str(summary_path),
        }
    payload = result.to_payload()
    role_map = None
    if isinstance(runtime_payload, dict):
        scoring = runtime_payload.get("scoring", {})
        if isinstance(scoring, dict) and isinstance(scoring.get("volume_roles"), dict):
            role_map = scoring["volume_roles"]
    role_stats = derive_role_stats(payload.get("scoring", {}).get("volume_stats"), role_map)
    if role_stats:
        payload["scoring"]["role_stats"] = role_stats
    payload["artifact_dir"] = str(artifact_dir)
    payload["run_summary_path"] = str(summary_path)
    return payload


class Geant4RuntimeAdapter(ABC):
    @abstractmethod
    def snapshot(self) -> RuntimeStateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def apply_config_patch(self, patch: dict[str, Any]) -> ExecutionObservation:
        raise NotImplementedError

    @abstractmethod
    def initialize_run(self) -> ExecutionObservation:
        raise NotImplementedError

    @abstractmethod
    def run_beam(self, events: int) -> ExecutionObservation:
        raise NotImplementedError

    @abstractmethod
    def get_last_log(self) -> ExecutionObservation:
        raise NotImplementedError


class InMemoryGeant4Adapter(Geant4RuntimeAdapter):
    """Design-time adapter for validating the MCP boundary before real Geant4 wiring exists."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._last_log: list[str] = []
        self._snapshot = RuntimeStateSnapshot(
            connected=True,
            runtime_phase=Geant4RuntimePhase.IDLE,
            available_actions=["get_runtime_state", "apply_config_patch"],
            metadata={"adapter": "in_memory"},
        )

    def snapshot(self) -> RuntimeStateSnapshot:
        return deepcopy(self._snapshot)

    def apply_config_patch(self, patch: dict[str, Any]) -> ExecutionObservation:
        self._config = _deep_merge(self._config, patch)
        self._snapshot.geometry_ready = bool(self._config.get("geometry"))
        self._snapshot.source_ready = bool(self._config.get("source"))
        self._snapshot.physics_ready = bool(self._config.get("physics_list") or self._config.get("physics"))
        self._snapshot.runtime_phase = Geant4RuntimePhase.CONFIGURED
        self._snapshot.last_action = "apply_config_patch"
        self._snapshot.available_actions = [
            "get_runtime_state",
            "apply_config_patch",
            "get_last_log",
            "initialize_run",
        ]
        self._last_log.append("config patch applied")
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Configuration patch applied.",
            payload={"config": deepcopy(self._config)},
            runtime_phase=self._snapshot.runtime_phase,
        )

    def initialize_run(self) -> ExecutionObservation:
        if not (self._snapshot.geometry_ready and self._snapshot.source_ready and self._snapshot.physics_ready):
            self._snapshot.last_error = "runtime not fully configured"
            return ExecutionObservation(
                status=RuntimeActionStatus.REJECTED,
                message="Cannot initialize Geant4 before geometry, source, and physics are configured.",
                errors=["missing_required_runtime_sections"],
                runtime_phase=self._snapshot.runtime_phase,
            )
        self._snapshot.runtime_phase = Geant4RuntimePhase.INITIALIZED
        self._snapshot.last_action = "initialize_run"
        self._snapshot.available_actions = [
            "get_runtime_state",
            "apply_config_patch",
            "get_last_log",
            "run_beam",
        ]
        self._last_log.append("runtime initialized")
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Geant4 runtime initialized.",
            runtime_phase=self._snapshot.runtime_phase,
        )

    def run_beam(self, events: int) -> ExecutionObservation:
        if self._snapshot.runtime_phase != Geant4RuntimePhase.INITIALIZED:
            return ExecutionObservation(
                status=RuntimeActionStatus.REJECTED,
                message="Cannot run beam before runtime initialization.",
                errors=["runtime_not_initialized"],
                runtime_phase=self._snapshot.runtime_phase,
            )
        self._snapshot.runtime_phase = Geant4RuntimePhase.RUNNING
        self._snapshot.last_action = "run_beam"
        self._last_log.append(f"run started with {events} events")
        payload = {"events": int(events), "status": "simulated"}
        self._snapshot.runtime_phase = Geant4RuntimePhase.INITIALIZED
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Beam run completed.",
            payload=payload,
            runtime_phase=self._snapshot.runtime_phase,
        )

    def get_last_log(self) -> ExecutionObservation:
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Runtime log fetched.",
            payload={"lines": list(self._last_log[-50:])},
            runtime_phase=self._snapshot.runtime_phase,
        )


class LocalProcessGeant4Adapter(Geant4RuntimeAdapter):
    """
    Execute a local Geant4 wrapper process while preserving the same MCP-facing API.

    The process is expected to accept:
    - `--events <int>`
    - `--config <path-to-json>`
    """

    def __init__(
        self,
        command: list[str],
        *,
        geant4_root: str = r"F:\Geant4",
        working_dir: str | None = None,
    ) -> None:
        self._command = list(command)
        self._geant4_root = Path(geant4_root)
        self._working_dir = working_dir
        self._config: dict[str, Any] = {}
        self._runtime_payload: dict[str, Any] = {}
        self._last_log: list[str] = []
        self._snapshot = RuntimeStateSnapshot(
            connected=bool(self._command),
            runtime_phase=Geant4RuntimePhase.IDLE if self._command else Geant4RuntimePhase.DETACHED,
            available_actions=["get_runtime_state", "apply_config_patch"] if self._command else [],
            metadata={
                "adapter": "local_process",
                "geant4_root": str(self._geant4_root),
                "command": list(self._command),
            },
        )

    def snapshot(self) -> RuntimeStateSnapshot:
        return deepcopy(self._snapshot)

    def apply_config_patch(self, patch: dict[str, Any]) -> ExecutionObservation:
        self._config = _deep_merge(self._config, patch)
        self._runtime_payload = build_runtime_payload(self._config)
        self._snapshot.geometry_ready = bool(self._config.get("geometry"))
        self._snapshot.source_ready = bool(self._config.get("source"))
        self._snapshot.physics_ready = bool(self._config.get("physics_list") or self._config.get("physics"))
        self._snapshot.runtime_phase = Geant4RuntimePhase.CONFIGURED
        self._snapshot.last_action = "apply_config_patch"
        self._snapshot.available_actions = [
            "get_runtime_state",
            "apply_config_patch",
            "get_last_log",
            "initialize_run",
        ]
        self._last_log.append("config patch applied")
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Configuration patch applied.",
            payload={
                "config": deepcopy(self._config),
                "runtime_payload": deepcopy(self._runtime_payload),
            },
            runtime_phase=self._snapshot.runtime_phase,
        )

    def initialize_run(self) -> ExecutionObservation:
        if not self._command:
            return ExecutionObservation(
                status=RuntimeActionStatus.FAILED,
                message="No local Geant4 command configured.",
                errors=["missing_runtime_command"],
                runtime_phase=self._snapshot.runtime_phase,
            )
        if not (self._snapshot.geometry_ready and self._snapshot.source_ready and self._snapshot.physics_ready):
            self._snapshot.last_error = "runtime not fully configured"
            return ExecutionObservation(
                status=RuntimeActionStatus.REJECTED,
                message="Cannot initialize Geant4 before geometry, source, and physics are configured.",
                errors=["missing_required_runtime_sections"],
                runtime_phase=self._snapshot.runtime_phase,
            )
        self._snapshot.runtime_phase = Geant4RuntimePhase.INITIALIZED
        self._snapshot.last_action = "initialize_run"
        self._snapshot.available_actions = [
            "get_runtime_state",
            "apply_config_patch",
            "get_last_log",
            "run_beam",
        ]
        self._last_log.append("runtime initialized")
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Geant4 runtime initialized.",
            runtime_phase=self._snapshot.runtime_phase,
        )

    def run_beam(self, events: int) -> ExecutionObservation:
        if self._snapshot.runtime_phase != Geant4RuntimePhase.INITIALIZED:
            return ExecutionObservation(
                status=RuntimeActionStatus.REJECTED,
                message="Cannot run beam before runtime initialization.",
                errors=["runtime_not_initialized"],
                runtime_phase=self._snapshot.runtime_phase,
            )

        config_path = None
        self._snapshot.runtime_phase = Geant4RuntimePhase.RUNNING
        self._snapshot.last_action = "run_beam"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="geant4_runtime_config_",
                delete=False,
                encoding="utf-8",
            ) as handle:
                import json

                runtime_payload = deepcopy(self._runtime_payload or build_runtime_payload(self._config))
                runtime_payload.pop("raw_config", None)
                json.dump(runtime_payload, handle, ensure_ascii=True, indent=2)
                config_path = handle.name

            completed = subprocess.run(
                [*self._command, "--events", str(int(events)), "--config", config_path],
                capture_output=True,
                text=True,
                cwd=self._working_dir,
                env=self._build_env(),
                check=False,
            )
            stdout_lines = completed.stdout.splitlines()
            stderr_lines = completed.stderr.splitlines()
            self._last_log.extend(stdout_lines[-100:])
            self._last_log.extend(stderr_lines[-100:])
            self._snapshot.runtime_phase = (
                Geant4RuntimePhase.INITIALIZED if completed.returncode == 0 else Geant4RuntimePhase.FAILED
            )
            if completed.returncode != 0:
                self._snapshot.last_error = f"runtime exited with code {completed.returncode}"
                return ExecutionObservation(
                    status=RuntimeActionStatus.FAILED,
                    message="Beam run failed.",
                    payload={
                        "events": int(events),
                        "returncode": completed.returncode,
                        "stdout_tail": stdout_lines[-20:],
                        "stderr_tail": stderr_lines[-20:],
                    },
                    errors=["runtime_process_failed"],
                    runtime_phase=self._snapshot.runtime_phase,
                )
            return ExecutionObservation(
                status=RuntimeActionStatus.COMPLETED,
                message="Beam run completed.",
                payload={
                    "events": int(events),
                    "returncode": completed.returncode,
                    "stdout_tail": stdout_lines[-20:],
                    "stderr_tail": stderr_lines[-20:],
                    "simulation_result": _load_run_summary_payload(
                        stdout_lines,
                        stderr_lines,
                        runtime_payload,
                    ),
                },
                runtime_phase=self._snapshot.runtime_phase,
            )
        except OSError as exc:
            self._snapshot.runtime_phase = Geant4RuntimePhase.FAILED
            self._snapshot.last_error = str(exc)
            return ExecutionObservation(
                status=RuntimeActionStatus.FAILED,
                message="Failed to launch local Geant4 process.",
                errors=["runtime_launch_failed", type(exc).__name__],
                payload={"exception": str(exc)},
                runtime_phase=self._snapshot.runtime_phase,
            )
        finally:
            if config_path:
                try:
                    Path(config_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def get_last_log(self) -> ExecutionObservation:
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Runtime log fetched.",
            payload={"lines": list(self._last_log[-100:])},
            runtime_phase=self._snapshot.runtime_phase,
        )

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        geant4_bin = self._geant4_root / "bin"
        data_dir = self._geant4_root / "share" / "Geant4" / "data"
        env["PATH"] = f"{geant4_bin}{os.pathsep}{env.get('PATH', '')}"
        env["GEANT4_DATA_DIR"] = str(data_dir)
        return env
