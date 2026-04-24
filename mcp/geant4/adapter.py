from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
from typing import Any

from core.runtime.types import (
    ExecutionObservation,
    Geant4RuntimePhase,
    RuntimeActionStatus,
    RuntimeStateSnapshot,
)
from core.orchestrator.path_ops import get_path
from core.simulation import build_simulation_spec, derive_role_stats, load_simulation_result
from core.validation.minimal_schema import get_minimal_required_paths
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
    role_map = _runtime_volume_roles(runtime_payload)
    role_stats = None
    if not result.scoring.role_stats:
        role_stats = derive_role_stats(result.scoring.volume_stats, role_map)

    payload = result.to_payload()
    if role_stats:
        payload["scoring"]["role_stats"] = role_stats
    payload["artifact_dir"] = str(artifact_dir)
    payload["run_summary_path"] = str(summary_path)
    return payload


def _runtime_volume_roles(runtime_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime_payload, dict):
        return None
    scoring = runtime_payload.get("scoring", {})
    if not isinstance(scoring, dict):
        return None
    volume_roles = scoring.get("volume_roles")
    return volume_roles if isinstance(volume_roles, dict) else None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (dict, list, tuple)) and len(value) == 0:
        return True
    return False


def _runtime_physics_present(config: dict[str, Any]) -> bool:
    physics_list = config.get("physics_list")
    if isinstance(physics_list, dict) and not _is_missing(physics_list.get("name")):
        return True
    if isinstance(physics_list, str) and physics_list.strip():
        return True
    return not _is_missing(get_path(config, "physics.physics_list"))


def _runtime_required_missing_paths(config: dict[str, Any]) -> list[str]:
    agent_only_paths = {"materials.volume_material_map", "output.format", "output.path", "physics.physics_list"}
    missing: list[str] = []
    for path in get_minimal_required_paths(config):
        if path in agent_only_paths:
            continue
        if _is_missing(get_path(config, path)):
            missing.append(path)
    if not _runtime_physics_present(config):
        missing.append("physics.physics_list")
    return list(dict.fromkeys(missing))


def _runtime_payload_preview(runtime_payload: dict[str, Any]) -> dict[str, Any]:
    preview_keys = (
        "structure",
        "material",
        "root_volume_name",
        "source_type",
        "particle",
        "energy",
        "position",
        "direction",
        "physics_list",
        "detector_enabled",
        "run",
        "run_manifest",
        "scoring",
    )
    return {key: deepcopy(runtime_payload.get(key)) for key in preview_keys if key in runtime_payload}


def _preflight_payload(config: dict[str, Any], *, events: int) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    missing_paths = _runtime_required_missing_paths(config)
    readiness = {
        "geometry": not any(path.startswith("geometry.") for path in missing_paths),
        "source": not any(path.startswith("source.") for path in missing_paths),
        "physics": "physics.physics_list" not in missing_paths,
    }
    payload: dict[str, Any] = {
        "ok": not missing_paths,
        "readiness": readiness,
        "missing_paths": list(missing_paths),
        "runtime_required_paths": [path for path in get_minimal_required_paths(config) if not path.startswith(("materials.", "output."))],
    }
    errors = [f"missing:{path}" for path in missing_paths]
    warnings: list[str] = []
    if not missing_paths:
        try:
            runtime_payload = build_runtime_payload(build_simulation_spec(config, events=max(1, int(events))))
        except Exception as exc:
            payload["ok"] = False
            payload["build_error"] = f"{type(exc).__name__}: {exc}"
            errors.append("runtime_payload_build_failed")
        else:
            payload["runtime_payload_preview"] = _runtime_payload_preview(runtime_payload)
            payload["runtime_payload_keys"] = sorted(runtime_payload.keys())
    else:
        warnings.append("runtime_payload_preview_omitted_until_required_fields_are_present")
    return bool(payload["ok"]), payload, errors, warnings


def _parse_runtime_command(raw_command: str) -> list[str]:
    value = str(raw_command or "").strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) and item for item in parsed):
            raise ValueError("GEANT4_RUNTIME_COMMAND_JSON must be a JSON string list")
        return list(parsed)
    return shlex.split(value, posix=False)


def build_geant4_adapter_from_env(env: dict[str, str] | None = None) -> Geant4RuntimeAdapter:
    """
    Build the default MCP adapter from environment configuration.

    If no runtime command is configured, the in-memory adapter remains the safe default.
    This keeps existing tests and UI flows deterministic while giving real Geant4 runs a
    standard activation path.
    """

    env_map = env if env is not None else os.environ
    raw_command = env_map.get("GEANT4_RUNTIME_COMMAND_JSON") or env_map.get("GEANT4_RUNTIME_COMMAND")
    command = _parse_runtime_command(raw_command or "")
    if not command:
        return InMemoryGeant4Adapter()
    return LocalProcessGeant4Adapter(
        command,
        geant4_root=env_map.get("GEANT4_ROOT", r"F:\Geant4"),
        working_dir=env_map.get("GEANT4_WORKING_DIR") or None,
    )


class Geant4RuntimeAdapter(ABC):
    @abstractmethod
    def snapshot(self) -> RuntimeStateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def apply_config_patch(self, patch: dict[str, Any]) -> ExecutionObservation:
        raise NotImplementedError

    @abstractmethod
    def validate_config(
        self,
        *,
        config: dict[str, Any] | None = None,
        patch: dict[str, Any] | None = None,
        events: int = 1,
    ) -> ExecutionObservation:
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

    def validate_config(
        self,
        *,
        config: dict[str, Any] | None = None,
        patch: dict[str, Any] | None = None,
        events: int = 1,
    ) -> ExecutionObservation:
        validation_config = deepcopy(config) if isinstance(config, dict) else deepcopy(self._config)
        if isinstance(patch, dict):
            validation_config = _deep_merge(validation_config, patch)
        ok, payload, errors, warnings = _preflight_payload(validation_config, events=max(1, int(events)))
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Configuration preflight passed." if ok else "Configuration preflight found missing runtime fields.",
            payload=payload,
            errors=errors,
            warnings=warnings,
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

    def validate_config(
        self,
        *,
        config: dict[str, Any] | None = None,
        patch: dict[str, Any] | None = None,
        events: int = 1,
    ) -> ExecutionObservation:
        validation_config = deepcopy(config) if isinstance(config, dict) else deepcopy(self._config)
        if isinstance(patch, dict):
            validation_config = _deep_merge(validation_config, patch)
        ok, payload, errors, warnings = _preflight_payload(validation_config, events=max(1, int(events)))
        return ExecutionObservation(
            status=RuntimeActionStatus.COMPLETED,
            message="Configuration preflight passed." if ok else "Configuration preflight found missing runtime fields.",
            payload=payload,
            errors=errors,
            warnings=warnings,
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
                runtime_payload.pop("payload_sha256", None)
                runtime_payload["payload_sha256"] = hashlib.sha256(
                    json.dumps(runtime_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
                ).hexdigest()
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
