from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.runtime.types import ToolCallRequest
from mcp.geant4.adapter import InMemoryGeant4Adapter, LocalProcessGeant4Adapter
from mcp.geant4.runtime_payload import build_runtime_payload
from mcp.geant4.server import Geant4McpServer


ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_WRAPPER = ROOT / "mcp" / "geant4" / "local_wrapper.py"
LOCAL_PROBE = ROOT / "legacy" / "tooling" / "geant4_minimal_probe" / "build" / "Release" / "geant4_minimal_probe.exe"
GEANT4_ROOT = Path(r"F:\Geant4")

_GEANT4_SERVER: Geant4McpServer | None = None
_LAST_VIEWER_PID: int | None = None


def _build_server() -> Geant4McpServer:
    if LOCAL_WRAPPER.exists():
        adapter = LocalProcessGeant4Adapter(
            [sys.executable, str(LOCAL_WRAPPER)],
            geant4_root=str(GEANT4_ROOT),
            working_dir=str(ROOT),
        )
        snapshot = adapter.snapshot()
        snapshot.metadata["wrapper_path"] = str(LOCAL_WRAPPER)
        snapshot.metadata["probe_path"] = str(LOCAL_PROBE)
        snapshot.metadata["wrapper_mode"] = "local_process"
        return Geant4McpServer(adapter=adapter)
    return Geant4McpServer(adapter=InMemoryGeant4Adapter())


def get_geant4_server() -> Geant4McpServer:
    global _GEANT4_SERVER
    if _GEANT4_SERVER is None:
        _GEANT4_SERVER = _build_server()
    return _GEANT4_SERVER


def geant4_state_payload() -> dict[str, Any]:
    obs = get_geant4_server().call_tool(ToolCallRequest(tool_name="get_runtime_state", arguments={}))
    payload = dict(obs.payload)
    payload["status"] = obs.status.value
    payload["message"] = obs.message
    payload["runtime_phase"] = obs.runtime_phase.value
    return payload


def handle_geant4_post(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    global _LAST_VIEWER_PID
    server = get_geant4_server()

    if path == "/api/geant4/viewer/open":
        patch = dict(payload.get("patch", {}))
        runtime_payload = build_runtime_payload(patch)
        adapter = server._adapter  # type: ignore[attr-defined]
        if not isinstance(adapter, LocalProcessGeant4Adapter):
            return 400, {
                "status": "failed",
                "message": "Live viewer requires the local process adapter.",
                "runtime_phase": adapter.snapshot().runtime_phase.value,
            }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="geant4_viewer_config_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            runtime_payload = dict(runtime_payload)
            runtime_payload.pop("raw_config", None)
            json.dump(runtime_payload, handle, ensure_ascii=True, indent=2)
            config_path = handle.name

        completed = subprocess.run(
            [
                *adapter._command,
                "--config",
                config_path,
                "--mode",
                "viewer",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=adapter._build_env(),
            check=False,
        )
        pid = None
        for line in completed.stdout.splitlines():
            if line.startswith("viewer_pid="):
                try:
                    pid = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pid = None
        _LAST_VIEWER_PID = pid
        return (
            200 if completed.returncode == 0 else 400,
            {
                "status": "completed" if completed.returncode == 0 else "failed",
                "message": "Geant4 viewer launched." if completed.returncode == 0 else "Failed to launch Geant4 viewer.",
                "payload": {
                    "viewer_pid": pid,
                    "stdout_tail": completed.stdout.splitlines()[-20:],
                    "stderr_tail": completed.stderr.splitlines()[-20:],
                },
                "runtime_phase": adapter.snapshot().runtime_phase.value,
            },
        )
    elif path == "/api/geant4/apply":
        obs = server.call_tool(
            ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": payload.get("patch", {})})
        )
    elif path == "/api/geant4/initialize":
        obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    elif path == "/api/geant4/run":
        obs = server.call_tool(
            ToolCallRequest(tool_name="run_beam", arguments={"events": int(payload.get("events", 1))})
        )
    elif path == "/api/geant4/log":
        obs = server.call_tool(ToolCallRequest(tool_name="get_last_log", arguments={}))
    else:
        obs = server.call_tool(ToolCallRequest(tool_name="get_runtime_state", arguments={}))

    body = asdict(obs)
    body["status"] = obs.status.value
    body["runtime_phase"] = obs.runtime_phase.value
    return (200 if obs.status.value in {"completed", "accepted"} else 400), body
