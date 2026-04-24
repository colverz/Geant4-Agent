from __future__ import annotations

from dataclasses import asdict

from core.runtime.types import ExecutionObservation, RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import Geant4RuntimeAdapter, build_geant4_adapter_from_env
from mcp.geant4.tools import get_default_tool_specs


class Geant4McpServer:
    """
    Lightweight MCP-facing boundary.

    This does not implement a transport itself; it provides the stable
    list-tools / call-tool behavior that can later be exposed through an
    actual MCP SDK, stdio bridge, or custom host runtime.
    """

    def __init__(self, adapter: Geant4RuntimeAdapter | None = None) -> None:
        self._adapter = adapter or build_geant4_adapter_from_env()
        self._tool_specs = {spec.name: spec for spec in get_default_tool_specs()}

    def list_tools(self) -> list[dict]:
        return [asdict(spec) for spec in self._tool_specs.values()]

    def call_tool(self, request: ToolCallRequest) -> ExecutionObservation:
        name = request.tool_name
        arguments = dict(request.arguments)
        if name == "get_runtime_state":
            snapshot = self._adapter.snapshot()
            return ExecutionObservation(
                status=RuntimeActionStatus.COMPLETED if snapshot.connected else RuntimeActionStatus.FAILED,
                message="Runtime state fetched.",
                payload=asdict(snapshot),
                runtime_phase=snapshot.runtime_phase,
            )
        if name == "apply_config_patch":
            return self._adapter.apply_config_patch(arguments.get("patch", {}))
        if name == "validate_config":
            return self._adapter.validate_config(
                config=arguments.get("config"),
                patch=arguments.get("patch"),
                events=int(arguments.get("events", 1) or 1),
            )
        if name == "initialize_run":
            return self._adapter.initialize_run()
        if name == "run_beam":
            return self._adapter.run_beam(int(arguments.get("events", 0)))
        if name == "summarize_last_result":
            return self._adapter.summarize_last_result()
        if name == "get_last_log":
            return self._adapter.get_last_log()
        snapshot = self._adapter.snapshot()
        return ExecutionObservation(
            status=RuntimeActionStatus.REJECTED,
            message=f"Unknown tool: {name}",
            errors=["unknown_tool"],
            runtime_phase=snapshot.runtime_phase,
        )
