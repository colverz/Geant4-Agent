from __future__ import annotations

import argparse
import sys

from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from mcp.geant4.adapter import InMemoryGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.server import Geant4McpServer


def _smoke_patch() -> dict:
    return {
        "geometry": {
            "structure": "single_box",
            "params": {"module_x": 10.0, "module_y": 20.0, "module_z": 30.0},
        },
        "materials": {"selected_materials": ["G4_Cu"]},
        "source": {
            "type": "point",
            "particle": "gamma",
            "energy": 1.0,
            "position": {"type": "vector", "value": [0.0, 0.0, -20.0]},
            "direction": {"type": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "physics_list": {"name": "FTFP_BERT"},
        "scoring": {"target_edep": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local Geant4 adapter smoke test when runtime env is configured.")
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--require-runtime", action="store_true", help="Fail instead of skipping when no runtime command is configured.")
    args = parser.parse_args()

    adapter = build_geant4_adapter_from_env()
    if isinstance(adapter, InMemoryGeant4Adapter):
        message = "SKIP: GEANT4_RUNTIME_COMMAND_JSON or GEANT4_RUNTIME_COMMAND is not configured."
        print(message)
        return 1 if args.require_runtime else 0

    server = Geant4McpServer(adapter=adapter)
    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": _smoke_patch()}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"apply_config_patch failed: {apply_obs.message}", file=sys.stderr)
        return 2

    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"initialize_run failed: {init_obs.message}", file=sys.stderr)
        return 3

    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": max(1, int(args.events))}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"run_beam failed: {run_obs.message}", file=sys.stderr)
        print(run_obs.payload or {}, file=sys.stderr)
        return 4

    simulation_result = (run_obs.payload or {}).get("simulation_result")
    print("OK: local Geant4 smoke completed.")
    if simulation_result:
        print(f"run_summary_path={simulation_result.get('run_summary_path', '')}")
        print(f"artifact_dir={simulation_result.get('artifact_dir', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
