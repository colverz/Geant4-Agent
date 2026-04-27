from __future__ import annotations

import argparse
import json
import sys

from core.runtime.types import RuntimeActionStatus, ToolCallRequest
from core.simulation import build_runtime_smoke_report
from mcp.geant4.adapter import InMemoryGeant4Adapter, build_geant4_adapter_from_env
from mcp.geant4.server import Geant4McpServer


def runtime_env_help() -> str:
    return (
        "Configure GEANT4_RUNTIME_COMMAND_JSON, for example "
        "'[\"runtime/geant4_local_app/build/Release/geant4_local_app.exe\"]', "
        "or GEANT4_RUNTIME_COMMAND before running live smoke. "
        "Pytest live smoke also requires GEANT4_LIVE_SMOKE=1."
    )


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
    parser.add_argument("--json", action="store_true", help="Emit a compact JSON report for automation.")
    parser.add_argument("--print-env-help", action="store_true", help="Print the required opt-in runtime environment variables and exit.")
    args = parser.parse_args()
    if args.print_env_help:
        print(runtime_env_help())
        return 0
    events = max(1, int(args.events))

    adapter = build_geant4_adapter_from_env()
    if isinstance(adapter, InMemoryGeant4Adapter):
        message = "SKIP: GEANT4_RUNTIME_COMMAND_JSON or GEANT4_RUNTIME_COMMAND is not configured."
        print(message)
        print(runtime_env_help())
        return 1 if args.require_runtime else 0

    server = Geant4McpServer(adapter=adapter)
    patch = _smoke_patch()
    validate_obs = server.call_tool(
        ToolCallRequest(tool_name="validate_config", arguments={"config": patch, "events": events})
    )
    if validate_obs.status != RuntimeActionStatus.COMPLETED or not (validate_obs.payload or {}).get("ok"):
        print(f"validate_config failed: {validate_obs.message}", file=sys.stderr)
        print(validate_obs.payload or {}, file=sys.stderr)
        return 2

    apply_obs = server.call_tool(ToolCallRequest(tool_name="apply_config_patch", arguments={"patch": patch}))
    if apply_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"apply_config_patch failed: {apply_obs.message}", file=sys.stderr)
        return 3

    init_obs = server.call_tool(ToolCallRequest(tool_name="initialize_run", arguments={}))
    if init_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"initialize_run failed: {init_obs.message}", file=sys.stderr)
        return 4

    run_obs = server.call_tool(ToolCallRequest(tool_name="run_beam", arguments={"events": events}))
    if run_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"run_beam failed: {run_obs.message}", file=sys.stderr)
        print(run_obs.payload or {}, file=sys.stderr)
        return 5

    summary_obs = server.call_tool(ToolCallRequest(tool_name="summarize_last_result", arguments={}))
    if summary_obs.status != RuntimeActionStatus.COMPLETED:
        print(f"summarize_last_result failed: {summary_obs.message}", file=sys.stderr)
        print(summary_obs.errors or [], file=sys.stderr)
        return 6

    report = build_runtime_smoke_report(
        events=events,
        run_payload=run_obs.payload,
        summary_payload=summary_obs.payload,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("OK: local Geant4 smoke completed.")
        print(f"schema_version={report['schema_version']}")
        print(f"events_requested={report['events_requested']}")
        print(f"events_completed={report['events_completed']}")
        print(f"run_summary_path={report['run_summary_path']}")
        print(f"artifact_dir={report['artifact_dir']}")
        print(f"target_edep_total_mev={report['key_metrics']['target_edep_total_mev']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
