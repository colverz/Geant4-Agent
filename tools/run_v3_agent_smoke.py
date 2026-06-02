from __future__ import annotations

import argparse
import json
import sys

from core.agent_v3 import AgentController, V3TurnInput
from core.agent_v3.reasoners import BasicGeant4Reasoner
from core.agent_v3.tools import build_default_geant4_tool_registry
from mcp.geant4.runtime_discovery import discover_local_geant4_runtime


V3_AGENT_SMOKE_SCHEMA_VERSION = "geant4_agent_v3_smoke.v1"
DEFAULT_TEXT = "我想评估铅屏蔽对 gamma 的透射效果，帮我设计一个 Geant4 模拟方案。"


def run_v3_agent_smoke(
    *,
    text: str = DEFAULT_TEXT,
    session_id: str = "v3-agent-smoke",
    locale: str = "zh-CN",
    accept_defaults: bool = False,
    events: int = 1000,
    run: bool = False,
    allow_in_memory: bool = False,
    auto_discover_runtime: bool = False,
) -> dict:
    discovery = discover_local_geant4_runtime() if auto_discover_runtime else None
    runtime_env = discovery.env() if discovery is not None and discovery.found else {}
    controller = AgentController(
        reasoner=BasicGeant4Reasoner(),
        tools=build_default_geant4_tool_registry(),
    )
    result = controller.run(
        V3TurnInput(
            session_id=session_id,
            user_text=text,
            locale=locale,
            metadata={
                "accept_defaults": accept_defaults or run,
                "events": max(1, int(events)),
                "run": run,
                "run_confirmed": run,
                "allow_in_memory": allow_in_memory,
                "runtime_env": runtime_env,
            },
        )
    )
    payload = {
        "schema_version": V3_AGENT_SMOKE_SCHEMA_VERSION,
        "ok": result.terminated_reason in {"final_answer", "waiting_user", "observed"},
        "terminated_reason": result.terminated_reason,
        "answer": result.answer.to_dict(),
        "state": result.state.to_dict(),
        "trace": result.trace,
        "observations": [item.to_dict() for item in result.observations],
    }
    if discovery is not None:
        payload["runtime_discovery"] = discovery.to_dict()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v3 clean-room Geant4 agent smoke path.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--session-id", default="v3-agent-smoke")
    parser.add_argument("--locale", default="zh-CN")
    parser.add_argument("--accept-defaults", action="store_true", help="Let the v3 agent draft SimulationSpec/runtime payload after design.")
    parser.add_argument("--run", action="store_true", help="Run preflight and execute Geant4 only when a real local-process runtime is configured.")
    parser.add_argument("--allow-in-memory", action="store_true", help="Allow in-memory runtime for wiring tests. This is not a real Geant4 result.")
    parser.add_argument("--auto-discover-runtime", action="store_true", help="Use repo-local compiled Geant4 runtime if available.")
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_v3_agent_smoke(
        text=args.text,
        session_id=args.session_id,
        locale=args.locale,
        accept_defaults=args.accept_defaults,
        events=args.events,
        run=args.run,
        allow_in_memory=args.allow_in_memory,
        auto_discover_runtime=args.auto_discover_runtime,
    )
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(result["answer"]["message"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
