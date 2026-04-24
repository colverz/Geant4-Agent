from __future__ import annotations

from core.runtime.types import ToolSpec


def _object_schema(*, required: list[str], properties: dict) -> dict:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


DEFAULT_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_runtime_state",
        description="Return the current Geant4 runtime state, health, and allowed next actions.",
        input_schema=_object_schema(required=[], properties={}),
    ),
    ToolSpec(
        name="apply_config_patch",
        description="Apply a structured patch to the current Geant4-side configuration snapshot.",
        input_schema=_object_schema(
            required=["patch"],
            properties={
                "patch": {
                    "type": "object",
                    "description": "Nested configuration fragment aligned with the agent config schema.",
                }
            },
        ),
    ),
    ToolSpec(
        name="validate_config",
        description="Preflight a Geant4 configuration before initialization or execution.",
        input_schema=_object_schema(
            required=[],
            properties={
                "config": {
                    "type": "object",
                    "description": "Optional full configuration to validate instead of the adapter's current snapshot.",
                },
                "patch": {
                    "type": "object",
                    "description": "Optional patch applied to the current or provided configuration before validation.",
                },
                "events": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of events used when building the runtime payload preview.",
                },
            },
        ),
    ),
    ToolSpec(
        name="initialize_run",
        description="Initialize the Geant4 runtime after geometry, source, and physics are ready.",
        input_schema=_object_schema(required=[], properties={}),
    ),
    ToolSpec(
        name="run_beam",
        description="Trigger a Geant4 beam/run execution for the requested number of events.",
        input_schema=_object_schema(
            required=["events"],
            properties={
                "events": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of primary events to execute.",
                }
            },
        ),
    ),
    ToolSpec(
        name="get_last_log",
        description="Return the latest runtime log, warnings, and errors from the Geant4 side.",
        input_schema=_object_schema(required=[], properties={}),
    ),
]


def get_default_tool_specs() -> list[ToolSpec]:
    return list(DEFAULT_TOOL_SPECS)
