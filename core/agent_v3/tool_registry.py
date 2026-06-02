from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import V3Observation, V3ObservationStatus, V3ToolCall, V3ToolRiskLevel

ToolHandler = Callable[[V3ToolCall], V3Observation]


@dataclass(slots=True)
class ToolSpec:
    name: str
    risk_level: V3ToolRiskLevel
    handler: ToolHandler
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    confirmation_required: bool = False
    idempotency_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "risk_level": self.risk_level.label,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "confirmation_required": self.confirmation_required,
            "idempotency_hint": self.idempotency_hint,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("tool name is required")
        self._tools[spec.name] = spec

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown v3 tool: {name}") from exc

    def list_tools(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._tools.values()]

    def validate_call(self, call: V3ToolCall) -> list[str]:
        if not self.has_tool(call.tool_name):
            return ["tool_not_registered"]
        return _validate_input_schema(call.arguments, self.get(call.tool_name).input_schema)

    def effective_risk(self, call: V3ToolCall) -> V3ToolRiskLevel:
        if not self.has_tool(call.tool_name):
            return call.risk_level
        return self.get(call.tool_name).risk_level

    def invoke(self, call: V3ToolCall) -> V3Observation:
        if not self.has_tool(call.tool_name):
            return V3Observation(
                source=call.tool_name,
                status=V3ObservationStatus.NOT_EVALUABLE,
                not_evaluable_reason="tool_not_registered",
                message=f"Tool is not registered: {call.tool_name}",
            )
        spec = self.get(call.tool_name)
        schema_errors = self.validate_call(call)
        if schema_errors:
            return V3Observation(
                source=call.tool_name,
                status=V3ObservationStatus.NOT_EVALUABLE,
                data={"schema_errors": schema_errors},
                not_evaluable_reason="input_schema_validation_failed",
                message=f"Tool input did not match schema for {call.tool_name}: {schema_errors[0]}",
            )
        if call.risk_level != spec.risk_level:
            call = V3ToolCall(
                tool_name=call.tool_name,
                arguments=call.arguments,
                risk_level=spec.risk_level,
                idempotency_key=call.idempotency_key,
                timeout_seconds=call.timeout_seconds,
            )
        return spec.handler(call)


def _validate_input_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    if not schema:
        return []
    if schema.get("type") != "object":
        return ["schema_root_type_not_object"]
    errors: list[str] = []
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    for key in required:
        if str(key) not in arguments:
            errors.append(f"missing_required:{key}")
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if schema.get("additionalProperties") is False:
        allowed = set(str(key) for key in properties)
        for key in arguments:
            if key not in allowed:
                errors.append(f"unexpected_property:{key}")
    for key, value in arguments.items():
        prop_schema = properties.get(key)
        if isinstance(prop_schema, dict):
            errors.extend(_validate_value(key, value, prop_schema))
    return errors


def _validate_value(path: str, value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_json_type(value, expected_type):
        errors.append(f"type_mismatch:{path}:expected_{expected_type}")
        return errors
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"enum_mismatch:{path}")
    if schema.get("minimum") is not None and isinstance(value, (int, float)) and value < float(schema["minimum"]):
        errors.append(f"minimum:{path}:{schema['minimum']}")
    return errors


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True
