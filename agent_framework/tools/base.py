from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Union

# Tool is basically the framework and a way for the agent to get help
# It has its own description and argument schema so the agent can be told
# what it may call without a human writing.
class ToolPermission(Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    ADMIN = "admin"

    @property
    def level(self) -> int:
        return _PERMISSION_LEVELS[self]

    @classmethod
    def coerce(cls, value: Union["ToolPermission", str]) -> "ToolPermission":
        """Turn a config string into a member, accepting the common spellings.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"permission must be a ToolPermission or str, got {type(value).__name__}"
            )

        key = value.strip().lower()
        if key in PERMISSION_ALIASES:
            return PERMISSION_ALIASES[key]
        raise ValueError(
            f"Unknown permission {value!r}. Valid values: "
            f"{sorted(PERMISSION_ALIASES)}"
        )

    @classmethod
    def coerce_set(
        cls, values: Optional[Iterable[Union["ToolPermission", str]]]
    ) -> set["ToolPermission"]:
        """Normalise a whole grant - None means 'holds nothing'."""
        if values is None:
            return set()
        if isinstance(values, (str, cls)):  # a bare grant, not a collection of them
            return {cls.coerce(values)}
        return {cls.coerce(value) for value in values}

    def satisfied_by(
        self, granted: Optional[Iterable[Union["ToolPermission", str]]]
    ) -> bool:
        """True when something in `granted` sits at or above this tier."""
        held = ToolPermission.coerce_set(granted)
        return any(permission.level >= self.level for permission in held)

    def __str__(self) -> str:
        return self.value


_PERMISSION_LEVELS = {
    ToolPermission.READ_ONLY: 0,
    ToolPermission.WRITE: 1,
    ToolPermission.ADMIN: 2,
}

# Spellings accepted in config and in agent grants. The scope-style names
# ("fs:read", "search") exist because that is how permissions are usually
# written down; they resolve onto the same three tiers.
PERMISSION_ALIASES: dict[str, ToolPermission] = {
    "read": ToolPermission.READ_ONLY,
    "read_only": ToolPermission.READ_ONLY,
    "readonly": ToolPermission.READ_ONLY,
    "fs:read": ToolPermission.READ_ONLY,
    "search": ToolPermission.READ_ONLY,
    "math": ToolPermission.READ_ONLY,
    "write": ToolPermission.WRITE,
    "fs:write": ToolPermission.WRITE,
    "admin": ToolPermission.ADMIN,
    "*": ToolPermission.ADMIN,
}


# How a tool call ended. Every path out of Tool.execute and
# ToolRegistry.execute_tool sets exactly one of these.
class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"  # the tool ran and raised
    UNAUTHORIZED = "unauthorized"  # the caller lacked the permission
    NOT_FOUND = "not_found"  # no tool registered under that name
    INVALID_ARGUMENTS = "invalid_arguments"  # never reached the tool at all

    def __str__(self) -> str:
        return self.value


# The one shape every tool call returns, successful or not.
@dataclass
class ToolResult:
    tool: str
    status: ToolStatus
    output: Any = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status is ToolStatus.OK

    @classmethod
    def succeeded(cls, tool: str, output: Any) -> "ToolResult":
        return cls(tool=tool, status=ToolStatus.OK, output=output)

    @classmethod
    def failed(
        cls, tool: str, error: str, status: ToolStatus = ToolStatus.ERROR
    ) -> "ToolResult":
        return cls(tool=tool, status=status, error=error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status.value,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }

    def __repr__(self) -> str:
        detail = self.error if self.error else repr(self.output)
        return f"ToolResult(tool={self.tool!r}, status={self.status.value!r}, {detail})"


# JSON-schema type names mapped onto what an isinstance check needs. "integer"
# excludes bool explicitly below - bool is a subclass of int in Python, and a
# tool asking for a count should not silently accept True as 1.
_JSON_TYPES: dict[str, tuple] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
    "null": (type(None),),
}


@dataclass
class Tool:
    """One callable capability, with its description, schema and permission.
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    required_permission: ToolPermission = ToolPermission.READ_ONLY

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool requires a non-empty name")
        if not callable(self.func):
            raise TypeError(f"Tool {self.name!r} func must be callable")
        self.required_permission = ToolPermission.coerce(self.required_permission)

    # --- Authorization ---

    def is_allowed_for(
        self, permissions: Optional[Iterable[Union[ToolPermission, str]]]
    ) -> bool:
        """True when a caller holding `permissions` may invoke this tool."""
        return self.required_permission.satisfied_by(permissions)

    # --- Argument checking ---

    @property
    def properties(self) -> dict[str, Any]:
        return self.parameters_schema.get("properties", {}) or {}

    @property
    def required_arguments(self) -> list[str]:
        return list(self.parameters_schema.get("required", []) or [])

    def validate_arguments(self, arguments: dict[str, Any]) -> Optional[str]:
        """Return an error message when `arguments` don't fit the schema."""
        properties = self.properties
        if not properties and not self.required_arguments:
            return None  # schema-less tool: whatever func accepts, it accepts

        for name in self.required_arguments:
            if name not in arguments:
                return f"missing required argument '{name}'"

        if not self.parameters_schema.get("additionalProperties", False):
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                return (
                    f"unexpected argument(s) {unknown}; "
                    f"accepted: {sorted(properties)}"
                )

        for name, value in arguments.items():
            declared = properties.get(name, {}).get("type")
            if declared is None:
                continue
            expected = _JSON_TYPES.get(declared)
            if expected is None:
                continue  # unknown type name in the schema: not the caller's fault
            if declared == "integer" and isinstance(value, bool):
                return f"argument '{name}' must be integer, got bool"
            if not isinstance(value, expected):
                return (
                    f"argument '{name}' must be {declared}, "
                    f"got {type(value).__name__}"
                )
        return None

    # --- Invocation ---

    def execute(self, **kwargs: Any) -> ToolResult:
        """Call the underlying function and always come back with a ToolResult.
        """
        invalid = self.validate_arguments(kwargs)
        if invalid is not None:
            return ToolResult.failed(
                self.name, invalid, status=ToolStatus.INVALID_ARGUMENTS
            )

        try:
            output = self.func(**kwargs)
        except TypeError as exc:
            # A signature mismatch the schema did not describe - report it as a
            # bad call rather than a tool malfunction, since the fix is the
            # caller's.
            return ToolResult.failed(
                self.name,
                f"{type(exc).__name__}: {exc}",
                status=ToolStatus.INVALID_ARGUMENTS,
            )
        except Exception as exc:
            return ToolResult.failed(self.name, f"{type(exc).__name__}: {exc}")

        return ToolResult.succeeded(self.name, output)

    # --- Description ---

    def describe(self) -> dict[str, Any]:
        """The tool as a JSON-serialisable spec (also the tool-calling shape)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "required_permission": self.required_permission.value,
        }

    def to_prompt_string(self) -> str:
        """One line per tool, for pasting into an agent's system prompt."""
        arguments = ", ".join(
            f"{name}: {spec.get('type', 'any')}"
            + ("" if name in self.required_arguments else " (optional)")
            for name, spec in self.properties.items()
        )
        return f"- {self.name}({arguments}): {self.description}"

    def __repr__(self) -> str:
        return (
            f"Tool(name={self.name!r}, "
            f"required_permission={self.required_permission.value!r})"
        )
