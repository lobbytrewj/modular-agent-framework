from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional, Union

from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.observability.tracer import resolve_tracer
from agent_framework.tools.base import (
    Tool,
    ToolPermission,
    ToolResult,
    ToolStatus,
)

logger = logging.getLogger(__name__)


# The one place a tool is looked up and the one place authorization happens.
#
# Agents never call `tool.func` and never call `tool.execute` directly - they
# go through `execute_tool`, which is what makes the permission check
# unavoidable rather than merely conventional. Anything that bypasses the
# registry bypasses the security model, so keeping the checks here (instead of
# inside each tool) means a new built-in tool is safe by construction.
class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        # Insertion-ordered, so list_tools() and the prompt block an agent
        # sees are stable between runs - a prompt that reshuffles itself makes
        # two otherwise identical runs incomparable.
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    # --- Registration ---

    def register(self, tool: Tool, overwrite: bool = False) -> None:
        """Add a tool. Registering a duplicate name is an error by default.

        Silent replacement is the wrong default for a security boundary: an
        agent config naming `file_write` should never end up bound to whatever
        happened to be registered last.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"register expects a Tool, got {type(tool).__name__}")
        if tool.name in self._tools and not overwrite:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                f"Pass overwrite=True to replace it."
            )
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool], overwrite: bool = False) -> None:
        for tool in tools:
            self.register(tool, overwrite=overwrite)

    def unregister(self, name: str) -> bool:
        """Remove a tool. True if it was there."""
        return self._tools.pop(name, None) is not None

    # --- Lookup ---

    def get(self, name: str) -> Optional[Tool]:
        """The tool registered under `name`, or None.

        Returns None rather than raising because the common caller is
        `execute_tool`, which turns a missing tool into a NOT_FOUND result an
        agent can read - the same shape as every other failure.
        """
        return self._tools.get(name)

    def list_tools(
        self,
        allowed_permissions: Optional[Iterable[Union[ToolPermission, str]]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> list[Tool]:
        """The registered tools, optionally narrowed twice over.

        `allowed_permissions` filters by what the caller is entitled to;
        `names` filters by what it was actually assigned. Both are applied,
        because they answer different questions - an agent granted WRITE is
        still only given the two tools its config lists, and an agent whose
        config lists `file_write` without the permission for it does not get
        it. Passing no filters lists everything, which is for inspection, not
        for handing to an agent.
        """
        selected = list(self._tools.values())

        if names is not None:
            wanted = set(names)
            selected = [tool for tool in selected if tool.name in wanted]

        if allowed_permissions is not None:
            held = ToolPermission.coerce_set(allowed_permissions)
            selected = [tool for tool in selected if tool.is_allowed_for(held)]

        return selected

    def describe_tools(
        self,
        allowed_permissions: Optional[Iterable[Union[ToolPermission, str]]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> list[dict[str, Any]]:
        return [
            tool.describe()
            for tool in self.list_tools(allowed_permissions, names=names)
        ]

    # --- Execution ---

    def execute_tool(
        self,
        agent_name: str,
        tool_name: str,
        agent_permissions: Optional[Iterable[Union[ToolPermission, str]]] = None,
        shared_memory: Optional[SharedWorkflowMemory] = None,
        raise_on_denied: bool = False,
        arguments: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one tool on behalf of one agent, if it is allowed to.

        The order is deliberate: existence, then permission, then arguments,
        then the call. Checking permission before arguments means a denied
        agent learns only that it was denied - not whether its arguments would
        have been accepted, which is the kind of detail a probing caller uses
        to map what it cannot reach.

        Denial is reported as an UNAUTHORIZED result by default rather than
        raised, so an agent loop can read the refusal, tell the model, and
        carry on. Set `raise_on_denied=True` at a call site where a denial is
        a programming error that should stop the run.

        Tool arguments may be passed as **kwargs, or in `arguments` when a name
        would collide with this method's own parameters (a tool taking an
        argument literally called `tool_name`, say).

        Returns the dict form of the ToolResult plus who called what - that
        record is both the return value and what gets written to the audit log.
        """
        call_arguments = dict(arguments or {})
        call_arguments.update(kwargs)
        started = time.perf_counter()

        tool = self.get(tool_name)
        if tool is None:
            result = ToolResult.failed(
                tool_name,
                f"No tool registered under name '{tool_name}'",
                status=ToolStatus.NOT_FOUND,
            )
            return self._finish(
                result, agent_name, call_arguments, shared_memory, required=None,
                started=started,
            )

        if not tool.is_allowed_for(agent_permissions):
            held = sorted(
                permission.value
                for permission in ToolPermission.coerce_set(agent_permissions)
            )
            message = (
                f"Agent '{agent_name}' is not permitted to use tool "
                f"'{tool_name}': it requires "
                f"'{tool.required_permission.value}' but holds {held or 'no permissions'}"
            )
            result = ToolResult.failed(
                tool_name, message, status=ToolStatus.UNAUTHORIZED
            )
            record = self._finish(
                result,
                agent_name,
                call_arguments,
                shared_memory,
                required=tool.required_permission,
                started=started,
            )
            if raise_on_denied:
                raise PermissionError(message)
            return record

        result = tool.execute(**call_arguments)
        return self._finish(
            result,
            agent_name,
            call_arguments,
            shared_memory,
            required=tool.required_permission,
            started=started,
        )

    def _finish(
        self,
        result: ToolResult,
        agent_name: str,
        arguments: dict[str, Any],
        shared_memory: Optional[SharedWorkflowMemory],
        required: Optional[ToolPermission],
        started: Optional[float] = None,
    ) -> dict[str, Any]:
        """Attach the call's provenance, log it, and hand back the record.

        Every attempt is logged, including the ones that were refused: a denial
        is the single most interesting thing this layer produces, and a log
        that only records successful calls cannot answer "did anything try to
        write to disk during that run?".

        The same record goes to the run's tracer when there is one, with the
        call's duration - so a refused call shows up in the trace as a
        TOOL_CALL event with status "unauthorized", next to the allowed ones.
        """
        duration = None if started is None else time.perf_counter() - started

        record = result.to_dict()
        record["agent"] = agent_name
        record["arguments"] = arguments
        record["required_permission"] = required.value if required else None
        record["duration"] = duration

        if shared_memory is not None:
            shared_memory.log_step(
                agent=agent_name,
                task=f"tool:{result.tool}",
                success=result.success,
                # The audit trail is read as text, so a tool's structured
                # output is stringified here rather than stored raw.
                output=None if result.output is None else str(result.output),
                error=result.error,
                kind="tool_call",
                tool=result.tool,
                status=result.status.value,
                arguments=arguments,
                required_permission=record["required_permission"],
            )

        tracer = resolve_tracer(shared_memory)
        if tracer is not None:
            tracer.record_tool_call(
                agent_name,
                result.tool,
                args=arguments,
                result=record,
                duration=duration,
            )

        if result.status is ToolStatus.UNAUTHORIZED:
            logger.warning("%s", result.error)
        elif not result.success:
            logger.info("tool '%s' called by '%s' failed: %s",
                        result.tool, agent_name, result.error)

        return record

    # --- Conveniences ---

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.names()})"


def build_default_registry(**builtin_options: Any) -> ToolRegistry:
    """A registry holding every built-in tool.

    Imported lazily: `agent_framework.tools.builtin` pulls in the file sandbox
    and the search corpus, and a program that only wants the Tool type should
    not pay for those.
    """
    from agent_framework.tools.builtin import build_builtin_tools

    return ToolRegistry(build_builtin_tools(**builtin_options))


_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def default_registry() -> ToolRegistry:
    """The process-wide registry that config-declared tool names resolve against.

    A shared default exists so `"tools": ["calculator"]` in agents.json means
    something without every factory being handed a registry. Anywhere the set
    of tools matters to a test, build an explicit ToolRegistry instead - this
    one is shared state, and a test that registers into it affects the next.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY
