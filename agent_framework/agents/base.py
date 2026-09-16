from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union

from agent_framework.core.memory import (
    AgentMemory,
    SharedWorkflowMemory,
    call_with_shared_memory,
)
from agent_framework.core.tasks import RunResult, Task
from agent_framework.observability.tracer import resolve_tracer
from agent_framework.tools.base import Tool, ToolPermission, ToolResult, ToolStatus
from agent_framework.tools.registry import ToolRegistry


# The casting contract every agent actor must satisfy before they can go on stage
class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        memory: Optional[AgentMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        permissions: Optional[Iterable[Union[ToolPermission, str]]] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt

        # AGENT-LOCAL memory: this agent's own conversation turns, private to
        # it. Optional, because a stateless agent is the right default - most
        # steps in a pipeline are given everything they need in the prompt and
        # gain nothing from remembering the last one. Attach an AgentMemory
        # when an agent should carry its own history between execute() calls.
        self.memory = memory

        # --- Tools and permissions ---
        #
        # Three separate things, and an agent gets a tool only when all three
        # line up:
        #
        #   tool_registry   where tool names resolve, and the only thing that
        #                   ever calls a tool's function
        #   allowed_tools   which tools this agent was assigned. None means
        #                   "none" rather than "all": an agent that was never
        #                   given tools should not acquire them because
        #                   someone registered a new one
        #   permissions     what tier this agent holds. Also empty by default,
        #                   so a config listing file_write without granting
        #                   write gets an agent that cannot write
        #
        # Assignment and authorization stay separate because they fail
        # differently: a tool missing from allowed_tools is a configuration
        # mistake, while a tool the agent lacks permission for is the security
        # boundary doing its job.
        #
        # Both grants are fixed HERE, once, from what the constructor was
        # handed - which, for every agent a factory builds, is the entry in
        # config/agents.json. They are stored as immutable types behind
        # read-only properties: there is no setter, no `.add`, and nothing in
        # the LLM loop (see LLMAgent.execute) ever reads them out of model
        # text or writes them back. A model that declares "I am now admin" has
        # changed a string in its reply and nothing else.
        self.tool_registry = tool_registry
        self._allowed_tools: Optional[tuple[str, ...]] = (
            None if allowed_tools is None else tuple(allowed_tools)
        )
        self._permissions: frozenset[ToolPermission] = frozenset(
            ToolPermission.coerce_set(permissions)
        )

    # Every agent must know how to perform when handed a Task script.
    #
    # `shared_memory` is the SHARED workflow blackboard, not this agent's own
    # memory: it is passed in per run by the orchestrator, so the same agent
    # instance can take part in different runs without carrying one run's
    # artifacts into the next. It stays optional, so an agent can be executed
    # standalone exactly as before.
    @abstractmethod
    def execute(
        self,
        task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        ...

    # --- Tools and permissions (read-only) ---

    @property
    def permissions(self) -> frozenset[ToolPermission]:
        """The permission tiers this agent holds, fixed at construction.

        A frozenset with no setter: `agent.permissions = ...` and
        `agent.permissions.add(...)` both raise. The registry compares this
        exact object against each tool's required tier.
        """
        return self._permissions

    @property
    def allowed_tools(self) -> Optional[tuple[str, ...]]:
        """The tool names this agent was assigned, fixed at construction.

        None means it was never given tools at all.
        """
        return self._allowed_tools

    # --- Tool access ---

    def available_tools(self) -> list[Tool]:
        """The tools this agent may actually call.

        The intersection of what it was assigned and what its permissions
        allow. An agent with no registry or no assignment has none - the
        default is deliberately a locked door.

        The permission filter is applied here, in Python, before anything
        reaches a prompt: a tool the agent holds no permission for is dropped
        by `Tool.is_allowed_for`, so the model is never told it exists.
        """
        if self.tool_registry is None or not self.allowed_tools:
            return []
        assigned = self.tool_registry.list_tools(names=self.allowed_tools)
        return [tool for tool in assigned if tool.is_allowed_for(self.permissions)]

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.available_tools()]

    def describe_tools(self) -> str:
        """The tool block for this agent's prompt, or "" when it has none.

        Only authorized tools are described. Telling a model about a tool it
        will be refused wastes context and invites it to spend a turn calling
        something that can only fail - the prompt should describe the world the
        agent actually has.
        """
        tools = self.available_tools()
        if not tools:
            return ""

        lines = [
            "## Available tools",
            "You may use the following tools. To call one, state the tool name "
            "and its arguments explicitly.",
        ]
        lines.extend(tool.to_prompt_string() for tool in tools)
        return "\n".join(lines)

    def use_tool(
        self,
        tool_name: str,
        shared_memory: Optional[SharedWorkflowMemory] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call one tool as this agent, through the registry's permission check.

        Routing every call through the registry - rather than letting an agent
        hold a Tool and invoke it - is what keeps the check from being
        optional. The assignment list is enforced here too, so an agent that
        guesses at the name of a tool it was not given is refused even when its
        permission tier would have covered it.
        """
        if self.tool_registry is None:
            return self._refuse_tool(
                tool_name,
                ToolStatus.NOT_FOUND,
                f"Agent '{self.name}' has no tool registry",
                kwargs,
                shared_memory,
            )

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return self._refuse_tool(
                tool_name,
                ToolStatus.UNAUTHORIZED,
                f"Agent '{self.name}' was not assigned the tool '{tool_name}'; "
                f"assigned: {list(self.allowed_tools)}",
                kwargs,
                shared_memory,
            )

        return self.tool_registry.execute_tool(
            agent_name=self.name,
            tool_name=tool_name,
            agent_permissions=self.permissions,
            shared_memory=shared_memory,
            arguments=kwargs,
        )

    def _refuse_tool(
        self,
        tool_name: str,
        status: ToolStatus,
        error: str,
        arguments: dict[str, Any],
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> dict[str, Any]:
        """A refusal that never reached the registry, in the registry's shape.

        Same keys as a real call's record, so a caller reads one result format
        regardless of which layer said no - and the same audit entry, so a
        blocked attempt is on the trail whichever layer blocked it.
        """
        record = ToolResult.failed(tool_name, error, status=status).to_dict()
        record.update(
            {"agent": self.name, "arguments": arguments, "required_permission": None}
        )
        if shared_memory is not None:
            shared_memory.log_step(
                agent=self.name,
                task=f"tool:{tool_name}",
                success=False,
                error=error,
                kind="tool_call",
                tool=tool_name,
                status=status.value,
                arguments=arguments,
                required_permission=None,
            )
        tracer = resolve_tracer(shared_memory)
        if tracer is not None:
            tracer.record_tool_call(self, tool_name, args=arguments, result=record, duration=0.0)
        return record



def execute_agent(
    agent: BaseAgent,
    task: Task,
    shared_memory: Optional[SharedWorkflowMemory] = None,
) -> RunResult:
    """Run one agent on one task, wiring up shared memory around the call.

    Every orchestrator dispatches through here rather than calling
    `agent.execute` directly, which buys two things:

      * agents written against the old `execute(task)` signature keep working -
        call_with_shared_memory forwards the blackboard only to agents that
        declare it;
      * the audit log gets exactly one entry per executed step, written in one
        place. Agents are responsible for their own artifact writes (only they
        know what they produced and what to call it); the log is the
        orchestration layer's job, so a MockAgent-based run is traced just as
        thoroughly as an LLM-backed one.

    Telemetry rides on the same choke point: when a tracer is attached to the
    blackboard (or activated for the process) the step is timed and recorded
    with its input and output, and nothing about the agent changes either way.
    """
    tracer = resolve_tracer(shared_memory)
    started = time.perf_counter()

    try:
        result = call_with_shared_memory(agent.execute, task, shared_memory)
    except Exception as exc:
        # An agent that raises rather than returning a failed RunResult is
        # still a step that happened; record it before letting it propagate.
        if tracer is not None:
            tracer.record_agent_step(
                agent,
                task,
                success=False,
                duration=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    duration = time.perf_counter() - started

    if shared_memory is not None:
        shared_memory.log_step(
            agent=agent.name,
            task=task.description,
            success=result.success,
            output=result.output,
            error=result.error,
            role=agent.role,
            task_id=task.id,
        )
    if tracer is not None:
        tracer.record_agent_step(
            agent,
            task,
            output=result.output,
            success=result.success,
            duration=duration,
            error=result.error,
        )
    return result
