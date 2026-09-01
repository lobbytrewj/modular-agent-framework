from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from agent_framework.core.memory import (
    AgentMemory,
    SharedWorkflowMemory,
    call_with_shared_memory,
)
from agent_framework.core.tasks import RunResult, Task


# The casting contract every agent actor must satisfy before they can go on stage
class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        memory: Optional[AgentMemory] = None,
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
    """
    result = call_with_shared_memory(agent.execute, task, shared_memory)

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
    return result
