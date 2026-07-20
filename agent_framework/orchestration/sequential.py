from __future__ import annotations

from typing import Optional, Sequence

from agent_framework.agents.base import BaseAgent
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.registry import AgentRegistry


# Runs a fixed lineup of agents one after another
# each agent's finish line (its RunResult.output) becomes the next agent's starting line (its Task.description).
class SequentialOrchestrator:
    def __init__(self, agents: Sequence[BaseAgent]):
        if not agents:
            raise ValueError("SequentialOrchestrator requires at least one agent")
        self.agents: list[BaseAgent] = list(agents)

        # holds every step's RunResult in order, so callers can inspect the whole pipeline, not just the final part.
        self.step_results: list[RunResult] = []

    @classmethod
    def from_registry(
        cls,
        agent_names: Sequence[str],
        registry: AgentRegistry,
        agent_kwargs: Optional[dict[str, dict]] = None,
    ) -> "SequentialOrchestrator":
        agent_kwargs = agent_kwargs or {}
        instances: list[BaseAgent] = []
        for name in agent_names:
            agent_cls = registry.get(name)
            kwargs = dict(agent_kwargs.get(name, {}))
            kwargs.setdefault("name", name)
            kwargs.setdefault("role", name)
            kwargs.setdefault(
                "system_prompt", f"You are the '{name}' step in a sequential pipeline."
            )
            instances.append(agent_cls(**kwargs))
        return cls(instances)

    def run(self, initial_task: Task) -> RunResult:
        # Data flow through the pipeline:
        #
        #   initial_task --> agents[0] --> RunResult_0
        #                                     |
        #                     RunResult_0.output becomes the
        #                     `description` of a new Task
        #                                     v
        #                  agents[1] --> RunResult_1
        #                                     |
        #                                    ...
        #
        # `input_data` from the *original* task is carried along unchanged
        # at every hop, so shared context (config, IDs, etc.) is available
        # to every agent in the chain, not just the first one.
        #
        # If any agent fails, we stop immediately instead of feeding a
        # failed/empty output to the next agent, and that failed RunResult
        # becomes the final result.
        self.step_results = []
        current_task = initial_task
        result: Optional[RunResult] = None

        for step_index, agent in enumerate(self.agents):
            result = agent.execute(current_task)
            self.step_results.append(result)

            if not result.success:
                break  # Halt the chain; don't propagate a failed step forward.

            is_last_step = step_index == len(self.agents) - 1
            if is_last_step:
                break

            next_agent = self.agents[step_index + 1]
            # The hand-off: this step's textual output becomes the next
            # step's input description. A fresh Task is built (rather than
            # mutating current_task) so each step has its own id/description
            # while still sharing the original input_data.
            current_task = Task(
                description=result.output or "",
                assigned_agent=next_agent.name,
                input_data=current_task.input_data,
            )

        # `result` is guaranteed to be set: self.agents is non-empty (checked
        # in __init__), so the loop always executes at least one iteration.
        assert result is not None
        return result
