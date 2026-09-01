from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from agent_framework.agents import BaseAgent, LLMAgent, execute_agent
from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.registry import AgentRegistry
from agent_framework.orchestration.sequential import (
    build_agent_kwargs,
    load_agent_config,
)


# Fans one task out to several agents at once, then fans their answers back in
# through a single synthesizer.
#
#                       initial_task
#                            |
#            +---------------+---------------+      <- fan-out
#            v               v               v
#        worker[0]       worker[1]       worker[2]   (concurrent)
#            |               |               |
#            +---------------+---------------+      <- fan-in
#                            v
#                    synthesizer_agent
#                            |
#                        RunResult
#
# Unlike SequentialOrchestrator, the workers never see each other's output:
# they all answer the same question independently. Only the synthesizer sees
# everything, which is what makes the concurrency safe in the first place.
class ParallelOrchestrator:
    def __init__(
        self,
        worker_agents: Sequence[BaseAgent],
        synthesizer_agent: BaseAgent,
        max_workers: int = 4,
    ):
        if not worker_agents:
            raise ValueError("ParallelOrchestrator requires at least one worker agent")
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")

        self.worker_agents: list[BaseAgent] = list(worker_agents)
        self.synthesizer_agent = synthesizer_agent
        self.max_workers = max_workers

        # Every worker's RunResult in worker order, followed by the
        # synthesizer's, so callers can inspect the whole fan-out, not just the
        # combined answer.
        self.step_results: list[RunResult] = []

    @classmethod
    def from_registry(
        cls,
        worker_names: Sequence[str],
        synthesizer_name: str,
        registry: AgentRegistry,
        agent_kwargs: Optional[dict[str, dict]] = None,
        max_workers: int = 4,
    ) -> "ParallelOrchestrator":
        agent_kwargs = agent_kwargs or {}

        def instantiate(name: str, default_prompt: str) -> BaseAgent:
            agent_cls = registry.get(name)
            kwargs = dict(agent_kwargs.get(name, {}))
            kwargs.setdefault("name", name)
            kwargs.setdefault("role", name)
            kwargs.setdefault("system_prompt", default_prompt)
            return agent_cls(**kwargs)

        workers = [
            instantiate(name, f"You are the '{name}' worker in a parallel pipeline.")
            for name in worker_names
        ]
        synthesizer = instantiate(
            synthesizer_name,
            f"You are the '{synthesizer_name}' step; combine the workers' findings.",
        )
        return cls(workers, synthesizer, max_workers=max_workers)

    @staticmethod
    def _run_worker(
        agent: BaseAgent,
        initial_task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        """Execute one worker on its own copy of the request.

        This is the function that actually runs on a pool thread. Each worker
        gets a fresh Task rather than the shared one: Task carries a unique
        id, and giving every branch its own means a RunResult can be traced
        back to the branch that produced it. `input_data` is passed along so
        shared context reaches every worker.

        The blackboard, unlike the Task, is deliberately NOT copied: the
        point of it is that all branches see one shared surface. That is
        also why SharedWorkflowMemory locks its writes - this function
        runs on a pool thread, several copies of it at once.
        """
        worker_task = Task(
            description=initial_task.description,
            assigned_agent=agent.name,
            input_data=initial_task.input_data,
        )
        try:
            return execute_agent(agent, worker_task, shared_memory)
        except Exception as exc:  # noqa: BLE001 - a crashed worker is just a failed branch
            return RunResult(
                task_id=worker_task.id,
                success=False,
                error=f"{agent.name} raised {type(exc).__name__}: {exc}",
            )

    def _build_synthesis_prompt(
        self, initial_task: Task, successes: list[tuple[BaseAgent, RunResult]]
    ) -> str:
        """Flatten the successful branches into one labelled prompt.

        Labelling each block with its worker matters for the same reason it
        does in the sequential pipeline: without it the synthesizer sees one
        undifferentiated wall of text and can't tell which claim came from
        which specialist.
        """
        blocks = [
            f"--- Worker {position} ({agent.name}) ---\n{result.output or ''}"
            for position, (agent, result) in enumerate(successes, start=1)
        ]
        return (
            f"Original Request:\n{initial_task.description}\n\n"
            f"Worker Findings:\n" + "\n\n".join(blocks)
        )

    def run(
        self,
        initial_task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        self.step_results = []

        # --- Fan-out -----------------------------------------------------
        #
        # ThreadPoolExecutor keeps a pool of at most `max_workers` OS threads
        # and a queue of pending work. `submit` hands it a callable plus its
        # arguments and returns immediately with a Future
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._run_worker, agent, initial_task, shared_memory)
                for agent in self.worker_agents
            ]

            # --- Fan-in --------------------------------------------------
            #
            # `future.result()` blocks the *calling* (main) thread until that
            # particular job is done, then returns whatever the callable
            # returned - or re-raises whatever it raised, in this thread.
            # _run_worker already converts agent errors into failed RunResults,
            # so in practice this just hands back a RunResult.
            results = [future.result() for future in futures]

        self.step_results.extend(results)

        successes = [
            (agent, result)
            for agent, result in zip(self.worker_agents, results)
            if result.success
        ]

        # Short-circuit: with nothing to synthesize, calling the synthesizer
        # would just burn a generation on an empty prompt. Report the failure
        # against the original task instead. The individual worker failures are
        # already in self.step_results for diagnosis.
        if not successes:
            errors = "; ".join(
                f"{agent.name}: {result.error}"
                for agent, result in zip(self.worker_agents, results)
            )
            return RunResult(
                task_id=initial_task.id,
                success=False,
                error=f"All {len(self.worker_agents)} workers failed - {errors}",
            )

        synthesis_task = Task(
            description=self._build_synthesis_prompt(initial_task, successes),
            assigned_agent=self.synthesizer_agent.name,
            input_data=initial_task.input_data,
        )
        synthesis_result = execute_agent(
            self.synthesizer_agent, synthesis_task, shared_memory
        )
        self.step_results.append(synthesis_result)
        return synthesis_result


def create_parallel_pipeline(
    config_path: str,
    worker_keys: list[str],
    synthesizer_key: str,
    max_workers: int = 4,
) -> ParallelOrchestrator:
    """Read the agent definitions off disk and wire up a fan-out pipeline.

    Same config file and same shape as create_sequential_pipeline - the only
    difference is that `worker_keys` run concurrently instead of in a chain,
    and `synthesizer_key` names the agent that merges their answers.
    """
    if not worker_keys:
        raise ValueError("create_parallel_pipeline requires at least one worker key")

    config = load_agent_config(config_path)

    registry = AgentRegistry()
    agent_kwargs: dict[str, dict] = {}

    for key in [*worker_keys, synthesizer_key]:
        if key not in config:
            raise KeyError(f"No agent configured under key '{key}' in {config_path}")

        entry = config[key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Config entry for '{key}' in {config_path} must be an object, "
                f"got {type(entry).__name__}"
            )

        # Real LLM-backed agents. build_llm_client caches by model settings, so
        # workers configured identically share one loaded copy of the weights
        # rather than each paying the load cost.
        registry.register(key, LLMAgent)
        agent_kwargs[key] = build_agent_kwargs(entry)

    return ParallelOrchestrator.from_registry(
        worker_names=worker_keys,
        synthesizer_name=synthesizer_key,
        registry=registry,
        agent_kwargs=agent_kwargs,
        max_workers=max_workers,
    )
