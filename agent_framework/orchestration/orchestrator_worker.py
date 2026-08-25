from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

from agent_framework.agents import BaseAgent, LLMAgent
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.hierarchical import Subtask, SubtaskPlanParser
from agent_framework.orchestration.registry import AgentRegistry
from agent_framework.orchestration.sequential import (
    AGENT_PARAMS,
    build_llm_client,
    load_agent_config,
)

logger = logging.getLogger(__name__)

# The signal the orchestrator emits when it judges the request fully answered.
DEFAULT_COMPLETION_PHRASE = "WORKFLOW_COMPLETE"


class OrchestratorWorkerPipeline(SubtaskPlanParser):
    def __init__(
        self,
        orchestrator_agent: BaseAgent,
        worker_agents: dict[str, BaseAgent],
        max_iterations: int = 3,
        completion_phrase: str = DEFAULT_COMPLETION_PHRASE,
    ):
        if not worker_agents:
            raise ValueError("OrchestratorWorkerPipeline requires at least one worker agent")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not completion_phrase.strip():
            raise ValueError("completion_phrase must be a non-empty string")

        self.orchestrator_agent = orchestrator_agent
        self.worker_agents = dict(worker_agents)
        self.max_iterations = max_iterations
        self.completion_phrase = completion_phrase.strip()

        self.iteration_history: list[dict] = []
        self.step_results: list[RunResult] = []

        self.completed_naturally: bool = False
        self.iterations_used: int = 0

    @classmethod
    def from_registry(
        cls,
        orchestrator_name: str,
        worker_names: Union[dict[str, str], Sequence[str]],
        registry: AgentRegistry,
        agent_kwargs: Optional[dict[str, dict]] = None,
        max_iterations: int = 3,
        completion_phrase: str = DEFAULT_COMPLETION_PHRASE,
    ) -> "OrchestratorWorkerPipeline":
        # Build the supervisor and its specialists from registered classes.
        agent_kwargs = agent_kwargs or {}

        def instantiate(name: str, default_prompt: str) -> BaseAgent:
            agent_cls = registry.get(name)
            kwargs = dict(agent_kwargs.get(name, {}))
            kwargs.setdefault("name", name)
            kwargs.setdefault("role", name)
            kwargs.setdefault("system_prompt", default_prompt)
            return agent_cls(**kwargs)

        if isinstance(worker_names, dict):
            key_to_name = dict(worker_names)
        else:
            key_to_name = {name: name for name in worker_names}

        workers = {
            key: instantiate(name, f"You are the '{key}' specialist on a team.")
            for key, name in key_to_name.items()
        }
        orchestrator = instantiate(
            orchestrator_name,
            f"You are '{orchestrator_name}', the orchestrator; assign work, review "
            f"what comes back, and reply {completion_phrase} once the request is met.",
        )
        return cls(
            orchestrator,
            workers,
            max_iterations=max_iterations,
            completion_phrase=completion_phrase,
        )

    # --- Step A: reviewing the prompt ---

    def _format_history(self) -> str:
        """Render every completed round the way the orchestrator needs to read it.
        """
        blocks: list[str] = []
        for round_record in self.iteration_history:
            round_number = round_record["iteration"] + 1
            for position, entry in enumerate(round_record["outputs"], start=1):
                worker = self.worker_agents[entry["agent"]]
                body = (
                    f"Result:\n{entry['output']}"
                    if entry["success"]
                    else f"FAILED: {entry['error']}"
                )
                blocks.append(
                    f"--- Round {round_number}, subtask {position}: "
                    f"{entry['agent']} ({worker.name}) ---\n"
                    f"Assigned: {entry['task']}\n{body}"
                )
        return "\n\n".join(blocks)

    def _build_review_task(self, initial_task: Task) -> Task:
        """Compose the orchestrator's turn for this iteration.
        """
        keys = ", ".join(self.worker_agents)
        sections = [
            f"Original request:\n{initial_task.description}",
            f"Available specialists:\n{self._roster()}",
        ]

        history = self._format_history()
        if history:
            sections.append(f"Work completed so far:\n{history}")
            sections.append(
                "Review the work above against the original request. If every "
                "requirement is now fully met, reply with "
                f"{self.completion_phrase} on its own first line, followed by "
                "the complete final answer.\n\n"
                "If anything is missing, wrong, or shallow, do NOT say "
                f"{self.completion_phrase}. Instead reply with ONLY a JSON array "
                "of the remaining subtasks, each assigned to the single "
                f"best-suited specialist ({keys}):\n"
                '[{"agent": "<specialist key>", "task": "<what they should do>"}]\n'
                "Only ask for work that is still outstanding - never repeat a "
                "subtask that has already been done well."
            )
        else:
            sections.append(
                "Break the request into subtasks and assign each one to the "
                f"single best-suited specialist. Use only these keys: {keys}. "
                "Skip any specialist the request does not need.\n\n"
                "Reply with ONLY a JSON array, no other text:\n"
                '[{"agent": "<specialist key>", "task": "<what they should do>"}]'
            )

        return Task(
            description="\n\n".join(sections),
            assigned_agent=self.orchestrator_agent.name,
            input_data=initial_task.input_data,
        )

    # --- Step B: check for termination ---

    def _signals_completion(self, text: str) -> bool:
        """True when the orchestrator declared the request satisfied."""
        return self.completion_phrase.lower() in (text or "").lower()

    def _strip_completion_phrase(self, text: str) -> str:
        """Return the synthesis with the control token removed."""
        cleaned = (text or "")
        lowered = cleaned.lower()
        marker = self.completion_phrase.lower()

        start = lowered.find(marker)
        while start != -1:
            cleaned = cleaned[:start] + cleaned[start + len(marker) :]
            lowered = cleaned.lower()
            start = lowered.find(marker)

        return cleaned.strip().lstrip(":-").strip()

    # --- Step C: use the workers ---

    def _build_worker_task(self, subtask: Subtask, initial_task: Task) -> Task:
        """Compose one specialist's Task: its assignment plus what exists so far.
        """
        worker = self.worker_agents[subtask.worker_key]

        history = self._format_history()
        if not history:
            description = (
                f"Original request:\n{initial_task.description}\n\n"
                f"Your assignment:\n{subtask.description}"
            )
        else:
            description = (
                f"Original request:\n{initial_task.description}\n\n"
                f"Work your team has already produced:\n{history}\n\n"
                f"Your assignment as the {worker.role} specialist:\n"
                f"{subtask.description}"
            )

        return Task(
            description=description,
            assigned_agent=worker.name,
            input_data=initial_task.input_data,
        )

    def _dispatch(self, subtasks: list[Subtask], initial_task: Task) -> list[dict]:
        """Run one round's subtasks and return a record of each outcome.
        """
        outputs: list[dict] = []
        for subtask in subtasks:
            worker = self.worker_agents[subtask.worker_key]
            worker_task = self._build_worker_task(subtask, initial_task)
            try:
                result = worker.execute(worker_task)
            except Exception as exc:  # noqa: BLE001 - one bad specialist isn't fatal
                result = RunResult(
                    task_id=worker_task.id,
                    success=False,
                    error=f"{worker.name} raised {type(exc).__name__}: {exc}",
                )

            self.step_results.append(result)
            outputs.append(
                {
                    "agent": subtask.worker_key,
                    "task": subtask.description,
                    "success": result.success,
                    "output": result.output or "",
                    "error": result.error,
                }
            )
        return outputs

    # --- Forced landing ---

    def _build_consolidation_task(self, initial_task: Task) -> Task:
        """The prompt used when the iteration budget runs out mid-review."""
        description = (
            f"Original request:\n{initial_task.description}\n\n"
            f"All work completed by your team:\n{self._format_history()}\n\n"
            f"The iteration budget of {self.max_iterations} round(s) is now "
            "exhausted, so no further subtasks can be assigned. Using only the "
            "work above, write the best possible final answer to the original "
            "request now. Note briefly at the end anything that remains "
            "incomplete. Do not ask for more work."
        )
        return Task(
            description=description,
            assigned_agent=self.orchestrator_agent.name,
            input_data=initial_task.input_data,
        )

    # --- The run loop ---

    def run(self, initial_task: Task) -> RunResult:
        self.iteration_history = []
        self.step_results = []
        self.completed_naturally = False
        self.iterations_used = 0

        for iteration in range(self.max_iterations):
            self.iterations_used = iteration + 1

            review_task = self._build_review_task(initial_task)
            review = self.orchestrator_agent.execute(review_task)
            self.step_results.append(review)

            if not review.success:
                return review

            raw = review.output or ""
            subtasks = self._parse_plan(raw, initial_task)

            signalled = self._signals_completion(raw)
            if signalled or not subtasks:
                self.iteration_history.append(
                    {
                        "iteration": iteration,
                        "prompt": review_task.description,
                        "tasks": [],
                        "outputs": [],
                        "evaluation": raw,
                        "completed": True,
                    }
                )
                self.completed_naturally = True
                if not signalled:
                    logger.info(
                        "Orchestrator produced no dispatchable subtask on round %d; "
                        "treating its reply as the final synthesis",
                        iteration + 1,
                    )

                synthesis = self._strip_completion_phrase(raw) if signalled else raw.strip()
                if not synthesis:
                    synthesis = raw.strip()

                return RunResult(
                    task_id=initial_task.id,
                    success=True,
                    output=synthesis,
                )
            
            outputs = self._dispatch(subtasks, initial_task)
            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "prompt": review_task.description,
                    "tasks": [
                        {"agent": subtask.worker_key, "task": subtask.description}
                        for subtask in subtasks
                    ],
                    "outputs": outputs,
                    "evaluation": raw,
                    "completed": False,
                }
            )

        if not any(
            entry["success"]
            for record in self.iteration_history
            for entry in record["outputs"]
        ):
            return RunResult(
                task_id=initial_task.id,
                success=False,
                error=(
                    f"No subtask succeeded across {self.max_iterations} iteration(s); "
                    "nothing to consolidate"
                ),
            )

        logger.info(
            "Reached max_iterations (%d) without a %s signal; forcing consolidation",
            self.max_iterations,
            self.completion_phrase,
        )
        final_result = self.orchestrator_agent.execute(
            self._build_consolidation_task(initial_task)
        )
        self.step_results.append(final_result)

        if not final_result.success:
            return RunResult(
                task_id=initial_task.id,
                success=False,
                error=(
                    f"Forced consolidation after {self.max_iterations} iteration(s) "
                    f"failed: {final_result.error}"
                ),
            )

        return RunResult(
            task_id=initial_task.id,
            success=True,
            output=self._strip_completion_phrase(final_result.output or ""),
        )

    # --- Plan parsing ---

    def _parse_plan(self, text: str, initial_task: Task) -> list[Subtask]:
        """Turn one orchestrator turn into dispatchable subtasks.
        """
        plan = self._parse_json_plan(text)
        if plan:
            return plan

        plan = self._parse_labelled_plan(text)
        if plan:
            logger.info(
                "Orchestrator reply was not JSON; recovered %d subtask(s) from prose",
                len(plan),
            )
            return plan

        keys = self._parse_key_list(text)
        if keys:
            logger.info(
                "Orchestrator named %d specialist(s) without instructions; "
                "assigning each the original request",
                len(keys),
            )
            return [Subtask(key, initial_task.description) for key in keys]

        return []


def create_orchestrator_worker_pipeline(
    config_path: str,
    orchestrator_key: str,
    worker_keys: list[str],
    max_iterations: int = 3,
) -> OrchestratorWorkerPipeline:
    """Read the agent definitions off disk and wire up an iterative team.
    """
    if not worker_keys:
        raise ValueError(
            "create_orchestrator_worker_pipeline requires at least one worker key"
        )

    config = load_agent_config(config_path)

    registry = AgentRegistry()
    agent_kwargs: dict[str, dict] = {}

    for key in [orchestrator_key, *worker_keys]:
        if key not in config:
            raise KeyError(f"No agent configured under key '{key}' in {config_path}")

        entry = config[key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Config entry for '{key}' in {config_path} must be an object, "
                f"got {type(entry).__name__}"
            )

        registry.register(key, LLMAgent)
        kwargs = {param: entry[param] for param in AGENT_PARAMS if param in entry}
        kwargs["llm_client"] = build_llm_client(entry)
        agent_kwargs[key] = kwargs

    return OrchestratorWorkerPipeline.from_registry(
        orchestrator_name=orchestrator_key,
        worker_names=list(worker_keys),
        registry=registry,
        agent_kwargs=agent_kwargs,
        max_iterations=max_iterations,
    )

