from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Optional, Sequence, Union

from agent_framework.agents import BaseAgent, LLMAgent
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.registry import AgentRegistry
from agent_framework.orchestration.sequential import (
    AGENT_PARAMS,
    build_llm_client,
    load_agent_config,
)

logger = logging.getLogger(__name__)

# Keys a leader might plausibly use for "who does this" and "what they do".
# Small models paraphrase field names constantly, so accepting the obvious
# synonyms is far cheaper than re-prompting until the JSON is exactly right.
_AGENT_FIELDS = ("agent", "specialist", "worker", "role", "assignee", "name")
_TASK_FIELDS = ("task", "subtask", "description", "instruction", "objective", "goal")


# One subtask: which specialist should do it, and what they were asked to do.
class Subtask:
    def __init__(self, worker_key: str, description: str):
        self.worker_key = worker_key
        self.description = description

    def __repr__(self) -> str:  # helpful when printing a plan in a test
        return f"Subtask(worker_key={self.worker_key!r}, description={self.description!r})"


# A manager agent that plans the work, hands pieces to specialists, then writes
# up what came back.
#
#                        initial_task
#                             |
#                             v
#                     leader (decompose)          <- step 1: PLAN
#                             |
#              "researcher does A, coder does B"
#                             |
#            +----------------+----------------+  <- step 2: DELEGATE
#            v                                 v
#      workers["researcher"]            workers["coder"]
#            |                                 |
#            +----------------+----------------+  <- step 3: AGGREGATE
#                             v
#                      leader (synthesize)
#                             |
#                         RunResult
class HierarchicalOrchestrator:
    def __init__(self, leader_agent: BaseAgent, worker_agents: dict[str, BaseAgent]):
        if not worker_agents:
            raise ValueError("HierarchicalOrchestrator requires at least one worker agent")

        self.leader_agent = leader_agent
        self.worker_agents = dict(worker_agents)

        self.step_results: list[RunResult] = []

        # Populated by run() for debugging: the raw planning result and the
        # parsed plan it produced.
        self.decomposition_result: Optional[RunResult] = None
        self.plan: list[Subtask] = []

    @classmethod
    def from_registry(
        cls,
        leader_name: str,
        worker_names: Union[dict[str, str], Sequence[str]],
        registry: AgentRegistry,
        agent_kwargs: Optional[dict[str, dict]] = None,
    ) -> "HierarchicalOrchestrator":
        """Build a team from registered agent classes.
        `worker_names` is either a list of registry names
        """
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
        leader = instantiate(
            leader_name,
            f"You are '{leader_name}', the team leader; plan work and summarize results.",
        )
        return cls(leader, workers)

    # --- Step 1: decomposition -------------------------------------------

    def _roster(self) -> str:
        """The menu of specialists shown to the leader.

        The leader can only delegate to names it knows exist, so the roster is
        generated from self.worker_agents rather than written by hand - add a
        specialist to the team and the prompt updates itself.
        """
        return "\n".join(
            f"- {key}: {agent.role} ({agent.name})"
            for key, agent in self.worker_agents.items()
        )

    def _build_decomposition_task(self, initial_task: Task) -> Task:
        """Ask the leader to split the request across the available specialists.

        JSON is requested because it's unambiguous to parse, but the parser
        below never assumes it arrives - see _parse_plan for why.
        """
        keys = ", ".join(self.worker_agents)
        description = (
            f"Request:\n{initial_task.description}\n\n"
            f"Available specialists:\n{self._roster()}\n\n"
            "Break the request into subtasks and assign each one to the single "
            "best-suited specialist. Use only the specialist keys listed above "
            f"({keys}). Skip any specialist the request does not need.\n\n"
            "Reply with ONLY a JSON array, no other text:\n"
            '[{"agent": "<specialist key>", "task": "<what they should do>"}]'
        )
        return Task(
            description=description,
            assigned_agent=self.leader_agent.name,
            input_data=initial_task.input_data,
        )

    # --- Step 2: parsing and routing --------------------------------------

    def _alias_lookup(self) -> dict[str, str]:
        """Map every plausible spelling of a specialist onto its canonical key.

        A leader asked for "researcher" may answer "Researcher", "research", or
        "ResearchAgent" - all of which mean the same team member. Resolving
        aliases here keeps the routing tolerant without letting an unknown name
        silently become a real assignment.
        """
        aliases: dict[str, str] = {}
        for key, agent in self.worker_agents.items():
            for alias in (key, agent.name, agent.role):
                aliases[alias.strip().lower()] = key
        return aliases

    def _resolve_worker(self, raw_name: str) -> Optional[str]:
        """Turn whatever the leader wrote into a real worker key, or None."""
        candidate = (raw_name or "").strip().lower()
        if not candidate:
            return None

        aliases = self._alias_lookup()
        if candidate in aliases:
            return candidate if candidate in self.worker_agents else aliases[candidate]

        # Substring match catches "the researcher agent" and "coder (python)".
        # Only accepted when exactly one specialist matches, so an ambiguous
        # name is dropped rather than routed to an arbitrary team member.
        hits = {key for alias, key in aliases.items() if alias in candidate}
        if len(hits) == 1:
            return hits.pop()
        return None

    @staticmethod
    def _first_value(entry: dict, fields: Iterable[str]) -> Optional[str]:
        for field in fields:
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _extract_json(text: str):
        """Pull a JSON array/object out of a model reply.

        Instruct models wrap JSON in prose ("Sure! Here's the plan:") or in a
        ```json fence even when told not to, so the payload is located by
        slicing between the outermost brackets rather than parsing the whole
        reply.
        """
        for opener, closer in (("[", "]"), ("{", "}")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    def _parse_json_plan(self, text: str) -> list[Subtask]:
        payload = self._extract_json(text)

        # Accept both a bare array and the {"subtasks": [...]} wrapper models
        # like to add unprompted.
        if isinstance(payload, dict):
            for field in ("subtasks", "tasks", "plan", "assignments"):
                if isinstance(payload.get(field), list):
                    payload = payload[field]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            return []

        plan: list[Subtask] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            worker_key = self._resolve_worker(self._first_value(entry, _AGENT_FIELDS) or "")
            description = self._first_value(entry, _TASK_FIELDS)
            # Both halves are required: an assignment with no owner can't be
            # routed, and an owner with no instruction has nothing to do.
            if worker_key and description:
                plan.append(Subtask(worker_key, description))
        return plan

    def _parse_labelled_plan(self, text: str) -> list[Subtask]:
        """Fallback for when the leader ignored the JSON instruction.

        Handles the common prose shapes: "researcher: do X", "1. Coder - do Y",
        "**analyst**: do Z". Anything whose label isn't a known specialist is
        skipped, so ordinary sentences don't become subtasks.
        """
        plan: list[Subtask] = []
        pattern = re.compile(r"^\s*(?:[-*\d.\)\s]*)?\**([\w /-]{2,40}?)\**\s*[:\-]\s+(.+)$")

        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            worker_key = self._resolve_worker(match.group(1))
            description = match.group(2).strip()
            if worker_key and description:
                plan.append(Subtask(worker_key, description))
        return plan

    def _parse_plan(self, text: str, initial_task: Task) -> list[Subtask]:
        """Turn the leader's reply into a routable plan, always returning one.

        Three tiers, cheapest and most reliable first:
          1. JSON, which is what we asked for.
          2. Labelled prose, which is what a small model often gives instead.
          3. Fan out the original request to every specialist.
        """
        plan = self._parse_json_plan(text)
        if plan:
            return plan

        plan = self._parse_labelled_plan(text)
        if plan:
            logger.info("Leader plan was not JSON; recovered %d subtasks from prose", len(plan))
            return plan

        logger.warning(
            "Could not parse a plan from the leader; delegating the original "
            "request to all %d specialists",
            len(self.worker_agents),
        )
        return [
            Subtask(key, initial_task.description) for key in self.worker_agents
        ]

    # --- Step 3: aggregation ----------------------------------------------

    def _build_aggregation_task(
        self, initial_task: Task, completed: list[tuple[Subtask, RunResult]]
    ) -> Task:
        """Hand the leader back everything its team produced.

        Each block records the subtask *as assigned* alongside the answer, so
        the leader can tell whether a specialist actually did what it was asked
        - information it needs to write an honest summary and which a bare
        concatenation of outputs would throw away.
        """
        blocks = []
        for position, (subtask, result) in enumerate(completed, start=1):
            agent = self.worker_agents[subtask.worker_key]
            blocks.append(
                f"--- Subtask {position}: {subtask.worker_key} ({agent.name}) ---\n"
                f"Assigned: {subtask.description}\n"
                f"Result:\n{result.output or ''}"
            )

        description = (
            f"Original Request:\n{initial_task.description}\n\n"
            f"Your team's completed work:\n" + "\n\n".join(blocks) + "\n\n"
            "Write the final unified response to the original request using "
            "your team's work above."
        )
        return Task(
            description=description,
            assigned_agent=self.leader_agent.name,
            input_data=initial_task.input_data,
        )

    # --- The run loop ------------------------------------------------------

    def run(self, initial_task: Task) -> RunResult:
        self.step_results = []
        self.plan = []
        self.decomposition_result = None

        # Step 1: PLAN. The leader decides who does what.
        decomposition = self.leader_agent.execute(
            self._build_decomposition_task(initial_task)
        )
        self.decomposition_result = decomposition
        if not decomposition.success:
            # No plan means nothing to delegate, so stop here rather than
            # guessing at an assignment the leader never made.
            return decomposition

        self.plan = self._parse_plan(decomposition.output or "", initial_task)

        # Step 2: DELEGATE. Each subtask goes to exactly the specialist named in
        # the plan. Subtasks run in plan order; unlike the parallel orchestrator
        # this is a sequential walk, because a leader may legitimately order
        # work so that later subtasks build on earlier ones.
        completed: list[tuple[Subtask, RunResult]] = []
        for subtask in self.plan:
            worker = self.worker_agents[subtask.worker_key]
            worker_task = Task(
                description=subtask.description,
                assigned_agent=worker.name,
                input_data=initial_task.input_data,
            )
            try:
                result = worker.execute(worker_task)
            except Exception as exc:  # noqa: BLE001 - one bad specialist isn't fatal
                result = RunResult(
                    task_id=worker_task.id,
                    success=False,
                    error=f"{worker.name} raised {type(exc).__name__}: {exc}",
                )

            self.step_results.append(result)
            if result.success:
                completed.append((subtask, result))

        # Same rule as the parallel orchestrator: with nothing completed there
        # is nothing to aggregate, so report the failure instead of asking the
        # leader to summarize an empty team.
        if not completed:
            errors = "; ".join(
                f"{subtask.worker_key}: {result.error}"
                for subtask, result in zip(self.plan, self.step_results)
            )
            return RunResult(
                task_id=initial_task.id,
                success=False,
                error=f"All {len(self.plan)} delegated subtasks failed - {errors}",
            )

        # Steps 3 & 4: AGGREGATE and LOG. The leader's summary is appended last,
        # so step_results reads as [worker, worker, ..., leader].
        final_result = self.leader_agent.execute(
            self._build_aggregation_task(initial_task, completed)
        )
        self.step_results.append(final_result)
        return final_result


def create_hierarchical_pipeline(
    config_path: str, leader_key: str, worker_keys: list[str]
) -> HierarchicalOrchestrator:
    """Read the agent definitions off disk and wire up a manager-worker team.

    Same config file and shape as the sequential and parallel factories; the
    difference is purely structural - `leader_key` names the agent that plans
    and summarizes, `worker_keys` name the specialists it may delegate to.
    """
    if not worker_keys:
        raise ValueError("create_hierarchical_pipeline requires at least one worker key")

    config = load_agent_config(config_path)

    registry = AgentRegistry()
    agent_kwargs: dict[str, dict] = {}

    for key in [leader_key, *worker_keys]:
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

    return HierarchicalOrchestrator.from_registry(
        leader_name=leader_key,
        worker_names=list(worker_keys),
        registry=registry,
        agent_kwargs=agent_kwargs,
    )
