from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from agent_framework.agents import BaseAgent, LLMAgent, MockAgent
from agent_framework.core.llm import LLMClient
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.registry import AgentRegistry

# The three fields BaseAgent's constructor takes
AGENT_PARAMS = ("name", "role", "system_prompt")

# Local model settings a config entry may carry, forwarded to LLMClient.
LLM_PARAMS = ("model", "temperature", "max_tokens", "device", "dtype")

# Clients are stateless between calls, so agents whose settings match can share
# one rather than constructing a duplicate.
_CLIENT_CACHE: dict[tuple, LLMClient] = {}


def build_llm_client(entry: dict) -> LLMClient:
    """Build (or reuse) the LLMClient for one agent from its config entry.

    Any of LLM_PARAMS the entry omits falls back to the LLMClient default.
    """
    settings = {param: entry[param] for param in LLM_PARAMS if param in entry}

    cache_key = tuple(sorted(settings.items()))
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = LLMClient(**settings)
    return _CLIENT_CACHE[cache_key]


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


def load_agent_config(config_path: str) -> dict[str, dict]:
    """Read the JSON agent config and hand back a key -> params mapping.

    Shared by every orchestration factory (see also router.py), so all of them
    read the same file format:

        {
          "researcher": {
            "name": "ResearchAgent",
            "role": "research",
            "system_prompt": "Gather raw findings on the given topic."
          },
          ...
        }
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")

    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Agent config {config_path} must be a JSON object mapping "
            f"agent keys to parameter objects, got {type(config).__name__}"
        )
    return config


def create_sequential_pipeline(
    config_path: str, agent_keys: list[str]
) -> SequentialOrchestrator:
    """Reads the agent definitions off disk, instantiates an LLMAgent per key
    in the order given by `agent_keys`, and wires them into an orchestrator.
    """
    if not agent_keys:
        raise ValueError("create_sequential_pipeline requires at least one agent key")

    config = load_agent_config(config_path)

    registry = AgentRegistry()
    agent_kwargs: dict[str, dict] = {}

    for key in agent_keys:
        if key not in config:
            raise KeyError(f"No agent configured under key '{key}' in {config_path}")

        entry = config[key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Config entry for '{key}' in {config_path} must be an object, "
                f"got {type(entry).__name__}"
            )

        # Previous behaviour, kept for reference:
        # registry.register(key, MockAgent)
        # agent_kwargs[key] = {k: entry[k] for k in AGENT_PARAMS if k in entry}

        # Real LLM-backed agents. Each gets its own client so per-agent model
        # settings (model, temperature, max_tokens) in the config are honoured.
        registry.register(key, LLMAgent)
        kwargs = {k: entry[k] for k in AGENT_PARAMS if k in entry}
        kwargs["llm_client"] = build_llm_client(entry)
        agent_kwargs[key] = kwargs

    return SequentialOrchestrator.from_registry(
        agent_names=agent_keys,
        registry=registry,
        agent_kwargs=agent_kwargs,
    )
