from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from agent_framework.agents import BaseAgent, LLMAgent, MockAgent, execute_agent
from agent_framework.core.llm import LLMClient
from agent_framework.core.memory import AgentMemory, SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.registry import AgentRegistry
from agent_framework.tools.base import ToolPermission
from agent_framework.tools.registry import ToolRegistry, default_registry

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


# Memory settings a config entry may carry, forwarded to LLMAgent.
#
#   "memory": true        give this agent its own AgentMemory, so it remembers
#                         its own turns between execute() calls
#   "history_window": n   how many of those turns get replayed into the prompt
#   "context_keys": [...] which SHARED artifacts it wants to see (default: all)
#   "output_key": "name"  where it publishes its own output on the blackboard
MEMORY_PARAMS = ("history_window", "context_keys", "output_key")

# Tool settings every config entry carries:
#
#   "tools": ["calculator"]       which registered tools this agent is assigned
#   "permissions": ["read_only"]  which permission tiers it holds
#
# Both are needed for a tool to actually be usable, and they are separate keys
# on purpose: assigning `file_write` without granting write is a config that
# says "this agent should eventually write files" and produces an agent that
# currently cannot. That reads as an oversight, which is the intent - the
# alternative, inferring the permission from the tool list, would mean naming a
# tool silently grants the right to use it.
#
# These two keys are the ONLY source of an agent's grants. They are read here,
# in Python, before the agent exists, and become immutable on the agent. No
# orchestrator, prompt or model reply can add to them afterwards - an entry
# that omits them gets an agent holding nothing, never one holding everything.
TOOL_PARAMS = ("tools", "permissions")


def _read_grant(entry: dict, key: str) -> list[str]:
    """One of TOOL_PARAMS off a config entry, as a list of strings.

    A missing key is an empty grant. Anything else that is not a list of
    strings is refused outright rather than coerced: the permission model
    should fail on a malformed config, not quietly widen or narrow around it.
    """
    values = entry.get(key, [])
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(
            f"config entry {entry.get('name', '?')!r}: {key!r} must be a list "
            f"of strings, got {type(values).__name__}"
        )
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                f"config entry {entry.get('name', '?')!r}: {key!r} entries must "
                f"be strings, got {type(value).__name__}"
            )
    return list(values)


def build_agent_kwargs(
    entry: dict,
    tool_registry: Optional[ToolRegistry] = None,
    llm_client: Optional[LLMClient] = None,
) -> dict:
    """Turn one config entry into the keyword arguments for an LLMAgent.

    Shared by every orchestration factory so a setting added to the config
    format reaches all of them at once, rather than five near-identical dict
    comprehensions drifting apart.

    Each agent gets a *fresh* AgentMemory when it asks for one: agent-local
    memory is private by definition, so two agents sharing one instance would
    be a bug that shows up as one agent quoting another's turns.

    The tool registry is the opposite case - it is shared deliberately, since
    every agent in a run should resolve `"file_write"` to the same tool over
    the same sandbox. `tool_registry` is only consulted for entries that
    actually declare tools, so a tool-free config never builds one.

    `llm_client` replaces the client the entry's model settings would build.
    Everything else - name, prompt, memory settings and, above all, the
    grants - still comes from the entry, which is what lets a demo or a test
    run a config-defined agent against a scripted client without loading a
    model or hand-copying its permissions.
    """
    kwargs = {param: entry[param] for param in AGENT_PARAMS if param in entry}
    kwargs.update({param: entry[param] for param in MEMORY_PARAMS if param in entry})

    if entry.get("memory"):
        kwargs["memory"] = AgentMemory()

    # Grants come straight off the entry, always, so every agent built from
    # config carries an explicit (possibly empty) assignment and tier. The
    # permission strings are validated against the known tiers here, so a
    # typo in agents.json fails at load time instead of leaving an agent
    # quietly holding nothing.
    allowed_tools = _read_grant(entry, "tools")
    permissions = _read_grant(entry, "permissions")
    ToolPermission.coerce_set(permissions)

    kwargs["allowed_tools"] = allowed_tools
    kwargs["permissions"] = permissions
    if allowed_tools:
        # The registry is only attached to agents that were assigned
        # something, so a tool-free config never builds the built-in tools.
        kwargs["tool_registry"] = tool_registry or default_registry()

    kwargs["llm_client"] = llm_client if llm_client is not None else build_llm_client(entry)
    return kwargs


# Runs a fixed lineup of agents one after another. Every step sees the original
# objective plus the output of *all* the steps before it, not just the last one.
class SequentialOrchestrator:
    def __init__(self, agents: Sequence[BaseAgent]):
        if not agents:
            raise ValueError("SequentialOrchestrator requires at least one agent")
        self.agents: list[BaseAgent] = list(agents)

        # holds every step's RunResult in order, so callers can inspect the whole pipeline, not just the final part.
        self.step_results: list[RunResult] = []

        # The exact Task handed to each agent, same order as self.agents. Useful
        # for debugging what context a given step actually saw.
        self.step_tasks: list[Task] = []

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

    def _build_step_task(
        self, agent: BaseAgent, initial_task: Task, transcript: list[tuple[BaseAgent, str]]
    ) -> Task:
        """Compose the Task for one step: the original objective plus every
        earlier step's output, each labelled with the agent that produced it.

        Labelling matters: without it the analyst can't tell the research
        findings apart from its own instructions, and the writer can't tell
        which block is raw research and which is the analysis.
        """
        if not transcript:
            # First step gets the user's request verbatim.
            return initial_task

        sections = [f"Original request:\n{initial_task.description}"]
        for position, (prior_agent, output) in enumerate(transcript, start=1):
            sections.append(
                f"--- Step {position} output from {prior_agent.name} "
                f"(role: {prior_agent.role}) ---\n{output}"
            )
        sections.append(
            f"Using the original request and all of the step output above, "
            f"do your job as the {agent.role} step."
        )

        return Task(
            description="\n\n".join(sections),
            assigned_agent=agent.name,
            input_data=initial_task.input_data,
        )

    def run(
        self,
        initial_task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        # Data flow through the pipeline:
        #
        #   initial_task --> agents[0] --> RunResult_0
        #                                     |
        #      original request + RunResult_0.output become the
        #      `description` of a new Task
        #                                     v
        #                  agents[1] --> RunResult_1
        #                                     |
        #      original request + RunResult_0.output + RunResult_1.output
        #                                     v
        #                  agents[2] --> RunResult_2
        #                                    ...
        #
        # Context accumulates rather than being replaced, so the analyst still
        # knows what topic was asked about, and the writer can quote the raw
        # research and not just the analyst's summary of it.
        #
        # `input_data` from the *original* task is carried along unchanged
        # at every hop, so shared context (config, IDs, etc.) is available
        # to every agent in the chain, not just the first one.
        #
        # If any agent fails, we stop immediately instead of feeding a
        # failed/empty output to the next agent, and that failed RunResult
        # becomes the final result.
        #
        # `shared_memory` is the cross-agent blackboard, and it travels
        # alongside that accumulating transcript rather than replacing it.
        # The transcript is this pipeline's own chain of prose; the
        # blackboard holds named deliverables that outlive the pipeline, so
        # a later workflow can ask for "api_code" by name instead of
        # re-reading three steps of narrative to find it.
        self.step_results = []
        self.step_tasks = []

        # (agent, output) for each step that succeeded, in order.
        transcript: list[tuple[BaseAgent, str]] = []
        result: Optional[RunResult] = None

        for agent in self.agents:
            current_task = self._build_step_task(agent, initial_task, transcript)
            self.step_tasks.append(current_task)

            result = execute_agent(agent, current_task, shared_memory)
            self.step_results.append(result)

            if not result.success:
                break  # Halt the chain; don't propagate a failed step forward.

            transcript.append((agent, result.output or ""))

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
        agent_kwargs[key] = build_agent_kwargs(entry)

    return SequentialOrchestrator.from_registry(
        agent_names=agent_keys,
        registry=registry,
        agent_kwargs=agent_kwargs,
    )
