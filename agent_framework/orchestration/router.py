from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Protocol, Union, runtime_checkable

from agent_framework.agents import BaseAgent, LLMAgent, MockAgent, execute_agent
from agent_framework.core.memory import SharedWorkflowMemory, call_with_shared_memory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.observability.tracer import resolve_tracer
from agent_framework.orchestration.sequential import (
    build_agent_kwargs,
    create_sequential_pipeline,
    load_agent_config,
)

logger = logging.getLogger(__name__)

# Basically checks and any class that has a run method taking a Task and returning
# a run result can be considered a Runnable
@runtime_checkable
class Runnable(Protocol):


    def run(
        self,
        task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        ...


# A place where the router can point to
Destination = Union[BaseAgent, Runnable]

# A custom decision function: inspects the task and returns the route *key*
# (a string) that should handle it. It does not resolve the destination itself
RouteFunc = Callable[[Task], str]


class RouterOrchestrator:
    """Gives a Task to exactly one of several destinations.

    Unlike SequentialOrchestrator which runs every step in a fixed order, 
    RouterOrchestrator picks a single destination per task.
    
    Routing is decided in 2 different ways:

      1. Keyword rules: each route key is registered with a list of
         keywords The first route whose keywords appear in the task
         description is chosen
      2. A custom `route_func(task) -> str`: full control over the decision.
         When set, this takes priority over keyword rules entirely.

    Any task that matches no rule is sent to fallback_destination.
    """

    # Internal sentinel key used in the routing log when nothing matched
    # and we had to fall back.
    _FALLBACK_KEY = "__fallback__"

    def __init__(
        self,
        fallback_destination: Destination,
        route_func: Optional[RouteFunc] = None,
    ):
        # key -> destination
        self.routes: dict[str, Destination] = {}

        # key -> lowercase keywords that should trigger that route. Only
        # used when no custom route_func is set.
        self._keywords: dict[str, list[str]] = {}

        self.fallback_destination = fallback_destination
        self.route_func = route_func

        # Every routing decision made by `run`, in order, so callers can
        # audit why a task went where it went
        self.routing_log: list[dict] = []

    def register_route(
        self,
        key: str,
        destination: Destination,
        keywords: Optional[list[str]] = None,
    ) -> None:
        """Register a destination with key

        Keywords trigger the keyword-based routing
        They're ignored if a custom `route_func` is set,
        since the route_func takes full responsibility for choosing the key.
        """
        self.routes[key] = destination
        if keywords:
            self._keywords[key] = [kw.lower() for kw in keywords]

    def set_route_func(self, route_func: RouteFunc) -> None:
        """Install a custom decision function, overriding keyword rules."""
        self.route_func = route_func

    def _decide_route(self, task: Task) -> str:
        """Work out which registered key should handle the task.

        Returns _FALLBACK_KEY if nothing matches
        """
        return self._decide_route_with_reason(task)[0]

    def _decide_route_with_reason(self, task: Task) -> tuple[str, str]:
        """The route key plus a one-line account of why it was chosen.

        The reason is what makes a routing decision debuggable after the
        fact: "went to 'coding'" is much less useful than "went to 'coding'
        because 'python' matched". It is written to the trace and to
        routing_log; the key alone is what `_decide_route` returns.
        """
        if self.route_func is not None:
            # Custom decision function is authoritative. If it names a key
            # we don't actually have a destination for, fall back rather than raising.
            key = self.route_func(task)
            if key in self.routes:
                return key, f"route_func chose '{key}'"
            logger.info(
                "route_func returned unknown key '%s' for task %s; falling back",
                key,
                task.id,
            )
            return self._FALLBACK_KEY, f"route_func returned unknown key '{key}'"

        # Keyword matching: first registered route whose keyword list
        # contains a substring hit in the task description wins.
        description_lower = task.description.lower()
        for key, keywords in self._keywords.items():
            matched = [keyword for keyword in keywords if keyword in description_lower]
            if matched:
                return key, f"keyword {matched[0]!r} matched"

        return self._FALLBACK_KEY, "no keyword rule matched"

    @staticmethod
    def _execute(
        destination: Destination,
        task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        """Run `task` against `destination`, whether it's a BaseAgent or an orchestrator.

        Either way the blackboard is handed on: a route that points at a
        sub-workflow must pass it all the way down, or artifacts would
        stop at the routing boundary - exactly the place a run crosses
        from one workflow into another.
        """
        if isinstance(destination, BaseAgent):
            return execute_agent(destination, task, shared_memory)
        if isinstance(destination, Runnable):
            return call_with_shared_memory(destination.run, task, shared_memory)
        raise TypeError(
            f"Destination {destination!r} is neither a BaseAgent (execute) "
            "nor a Runnable orchestrator (run)"
        )

    def run(
        self,
        task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        """Route the task to a single destination and return its RunResult.

        Steps:
          1. Decide the route key using custom function rules or routing rules.
          2. Resolve that key to a destination and falls back to
             `fallback_destination` if the key was unmatched.
          3. Log the decision using 'logging` and in `self.routing_log'.
          4. Execute the chosen destination and return its result directly
        """
        route_key, reasoning = self._decide_route_with_reason(task)
        used_fallback = route_key == self._FALLBACK_KEY
        destination = self.fallback_destination if used_fallback else self.routes[route_key]

        decision_record = {
            "task_id": task.id,
            "route": "fallback" if used_fallback else route_key,
            "description": task.description,
            "reasoning": reasoning,
        }
        self.routing_log.append(decision_record)
        logger.info(
            "Routing task %s -> '%s' (description=%r)",
            task.id,
            decision_record["route"],
            task.description,
        )

        # The routing decision itself goes on the shared audit trail, not
        # just in self.routing_log: read back later, "which route did this
        # run take" is only answerable if the choice is interleaved with
        # the steps it caused.
        if shared_memory is not None:
            shared_memory.log_step(
                agent="router",
                task=task.description,
                workflow="router",
                route=decision_record["route"],
                task_id=task.id,
            )

        tracer = resolve_tracer(shared_memory)
        if tracer is not None:
            tracer.record_route(
                "router",
                decision_record["route"],
                task.description,
                reasoning=reasoning,
                task_id=task.id,
                destination=_destination_name(destination),
            )
            # Handing off to a whole sub-workflow is a delegation in its own
            # right - the steps it runs will show up under their own agents,
            # and this event is what ties them back to the routing choice.
            if not isinstance(destination, BaseAgent):
                tracer.record_delegation(
                    "router",
                    _destination_name(destination),
                    task,
                    reason=f"route '{decision_record['route']}' points at a sub-workflow",
                    workflow="router",
                )

        return self._execute(destination, task, shared_memory)


def _destination_name(destination: Destination) -> str:
    """A label for a route target: the agent's name, or the orchestrator's class."""
    name = getattr(destination, "name", None)
    return name if isinstance(name, str) else type(destination).__name__


DEFAULT_AGENTS_CONFIG = "config/agents.json"
DEFAULT_ROUTER_CONFIG = "config/router_config.json"


def load_router_config(router_config_path: str) -> dict:
    """Read router_config.json and hand back its four sections.

    Expected shape:

        {
          "fallback_key": "fallback",
          "route_rules":     {"coding": ["python", "code", "bug"], ...},
          "route_agents":    {"coding": "coder", ...},
          "route_workflows": {"research": ["researcher", "writer"], ...}
        }
    """
    path = Path(router_config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Router config file not found: {router_config_path}")

    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Router config {router_config_path} must be a JSON object, "
            f"got {type(config).__name__}"
        )

    for section in ("route_rules", "route_agents", "route_workflows"):
        value = config.setdefault(section, {})
        if not isinstance(value, dict):
            raise ValueError(
                f"'{section}' in {router_config_path} must be an object, "
                f"got {type(value).__name__}"
            )

    fallback_key = config.setdefault("fallback_key", "fallback")
    if not isinstance(fallback_key, str):
        raise ValueError(
            f"'fallback_key' in {router_config_path} must be a string, "
            f"got {type(fallback_key).__name__}"
        )

    return config


def _build_agent(config: dict[str, dict], key: str, config_path: str) -> BaseAgent:
    """Instantiate a single LLMAgent from its config entry."""
    if key not in config:
        raise KeyError(f"No agent configured under key '{key}' in {config_path}")

    entry = config[key]
    if not isinstance(entry, dict):
        raise ValueError(
            f"Config entry for '{key}' in {config_path} must be an object, "
            f"got {type(entry).__name__}"
        )

    kwargs = build_agent_kwargs(entry)
    kwargs.setdefault("name", key)
    kwargs.setdefault("role", key)
    kwargs.setdefault("system_prompt", f"You are the '{key}' destination of a router.")

    # Previous behaviour, kept for reference: a mock agent that echoes the
    # task back and never touches the network.
    # return MockAgent(**kwargs)

    # Real LLM-backed agent. It gets its own client so per-agent model settings
    # (model, temperature, max_tokens) in the config are honoured.
    return LLMAgent(**kwargs)


def _resolve_destination(
    route_key: str,
    agents_config: dict[str, dict],
    agents_config_path: str,
    route_workflows: dict[str, list[str]],
    route_agents: dict[str, str],
) -> Destination:
    """Work out what a route key points at: a sub-workflow or a single agent.
    """
    chain = route_workflows.get(route_key)
    if chain is not None:
        return create_sequential_pipeline(agents_config_path, chain)

    agent_key = route_agents.get(route_key, route_key)
    return _build_agent(agents_config, agent_key, agents_config_path)


def create_router_pipeline(
    agents_config_path: str = DEFAULT_AGENTS_CONFIG,
    router_config_path: str = DEFAULT_ROUTER_CONFIG,
    route_rules: Optional[dict[str, list[str]]] = None,
) -> RouterOrchestrator:
    #Build a RouterOrchestrator from the two JSON config files.
    router_config = load_router_config(router_config_path)
    agents_config = load_agent_config(agents_config_path)

    rules = router_config["route_rules"] if route_rules is None else route_rules
    route_agents = router_config["route_agents"]
    route_workflows = router_config["route_workflows"]

    router = RouterOrchestrator(
        fallback_destination=_build_agent(
            agents_config, router_config["fallback_key"], agents_config_path
        )
    )

    # Registration order is the matching order, so it follows the order the
    # rules appear in the config file.
    for route_key, keywords in rules.items():
        destination = _resolve_destination(
            route_key,
            agents_config,
            agents_config_path,
            route_workflows,
            route_agents,
        )
        router.register_route(route_key, destination, keywords=list(keywords))

    return router
