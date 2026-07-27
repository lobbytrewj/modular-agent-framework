from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol, Union, runtime_checkable

from agent_framework.agents.base import BaseAgent
from agent_framework.core.tasks import RunResult, Task

logger = logging.getLogger(__name__)

# Basically checks and any class that has a run method taking a Task and returning
# a run result can be considered a Runnable
@runtime_checkable
class Runnable(Protocol):


    def run(self, task: Task) -> RunResult:
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
        if self.route_func is not None:
            # Custom decision function is authoritative. If it names a key
            # we don't actually have a destination for, fall back rather than raising.
            key = self.route_func(task)
            if key in self.routes:
                return key
            logger.info(
                "route_func returned unknown key '%s' for task %s; falling back",
                key,
                task.id,
            )
            return self._FALLBACK_KEY

        # Keyword matching: first registered route whose keyword list
        # contains a substring hit in the task description wins.
        description_lower = task.description.lower()
        for key, keywords in self._keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return key

        return self._FALLBACK_KEY

    @staticmethod
    def _execute(destination: Destination, task: Task) -> RunResult:
        """Run `task` against `destination`, whether it's a BaseAgent or an orchestrator."""
        if isinstance(destination, BaseAgent):
            return destination.execute(task)
        if isinstance(destination, Runnable):
            return destination.run(task)
        raise TypeError(
            f"Destination {destination!r} is neither a BaseAgent (execute) "
            "nor a Runnable orchestrator (run)"
        )

    def run(self, task: Task) -> RunResult:
        """Route the task to a single destination and return its RunResult.

        Steps:
          1. Decide the route key using custom function rules or routing rules.
          2. Resolve that key to a destination and falls back to
             `fallback_destination` if the key was unmatched.
          3. Log the decision using 'logging` and in `self.routing_log'.
          4. Execute the chosen destination and return its result directly
        """
        route_key = self._decide_route(task)
        used_fallback = route_key == self._FALLBACK_KEY
        destination = self.fallback_destination if used_fallback else self.routes[route_key]

        decision_record = {
            "task_id": task.id,
            "route": "fallback" if used_fallback else route_key,
            "description": task.description,
        }
        self.routing_log.append(decision_record)
        logger.info(
            "Routing task %s -> '%s' (description=%r)",
            task.id,
            decision_record["route"],
            task.description,
        )

        return self._execute(destination, task)
