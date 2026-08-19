from __future__ import annotations

import time
import warnings

from agent_framework.agents import BaseAgent
from agent_framework.core.tasks import Task
from agent_framework.orchestration.router import create_router_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

AGENTS_CONFIG = "config/agents.json"
ROUTER_CONFIG = "config/router_config.json"

CASES = [
    (
        "Test 1: single BaseAgent destination",
        "Write a python function to compute the Fibonacci sequence efficiently.",
        "coding",
    ),
    (
        "Test 2: workflow pipeline destination",
        "Research recent quantum computing breakthroughs and report the key findings.",
        "research",
    ),
    (
        "Test 3: fallback destination",
        "What is the tallest mountain in Japan?",
        "fallback",
    ),
]


def snippet(text: str, limit: int = 240) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def resolve_destination(router, route_key: str):
    """The object a route key points at, mirroring what run() would pick."""
    if route_key == "fallback":
        return router.fallback_destination
    return router.routes[route_key]


def describe_destination(destination) -> str:
    """Say what a destination is, and how the router will call it."""
    kind = type(destination).__name__

    # Mirrors the branch order inside RouterOrchestrator._execute.
    if isinstance(destination, BaseAgent):
        return f"{kind} (single agent, dispatched via .execute())"
    return f"{kind} (composite workflow, dispatched via .run())"


def test_real_router_pipeline() -> None:
    router = create_router_pipeline(
        agents_config_path=AGENTS_CONFIG,
        router_config_path=ROUTER_CONFIG,
    )

    print("---- Live Router Trace ----")
    print(f"registered routes: {list(router.routes)}")
    print(f"fallback: {type(router.fallback_destination).__name__}\n")

    for case_number, (label, prompt, expected_route) in enumerate(CASES, start=1):
        task = Task(description=prompt, assigned_agent="router")

        started = time.perf_counter()
        result = router.run(task)
        elapsed = time.perf_counter() - started

        entry = router.routing_log[-1]
        destination = resolve_destination(router, entry["route"])

        print(f"[{label}]")
        print(f"  prompt:      {prompt}")
        print(f"  route:       {entry['route']} (expected: {expected_route})")
        print(f"  destination: {describe_destination(destination)}")
        print(f"  elapsed:     {elapsed:.2f}s")
        if result.success:
            print(f"  output:      {snippet(result.output)}\n")
        else:
            print(f"  error:       {result.error}\n")

        # The destination actually produced real text.
        assert result.success is True, result.error
        assert result.output, f"{label}: empty output"

        # The routing decision was the one we expected, and it was recorded
        # against the task that caused it.
        assert entry["route"] == expected_route, entry
        assert entry["task_id"] == task.id, entry

        if isinstance(destination, BaseAgent):
            assert result.task_id == task.id, result

        assert len(router.routing_log) == case_number, router.routing_log

    print("---- Routing Log ----")
    print(f"{'#':<3} {'route':<12} {'task_id':<38} description")
    for index, entry in enumerate(router.routing_log, start=1):
        print(
            f"{index:<3} {entry['route']:<12} {entry['task_id']:<38} "
            f"{snippet(entry['description'], 60)}"
        )

    assert len(router.routing_log) == len(CASES), router.routing_log


if __name__ == "__main__":
    test_real_router_pipeline()
    print("\nAll assertions passed.")
