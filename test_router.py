from __future__ import annotations

from agent_framework.core.tasks import Task
from agent_framework.orchestration.router import create_router_pipeline

SCENARIOS: dict[str, str] = {
    "Fix a bug in my python code": "coding",
    "Calculate the sum of primes": "math",
    "Research and investigate AI market trends": "research",
    "What is the color of the sky?": "fallback",
}


def test_router_pipeline() -> None:
    router = create_router_pipeline()

    # Every route the scenarios expect (bar the fallback) is registered.
    expected_keys = {route for route in SCENARIOS.values() if route != "fallback"}
    assert expected_keys <= set(router.routes)

    print("---- Router Execution Trace ----")
    for description, expected_route in SCENARIOS.items():
        task = Task(description=description, assigned_agent="router")
        result = router.run(task)
        decision = router.routing_log[-1]

        assert result.success is True
        assert result.error is None
        assert decision["route"] == expected_route
        assert decision["task_id"] == task.id

        status = "OK" if result.success else "FAILED"
        print(f"\nTask: {task.description!r}")
        print(f"  -> route selected: {decision['route']} ({status})")
        print(f"  -> output: {result.output}")

    # One logged decision per task, in order.
    assert len(router.routing_log) == len(SCENARIOS)

    print("\n---- Full Routing Log ----")
    for entry in router.routing_log:
        print(entry)


if __name__ == "__main__":
    test_router_pipeline()
    print("\nAll assertions passed.")
