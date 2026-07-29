from __future__ import annotations

from agent_framework.core.tasks import Task
from agent_framework.orchestration.sequential import create_sequential_pipeline


# This file picks the agents, runs a task, and checks the result.
def test_sequential_pipeline() -> None:
    pipeline = create_sequential_pipeline(
        "config/agents.json", ["researcher", "analyst", "writer"]
    )

    task = Task(
        description="Investigate electric vehicle market trends",
        assigned_agent="researcher",
    )
    result = pipeline.run(task)

    assert result.success is True
    assert result.error is None
    # Every agent in the pipeline should have contributed a step.
    assert len(pipeline.step_results) == len(pipeline.agents)
    assert all(step.success for step in pipeline.step_results)

    print("---- Sequential Pipeline Execution Trace ----")
    for agent, step in zip(pipeline.agents, pipeline.step_results):
        status = "OK" if step.success else "FAILED"
        print(f"\n[{agent.name}] ({agent.role}) ({status})")
        print(f"  output: {step.output}")

    print("\n---- Final Result ----")
    print(result)


if __name__ == "__main__":
    test_sequential_pipeline()
    print("\nAll assertions passed.")
