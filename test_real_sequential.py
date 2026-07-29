from __future__ import annotations

from agent_framework.core.tasks import Task
from agent_framework.orchestration.sequential import create_sequential_pipeline

# Live test: this really runs the local model configured in config/agents.json
AGENT_KEYS = ["researcher", "analyst", "writer"]
TOPIC = "Investigate electric vehicle market trends"


def test_real_sequential_pipeline() -> None:
    pipeline = create_sequential_pipeline("config/agents.json", AGENT_KEYS)

    assert len(pipeline.agents) == len(AGENT_KEYS)

    task = Task(description=TOPIC, assigned_agent=AGENT_KEYS[0])
    result = pipeline.run(task)

    print("---- Live Sequential Pipeline Trace ----")
    print(f"topic: {TOPIC}\n")
    for step_number, (agent, step) in enumerate(
        zip(pipeline.agents, pipeline.step_results), start=1
    ):
        client = agent.llm_client
        status = "OK" if step.success else "FAILED"
        print(f"[step {step_number}] {agent.name} ({agent.role}) ({status})")
        print(f"  model: {client.model} | device: {client.device} ({client.dtype})")
        if step.success:
            print(f"  output: {step.output}\n")
        else:
            print(f"  error: {step.error}\n")

    # Every step ran and produced real text.
    assert len(pipeline.step_results) == len(AGENT_KEYS), pipeline.step_results
    for step in pipeline.step_results:
        assert step.success is True, step.error
        assert step.output

    assert result.success is True, result.error
    assert result.error is None

    print("---- Final Report ----")
    print(result.output)


if __name__ == "__main__":
    test_real_sequential_pipeline()
    print("\nAll assertions passed.")
