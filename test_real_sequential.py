from __future__ import annotations

from agent_framework.core.tasks import Task
from agent_framework.orchestration.sequential import create_sequential_pipeline

AGENT_KEYS = ["researcher", "analyst", "writer"]
TOPIC = "Investigate best gpu's for efficiency"


def indent(text: str, prefix: str = "    ") -> str:
    """Indent a multi-line prompt so it reads as one block in the trace."""
    return "\n".join(prefix + line for line in text.splitlines())


def test_real_sequential_pipeline() -> None:
    pipeline = create_sequential_pipeline("config/agents.json", AGENT_KEYS)

    assert len(pipeline.agents) == len(AGENT_KEYS)

    task = Task(description=TOPIC, assigned_agent=AGENT_KEYS[0])
    result = pipeline.run(task)

    print("---- Live Sequential Pipeline Trace ----")
    print(f"topic: {TOPIC}\n")
    for step_number, (agent, step_task, step) in enumerate(
        zip(pipeline.agents, pipeline.step_tasks, pipeline.step_results), start=1
    ):
        client = agent.llm_client
        status = "OK" if step.success else "FAILED"
        print(f"[step {step_number}] {agent.name} ({agent.role}) ({status})")
        print(f"  model: {client.model} | device: {client.device} ({client.dtype})")
        #print(f"  saw:\n{indent(step_task.description)}")
        if step.success:
            print(f"  output: {step.output}\n")
        else:
            print(f"  error: {step.error}\n")

    # Every step ran and produced real text.
    assert len(pipeline.step_results) == len(AGENT_KEYS), pipeline.step_results
    for step in pipeline.step_results:
        assert step.success is True, step.error
        assert step.output

    # Downstream agents must be able to see upstream work: every step after the
    # first carries the original topic plus each earlier agent's output.
    for step_number, step_task in enumerate(pipeline.step_tasks[1:], start=2):
        assert TOPIC in step_task.description, step_task.description
        for prior_agent, prior_step in zip(
            pipeline.agents[: step_number - 1], pipeline.step_results
        ):
            assert prior_agent.name in step_task.description, step_task.description
            assert prior_step.output in step_task.description, step_task.description

    assert result.success is True, result.error
    assert result.error is None

    print("---- Final Report ----")
    print(result.output)


if __name__ == "__main__":
    test_real_sequential_pipeline()
    print("\nAll assertions passed.")
