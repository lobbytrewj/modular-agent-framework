from __future__ import annotations

import time
import warnings

from agent_framework.core.tasks import Task
from agent_framework.orchestration.orchestrator_worker import (
    create_orchestrator_worker_pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)

AGENTS_CONFIG = "config/agents.json"
ORCHESTRATOR_KEY = "orchestrator"
WORKER_KEYS = ["coder", "reviewer"]
MAX_ITERATIONS = 2

# Deliberately two-phase: the review cannot happen until the code exists, so a
# single round of work genuinely cannot satisfy the request. That is the case
# Goal 7's one-shot decomposition structurally cannot serve.
REQUEST = (
    "Write a Python function to calculate factorial. "
    "The coder must write it, and the reviewer must check it."
)


def snippet(text: str, limit: int = 240) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def test_real_orchestrator_worker_pipeline() -> None:
    orchestrator = create_orchestrator_worker_pipeline(
        AGENTS_CONFIG,
        orchestrator_key=ORCHESTRATOR_KEY,
        worker_keys=WORKER_KEYS,
        max_iterations=MAX_ITERATIONS,
    )

    assert set(orchestrator.worker_agents) == set(WORKER_KEYS)
    assert orchestrator.max_iterations == MAX_ITERATIONS

    task = Task(
        description=REQUEST,
        assigned_agent=orchestrator.orchestrator_agent.name,
    )

    started = time.perf_counter()
    result = orchestrator.run(task)
    elapsed = time.perf_counter() - started


    print("---- Live Orchestrator-Worker Trace ----")
    print(
        f"orchestrator: {orchestrator.orchestrator_agent.name} "
        f"({orchestrator.orchestrator_agent.role})"
    )
    for key, worker in orchestrator.worker_agents.items():
        print(f"worker:       {key} -> {worker.name} ({worker.role})")
    print(f"budget:       {MAX_ITERATIONS} iteration(s)")
    print(f"request:      {REQUEST}\n")

    for record in orchestrator.iteration_history:
        turn = record["iteration"] + 1

        if record["completed"]:
            signalled = orchestrator.completion_phrase in (record["evaluation"] or "")
            decision = (
                f"{orchestrator.completion_phrase} - request judged satisfied"
                if signalled
                else "no new subtasks - reply taken as the final answer"
            )
        else:
            routed = ", ".join(entry["agent"] for entry in record["tasks"])
            decision = f"routed {len(record['tasks'])} subtask(s) -> {routed}"

        print(f"[Turn {turn}] orchestrator ({'review' if turn > 1 else 'initial plan'})")
        print(f"  decision:  {decision}")
        print(f"  raw reply: {snippet(record['evaluation'])}")

        for position, entry in enumerate(record["outputs"], start=1):
            worker = orchestrator.worker_agents[entry["agent"]]
            status = "OK" if entry["success"] else "FAILED"
            print(f"  [{turn}.{position}] {worker.name} ({entry['agent']}) ({status})")
            print(f"      assigned: {snippet(entry['task'], 140)}")
            if entry["success"]:
                print(f"      output:   {snippet(entry['output'])}")
            else:
                print(f"      error:    {entry['error']}")
        print()



    assert result.success is True, result.error
    assert result.output, "orchestrator returned an empty final answer"

    assert len(orchestrator.iteration_history) >= 1, orchestrator.iteration_history

    assert len(orchestrator.step_results) >= 2, len(orchestrator.step_results)

    final_turn = orchestrator.step_results[-1].output or ""
    assert result.output == orchestrator._strip_completion_phrase(final_turn), (
        f"final answer diverged from the last orchestrator turn:\n"
        f"  returned:   {snippet(result.output)}\n"
        f"  last turn:  {snippet(final_turn)}"
    )
    assert orchestrator.completion_phrase not in result.output

    assert 1 <= orchestrator.iterations_used <= MAX_ITERATIONS
    assert len(orchestrator.iteration_history) == orchestrator.iterations_used

    for record in orchestrator.iteration_history[:-1]:
        assert not record["completed"], record["iteration"]
        assert record["tasks"], record["iteration"]

    prompts = [record["prompt"] for record in orchestrator.iteration_history]
    for earlier, later in zip(prompts, prompts[1:]):
        assert len(later) > len(earlier), "review prompt did not accumulate context"
    if len(prompts) > 1:
        assert "Work completed so far" in prompts[1]
        assert "Work completed so far" not in prompts[0]

    # Every dispatched subtask went to a real specialist.
    dispatched = sum(len(record["tasks"]) for record in orchestrator.iteration_history)
    for record in orchestrator.iteration_history:
        for entry in record["outputs"]:
            assert entry["agent"] in orchestrator.worker_agents, entry

    expected_steps = orchestrator.iterations_used + dispatched
    if not orchestrator.completed_naturally:
        expected_steps += 1
    assert len(orchestrator.step_results) == expected_steps, (
        f"expected {expected_steps} step result(s), got {len(orchestrator.step_results)}"
    )

    if orchestrator.completed_naturally:
        last = orchestrator.iteration_history[-1]["evaluation"] or ""
        reason = (
            f"orchestrator signalled {orchestrator.completion_phrase}"
            if orchestrator.completion_phrase in last
            else "orchestrator produced no new subtasks"
        )
    else:
        reason = (
            f"max_iterations ({MAX_ITERATIONS}) reached - forced consolidation"
        )

    print("[termination]")
    print(f"  reason:   {reason}")
    print(f"  turns:    {orchestrator.iterations_used}")
    print(
        f"  subtasks: {dispatched} dispatched across "
        f"{len(orchestrator.step_results)} agent call(s)"
    )
    print(f"  elapsed:  {elapsed:.2f}s total\n")

    if orchestrator._parse_plan(result.output, task):
        print(
            "  WARNING: the final answer parses as a subtask plan, not an "
            "answer. The orchestrator kept assigning work on its closing turn "
            "- check clause (3) of its system prompt.\n"
        )

    print("---- Final Consolidated Output ----")
    print(result.output)


if __name__ == "__main__":
    test_real_orchestrator_worker_pipeline()
    print("\nAll assertions passed.")
