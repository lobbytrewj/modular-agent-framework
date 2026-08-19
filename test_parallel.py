from __future__ import annotations

import threading
import time

from agent_framework.agents import MockAgent
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration import ParallelOrchestrator

TOPIC = "Investigate renewable energy adoption"
WORKER_NAMES = ["tech_worker", "market_worker", "policy_worker"]

# How long the concurrency probe waits for its peers before giving up. Long
# enough that a loaded machine won't trip it, short enough that the negative
# control (which is *supposed* to time out) doesn't stall the suite.
BARRIER_TIMEOUT = 1.0


# --- Test doubles -------------------------------------------------------
#
# All of these are MockAgents: they return canned results instantly and never
# touch a model. MockAgent already echoes the task description into its output
# (`MockAgent {name} successfully completed task: {description}`), which is what
# lets us prove which prompt an agent actually saw.


class RecordingMockAgent(MockAgent):
    """A MockAgent that keeps every Task it was handed.

    MockAgent always succeeds, so on its own it can't tell us whether it ran
    *zero* times or once. Recording the tasks gives us both the call count and
    the exact prompt text, which the short-circuit and fan-in tests need.
    """

    def __init__(self, name: str, role: str, system_prompt: str):
        super().__init__(name, role, system_prompt)
        self.received_tasks: list[Task] = []

    def execute(self, task: Task) -> RunResult:
        self.received_tasks.append(task)
        return super().execute(task)


class FailingMockAgent(RecordingMockAgent):
    """A worker that always reports failure, for the short-circuit test."""

    def execute(self, task: Task) -> RunResult:
        self.received_tasks.append(task)
        return RunResult(task_id=task.id, success=False, error=f"{self.name} failed")


class BarrierMockAgent(RecordingMockAgent):
    """A worker that blocks until every other barrier worker has also started.

    This is how we prove *actual* concurrency rather than inferring it from a
    stopwatch. If the pool runs these serially, worker 1 sits at the barrier
    waiting for peers that can't start until it returns, and the wait times out.
    Timing assertions would be flaky on a busy machine; this is deterministic.
    """

    def __init__(self, name: str, role: str, system_prompt: str, barrier: threading.Barrier):
        super().__init__(name, role, system_prompt)
        self.barrier = barrier

    def execute(self, task: Task) -> RunResult:
        # Raises BrokenBarrierError on timeout, which the orchestrator converts
        # into a failed RunResult, so a serialized run shows up as failures.
        self.barrier.wait(timeout=BARRIER_TIMEOUT)
        return super().execute(task)


class SlowMockAgent(RecordingMockAgent):
    """A worker that sleeps a fixed amount before succeeding."""

    def __init__(self, name: str, role: str, system_prompt: str, delay: float):
        super().__init__(name, role, system_prompt)
        self.delay = delay

    def execute(self, task: Task) -> RunResult:
        time.sleep(self.delay)
        return super().execute(task)


def build_workers() -> list[RecordingMockAgent]:
    return [
        RecordingMockAgent(name, name.replace("_worker", ""), f"You are the {name}.")
        for name in WORKER_NAMES
    ]


# --- Tests --------------------------------------------------------------


def test_fan_out_and_synthesis() -> None:
    """The happy path: 3 workers fan out, the synthesizer fans them back in."""
    workers = build_workers()
    synthesizer = RecordingMockAgent("synth_worker", "synthesizer", "Combine findings.")
    orchestrator = ParallelOrchestrator(workers, synthesizer, max_workers=3)

    task = Task(description=TOPIC, assigned_agent="parallel", input_data={"region": "US"})
    result = orchestrator.run(task)

    # (a) Every worker ran exactly once and reported success.
    worker_steps = orchestrator.step_results[: len(workers)]
    for worker, step in zip(workers, worker_steps):
        assert len(worker.received_tasks) == 1, f"{worker.name} ran {len(worker.received_tasks)}x"
        assert step.success is True, step.error
        assert step.output

    # Each worker sees the ORIGINAL request, not another worker's output. This
    # is the defining difference from the sequential pipeline, and it's what
    # makes running them at the same time correct rather than just fast.
    for worker in workers:
        assert worker.received_tasks[0].description == TOPIC

    # ...on its own Task id, so a result traces back to the branch that made it.
    branch_ids = {worker.received_tasks[0].id for worker in workers}
    assert len(branch_ids) == len(workers), branch_ids
    assert task.id not in branch_ids

    # input_data rides along to every branch (same guarantee as sequential).
    for worker in workers:
        assert worker.received_tasks[0].input_data == {"region": "US"}

    # (b) One entry per worker, plus one for the synthesizer, in that order.
    assert len(orchestrator.step_results) == len(workers) + 1, orchestrator.step_results

    # (c) The synthesizer ran once, and its prompt carries the original request
    # plus every worker's output verbatim under a label naming that worker.
    assert len(synthesizer.received_tasks) == 1
    prompt = synthesizer.received_tasks[0].description
    assert prompt.startswith(f"Original Request:\n{TOPIC}")
    for position, (worker, step) in enumerate(zip(workers, worker_steps), start=1):
        assert f"--- Worker {position} ({worker.name}) ---" in prompt, prompt
        assert step.output in prompt, prompt

    # (d) What run() returns IS the synthesizer's result, and it's the last
    # entry in step_results, not a copy or a rewrap of it.
    assert result is orchestrator.step_results[-1]
    assert result.success is True
    assert result.task_id == synthesizer.received_tasks[0].id
    assert result.output == f"MockAgent {synthesizer.name} successfully completed task: {prompt}"

    print("---- Fan-Out / Fan-In Trace ----")
    for step_number, (agent, step) in enumerate(
        zip([*workers, synthesizer], orchestrator.step_results), start=1
    ):
        status = "OK" if step.success else "FAILED"
        print(f"[{step_number}] {agent.name} ({agent.role}) ({status})")
    print("\n---- Prompt handed to the synthesizer ----")
    print(prompt)


def test_all_workers_fail_short_circuits() -> None:
    """When every branch fails there is nothing to synthesize, so don't try."""
    workers = [
        FailingMockAgent(name, name.replace("_worker", ""), f"You are the {name}.")
        for name in WORKER_NAMES
    ]
    synthesizer = RecordingMockAgent("synth_worker", "synthesizer", "Combine findings.")
    orchestrator = ParallelOrchestrator(workers, synthesizer, max_workers=3)

    task = Task(description=TOPIC, assigned_agent="parallel")
    result = orchestrator.run(task)

    # The workers still ran - short-circuiting happens at the fan-in, not by
    # skipping the fan-out.
    assert all(len(worker.received_tasks) == 1 for worker in workers)
    assert len(orchestrator.step_results) == len(workers), orchestrator.step_results
    assert all(step.success is False for step in orchestrator.step_results)

    # The synthesizer was never invoked - the whole point of the short-circuit.
    assert synthesizer.received_tasks == [], "synthesizer must not run on total failure"

    # The failure is reported against the ORIGINAL task, since no synthesizer
    # task was ever created to attribute it to, and it names every branch so
    # the caller can see all three causes rather than just the first.
    assert result.success is False
    assert result.output is None
    assert result.task_id == task.id
    for worker in workers:
        assert f"{worker.name} failed" in result.error, result.error

    print(f"\n---- Short-Circuit ----\n{result.error}")


def test_partial_failure_still_synthesizes() -> None:
    """One dead branch shouldn't discard the branches that worked."""
    healthy = RecordingMockAgent("tech_worker", "tech", "You are the tech worker.")
    broken = FailingMockAgent("market_worker", "market", "You are the market worker.")
    synthesizer = RecordingMockAgent("synth_worker", "synthesizer", "Combine findings.")
    orchestrator = ParallelOrchestrator([healthy, broken], synthesizer)

    result = orchestrator.run(Task(description=TOPIC, assigned_agent="parallel"))

    assert result.success is True
    assert len(orchestrator.step_results) == 3

    # Only the surviving branch reaches the synthesizer, and it's renumbered to
    # "Worker 1" - positions count successes, not original slots.
    prompt = synthesizer.received_tasks[0].description
    assert f"--- Worker 1 ({healthy.name}) ---" in prompt, prompt
    assert broken.name not in prompt, prompt
    assert prompt.count("--- Worker") == 1, prompt

    print(f"\n---- Partial Failure ----\n{prompt}")


def test_workers_run_concurrently() -> None:
    """Prove that the worker agents are actually running at the exact same time, and prove that our test isn't fake or giving a false pass."""
    # Positive case: 3 threads for 3 workers. Every worker can reach the
    # barrier, so all three release and succeed.
    barrier = threading.Barrier(len(WORKER_NAMES))
    workers = [
        BarrierMockAgent(name, name.replace("_worker", ""), f"You are the {name}.", barrier)
        for name in WORKER_NAMES
    ]
    orchestrator = ParallelOrchestrator(
        workers, RecordingMockAgent("synth_worker", "synthesizer", "Combine."), max_workers=3
    )
    result = orchestrator.run(Task(description=TOPIC, assigned_agent="parallel"))

    assert result.success is True
    assert all(step.success for step in orchestrator.step_results), [
        step.error for step in orchestrator.step_results
    ]
    print(f"\n---- Concurrency ----\n{len(WORKER_NAMES)} workers met at the barrier")

    # Negative control: same agents, one thread. Worker 1 waits for peers that
    # cannot start until it returns, so the barrier times out and breaks. If
    # this DIDN'T fail, the test above would be proving nothing.
    serial_barrier = threading.Barrier(len(WORKER_NAMES))
    serial_workers = [
        BarrierMockAgent(name, name.replace("_worker", ""), f"You are the {name}.", serial_barrier)
        for name in WORKER_NAMES
    ]
    serial = ParallelOrchestrator(
        serial_workers,
        RecordingMockAgent("synth_worker", "synthesizer", "Combine."),
        max_workers=1,
    )
    serial_result = serial.run(Task(description=TOPIC, assigned_agent="parallel"))

    assert serial_result.success is False, "max_workers=1 must not satisfy the barrier"
    assert "BrokenBarrierError" in serial_result.error, serial_result.error
    print("max_workers=1 broke the barrier, as expected (probe is not vacuous)")


def test_results_stay_in_worker_order() -> None:
    """step_results must follow worker order, not the order threads finished."""
    delays = [0.30, 0.15, 0.0]
    workers = [
        SlowMockAgent(name, name.replace("_worker", ""), f"You are the {name}.", delay)
        for name, delay in zip(WORKER_NAMES, delays)
    ]
    synthesizer = RecordingMockAgent("synth_worker", "synthesizer", "Combine findings.")
    orchestrator = ParallelOrchestrator(workers, synthesizer, max_workers=3)

    start = time.perf_counter()
    orchestrator.run(Task(description=TOPIC, assigned_agent="parallel"))
    elapsed = time.perf_counter() - start

    for worker, step in zip(workers, orchestrator.step_results):
        assert worker.name in step.output, f"{step.output} is not {worker.name}'s"

    # The synthesizer's labels agree with that ordering.
    prompt = synthesizer.received_tasks[0].description
    positions = [prompt.index(worker.name) for worker in workers]
    assert positions == sorted(positions), prompt

    # Sanity check that they really did overlap: run serially this would take
    # at least the sum of the delays.
    assert elapsed < sum(delays), f"{elapsed:.2f}s >= {sum(delays):.2f}s serial"
    print(f"\n---- Ordering ----\nslowest-first workers finished in {elapsed:.2f}s "
          f"(serial would be {sum(delays):.2f}s), order preserved")


def test_rejects_invalid_construction() -> None:
    synthesizer = RecordingMockAgent("synth_worker", "synthesizer", "Combine findings.")

    for description, build in (
        ("no workers", lambda: ParallelOrchestrator([], synthesizer)),
        ("max_workers=0", lambda: ParallelOrchestrator(build_workers(), synthesizer, 0)),
    ):
        try:
            build()
        except ValueError as exc:
            print(f"rejected {description}: {exc}")
        else:
            raise AssertionError(f"{description} should have raised ValueError")


if __name__ == "__main__":
    test_fan_out_and_synthesis()
    test_all_workers_fail_short_circuits()
    test_partial_failure_still_synthesizes()
    test_workers_run_concurrently()
    test_results_stay_in_worker_order()
    print("\n---- Validation ----")
    test_rejects_invalid_construction()
    print("\nAll assertions passed.")
