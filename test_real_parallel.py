from __future__ import annotations

import time
import warnings

# transformers builds a few tensors during model load that torch flags as
# UserWarnings (dtype/meta-device notices). They're noise here and they fire
# once per worker thread, so silence them before the model is touched.
warnings.filterwarnings("ignore", category=UserWarning)

from agent_framework.core.tasks import Task
from agent_framework.orchestration.parallel import create_parallel_pipeline

# Live test: this really runs the local model configured in config/agents.json
WORKER_KEYS = ["tech_researcher", "market_researcher"]
SYNTHESIZER_KEY = "synthesizer"
TOPIC = "Investigate CPUs from AMD"

# 2 workers + 1 synthesizer.
EXPECTED_STEPS = len(WORKER_KEYS) + 1


def snippet(text: str, limit: int = 240) -> str:
    """First `limit` characters on one line, so the fan-out stays scannable."""
    flattened = " ".join((text or "").split())
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit] + " ..."


def test_real_parallel_pipeline() -> None:
    # create_parallel_pipeline calls build_llm_client once per config key, and
    # that call is where the memory story lives. Two caches sit behind it:
    #
    #   1. build_llm_client (sequential.py) caches LLMClient objects by their
    #      full settings tuple. tech_researcher and market_researcher are
    #      configured identically, so they get back the same client object.
    #   2. LLMClient itself caches the loaded transformers pipeline by
    #      (model, device, dtype).

    orchestrator = create_parallel_pipeline(
        "config/agents.json",
        worker_keys=WORKER_KEYS,
        synthesizer_key=SYNTHESIZER_KEY,
        max_workers=2,
    )

    assert len(orchestrator.worker_agents) == len(WORKER_KEYS)

    # Proof of the sharing described above: identical config -> identical client.
    tech_agent, market_agent = orchestrator.worker_agents
    assert tech_agent.llm_client is market_agent.llm_client

    task = Task(description=TOPIC, assigned_agent=tech_agent.name)

    # perf_counter is the clock - it isn't affected by
    # system clock adjustments, which is what you want for measuring a duration.
    started = time.perf_counter()
    result = orchestrator.run(task)
    elapsed = time.perf_counter() - started

    assert result.success is True, result.error
    assert result.error is None

    # Every worker plus the synthesizer recorded a step.
    assert len(orchestrator.step_results) == EXPECTED_STEPS, orchestrator.step_results

    print("---- Live Parallel Pipeline Trace ----")
    print(f"topic: {TOPIC}")
    print(f"max_workers: {orchestrator.max_workers}\n")

    # step_results is [worker_0, worker_1, ..., synthesizer], in worker order.
    worker_results = orchestrator.step_results[: len(orchestrator.worker_agents)]
    for position, (agent, step) in enumerate(
        zip(orchestrator.worker_agents, worker_results), start=1
    ):
        client = agent.llm_client
        status = "OK" if step.success else "FAILED"
        print(f"[worker {position}] {agent.name} ({agent.role}) ({status})")
        print(f"  model: {client.model} | device: {client.device} ({client.dtype})")
        if step.success:
            print(f"  output: {step.output}\n")
        else:
            print(f"  error: {step.error}\n")

    synth_step = orchestrator.step_results[-1]
    synth_status = "OK" if synth_step.success else "FAILED"
    print(f"[synthesizer] {orchestrator.synthesizer_agent.name} ({synth_status})\n")

    # Each branch ran and produced real text - a failed worker is silently
    # dropped from the synthesis prompt, so this has to be checked explicitly
    # rather than inferred from result.success.
    for step in orchestrator.step_results:
        assert step.success is True, step.error
        assert step.output
        assert step.output.strip()

    # The orchestrator returns the synthesizer's result, not a copy of it.
    assert result.output == orchestrator.step_results[-1].output

    print("---- Final Synthesized Report ----")
    print(result.output)
    print(f"\ntotal elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    test_real_parallel_pipeline()
    print("\nAll assertions passed.")
