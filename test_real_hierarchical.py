from __future__ import annotations

import time
import warnings

from agent_framework.core.tasks import Task
from agent_framework.orchestration.hierarchical import create_hierarchical_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

AGENTS_CONFIG = "config/agents.json"
LEADER_KEY = "manager"
WORKER_KEYS = ["researcher", "coder"]

# Deliberately composite: one half is a coding job, the other is a research
# write-up. A single specialist can't serve both well, so the leader has a real
# reason to split the work across its roster.
REQUEST = (
    "Design a REST API endpoint in Python FastAPI to track gym workout sets "
    "and summarize best practices for database indexing."
)


def snippet(text: str, limit: int = 240) -> str:
    """First limit characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def identify_parse_tier(orchestrator) -> str:
    """Report which of _parse_plan's three tiers produced the live plan.

 
      Tier 1  _parse_json_plan - what we asked for. _extract_json slices
              between the outermost brackets rather than parsing the whole
              reply

      Tier 2  _parse_labelled_plan  - the leader wrote prose instead. A line only
              becomes a subtask if its label resolves to a real specialist via
              _resolve_worker, so ordinary sentences can't sneak in as work.

      Tier 3  full fan-out - nothing parsed at all
    """
    raw = orchestrator.decomposition_result.output or ""

    if orchestrator._parse_json_plan(raw):
        return "tier 1 - strict JSON (_parse_json_plan)"
    if orchestrator._parse_labelled_plan(raw):
        return "tier 2 - labelled prose (_parse_labelled_plan)"
    return "tier 3 - full fan-out (leader produced no parseable plan)"


def test_real_hierarchical_pipeline() -> None:
    orchestrator = create_hierarchical_pipeline(
        AGENTS_CONFIG,
        leader_key=LEADER_KEY,
        worker_keys=WORKER_KEYS,
    )

    assert set(orchestrator.worker_agents) == set(WORKER_KEYS)

    task = Task(description=REQUEST, assigned_agent=orchestrator.leader_agent.name)

    started = time.perf_counter()
    result = orchestrator.run(task)
    elapsed = time.perf_counter() - started

    print("---- Live Hierarchical Pipeline Trace ----")
    print(f"leader:  {orchestrator.leader_agent.name} ({orchestrator.leader_agent.role})")
    for key, worker in orchestrator.worker_agents.items():
        print(f"worker:  {key} -> {worker.name} ({worker.role})")
    print(f"request: {REQUEST}\n")

    # --- Step 1: PLAN ---
    decomposition = orchestrator.decomposition_result
    assert decomposition is not None, "run() never recorded a decomposition result"
    assert decomposition.success is True, decomposition.error

    print("[step 1] decomposition")
    print(f"  parser:      {identify_parse_tier(orchestrator)}")
    print(f"  raw reply:   {snippet(decomposition.output)}")
    print(f"  plan:        {len(orchestrator.plan)} subtask(s)")
    for position, subtask in enumerate(orchestrator.plan, start=1):
        print(f"    {position}. {subtask.worker_key}: {snippet(subtask.description, 140)}")
    print()

    assert len(orchestrator.plan) >= 1, orchestrator.plan
    for subtask in orchestrator.plan:
        assert subtask.worker_key in orchestrator.worker_agents, subtask
        assert subtask.description.strip(), subtask

    # --- Step 2: DELEGATE ---
    # step_results is [worker, worker, ..., leader]: one entry per planned subtask
    assert len(orchestrator.step_results) == len(orchestrator.plan) + 1, (
        f"expected {len(orchestrator.plan)} worker result(s) + 1 leader summary, "
        f"got {len(orchestrator.step_results)}"
    )

    subtask_results = orchestrator.step_results[:-1]
    for position, (subtask, step) in enumerate(
        zip(orchestrator.plan, subtask_results), start=1
    ):
        worker = orchestrator.worker_agents[subtask.worker_key]
        status = "OK" if step.success else "FAILED"
        print(f"[step 2.{position}] {worker.name} ({subtask.worker_key}) ({status})")
        print(f"  assigned: {snippet(subtask.description, 140)}")
        if step.success:
            print(f"  output:   {snippet(step.output)}\n")
        else:
            print(f"  error:    {step.error}\n")

    # --- Step 3: AGGREGATE ---
    assert result.success is True, result.error
    assert result.output, "leader returned an empty final report"

    # run() returns the leader's aggregation verbatim
    assert result.output == orchestrator.step_results[-1].output
    assert result is orchestrator.step_results[-1]

    print("[step 3] aggregation")
    print(f"  leader:   {orchestrator.leader_agent.name}")
    print(f"  synthesized from {sum(1 for s in subtask_results if s.success)} "
          f"of {len(orchestrator.plan)} subtask result(s)")
    print(f"  elapsed:  {elapsed:.2f}s total\n")

    print("---- Final Report ----")
    print(result.output)


if __name__ == "__main__":
    test_real_hierarchical_pipeline()
    print("\nAll assertions passed.")
