"""Live integration test for the auto-routed verifier pipeline.

Runs real local inference through create_routed_verified_pipeline - no mocks -
so it exercises the two halves together: a router picking a destination from
the task text, and an outer QA gate deciding whether what came back actually
answered the request.

Required config/agents.json entry (already applied):

    "verifier": {
      "name": "VerifierAgent",
      "role": "quality_assurance",
      "system_prompt": "You are a quality-assurance verifier ... Reply with
        ONLY a JSON object: {\"is_complete\": true or false, \"feedback\":
        \"<what is missing or wrong>\"} ...",
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "device": "mps",
      "temperature": 0.1,
      "max_tokens": 128
    }

Routing comes from config/router_config.json unchanged: the request below
contains "python", which matches the "coding" rule and resolves through
route_agents to the "coder" agent.

Use "device": "cpu" instead of "mps" on a machine without Apple Silicon.
"""

from __future__ import annotations

import time
import warnings

from agent_framework.core.tasks import Task
from agent_framework.orchestration.routed_verified_pipeline import (
    create_routed_verified_pipeline,
)

# Silences the "To copy construct from a tensor ..." notices transformers emits
# on every pipeline load.
warnings.filterwarnings("ignore", category=UserWarning)

AGENTS_CONFIG = "config/agents.json"
ROUTER_CONFIG = "config/router_config.json"
VERIFIER_KEY = "verifier"
MAX_ATTEMPTS = 2

# "Python" is the routing keyword; "memoization" is the requirement the
# verifier can actually check for. Both matter: one decides where the task
# goes, the other decides whether it comes back done.
REQUEST = "Write a Python function to compute the Fibonacci sequence with memoization."


def snippet(text: str, limit: int = 240) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def test_real_routed_verifier_pipeline() -> None:
    pipeline = create_routed_verified_pipeline(
        AGENTS_CONFIG,
        ROUTER_CONFIG,
        verifier_key=VERIFIER_KEY,
        max_attempts=MAX_ATTEMPTS,
    )

    assert pipeline.max_attempts == MAX_ATTEMPTS
    assert pipeline.router.routes, "router was built with no registered routes"

    task = Task(description=REQUEST, assigned_agent=pipeline.verifier_agent.name)

    started = time.perf_counter()
    result = pipeline.run(task)
    elapsed = time.perf_counter() - started

    print("---- Live Auto-Routed Verifier Trace ----")
    print(f"verifier:      {pipeline.verifier_agent.name} ({pipeline.verifier_agent.role})")
    print(f"routes known:  {', '.join(pipeline.router.routes)}")
    print(f"max_attempts:  {MAX_ATTEMPTS}")
    print(f"request:       {REQUEST}\n")

    for entry in pipeline.history:
        verdict = entry["verdict"]
        status = "COMPLETE" if verdict.is_complete else "INCOMPLETE"
        print(f"[Attempt {entry['attempt']}] route selected: '{entry['chosen_route']}'")
        print(f"  output:    {snippet(entry['workflow_output'])}")
        print(f"  verdict:   {status}")
        print(f"  raw JSON:  {snippet(verdict.raw_response, 160)}")
        print(f"  feedback:  {snippet(verdict.feedback, 200)}")
        if not verdict.is_complete and entry["attempt"] < len(pipeline.history):
            print("  -> retrying: critique injected, task re-routed from scratch")
        print()

    # --- Required assertions ---------------------------------------------

    # NOTE: like the Goal 9 test, this asserts a QUALITY outcome. It fails if
    # the verifier never judges the output complete within MAX_ATTEMPTS - the
    # gate doing its job, not the loop misbehaving. The invariants below hold
    # either way, so read them first when this line goes red.
    assert result.success is True, (
        f"output was never verified in {len(pipeline.history)} attempt(s). "
        f"Status: {pipeline.status_summary()}. "
        f"Last verdict feedback: "
        f"{snippet(pipeline.history[-1]['verdict'].feedback) if pipeline.history else 'n/a'}"
    )
    assert result.output, "pipeline returned an empty deliverable"
    assert result.output.strip(), "pipeline returned only whitespace"
    assert len(pipeline.history) >= 1, pipeline.history

    # --- Loop invariants --------------------------------------------------

    assert pipeline.verified is True
    assert 1 <= pipeline.attempts_used <= MAX_ATTEMPTS
    assert len(pipeline.history) == pipeline.attempts_used

    # Attempts are numbered 1..N with no gaps.
    assert [entry["attempt"] for entry in pipeline.history] == list(
        range(1, len(pipeline.history) + 1)
    )

    # Every attempt records a real route and a fully-populated verdict. A route
    # is either one the router knows about, or the fallback label.
    known_routes = set(pipeline.router.routes) | {"fallback"}
    for entry in pipeline.history:
        assert entry["chosen_route"] in known_routes, entry["chosen_route"]
        assert entry["workflow_output"].strip(), f"attempt {entry['attempt']} produced nothing"
        verdict = entry["verdict"]
        assert isinstance(verdict.is_complete, bool)
        assert verdict.feedback.strip(), f"attempt {entry['attempt']} had an empty critique"
        assert verdict.raw_response, f"attempt {entry['attempt']} recorded no raw reply"

    # Only the final attempt may pass; had an earlier one passed, run() would
    # have returned and never routed again.
    for entry in pipeline.history[:-1]:
        assert not entry["verdict"].is_complete, entry["attempt"]

    # The deliverable is the accepted attempt's output verbatim - run() never
    # rewrites or summarises what the route produced.
    assert result.output == pipeline.history[-1]["workflow_output"]

    # The router logged one decision per attempt, and the pipeline's recorded
    # routes agree with the router's own log.
    assert len(pipeline.router.routing_log) == pipeline.attempts_used
    assert [record["route"] for record in pipeline.router.routing_log] == [
        entry["chosen_route"] for entry in pipeline.history
    ]

    # step_results accounts for every call: one route + one verify per attempt.
    assert len(pipeline.step_results) == 2 * pipeline.attempts_used
    assert all(step.success for step in pipeline.step_results)

    # --- Summary -----------------------------------------------------------

    retries = len(pipeline.history) - 1
    print("[termination]")
    print(f"  status:   {pipeline.status_summary()}")
    print(f"  attempts: {pipeline.attempts_used} ({retries} retry/retries after the first)")
    print(f"  route(s): {' -> '.join(e['chosen_route'] for e in pipeline.history)}")
    print(f"  calls:    {len(pipeline.step_results)} agent call(s)")
    print(f"  elapsed:  {elapsed:.2f}s total\n")

    # The assertions above check routing and gate mechanics. They do NOT check
    # that the Fibonacci code is correct - only that a verifier said it was.
    print("---- Final Verified Output ----")
    print(result.output)


if __name__ == "__main__":
    test_real_routed_verifier_pipeline()
    print("\nAll assertions passed.")
