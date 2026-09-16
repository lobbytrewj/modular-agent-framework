
from __future__ import annotations

import re
import time
import warnings
from typing import Optional


warnings.filterwarnings("ignore", category=UserWarning)

from agent_framework.agents import LLMAgent, execute_agent
from agent_framework.core.memory import AgentMemory, SharedWorkflowMemory
from agent_framework.core.tasks import Task
from agent_framework.orchestration.sequential import build_agent_kwargs, load_agent_config
from agent_framework.utils.memory_inspector import print_full_memory_trace

AGENTS_CONFIG = "config/agents.json"

GENERATOR_KEY = "coder_generator"   # step 1: writes the helper
EVALUATOR_KEY = "code_evaluator"    # step 2: reads it off the blackboard

# The one string both steps agree on. It is the entire interface between them.
ARTIFACT_KEY = "code_artifact"
REVIEW_KEY = "review_artifact"

# Small enough that a 1.5B model finishes it inside 128 tokens.
GENERATION_REQUEST = (
    "Write a single Python helper function named add_numbers(a, b) that "
    "returns the sum of two numbers. Include type annotations and a one-line "
    "docstring. Reply with the function definition only - no examples, no "
    "tests, no commentary."
)

REVIEW_REQUEST = (
    "The shared workflow context above contains a helper function an earlier "
    "step of this workflow already wrote. Review it. Say what it does, whether "
    "it is correct, and give one example call with the result it returns. "
    "Refer to the function by the exact name it was defined with. Do not "
    "rewrite the function."
)


def snippet(text: str, limit: int = 200) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def first_function_name(code: str) -> Optional[str]:
    """The name of the first function defined in `code`, if there is one.
    """
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code or "")
    return match.group(1) if match else None


def build_agent(key: str, **overrides) -> LLMAgent:
    """Instantiate one configured agent, with per-test overrides applied."""
    entry = {**load_agent_config(AGENTS_CONFIG)[key], **overrides}
    return LLMAgent(**build_agent_kwargs(entry))


def test_trace_memory_transfer() -> None:
    started = time.perf_counter()

    # One blackboard for the whole run, created empty. Nothing else is shared
    # between the two steps below - not an orchestrator, not an agent.
    shared = SharedWorkflowMemory()
    assert shared.artifacts == {} and shared.execution_log == []
    assert shared.to_prompt_context() == "", "an empty board must render nothing"

    # Step 1 - the generator writes the helper
    generator = build_agent(
        GENERATOR_KEY,
        max_tokens=128,
        device="mps",
        memory=True,            # its own AgentMemory, private to it
        output_key=ARTIFACT_KEY,  # publishes itself on the way out
    )
    assert isinstance(generator.memory, AgentMemory) and len(generator.memory) == 0

    task_1 = Task(description=GENERATION_REQUEST, assigned_agent=generator.name)

    step_1_started = time.perf_counter()
    # execute_agent rather than generator.execute: it forwards the blackboard
    # and writes this step's entry on the shared audit log.
    result_1 = execute_agent(generator, task_1, shared)
    step_1_elapsed = time.perf_counter() - step_1_started

    assert result_1.success is True, f"generation failed: {result_1.error}"
    assert result_1.output and result_1.output.strip(), "generator returned nothing"

    # The artifact landed on the board under the agreed name. (output_key did
    # this; shared.set(ARTIFACT_KEY, result_1.output) is the explicit form.)
    assert shared.get(ARTIFACT_KEY) == result_1.output

    function_name = first_function_name(result_1.output)

    # Step 2 - the evaluator reads what step 1 left behind
    evaluator = build_agent(
        EVALUATOR_KEY,
        max_tokens=128,
        device="mps",
        memory=True,
        context_keys=[ARTIFACT_KEY],  # a narrowed view of the board
        output_key=REVIEW_KEY,
    )
    assert evaluator is not generator
    assert evaluator.memory is not generator.memory

    task_2 = Task(description=REVIEW_REQUEST, assigned_agent=evaluator.name)

    # The claim the whole test rests on: step 2's prompt, as the test wrote
    # it, contains none of step 1's work.
    assert "def " not in task_2.description
    if function_name:
        assert function_name not in task_2.description, (
            "the review request names the function, so this would pass even "
            "with an empty blackboard"
        )

    step_2_started = time.perf_counter()
    result_2 = execute_agent(evaluator, task_2, shared)
    step_2_elapsed = time.perf_counter() - step_2_started

    assert result_2.success is True, f"evaluation failed: {result_2.error}"
    assert result_2.output and result_2.output.strip(), "evaluator returned nothing"

    elapsed = time.perf_counter() - started

    # The trace - blackboard, injected prompt, audit trail, private memories
    print()
    print(f"[step 1] {generator.name} generated in {step_1_elapsed:.2f}s")
    print(result_1.output)
    print()
    print(f"[step 2] {evaluator.name} reviewed in {step_2_elapsed:.2f}s, having "
          f"been told nothing about the function except where to find it")
    print(result_2.output)

    print_full_memory_trace(shared, agents=[generator, evaluator], truncate_chars=300)

    # Assertions - the blackboard
    assert shared.has(ARTIFACT_KEY) is True
    assert ARTIFACT_KEY in shared  # __contains__ is has()
    assert shared.has(REVIEW_KEY) is True
    assert shared.get(REVIEW_KEY) == result_2.output
    assert shared.get("never_written", "absent") == "absent"
    assert "def " in shared.get(ARTIFACT_KEY), (
        "step 1 produced no function definition, so there was nothing for "
        f"step 2 to read: {snippet(result_1.output)}"
    )
    assert function_name is not None, (
        f"no function definition found in step 1's output: {snippet(result_1.output)}"
    )

    # Assertions - what was injected into step 2's prompt
    context = shared.to_prompt_context([ARTIFACT_KEY])
    assert "## Shared workflow context" in context
    assert f"### {ARTIFACT_KEY}" in context
    assert f"def {function_name}" in context
    assert REVIEW_KEY not in context, "context_keys must narrow the view"

    # The prompt the evaluator was actually sent, recorded in its own memory,
    # carries the code even though the Task the test wrote did not. This is
    # the mechanical half of the proof and holds regardless of review quality.
    sent_prompt = evaluator.memory.messages[0].content
    assert "## Shared workflow context" in sent_prompt
    assert f"def {function_name}" in sent_prompt
    assert sent_prompt.rstrip().endswith(REVIEW_REQUEST), (
        "the instruction must come last, after the injected context"
    )

    # Assertions - the audit trail spans both steps
    log = shared.execution_log
    assert len(log) >= 1
    assert len(log) == 2, [record["agent"] for record in log]
    assert [record["step"] for record in log] == [1, 2]
    assert [record["agent"] for record in log] == [generator.name, evaluator.name]
    assert all(record["success"] for record in log)
    assert log[0]["output"] == result_1.output
    assert log[1]["output"] == result_2.output
    assert log[0]["timestamp"] <= log[1]["timestamp"]
    assert len(shared.log_for(evaluator.name)) == 1

    # Assertions - agent-local memory stayed private
    for agent, result in ((generator, result_1), (evaluator, result_2)):
        assert isinstance(agent.memory, AgentMemory)
        assert [m.role for m in agent.memory.messages] == ["user", "assistant"]
        assert agent.memory.messages[1].content == result.output

    generator_history = generator.memory.get_context_string()
    evaluator_history = evaluator.memory.get_context_string()
    assert REVIEW_REQUEST not in generator_history
    assert GENERATION_REQUEST not in evaluator_history
    assert result_2.output not in generator_history

    print()
    print("[summary]")
    print(f"  artifacts: {shared.keys()}, function {function_name!r}")
    print(f"  log:       {len(log)} step(s) on one shared trail")
    print(f"  memory:    generator {len(generator.memory)} turn(s), "
          f"evaluator {len(evaluator.memory)} turn(s), kept separate")
    print(f"  elapsed:   {step_1_elapsed:.2f}s + {step_2_elapsed:.2f}s = {elapsed:.2f}s total")


if __name__ == "__main__":
    test_trace_memory_transfer()
    print("\nAll assertions passed.")
