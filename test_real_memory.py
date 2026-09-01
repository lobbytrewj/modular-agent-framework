"""Live integration test for Goal 10: agent-local memory and shared state.

Runs real local inference - no mocks, no fake clients - so what it proves is
that a deliverable produced by one agent in one workflow execution can be
picked up, by name, by a *different* agent in a *separate* execution, with the
code travelling through SharedWorkflowMemory rather than through the prompt the
test wrote.

The two things being demonstrated are easy to conflate, so the test keeps them
visibly apart:

    AGENT-LOCAL MEMORY   each agent's own AgentMemory - the user/assistant
                         turns it personally spoke. Private: the tester never
                         sees the generator's turns.
    SHARED WORKFLOW      one SharedWorkflowMemory scratchpad, created once and
    MEMORY               handed to both executions. Holds the deliverable
                         ("code_artifact") plus the audit log of every step.

Design of the proof:

    Workflow A  create_sequential_pipeline([coder_generator]) writes a Python
                function. The test publishes it as "code_artifact".
    Workflow B  a separately constructed CodingAgent - different config key,
                different LLMAgent instance, different client, its own memory,
                executed on its own outside any orchestrator - is asked to
                write unit tests. Its Task description does NOT contain the
                code, and the test asserts that. The only path by which the
                function can reach it is to_prompt_context(), which
                LLMAgent._build_user_message prepends automatically.

    So if the tester's output names the function, the artifact crossed the
    boundary between two independent executions. That is the whole claim.

Required config/agents.json entries (already present):

    "coder_generator": {"name": "CoderGeneratorAgent", "role": "generator",
                        "model": "Qwen/Qwen2.5-1.5B-Instruct",
                        "device": "mps", "temperature": 0.1, "max_tokens": 256}
    "coder":           {"name": "CodingAgent", "role": "coding",
                        ... same model/device ...}

Use "device": "cpu" instead of "mps" on a machine without Apple Silicon.
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Optional

# transformers builds a few tensors during model load that torch flags as
# UserWarnings (dtype/meta-device notices). Silence them before the model is
# touched, or they bury the trace this test exists to print.
warnings.filterwarnings("ignore", category=UserWarning)

from agent_framework.agents import LLMAgent, execute_agent
from agent_framework.core.memory import AgentMemory, SharedWorkflowMemory
from agent_framework.core.tasks import Task
from agent_framework.orchestration.sequential import (
    build_agent_kwargs,
    create_sequential_pipeline,
    load_agent_config,
)

AGENTS_CONFIG = "config/agents.json"

# Two different config keys, so the two halves of the run are genuinely
# different agents rather than one agent called twice.
GENERATOR_KEY = "coder_generator"   # workflow A: writes the function
TESTER_KEY = "coder"                # workflow B: writes tests for it

# The name of the artifact on the blackboard. Both halves agree on this string
# and on nothing else - that agreement is the entire interface between them.
ARTIFACT_KEY = "code_artifact"

GENERATION_REQUEST = (
    "Write a single Python function named moving_average(values, window) that "
    "returns the simple moving average of a list of numbers over the given "
    "window size. Include type annotations and a docstring, and raise "
    "ValueError when window is not a positive integer or is larger than the "
    "list. Reply with the function definition only - no examples, no tests, "
    "no commentary."
)

# Deliberately says nothing about what the function is called, what it does, or
# what it looks like. Every one of those facts has to arrive from the
# scratchpad, which is what makes this a test of shared state rather than of
# the model's ability to follow a self-contained prompt.
TESTING_REQUEST = (
    "The shared workflow context above contains a function that an earlier "
    "step of this workflow already wrote. Write Python unittest test cases for "
    "that exact function. Call it by the name it was actually defined with, "
    "and cover both a normal case and the error case it raises. Reply with "
    "test code only."
)


def snippet(text: str, limit: int = 240) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def first_function_name(code: str) -> Optional[str]:
    """The name of the first function defined in `code`, if there is one.

    Read out of the generated text rather than assumed from the prompt: a small
    model asked for `moving_average` may still hand back `movingAverage`, and
    the point of the test is whether the tester saw the SAME function, not
    whether the generator obeyed a naming instruction.
    """
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code or "")
    return match.group(1) if match else None


def test_real_cross_workflow_memory() -> None:
    started = time.perf_counter()

    # One scratchpad for the whole run. Nothing else is shared between the two
    # workflows below - not an orchestrator, not an agent, not a client.
    shared = SharedWorkflowMemory()
    assert shared.artifacts == {} and shared.execution_log == []
    assert shared.to_prompt_context() == "", "an empty board must render nothing"

    # =====================================================================
    # Workflow A - generation
    # =====================================================================
    workflow_a = create_sequential_pipeline(AGENTS_CONFIG, [GENERATOR_KEY])
    generator = workflow_a.agents[0]

    # Agent-local memory, attached after construction. (The same thing can be
    # asked for in config/agents.json with "memory": true - workflow B below
    # does it that way.)
    generator.memory = AgentMemory()

    task_a = Task(description=GENERATION_REQUEST, assigned_agent=generator.name)

    step_a_started = time.perf_counter()
    result_a = workflow_a.run(task_a, shared_memory=shared)
    step_a_elapsed = time.perf_counter() - step_a_started

    assert result_a.success is True, f"generation failed: {result_a.error}"
    assert result_a.output and result_a.output.strip(), "generator returned nothing"

    # Publish the deliverable under the agreed name. This is the explicit form;
    # an agent configured with "output_key": "code_artifact" publishes itself
    # on the way out, without the caller having to remember to.
    shared.set(ARTIFACT_KEY, result_a.output)

    function_name = first_function_name(result_a.output)

    # =====================================================================
    # Workflow B - a separate execution, reading what A left behind
    # =====================================================================
    config = load_agent_config(AGENTS_CONFIG)
    tester_entry = {
        **config[TESTER_KEY],
        # Unit tests need more room than the 128 tokens this agent is
        # configured with for one-shot fixes.
        "max_tokens": 256,
        # Config-driven memory: its own AgentMemory, and a narrowed view of the
        # board so only the artifact it actually needs reaches its prompt.
        "memory": True,
        "context_keys": [ARTIFACT_KEY],
    }
    tester = LLMAgent(**build_agent_kwargs(tester_entry))

    assert tester is not generator
    assert tester.llm_client is not generator.llm_client
    assert isinstance(tester.memory, AgentMemory) and len(tester.memory) == 0

    task_b = Task(description=TESTING_REQUEST, assigned_agent=tester.name)

    # The claim this test rests on: the second half's prompt, as written by the
    # test, contains none of the first half's work.
    assert "def " not in task_b.description
    if function_name:
        assert function_name not in task_b.description, (
            "the testing request names the function, so the test would pass "
            "even with an empty scratchpad"
        )

    # execute_agent rather than tester.execute: workflow B has no orchestrator,
    # and this is the call that both forwards the board and appends workflow
    # B's step to the same audit log workflow A wrote to.
    step_b_started = time.perf_counter()
    result_b = execute_agent(tester, task_b, shared)
    step_b_elapsed = time.perf_counter() - step_b_started

    elapsed = time.perf_counter() - started

    # =====================================================================
    # Trace
    # =====================================================================
    print("---- Live Cross-Workflow Memory Trace ----")
    print(f"workflow A: {generator.name} ({generator.role}) via SequentialOrchestrator")
    print(f"workflow B: {tester.name} ({tester.role}) executed standalone")
    print(f"artifact key: {ARTIFACT_KEY!r}, function detected: {function_name!r}\n")

    print(f"[workflow A] generated in {step_a_elapsed:.2f}s")
    print(result_a.output)
    print()

    print("---- SharedWorkflowMemory scratchpad ----")
    print(f"artifact keys: {shared.keys()}")
    print(f"has({ARTIFACT_KEY!r}): {shared.has(ARTIFACT_KEY)}")
    print("\n-- to_prompt_context() as workflow B received it --")
    print(shared.to_prompt_context([ARTIFACT_KEY]))
    print("\n-- execution_log (both workflows, one ordered trail) --")
    for record in shared.execution_log:
        status = "OK" if record["success"] else "FAILED"
        print(
            f"  step {record['step']}: {record['agent']} ({record.get('role')}) "
            f"({status}) - {snippet(record['task'], 80)}"
        )
    print()

    print(f"[workflow B] wrote tests in {step_b_elapsed:.2f}s, having been told "
          f"nothing about the function except where to find it")
    print(result_b.output)
    print()

    # =====================================================================
    # Assertions - the artifact
    # =====================================================================
    assert shared.has(ARTIFACT_KEY) is True
    assert shared.get(ARTIFACT_KEY) == result_a.output
    assert "def " in shared.get(ARTIFACT_KEY), (
        "workflow A produced no function definition, so there is nothing for "
        f"workflow B to have read: {snippet(result_a.output)}"
    )
    assert shared.get("never_written", "absent") == "absent"
    assert ARTIFACT_KEY in shared  # __contains__ is has()

    assert function_name is not None, (
        f"no function definition found in workflow A's output: {snippet(result_a.output)}"
    )

    # The rendered context is what actually reached the model: a labelled
    # markdown section carrying the artifact verbatim.
    context = shared.to_prompt_context([ARTIFACT_KEY])
    assert "## Shared workflow context" in context
    assert f"### {ARTIFACT_KEY}" in context
    assert f"def {function_name}" in context

    # =====================================================================
    # Assertions - the artifact crossed the workflow boundary
    # =====================================================================
    assert result_b.success is True, f"tester failed: {result_b.error}"
    assert result_b.output and result_b.output.strip(), "tester returned nothing"

    # The prompt the tester was actually sent - recorded in its own memory -
    # contains the code, even though the Task the test wrote did not. This is
    # the mechanical half of the proof, and it holds regardless of how well the
    # model then performed.
    sent_prompt = tester.memory.messages[0].content
    assert "## Shared workflow context" in sent_prompt
    assert f"def {function_name}" in sent_prompt
    assert TESTING_REQUEST in sent_prompt
    assert sent_prompt.rstrip().endswith(TESTING_REQUEST), (
        "the instruction must come last, after the injected context"
    )

    # The behavioural half: the model used what it was given. NOTE this asserts
    # a QUALITY outcome - it fails if a 1.5B model wrote tests for a function it
    # renamed or invented, which is a fact about the model, not a defect in the
    # memory plumbing. The mechanical assertions above hold either way, so read
    # those first when this line goes red.
    assert function_name in result_b.output, (
        f"workflow B never mentioned {function_name!r}, the function workflow A "
        f"defined and the scratchpad handed it: {snippet(result_b.output)}"
    )

    # =====================================================================
    # Assertions - agent-local memory
    # =====================================================================
    # One user turn and one assistant turn per execution, in that order, with
    # the reply recorded verbatim.
    for agent, result in ((generator, result_a), (tester, result_b)):
        assert isinstance(agent.memory, AgentMemory)
        assert len(agent.memory) == 2, [m.role for m in agent.memory.messages]
        assert [m.role for m in agent.memory.messages] == ["user", "assistant"]
        assert agent.memory.messages[1].content == result.output
        assert agent.memory.messages[0].timestamp <= agent.memory.messages[1].timestamp

    assert GENERATION_REQUEST in generator.memory.messages[0].content
    assert TESTING_REQUEST in tester.memory.messages[0].content

    # Private by construction: neither agent's history contains the other's.
    generator_history = generator.memory.get_context_string()
    tester_history = tester.memory.get_context_string()
    assert TESTING_REQUEST not in generator_history
    assert GENERATION_REQUEST not in tester_history
    assert result_b.output not in generator_history
    assert generator.memory is not tester.memory

    # get_context_string is the flattened form, and it renders both turns.
    assert generator_history.startswith("Conversation so far:")
    assert generator.memory.get_context_string(last_k=1).count("Assistant:") == 1
    assert "User:" not in generator.memory.get_context_string(last_k=1)

    # =====================================================================
    # Assertions - the shared audit log spans both workflows
    # =====================================================================
    log = shared.execution_log
    assert len(log) == 2, [record["agent"] for record in log]
    assert [record["step"] for record in log] == [1, 2]
    assert [record["agent"] for record in log] == [generator.name, tester.name]
    assert all(record["success"] for record in log)
    assert log[0]["output"] == result_a.output
    assert log[1]["output"] == result_b.output
    assert log[0]["timestamp"] <= log[1]["timestamp"]
    assert len(shared.log_for(tester.name)) == 1

    # =====================================================================
    # Summary
    # =====================================================================
    print("[summary]")
    print(f"  artifact:  {ARTIFACT_KEY!r} -> {len(shared.get(ARTIFACT_KEY))} chars, "
          f"function {function_name!r}")
    print(f"  log:       {len(log)} step(s) across 2 independent executions")
    print(f"  memory:    generator {len(generator.memory)} turn(s), "
          f"tester {len(tester.memory)} turn(s), kept separate")
    print(f"  elapsed:   {step_a_elapsed:.2f}s + {step_b_elapsed:.2f}s = {elapsed:.2f}s total")


if __name__ == "__main__":
    test_real_cross_workflow_memory()
    print("\nAll assertions passed.")
