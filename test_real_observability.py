"""End-to-end trace for Goal 12 (observability) on the local Qwen model.

Run it directly - `python3 test_real_observability.py` - or through pytest.

A two-step sequential workflow runs with a Tracer attached to its shared
memory, exactly as a production run would be wired:

    step 1  AnalystAgent   reads the figures, works out the total through the
                           calculator tool, and publishes the result to shared
                           memory
    step 2  ReportAgent    reads that artifact off the blackboard, summarises
                           it, and writes the summary to a sandboxed file

Between them, one write is attempted with the analyst's read-only permission
set, so the trace carries a security refusal next to the allowed calls. The
whole run is then exported to run_trace.json and printed as a timeline.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

# transformers builds a few tensors during model load that torch flags as
# UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

from agent_framework.agents import LLMAgent
from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.observability import (
    EventType,
    Tracer,
    WorkflowTrace,
    print_run_summary,
    resolve_tracer,
)
from agent_framework.orchestration.sequential import (
    LLM_PARAMS,
    SequentialOrchestrator,
    build_agent_kwargs,
    load_agent_config,
)
from agent_framework.tools import ToolRegistry, ToolStatus
from agent_framework.tools.builtin import FileSandbox, build_builtin_tools

AGENTS_CONFIG = "config/agents.json"

# The two config entries under test. Names, prompts, tool assignments and
# permission grants all come from agents.json; only model settings are
# overridden here.
ANALYST_KEY = "analyst"  # calculator + file_read, read_only
WRITER_KEY = "writer"  # file_read + file_list + file_write, read_only + write

# Where the trace is exported. Relative to the working directory, as the
# requirement states it.
RUN_TRACE = Path("run_trace.json")

# The figures the analyst is handed, and the arithmetic they imply. The test
# computes the total itself so no assertion depends on the model's arithmetic.
FIGURES = {"north": 1240, "south": 860, "east": 1575, "west": 930}
CANONICAL_EXPRESSION = " + ".join(str(value) for value in FIGURES.values())
EXPECTED_TOTAL = sum(FIGURES.values())

# The artifact step 1 publishes and step 2 consumes.
RESULT_KEY = "q3_total"

# Where step 2 writes its summary, inside the sandbox.
SUMMARY_FILE = "reports/q3_summary.md"

# The path the unauthorized attempt targets. Nothing must ever create it.
FORBIDDEN_FILE = "reports/analyst_leak.md"


def snippet(text: str, limit: int = 220) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def extract_expression(reply: str) -> Optional[str]:
    """Pull a `calculator(...)` expression out of the model's reply, if it made one."""
    match = re.search(r"calculator\(\s*([^)]+?)\s*\)", reply or "")
    if not match:
        return None
    candidate = match.group(1).strip().strip("'\"")
    if not candidate or not re.fullmatch(r"[\d\s+\-*/%.()]+", candidate):
        return None
    return candidate


def build_registry(root: Path) -> ToolRegistry:
    """The built-in tools over a sandbox rooted at `root`."""
    return ToolRegistry(build_builtin_tools(sandbox=FileSandbox(root)))


def agent_kwargs(registry: ToolRegistry, key: str, **model_overrides) -> dict:
    """Constructor arguments for one config entry, with model settings overridden.

    Passing "tools" or "permissions" here is refused: the grants an agent
    carries into the trace must be exactly the ones in its config entry.
    """
    illegal = set(model_overrides) - set(LLM_PARAMS)
    if illegal:
        raise ValueError(f"only model settings may be overridden, not {sorted(illegal)}")
    entry = {**load_agent_config(AGENTS_CONFIG)[key], **model_overrides}
    return build_agent_kwargs(entry, tool_registry=registry)


# --- The two workflow steps ---
#
# LLMAgent produces text; it does not act on it. These two subclasses add the
# tool use each step needs, going through `use_tool` so every call is
# permission-checked, audit-logged and traced like any other.


class CalculatingAnalyst(LLMAgent):
    """Step 1: asks the model for the arithmetic, runs it, publishes the total."""

    def execute(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None
    ) -> RunResult:
        reply = super().execute(task, shared_memory)
        if not reply.success:
            return reply

        # The model's own expression when it produced a parseable one; the
        # canonical one otherwise. A 1.5B checkpoint formatting its arithmetic
        # as asked is not what this test verifies.
        expression = extract_expression(reply.output) or CANONICAL_EXPRESSION
        calculated = self.use_tool(
            "calculator", shared_memory=shared_memory, expression=expression
        )
        if not calculated["success"]:
            return RunResult(task_id=task.id, success=False, error=calculated["error"])

        total = calculated["output"]
        if shared_memory is not None:
            shared_memory.set(
                RESULT_KEY,
                {"expression": expression, "total": total, "analysis": reply.output},
            )
            # The blackboard the moment the artifact landed: this is the state
            # step 2 will be handed.
            tracer = resolve_tracer(shared_memory)
            if tracer is not None:
                tracer.record_snapshot(shared_memory, label="after step 1")

        return RunResult(
            task_id=task.id,
            success=True,
            output=f"{reply.output}\n\ncalculator({expression}) = {total}",
        )


class SummarisingWriter(LLMAgent):
    """Step 2: reads the published total, summarises it, writes the summary to disk."""

    def execute(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None
    ) -> RunResult:
        artifact = shared_memory.get(RESULT_KEY) if shared_memory is not None else None
        if artifact is None:
            return RunResult(
                task_id=task.id, success=False, error=f"no '{RESULT_KEY}' artifact to summarise"
            )

        # LLMAgent pastes the blackboard into the prompt already; the task
        # text just says what to do with it.
        reply = super().execute(task, shared_memory)
        if not reply.success:
            return reply

        content = (
            f"# Q3 summary\n\nTotal: {artifact['total']} "
            f"(from {artifact['expression']})\n\n{reply.output.strip()}\n"
        )
        written = self.use_tool(
            "file_write", shared_memory=shared_memory, path=SUMMARY_FILE, content=content
        )
        if not written["success"]:
            return RunResult(task_id=task.id, success=False, error=written["error"])

        if shared_memory is not None:
            shared_memory.set("summary_file", written["output"]["path"])
        return RunResult(task_id=task.id, success=True, output=content)


def test_real_observability_end_to_end() -> None:
    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = build_registry(root)

        # One blackboard for the run, with the tracer attached to it: every
        # agent step, tool call and snapshot lands on the same trace whichever
        # component produced it.
        shared = SharedWorkflowMemory()
        tracer = Tracer.for_memory(
            shared, run_id="real-observability", workflow="sequential", model="qwen2.5-1.5b"
        )
        assert shared.tracer is tracer and resolve_tracer(shared) is tracer

        analyst = CalculatingAnalyst(**agent_kwargs(registry, ANALYST_KEY, max_tokens=160))
        writer = SummarisingWriter(**agent_kwargs(registry, WRITER_KEY, max_tokens=160))
        assert analyst.tool_names() == ["calculator", "file_read"]
        assert "file_write" in writer.tool_names()

        pipeline = SequentialOrchestrator([analyst, writer])

        print("---- Live Observability Trace ----")
        print(f"sandbox:  {root}")
        print(f"run id:   {tracer.trace.run_id}")
        for agent in pipeline.agents:
            print(f"{agent.name + ':':<14}tools={agent.tool_names()} "
                  f"holds={sorted(p.value for p in agent.permissions)}")
        print()

        tracer.record_snapshot(shared, label="before run")

        # --- The workflow ---
        figures = "\n".join(f"{region}: {value}" for region, value in FIGURES.items())
        task = Task(
            description=(
                "Here are the Q3 regional revenue figures (thousands USD):\n\n"
                f"{figures}\n\n"
                "State the total revenue across all four regions. Give the "
                "calculator expression you would evaluate, on its own line in "
                "the form calculator(<expression>), then the answer."
            ),
            assigned_agent=analyst.name,
        )
        run_started = time.perf_counter()
        result = pipeline.run(task, shared)
        run_elapsed = time.perf_counter() - run_started

        assert result.success is True, result.error
        assert len(pipeline.step_results) == 2
        assert all(step.success for step in pipeline.step_results)
        for agent, step in zip(pipeline.agents, pipeline.step_results):
            print(f"[{agent.name}] {snippet(step.output)}")

        # Step 1 published the total; step 2 read it and wrote the file.
        artifact = shared.get(RESULT_KEY)
        assert artifact is not None and artifact["total"] == EXPECTED_TOTAL, artifact
        summary_path = root / SUMMARY_FILE
        assert summary_path.is_file()
        assert summary_path.read_text(encoding="utf-8") == pipeline.step_results[1].output
        assert str(EXPECTED_TOTAL) in summary_path.read_text(encoding="utf-8")
        assert shared.get("summary_file") == SUMMARY_FILE
        print(f"\n{RESULT_KEY} = {artifact['total']}  (via calculator({artifact['expression']}))")
        print(f"summary written to {SUMMARY_FILE}")

        # --- One unauthorized attempt ---
        # The analyst holds read_only. A write made with its own (immutable)
        # permission set is refused at the registry's tier check, and that
        # refusal is timed and traced like every other call.
        blocked = registry.execute_tool(
            analyst.name,
            "file_write",
            analyst.permissions,
            shared_memory=shared,
            path=FORBIDDEN_FILE,
            content=f"total={artifact['total']}",
        )
        assert blocked["success"] is False
        assert blocked["status"] == ToolStatus.UNAUTHORIZED.value
        assert not (root / FORBIDDEN_FILE).exists()
        print(f"blocked: {analyst.name} -> file_write ({snippet(blocked['error'], 120)})")

        tracer.record_snapshot(shared, label="after run")
        trace = tracer.finish()

        # --- Export and print ---
        trace.export_json(RUN_TRACE)
        assert RUN_TRACE.is_file()
        payload = json.loads(RUN_TRACE.read_text(encoding="utf-8"))
        assert payload["run_id"] == trace.run_id
        assert len(payload["events"]) == len(trace.events)
        restored = WorkflowTrace.load_json(RUN_TRACE)
        assert restored.to_dict() == trace.to_dict()

        print_run_summary(trace, json_path=RUN_TRACE, export=False)

        # --- Assertions on the trace ---
        assert len(trace.events) >= 3
        assert trace.status == "completed"

        kinds = {event.event_type for event in trace.events}
        assert EventType.AGENT_CALL in kinds
        assert EventType.TOOL_CALL in kinds
        assert EventType.STATE_SNAPSHOT in kinds

        # Both steps were traced, in order, as successes.
        agent_calls = trace.events_of(EventType.AGENT_CALL)
        assert [event.agent_name for event in agent_calls] == [analyst.name, writer.name]
        assert all(event.success is True for event in agent_calls)
        assert agent_calls[0].data["output"] == pipeline.step_results[0].output

        # The calculator, the write, and the refusal - in that order.
        tool_calls = trace.events_of(EventType.TOOL_CALL)
        assert [(event.data["tool"], event.data["status"]) for event in tool_calls] == [
            ("calculator", "ok"),
            ("file_write", "ok"),
            ("file_write", "unauthorized"),
        ]
        assert tool_calls[0].agent_name == analyst.name
        assert tool_calls[0].data["output"] == EXPECTED_TOTAL
        assert tool_calls[1].agent_name == writer.name
        assert tool_calls[1].data["output"]["path"] == SUMMARY_FILE

        refused = tool_calls[2]
        assert refused.agent_name == analyst.name
        assert refused.success is False
        assert refused.data["required_permission"] == "write"
        assert "not permitted" in refused.error
        assert refused.data["arguments"]["path"] == FORBIDDEN_FILE
        assert [event.event_type for event in trace.errors()] == [EventType.TOOL_CALL]

        # The snapshots show the artifact appearing between the two steps.
        snapshots = trace.events_of(EventType.STATE_SNAPSHOT)
        assert [snap.data["label"] for snap in snapshots] == [
            "before run", "after step 1", "after run",
        ]
        assert snapshots[0].data["artifact_keys"] == []
        assert snapshots[1].data["artifact_keys"] == [RESULT_KEY]
        assert snapshots[1].data["artifacts"][RESULT_KEY]["total"] == EXPECTED_TOTAL
        assert snapshots[2].data["artifact_keys"] == [RESULT_KEY, "summary_file"]

        # Every timed event took real time. Snapshots are instants and carry
        # no duration.
        timed = [event for event in trace.events if event.duration is not None]
        assert len(timed) == len(agent_calls) + len(tool_calls)
        for event in timed:
            assert event.duration > 0, event
        assert all(snap.duration is None for snap in snapshots)
        # A real model call dwarfs a tool call.
        assert min(e.duration for e in agent_calls) > max(e.duration for e in tool_calls)
        assert trace.end_time is not None and trace.wall_time >= sum(e.duration for e in agent_calls)

        # The summary agrees with the events.
        summary = trace.summary()
        assert summary["status"] == "completed"
        assert summary["counts"]["agent_call"] == 2
        assert summary["counts"]["tool_call"] == 3
        assert summary["counts"]["state_snapshot"] == 3
        assert summary["tools"] == {
            "total_calls": 3, "allowed": 2, "blocked": 1, "failed": 0,
            "by_tool": {
                "calculator": {"calls": 1, "allowed": 1, "blocked": 0, "failed": 0},
                "file_write": {"calls": 2, "allowed": 1, "blocked": 1, "failed": 0},
            },
        }
        assert set(summary["agents"]) == {analyst.name, writer.name}
        assert summary["metadata"] == {"workflow": "sequential", "model": "qwen2.5-1.5b"}

        # The audit log is untouched by tracing: two steps and three tool calls.
        kinds_logged = [record.get("kind", "step") for record in shared.execution_log]
        assert kinds_logged == ["tool_call", "step", "tool_call", "step", "tool_call"]

        # Nothing but the summary exists in the sandbox.
        on_disk = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        assert on_disk == [SUMMARY_FILE], on_disk

        elapsed = time.perf_counter() - started
        print(f"\n[summary] {len(trace.events)} events, "
              f"{summary['tools']['allowed']} allowed / {summary['tools']['blocked']} blocked "
              f"tool call(s), workflow {run_elapsed:.2f}s, {elapsed:.2f}s total")


if __name__ == "__main__":
    test_real_observability_end_to_end()
    print("\nAll assertions passed.")
