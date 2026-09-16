"""Trace for Goal 12: observability and debugging.

Run it directly - `python3 tests/test_observability.py`
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_framework.agents import MockAgent, execute_agent
from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.observability import (
    EventType,
    TraceEvent,
    Tracer,
    WorkflowTrace,
    format_run_summary,
    get_active_tracer,
    print_run_summary,
    resolve_tracer,
)
from agent_framework.orchestration.parallel import ParallelOrchestrator
from agent_framework.orchestration.routed_verified_pipeline import AutoRoutedVerifiedPipeline
from agent_framework.orchestration.router import RouterOrchestrator
from agent_framework.tools import ToolRegistry, ToolStatus
from agent_framework.tools.builtin import FileSandbox, build_builtin_tools


def make_registry(root: Path) -> ToolRegistry:
    return ToolRegistry(build_builtin_tools(sandbox=FileSandbox(root)))


class ScriptedVerifier(MockAgent):
    """A verifier that replies with a scripted sequence of JSON verdicts."""

    def __init__(self, verdicts: list[str], **kwargs):
        super().__init__(**kwargs)
        self.verdicts = list(verdicts)

    def execute(self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None) -> RunResult:
        reply = self.verdicts.pop(0) if self.verdicts else '{"is_complete": true, "feedback": ""}'
        return RunResult(task_id=task.id, success=True, output=reply)


class ExplodingAgent(MockAgent):
    def execute(self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None) -> RunResult:
        raise RuntimeError("model fell over")


# --- WorkflowTrace ---


def test_trace_serializes_to_and_from_json() -> None:
    trace = WorkflowTrace(run_id="run-1", metadata={"workflow": "demo", "seed": 7})
    trace.record_event(EventType.AGENT_CALL, agent_name="A", duration=0.25, task="t", output="o", success=True)
    trace.record_event(EventType.TOOL_CALL, agent_name="A", duration=0.01, tool="calculator",
                       arguments={"expression": "1+1"}, status="ok", success=True, output=2)
    trace.record_event(EventType.ROUTE_DECISION, agent_name="router", route="math", reasoning="keyword 'sum'")
    trace.record_event(EventType.DELEGATION, agent_name="router", to="worker", task="sub")
    trace.record_event(EventType.VERIFICATION, agent_name="V", attempt=1, is_complete=False, feedback="short")
    trace.record_event("state_snapshot", artifact_keys=["draft"], artifacts={"draft": "x" * 10})
    trace.finish()

    # Every event type has a stable id, epoch timestamp and typed kind.
    assert [event.event_type for event in trace.events] == list(EventType)
    assert all(event.event_id.startswith("run-1-") for event in trace.events)
    assert all(isinstance(event.timestamp, float) for event in trace.events)
    assert trace.end_time is not None and trace.wall_time >= 0

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "trace.json"
        trace.export_json(path)  # creates the parent directory
        assert path.is_file()

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["run_id"] == "run-1" and raw["metadata"] == {"workflow": "demo", "seed": 7}
        assert [item["event_type"] for item in raw["events"]] == [kind.value for kind in EventType]

        restored = WorkflowTrace.load_json(path)

    # The round trip is lossless: same dict, same typed events, same summary.
    assert restored.to_dict() == trace.to_dict()
    assert isinstance(restored.events[0], TraceEvent)
    assert restored.events[0].event_type is EventType.AGENT_CALL
    assert restored.events[1].data["arguments"] == {"expression": "1+1"}
    assert restored.summary() == trace.summary()

    summary = trace.summary()
    assert summary["status"] == "completed"
    assert summary["counts"] == {kind.value: 1 for kind in EventType}
    assert summary["durations"]["agent_call"]["total"] == 0.25
    assert summary["tools"] == {
        "total_calls": 1, "allowed": 1, "blocked": 0, "failed": 0,
        "by_tool": {"calculator": {"calls": 1, "allowed": 1, "blocked": 0, "failed": 0}},
    }
    assert summary["routes"] == [{"router": "router", "route": "math", "reasoning": "keyword 'sum'"}]
    assert summary["verdicts"] == [{"verifier": "V", "attempt": 1, "is_complete": False}]

    # Unserialisable data is coerced, never fatal, and long text is capped.
    trace.record_event(EventType.AGENT_CALL, agent_name="B", weird=object(), output="y" * 10_000)
    payload = json.loads(trace.to_json())
    assert payload["events"][-1]["data"]["weird"].startswith("<object")
    assert "truncated, 10000 chars" in payload["events"][-1]["data"]["output"]

    print("[trace] WorkflowTrace exports to JSON and loads back losslessly")


# --- Tracer ---


def test_span_times_the_block_and_records_exceptions() -> None:
    tracer = Tracer(run_id="spans")

    with tracer.span("agent_execution", agent="Analyst", workflow="seq", task="sum") as span:
        time.sleep(0.01)
        span.set(success=True, output="4605")

    event = span.event
    assert event is not None and event.event_type is EventType.AGENT_CALL
    assert event.agent_name == "Analyst" and event.workflow == "seq"
    assert event.duration is not None and event.duration >= 0.01
    assert event.data["task"] == "sum" and event.data["output"] == "4605" and event.success is True
    assert event.data["span"] == "agent_execution"

    # The type is inferred from the name, or given explicitly.
    assert tracer.span("tool_lookup").event_type is EventType.TOOL_CALL
    assert tracer.span("routing").event_type is EventType.ROUTE_DECISION
    assert tracer.span("custom", event_type="delegation").event_type is EventType.DELEGATION
    try:
        tracer.span("mystery")
    except ValueError as exc:
        assert "cannot infer" in str(exc)
    else:
        raise AssertionError("an un-inferable span name must be refused")

    # An exception inside the block is recorded as a failure and re-raised.
    try:
        with tracer.span("agent_execution", agent="Crasher"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("span must not swallow exceptions")

    crashed = tracer.events[-1]
    assert crashed.success is False and crashed.error == "RuntimeError: boom"
    assert tracer.trace.status == "running"  # not finished yet
    tracer.finish()
    assert tracer.trace.status == "failed"

    print("[tracer] spans time their block and record exceptions before re-raising")


def test_events_are_captured_across_agent_calls_tools_and_snapshots() -> None:
    with tempfile.TemporaryDirectory() as directory:
        registry = make_registry(Path(directory))
        shared = SharedWorkflowMemory()
        tracer = Tracer.for_memory(shared, run_id="capture", workflow="demo")
        assert shared.tracer is tracer and resolve_tracer(shared) is tracer

        writer = MockAgent(
            name="Writer", role="report", system_prompt="Write.",
            tool_registry=registry, allowed_tools=["file_write", "file_read"], permissions=["write"],
        )

        # 1. An agent step, through the same choke point every orchestrator uses.
        task = Task(description="Draft the report", assigned_agent="Writer", input_data={"topic": "q3"})
        result = execute_agent(writer, task, shared)
        assert result.success

        # 2. A tool call, through the registry.
        stored = writer.use_tool("file_write", shared_memory=shared, path="report.md", content="# Q3")
        assert stored["success"] and isinstance(stored["duration"], float)

        # 3. The blackboard, at this moment.
        shared.set("report", "# Q3")
        tracer.record_snapshot(shared, label="after write")

        events = tracer.events
        assert [e.event_type for e in events] == [
            EventType.AGENT_CALL, EventType.TOOL_CALL, EventType.STATE_SNAPSHOT,
        ]

        step, call, snapshot = events
        assert step.agent_name == "Writer" and step.data["role"] == "report"
        assert step.data["task"] == "Draft the report" and step.data["task_id"] == task.id
        assert step.data["input_data"] == {"topic": "q3"}
        assert step.data["output"] == result.output and step.success is True
        assert step.duration is not None and step.duration >= 0

        assert call.agent_name == "Writer" and call.data["tool"] == "file_write"
        assert call.data["arguments"] == {"path": "report.md", "content": "# Q3"}
        assert call.data["status"] == "ok" and call.success is True
        assert call.data["required_permission"] == "write"
        assert call.data["output"]["path"] == "report.md"
        assert call.duration is not None and call.duration >= 0

        assert snapshot.data["label"] == "after write"
        assert snapshot.data["artifact_keys"] == ["report"]
        assert snapshot.data["artifacts"] == {"report": "# Q3"}
        assert snapshot.data["execution_log_length"] == 2  # the step and the tool call

        # The audit log is untouched by tracing: same two records as before Goal 12.
        assert [r.get("kind", "step") for r in shared.execution_log] == ["step", "tool_call"]

        # An agent that raises is recorded as a failed step, then the error propagates.
        try:
            execute_agent(ExplodingAgent(name="Boom", role="x", system_prompt="x"), task, shared)
        except RuntimeError:
            pass
        else:
            raise AssertionError("execute_agent must not swallow agent exceptions")
        failed = tracer.events[-1]
        assert failed.event_type is EventType.AGENT_CALL and failed.agent_name == "Boom"
        assert failed.success is False and "model fell over" in failed.error

    print("[capture] agent steps, tool calls and snapshots land in the trace with timing")


def test_unauthorized_tool_calls_generate_unauthorized_events() -> None:
    with tempfile.TemporaryDirectory() as directory:
        registry = make_registry(Path(directory))
        shared = SharedWorkflowMemory()
        tracer = Tracer.for_memory(shared, run_id="denied")

        reader = MockAgent(
            name="Reader", role="analyst", system_prompt="Read.",
            tool_registry=registry, allowed_tools=["file_read", "file_write"], permissions=["read_only"],
        )

        # Refused by the registry's tier check: assigned, but not permitted.
        blocked = reader.use_tool("file_write", shared_memory=shared, path="x.md", content="no")
        assert blocked["status"] == ToolStatus.UNAUTHORIZED.value

        # Refused by the agent's assignment check: never assigned at all.
        guessed = reader.use_tool("calculator", shared_memory=shared, expression="1+1")
        assert guessed["status"] == ToolStatus.UNAUTHORIZED.value

        # Refused straight at the registry with the agent's own permission set.
        direct = registry.execute_tool(
            "Reader", "file_write", reader.permissions, shared_memory=shared, path="y.md", content="no"
        )
        assert direct["status"] == ToolStatus.UNAUTHORIZED.value

        # And one allowed call, for contrast.
        registry.execute_tool("Reader", "file_write", {"write"}, shared_memory=shared, path="ok.md", content="yes")
        allowed = reader.use_tool("file_read", shared_memory=shared, path="ok.md")
        assert allowed["success"]

        tool_events = tracer.trace.events_of(EventType.TOOL_CALL)
        assert len(tool_events) == 5
        statuses = [e.data["status"] for e in tool_events]
        assert statuses == ["unauthorized", "unauthorized", "unauthorized", "ok", "ok"]

        registry_denial, agent_denial, direct_denial = tool_events[:3]
        assert registry_denial.data["tool"] == "file_write"
        assert registry_denial.data["required_permission"] == "write"
        assert registry_denial.success is False
        assert "not permitted" in registry_denial.error
        assert registry_denial.data["arguments"] == {"path": "x.md", "content": "no"}

        assert agent_denial.data["tool"] == "calculator"
        assert agent_denial.data["required_permission"] is None  # never reached the registry
        assert "not assigned" in agent_denial.error

        assert direct_denial.data["required_permission"] == "write"

        # Nothing the refused calls asked for exists on disk.
        assert sorted(p.name for p in Path(directory).iterdir()) == ["ok.md"]

        # The summary counts allowed against blocked, per tool and overall.
        tools = tracer.summary()["tools"]
        assert tools["total_calls"] == 5 and tools["blocked"] == 3 and tools["allowed"] == 2
        assert tools["by_tool"]["file_write"] == {"calls": 3, "allowed": 1, "blocked": 2, "failed": 0}
        assert tools["by_tool"]["calculator"] == {"calls": 1, "allowed": 0, "blocked": 1, "failed": 0}
        assert [e["event_type"] for e in tracer.summary()["errors"]] == ["tool_call"] * 3

        # A refusal is the system working, not a failed run.
        tracer.finish()
        assert tracer.trace.status == "completed"

    print("[security] unauthorized tool calls are traced as UNAUTHORIZED events, never executed")


def test_router_and_verified_pipeline_record_decisions_retries_and_verdicts() -> None:
    shared = SharedWorkflowMemory()
    tracer = Tracer.for_memory(shared, run_id="routed")

    coder = MockAgent(name="Coder", role="coding", system_prompt="Code.")
    fallback = MockAgent(name="Fallback", role="fallback", system_prompt="Anything.")
    router = RouterOrchestrator(fallback_destination=fallback)
    router.register_route("coding", coder, keywords=["python", "bug"])

    # First verdict: incomplete, so the pipeline retries. Second: complete.
    verifier = ScriptedVerifier(
        ['{"is_complete": false, "feedback": "no tests"}', '{"is_complete": true, "feedback": ""}'],
        name="Verifier", role="qa", system_prompt="Verify.",
    )
    pipeline = AutoRoutedVerifiedPipeline(router=router, verifier_agent=verifier, max_attempts=3)

    result = pipeline.run(Task(description="Fix the python bug", assigned_agent="router"), shared)
    assert result.success and pipeline.verified and pipeline.attempts_used == 2

    kinds = [e.event_type for e in tracer.events]
    assert kinds == [
        EventType.ROUTE_DECISION, EventType.AGENT_CALL,  # attempt 1: route, coder
        EventType.AGENT_CALL, EventType.VERIFICATION,  # verifier runs, verdict: incomplete
        EventType.DELEGATION,  # retry task handed back to the router
        EventType.ROUTE_DECISION, EventType.AGENT_CALL,  # attempt 2
        EventType.AGENT_CALL, EventType.VERIFICATION,  # verdict: complete
    ]

    routes = tracer.trace.events_of(EventType.ROUTE_DECISION)
    assert [r.data["route"] for r in routes] == ["coding", "coding"]
    assert routes[0].data["reasoning"] == "keyword 'python' matched"
    assert routes[0].data["destination"] == "Coder"
    assert routes[0].agent_name == "router"
    # routing_log carries the same reasoning, additively - existing keys unchanged.
    assert router.routing_log[0]["reasoning"] == "keyword 'python' matched"
    assert {"task_id", "route", "description"} <= set(router.routing_log[0])

    verdicts = tracer.trace.events_of(EventType.VERIFICATION)
    assert [(v.data["attempt"], v.data["is_complete"]) for v in verdicts] == [(1, False), (2, True)]
    assert verdicts[0].data["feedback"] == "no tests"
    assert verdicts[0].data["chosen_route"] == "coding"
    assert verdicts[0].agent_name == "Verifier"

    retry = tracer.trace.events_of(EventType.DELEGATION)[0]
    assert retry.data["from"] == "Verifier" and retry.data["to"] == "router"
    assert retry.data["attempt"] == 2 and retry.data["feedback"] == "no tests"
    assert "no tests" in retry.data["task"]  # the retry task carries the critique

    # Fallback and route_func decisions explain themselves too.
    router.run(Task(description="tell me a joke", assigned_agent="router"), shared)
    assert tracer.events[-2].data["route"] == "fallback"
    assert tracer.events[-2].data["reasoning"] == "no keyword rule matched"
    router.set_route_func(lambda task: "coding")
    router.run(Task(description="anything", assigned_agent="router"), shared)
    assert tracer.events[-2].data["reasoning"] == "route_func chose 'coding'"

    # A route that points at a sub-workflow is recorded as a delegation.
    inner = RouterOrchestrator(fallback_destination=fallback)
    outer = RouterOrchestrator(fallback_destination=fallback)
    outer.register_route("nested", inner, keywords=["nested"])
    outer.run(Task(description="a nested request", assigned_agent="outer"), shared)
    handoff = [e for e in tracer.events if e.event_type is EventType.DELEGATION][-1]
    assert handoff.data["to"] == "RouterOrchestrator" and "sub-workflow" in handoff.data["reason"]

    summary = tracer.summary()
    assert summary["verdicts"] == [
        {"verifier": "Verifier", "attempt": 1, "is_complete": False},
        {"verifier": "Verifier", "attempt": 2, "is_complete": True},
    ]
    assert summary["agents"]["Coder"]["calls"] == 3  # 2 pipeline attempts + the route_func run
    assert summary["agents"]["Verifier"]["calls"] == 2

    print("[router] route choices, retry delegations and verifier verdicts are traced")


def test_attached_tracer_reaches_parallel_worker_threads() -> None:
    shared = SharedWorkflowMemory()
    tracer = Tracer.for_memory(shared, run_id="parallel")

    workers = [MockAgent(name=f"W{i}", role="worker", system_prompt="Work.") for i in range(4)]
    synthesizer = MockAgent(name="Synth", role="synth", system_prompt="Merge.")
    orchestrator = ParallelOrchestrator(workers, synthesizer, max_workers=4)

    result = orchestrator.run(Task(description="fan out", assigned_agent="parallel"), shared)
    assert result.success

    calls = tracer.trace.events_of(EventType.AGENT_CALL)
    assert sorted(e.agent_name for e in calls) == ["Synth", "W0", "W1", "W2", "W3"]
    assert all(e.duration is not None for e in calls)
    assert len(tracer.trace) == len(calls)  # nothing lost, nothing duplicated

    # Concurrent appends from arbitrary threads stay consistent.
    trace = WorkflowTrace(run_id="threads")

    def hammer() -> None:
        for _ in range(200):
            trace.record_event(EventType.AGENT_CALL, agent_name=threading.current_thread().name)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(trace) == 1600
    assert len({e.event_id for e in trace.events}) == 1600

    print("[threads] an attached tracer collects from parallel workers; appends are thread-safe")


def test_active_tracer_and_absence_of_any_tracer() -> None:
    agent = MockAgent(name="Solo", role="x", system_prompt="x")
    task = Task(description="hello", assigned_agent="Solo")

    # No tracer anywhere: identical behaviour to before Goal 12.
    assert get_active_tracer() is None and resolve_tracer(None) is None
    shared = SharedWorkflowMemory()
    assert shared.tracer is None
    result = execute_agent(agent, task, shared)
    assert result.success and len(shared.execution_log) == 1

    # Activated for the process: no blackboard needed.
    tracer = Tracer(run_id="active")
    with tracer.activate() as active:
        assert active is tracer and get_active_tracer() is tracer
        execute_agent(agent, task)  # no shared memory at all
        # Explicit beats attached beats active.
        other = Tracer(run_id="other")
        assert resolve_tracer(SharedWorkflowMemory(tracer=other)) is other
        assert resolve_tracer(None, tracer=other) is other
        assert resolve_tracer(None) is tracer

    assert get_active_tracer() is None
    assert tracer.trace.end_time is not None  # leaving the block ends the run
    assert [e.agent_name for e in tracer.events] == ["Solo"]

    print("[active] a process-wide tracer works without a blackboard and leaves nothing behind")


def test_print_run_summary_renders_timeline_and_exports_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        registry = make_registry(Path(directory))
        shared = SharedWorkflowMemory()
        tracer = Tracer.for_memory(shared, run_id="report", workflow="demo")

        analyst = MockAgent(
            name="Analyst", role="analysis", system_prompt="Analyse.",
            tool_registry=registry, allowed_tools=["calculator", "file_write"], permissions=["read_only"],
        )
        execute_agent(analyst, Task(description="sum the figures", assigned_agent="Analyst"), shared)
        analyst.use_tool("calculator", shared_memory=shared, expression="2+2")
        analyst.use_tool("file_write", shared_memory=shared, path="x.md", content="no")
        tracer.record_route("router", "analysis", "sum the figures", reasoning="keyword 'sum' matched")
        tracer.record_verification("Verifier", True, "", attempt=1)
        tracer.record_snapshot(shared, label="end")
        tracer.finish()

        json_path = Path(directory) / "out" / "trace.json"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exported = print_run_summary(tracer, json_path=json_path)
        text = buffer.getvalue()

        assert exported == json_path and json_path.is_file()
        assert WorkflowTrace.load_json(json_path).run_id == "report"

        # Header: run id, wall time, status.
        assert "RUN SUMMARY  report" in text
        assert "status:     COMPLETED" in text and "wall time:" in text

        # Timeline table with the requested columns, one row per event, in order.
        assert "Time      | Step Type      | Entity             | Duration | Status     | Summary" in text
        rows = [line for line in text.splitlines() if line.startswith("+")]
        assert [row.split("|")[1].strip() for row in rows] == [
            "agent_call", "tool_call", "tool_call", "route_decision", "verification", "state_snapshot",
        ]
        assert "BLOCKED" in rows[2] and "file_write(" in rows[2]
        assert "calculator(expression='2+2') -> 4" in rows[1]
        assert "-> analysis (keyword 'sum' matched)" in rows[3]
        assert "PASS" in rows[4]
        assert "end: 0 artifact(s)" in rows[5]

        # Tool statistics, allowed against blocked.
        assert "total calls: 2   allowed: 1   blocked: 1   failed: 0" in text
        assert "ERRORS AND REFUSALS (1)" in text

        # The export location.
        assert f"JSON trace: {json_path.resolve()}" in text

        # Without a path, the summary says how to get one instead of pointing nowhere.
        assert "not exported" in format_run_summary(tracer.trace)

    print("[summary] print_run_summary shows header, timeline, tool stats and the JSON path")


TESTS = (
    test_trace_serializes_to_and_from_json,
    test_span_times_the_block_and_records_exceptions,
    test_events_are_captured_across_agent_calls_tools_and_snapshots,
    test_unauthorized_tool_calls_generate_unauthorized_events,
    test_router_and_verified_pipeline_record_decisions_retries_and_verdicts,
    test_attached_tracer_reaches_parallel_worker_threads,
    test_active_tracer_and_absence_of_any_tracer,
    test_print_run_summary_renders_timeline_and_exports_json,
)


if __name__ == "__main__":
    print("---- Observability and Debugging Trace ----")
    for test in TESTS:
        test()
    print("\nAll assertions passed.")
