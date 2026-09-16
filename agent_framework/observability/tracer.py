from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.observability.trace import (
    EventType,
    TraceEvent,
    WorkflowTrace,
    jsonable,
    truncate_text,
)

# The Tracer is the write side of observability: one per run, shared by every
# component that takes part in that run. Components never construct events
# themselves - they call one of the record_* methods, which know what an
# agent step or a tool call should look like in the trace, so every run's
# telemetry has the same shape whichever orchestrator produced it.


# --- Active tracer (process-wide) ---

_ACTIVE: list["Tracer"] = []
_ACTIVE_LOCK = threading.Lock()


def get_active_tracer() -> Optional["Tracer"]:
    """The innermost activated tracer, or None."""
    with _ACTIVE_LOCK:
        return _ACTIVE[-1] if _ACTIVE else None


def resolve_tracer(
    shared_memory: Optional[SharedWorkflowMemory] = None,
    tracer: Optional["Tracer"] = None,
) -> Optional["Tracer"]:
    """The tracer a component should record on, if any.

    Explicit beats attached beats active. Returns None when there is nothing
    to record on, so callers can write `if tracer:` and skip the work.
    """
    if tracer is not None:
        return tracer
    attached = getattr(shared_memory, "tracer", None) if shared_memory is not None else None
    if attached is not None:
        return attached
    return get_active_tracer()


# --- Spans ---

# What kind of event a span records, inferred from its name when the caller
# does not say. Matching is by substring so "agent_execution", "agent-step"
# and "verifier_agent_call" all land where a reader would expect.
_SPAN_TYPES: tuple[tuple[str, EventType], ...] = (
    ("tool", EventType.TOOL_CALL),
    ("rout", EventType.ROUTE_DECISION),
    ("deleg", EventType.DELEGATION),
    ("verif", EventType.VERIFICATION),
    ("eval", EventType.VERIFICATION),
    ("snapshot", EventType.STATE_SNAPSHOT),
    ("agent", EventType.AGENT_CALL),
)


def _infer_span_type(name: str) -> Optional[EventType]:
    lowered = name.lower()
    for needle, kind in _SPAN_TYPES:
        if needle in lowered:
            return kind
    return None


class Span:
    """One timed operation, recorded as a single event when it closes.

        with tracer.span("agent_execution", agent="Analyst") as span:
            result = agent.execute(task)
            span.set(success=result.success, output=result.output)

    Anything put on the span with `set` (or as keyword arguments up front)
    lands in the event's data. An exception escaping the block is recorded
    as a failure with the exception's text, then re-raised: the trace should
    show the step that blew up, not stop just before it.
    """

    def __init__(
        self,
        tracer: "Tracer",
        name: str,
        event_type: EventType,
        agent_name: Optional[str],
        workflow: Optional[str],
        data: dict[str, Any],
    ):
        self.tracer = tracer
        self.name = name
        self.event_type = event_type
        self.agent_name = agent_name
        self.workflow = workflow
        self.data = dict(data)
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.event: Optional[TraceEvent] = None

    def set(self, **data: Any) -> "Span":
        self.data.update(data)
        return self

    @property
    def duration(self) -> Optional[float]:
        if self.start is None:
            return None
        return (self.end if self.end is not None else time.perf_counter()) - self.start

    def __enter__(self) -> "Span":
        self.start = time.perf_counter()
        self._wall_start = time.time()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.end = time.perf_counter()
        if exc is not None:
            self.data.setdefault("success", False)
            self.data.setdefault("error", f"{exc_type.__name__}: {exc}")
        self.event = self.tracer.trace.record_event(
            self.event_type,
            agent_name=self.agent_name,
            workflow=self.workflow,
            duration=self.end - self.start,
            timestamp=self._wall_start,
            span=self.name,
            **self.data,
        )
        return False  # never swallow


# --- The collector ---


class Tracer:
    """Thread-safe telemetry collector for one workflow run.

    Holds a WorkflowTrace and knows how each framework event should be
    written into it. Pass it into a run by attaching it to the run's
    SharedWorkflowMemory (`Tracer.for_memory` or `tracer.attach`), or
    activate it for the process with `with tracer.activate():`.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        trace: Optional[WorkflowTrace] = None,
        **metadata: Any,
    ):
        if trace is None:
            trace = WorkflowTrace(**({"run_id": run_id} if run_id else {}))
        self.trace = trace
        self.trace.metadata.update(metadata)
        self._lock = threading.RLock()

    @classmethod
    def for_memory(cls, shared_memory: SharedWorkflowMemory, **kwargs: Any) -> "Tracer":
        """Build a tracer and attach it to `shared_memory` in one step."""
        tracer = cls(**kwargs)
        tracer.attach(shared_memory)
        return tracer

    # --- Wiring ---

    def attach(self, shared_memory: SharedWorkflowMemory) -> "Tracer":
        """Make this the tracer every component sees through `shared_memory`."""
        shared_memory.tracer = self
        return self

    @contextmanager
    def activate(self) -> Iterator["Tracer"]:
        """Make this the process's active tracer for the duration of the block.

        Ends the trace on exit, so the block is the run. Note that a step run
        on another thread (the parallel orchestrator's workers) only reaches
        an *attached* tracer; activation is for single-threaded scripts.
        """
        with _ACTIVE_LOCK:
            _ACTIVE.append(self)
        try:
            yield self
        finally:
            with _ACTIVE_LOCK:
                # Remove this tracer specifically, not whatever is on top, so
                # a mis-nested exit cannot pop someone else's tracer.
                for index in range(len(_ACTIVE) - 1, -1, -1):
                    if _ACTIVE[index] is self:
                        del _ACTIVE[index]
                        break
            self.finish()

    def finish(self) -> WorkflowTrace:
        """Close the trace (sets end_time) and return it."""
        self.trace.finish()
        return self.trace

    # --- Timing ---

    def span(
        self,
        name: str,
        event_type: Optional[Union[EventType, str]] = None,
        agent: Any = None,
        workflow: Optional[str] = None,
        **data: Any,
    ) -> Span:
        """A context manager that times a block and records it as one event.

        `event_type` is inferred from `name` when not given ("agent_execution"
        is an AGENT_CALL, "tool_call" a TOOL_CALL, ...). `agent` may be an
        agent object or its name.
        """
        if event_type is None:
            event_type = _infer_span_type(name)
            if event_type is None:
                raise ValueError(
                    f"cannot infer an EventType from span name {name!r}; "
                    f"pass event_type= explicitly"
                )
        return Span(
            self,
            name,
            EventType(event_type),
            agent_name=_entity_name(agent),
            workflow=workflow,
            data=data,
        )

    # --- Recording ---

    def record_event(self, event_type: Union[EventType, str], **kwargs: Any) -> TraceEvent:
        """Record an arbitrary event. The typed record_* methods are preferred."""
        return self.trace.record_event(event_type, **kwargs)

    def record_agent_step(
        self,
        agent: Any,
        task: Any,
        output: Optional[str] = None,
        success: Optional[bool] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> TraceEvent:
        """One agent executed one task.

        `agent` is a BaseAgent or a name; `task` a Task or a description. The
        task's input_data and the output are stored (truncated), so a trace
        shows what each step was given and what it produced.
        """
        data: dict[str, Any] = {
            "role": getattr(agent, "role", None),
            "task_id": getattr(task, "id", None),
            "task": truncate_text(_task_text(task)),
            "input_data": jsonable(getattr(task, "input_data", None)),
            "output": truncate_text(output),
            "success": success,
            "error": error,
        }
        data.update(extra)
        return self.trace.record_event(
            EventType.AGENT_CALL,
            agent_name=_entity_name(agent),
            workflow=workflow,
            duration=duration,
            **data,
        )

    def record_tool_call(
        self,
        agent: Any,
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
        result: Any = None,
        duration: Optional[float] = None,
        status: Optional[str] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        required_permission: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> TraceEvent:
        """One tool invocation, whether it ran, failed, or was refused.

        `result` may be the record dict ToolRegistry.execute_tool returns (its
        status/output/error fields are read off it) or the raw output.
        """
        output: Any = result
        if isinstance(result, dict) and "status" in result and "tool" in result:
            status = status or result.get("status")
            success = result.get("success") if success is None else success
            error = error or result.get("error")
            required_permission = required_permission or result.get("required_permission")
            output = result.get("output")
        if success is None and status is not None:
            success = status == "ok"

        data: dict[str, Any] = {
            "tool": tool_name,
            "arguments": jsonable(args or {}),
            "status": status,
            "success": success,
            "output": jsonable(output),
            "error": error,
            "required_permission": required_permission,
        }
        data.update(extra)
        return self.trace.record_event(
            EventType.TOOL_CALL,
            agent_name=_entity_name(agent),
            workflow=workflow,
            duration=duration,
            **data,
        )

    def record_route(
        self,
        router_name: str,
        route_chosen: str,
        task_description: str,
        reasoning: Optional[str] = None,
        task_id: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> TraceEvent:
        """A router picked a destination, and why."""
        data: dict[str, Any] = {
            "route": route_chosen,
            "task_id": task_id,
            "task": truncate_text(task_description),
            "reasoning": reasoning,
        }
        data.update(extra)
        return self.trace.record_event(
            EventType.ROUTE_DECISION,
            agent_name=router_name,
            workflow=workflow or router_name,
            **data,
        )

    def record_delegation(
        self,
        from_entity: Any,
        to_entity: Any,
        task: Any,
        reason: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> TraceEvent:
        """Work handed from one entity to another (a retry, a subtask, a sub-workflow)."""
        data: dict[str, Any] = {
            "from": _entity_name(from_entity),
            "to": _entity_name(to_entity),
            "task_id": getattr(task, "id", None),
            "task": truncate_text(_task_text(task)),
            "reason": reason,
        }
        data.update(extra)
        return self.trace.record_event(
            EventType.DELEGATION,
            agent_name=_entity_name(from_entity),
            workflow=workflow,
            **data,
        )

    def record_verification(
        self,
        verifier: Any,
        is_complete: Optional[bool],
        feedback: Optional[str] = None,
        attempt: Optional[int] = None,
        raw_response: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> TraceEvent:
        """A verifier or evaluator passed judgement on some output."""
        data: dict[str, Any] = {
            "attempt": attempt,
            "is_complete": is_complete,
            "feedback": truncate_text(feedback),
            "raw_response": truncate_text(raw_response),
        }
        data.update(extra)
        return self.trace.record_event(
            EventType.VERIFICATION,
            agent_name=_entity_name(verifier),
            workflow=workflow,
            **data,
        )

    def record_snapshot(
        self,
        shared_memory: SharedWorkflowMemory,
        label: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> TraceEvent:
        """The blackboard as it is right now: every artifact, and the log length."""
        state = shared_memory.snapshot()
        data: dict[str, Any] = {
            "label": label,
            "artifact_keys": list(state["artifacts"]),
            "artifacts": jsonable(state["artifacts"]),
            "execution_log_length": len(state["execution_log"]),
        }
        data.update(extra)
        return self.trace.record_event(
            EventType.STATE_SNAPSHOT, workflow=workflow, **data
        )

    # --- Reading and exporting ---

    @property
    def events(self) -> list[TraceEvent]:
        return list(self.trace.events)

    def summary(self) -> dict[str, Any]:
        return self.trace.summary()

    def export_json(self, path: Union[str, Path]) -> Path:
        """Write the trace to `path` and return it, so a caller can print the location."""
        self.trace.export_json(path)
        return Path(path)

    def __repr__(self) -> str:
        return f"Tracer({self.trace!r})"


# --- Helpers ---


def _entity_name(entity: Any) -> Optional[str]:
    """A name for an agent, orchestrator, or plain string."""
    if entity is None:
        return None
    if isinstance(entity, str):
        return entity
    name = getattr(entity, "name", None)
    if isinstance(name, str):
        return name
    return type(entity).__name__


def _task_text(task: Any) -> Optional[str]:
    if task is None:
        return None
    if isinstance(task, str):
        return task
    return getattr(task, "description", None) or str(task)
