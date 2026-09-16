from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Union

# The structured record of one workflow run.


class EventType(str, Enum):
    AGENT_CALL = "agent_call"  # one agent executed one task
    TOOL_CALL = "tool_call"  # one tool invocation, allowed or refused
    ROUTE_DECISION = "route_decision"  # a router chose a destination
    DELEGATION = "delegation"  # work was handed from one entity to another
    VERIFICATION = "verification"  # a verifier or evaluator passed judgement
    STATE_SNAPSHOT = "state_snapshot"  # the blackboard at a point in time

    def __str__(self) -> str:
        return self.value


# How much of a string lands in an event before it is cut. Traces are read by
# people and written to disk; a 40 KB model reply repeated across ten events
# helps neither.
MAX_EVENT_TEXT = 4000


def truncate_text(value: Any, limit: int = MAX_EVENT_TEXT) -> Any:
    """Cap a string at `limit` chars, saying how much was dropped. Non-strings pass through."""
    if not isinstance(value, str) or limit is None or len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated, {len(value)} chars total]"


def jsonable(value: Any, limit: int = MAX_EVENT_TEXT) -> Any:
    """Coerce `value` into something json.dumps accepts, recursively.

    Dataclasses, pydantic models and enums are common in this framework's
    results; anything else unknown becomes its repr rather than raising, so a
    tool returning an exotic object cannot break the run's telemetry.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return truncate_text(value, limit)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): jsonable(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item, limit) for item in value]
    if hasattr(value, "model_dump"):  # pydantic
        return jsonable(value.model_dump(), limit)
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(
            {name: getattr(value, name) for name in value.__dataclass_fields__}, limit
        )
    return truncate_text(repr(value), limit)


@dataclass
class TraceEvent:
    """One thing that happened during a run."""

    event_type: EventType
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    workflow: Optional[str] = None
    agent_name: Optional[str] = None
    duration: Optional[float] = None  # seconds
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = EventType(self.event_type)

    # --- Convenience readers ---

    @property
    def success(self) -> Optional[bool]:
        """The event's outcome, when it recorded one (None for e.g. a snapshot)."""
        return self.data.get("success")

    @property
    def status(self) -> str:
        """A short status word: the tool status if there is one, else ok/failed/-."""
        if "status" in self.data:
            return str(self.data["status"])
        success = self.success
        if success is None:
            return "-"
        return "ok" if success else "failed"

    @property
    def error(self) -> Optional[str]:
        return self.data.get("error")

    # --- Serialisation ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "workflow": self.workflow,
            "agent_name": self.agent_name,
            "duration": self.duration,
            "data": jsonable(self.data),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEvent":
        return cls(
            event_type=EventType(payload["event_type"]),
            event_id=payload["event_id"],
            timestamp=payload["timestamp"],
            workflow=payload.get("workflow"),
            agent_name=payload.get("agent_name"),
            duration=payload.get("duration"),
            data=dict(payload.get("data") or {}),
        )

    def __repr__(self) -> str:
        who = self.agent_name or self.workflow or "-"
        took = f", {self.duration:.3f}s" if self.duration is not None else ""
        return f"TraceEvent({self.event_type.value}, {who}, {self.status}{took})"


@dataclass
class WorkflowTrace:
    """Every event of one run, in the order it happened."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Events arrive from whichever thread ran the step (the parallel
        # orchestrator fans out over a pool), so appends are serialised.
        self._lock = threading.RLock()
        self._counter = len(self.events)

    # --- Recording ---

    def record_event(self, event_type: Union[EventType, str], **kwargs: Any) -> TraceEvent:
        """Append one event and return it.

        Keyword arguments naming a TraceEvent field (workflow, agent_name,
        duration, timestamp, event_id) set that field; everything else goes
        into `data`. So `record_event(EventType.AGENT_CALL, agent_name="x",
        output="...")` puts the output where a reader expects it without the
        caller building the dict by hand.
        """
        fields = {}
        data: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("workflow", "agent_name", "duration", "timestamp", "event_id"):
                fields[key] = value
            elif key == "data" and isinstance(value, dict):
                data.update(value)
            else:
                data[key] = value

        with self._lock:
            self._counter += 1
            fields.setdefault("event_id", f"{self.run_id}-{self._counter:04d}")
            event = TraceEvent(event_type=EventType(event_type), data=data, **fields)
            self.events.append(event)
        return event

    def finish(self) -> None:
        """Mark the run over. Idempotent: the first end time sticks."""
        with self._lock:
            if self.end_time is None:
                self.end_time = time.time()

    # --- Reading ---

    @property
    def wall_time(self) -> float:
        """Seconds from start to end - or to now, while the run is still open."""
        return (self.end_time if self.end_time is not None else time.time()) - self.start_time

    @property
    def status(self) -> str:
        """running / completed / failed, from the end time and the events."""
        with self._lock:
            if self.end_time is None:
                return "running"
            failed = any(
                event.success is False
                for event in self.events
                if event.event_type in (EventType.AGENT_CALL, EventType.VERIFICATION)
            )
        # A refused tool call or a false verdict mid-run is the system working,
        # so only a step that actually failed marks the run as failed. The
        # final verdict, if any, is what settles a verified pipeline.
        if failed:
            return "failed"
        return "completed"

    def events_of(self, event_type: Union[EventType, str]) -> list[TraceEvent]:
        wanted = EventType(event_type)
        with self._lock:
            return [event for event in self.events if event.event_type is wanted]

    def events_for(self, agent_name: str) -> list[TraceEvent]:
        with self._lock:
            return [event for event in self.events if event.agent_name == agent_name]

    def errors(self) -> list[TraceEvent]:
        """Events that carry an error, refused tool calls included."""
        with self._lock:
            return [event for event in self.events if event.error]

    def summary(self) -> dict[str, Any]:
        """Counts, durations, tools used and errors - the run at a glance."""
        with self._lock:
            events = list(self.events)
            end_time = self.end_time

        counts: dict[str, int] = {kind.value: 0 for kind in EventType}
        durations: dict[str, dict[str, float]] = {}
        for event in events:
            counts[event.event_type.value] += 1
            if event.duration is not None:
                bucket = durations.setdefault(
                    event.event_type.value, {"total": 0.0, "max": 0.0, "count": 0}
                )
                bucket["total"] += event.duration
                bucket["max"] = max(bucket["max"], event.duration)
                bucket["count"] += 1
        for bucket in durations.values():
            bucket["mean"] = bucket["total"] / bucket["count"]

        agent_calls = [e for e in events if e.event_type is EventType.AGENT_CALL]
        tool_calls = [e for e in events if e.event_type is EventType.TOOL_CALL]

        agents: dict[str, dict[str, Any]] = {}
        for event in agent_calls:
            entry = agents.setdefault(
                event.agent_name or "?", {"calls": 0, "failed": 0, "total_duration": 0.0}
            )
            entry["calls"] += 1
            if event.success is False:
                entry["failed"] += 1
            if event.duration is not None:
                entry["total_duration"] += event.duration

        tools: dict[str, dict[str, int]] = {}
        blocked = 0
        for event in tool_calls:
            name = str(event.data.get("tool", "?"))
            entry = tools.setdefault(name, {"calls": 0, "allowed": 0, "blocked": 0, "failed": 0})
            entry["calls"] += 1
            status = str(event.data.get("status", ""))
            if status == "unauthorized":
                entry["blocked"] += 1
                blocked += 1
            elif event.success:
                entry["allowed"] += 1
            else:
                entry["failed"] += 1

        routes = [
            {
                "router": e.agent_name,
                "route": e.data.get("route"),
                "reasoning": e.data.get("reasoning"),
            }
            for e in events
            if e.event_type is EventType.ROUTE_DECISION
        ]
        verdicts = [
            {
                "verifier": e.agent_name,
                "attempt": e.data.get("attempt"),
                "is_complete": e.data.get("is_complete"),
            }
            for e in events
            if e.event_type is EventType.VERIFICATION
        ]

        return {
            "run_id": self.run_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": end_time,
            "wall_time": self.wall_time,
            "event_count": len(events),
            "counts": counts,
            "durations": durations,
            "agents": agents,
            "tools": {
                "total_calls": len(tool_calls),
                "allowed": sum(t["allowed"] for t in tools.values()),
                "blocked": blocked,
                "failed": sum(t["failed"] for t in tools.values()),
                "by_tool": tools,
            },
            "routes": routes,
            "verdicts": verdicts,
            "errors": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "entity": e.agent_name or e.workflow,
                    "error": e.error,
                }
                for e in events
                if e.error
            ],
            "metadata": jsonable(self.metadata),
        }

    # --- Serialisation ---

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "metadata": jsonable(self.metadata),
                "events": [event.to_dict() for event in self.events],
            }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowTrace":
        return cls(
            run_id=payload["run_id"],
            start_time=payload["start_time"],
            end_time=payload.get("end_time"),
            events=[TraceEvent.from_dict(item) for item in payload.get("events", [])],
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def export_json(self, path: Union[str, Path]) -> None:
        """Write the whole trace to `path` as JSON, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "WorkflowTrace":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # --- Conveniences ---

    def __len__(self) -> int:
        with self._lock:
            return len(self.events)

    def __iter__(self) -> Iterable[TraceEvent]:
        with self._lock:
            return iter(list(self.events))

    def __repr__(self) -> str:
        return (
            f"WorkflowTrace(run_id={self.run_id!r}, events={len(self)}, "
            f"status={self.status!r})"
        )
