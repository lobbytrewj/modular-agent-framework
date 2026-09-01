from __future__ import annotations

import inspect
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Two different kinds of state live in this module, and keeping them apart is
# the whole point of it:
#
#   AgentMemory
#       Private to one agent. It is that agent's own conversation and no one else can read it
#       Two agents in the same workflow keep their own, so the analyst remembering what
#       it said last round never leaks into the writer's prompt.
#
#   SHARED WORKFLOW MEMORY (SharedWorkflowMemory / WorkflowScratchpad)
#       A blackboard every agent in a run can read from and write to. It holds
#       finished deliverables plus one audit log of every step that executed. 
#       This is how a step in one workflow hands something concrete to a step 
#       in another.

# The roles a chat model understands.
VALID_ROLES = ("system", "user", "assistant")

# How much of a single artifact gets pasted into a prompt before it is cut short
MAX_ARTIFACT_CHARS = 2000


# One turn of a conversation: who spoke, what they said, and when.
@dataclass
class Message:
    role: str  # "system", "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"role must be one of {VALID_ROLES}, got {self.role!r}"
            )

    def as_chat_message(self) -> dict:
        """The shape LLMClient.complete expects in its `history` argument."""
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:  # readable in a test trace
        preview = self.content.replace("\n", " ")[:60]
        return f"Message(role={self.role!r}, content={preview!r})"


# --- Agent-local memory ---
@dataclass
class AgentMemory:
    messages: list[Message] = field(default_factory=list)
    max_messages: Optional[int] = None

    # --- Writing ----

    def add_message(self, role: str, content: str) -> Message:
        """Append one turn and return it."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self._trim()
        return message

    def add_system_message(self, content: str) -> Message:
        return self.add_message("system", content)

    def add_user_message(self, content: str) -> Message:
        """Record what the agent was asked (its prompt for this turn)."""
        return self.add_message("user", content)

    def add_assistant_message(self, content: str) -> Message:
        """Record what the agent answered."""
        return self.add_message("assistant", content)

    def _trim(self) -> None:
        if self.max_messages is not None and len(self.messages) > self.max_messages:
            # Drop from the front: the oldest turn is the one whose absence
            # costs the least context.
            del self.messages[: len(self.messages) - self.max_messages]

    def clear(self) -> None:
        """Forget everything. The agent starts its next task cold."""
        self.messages.clear()

    # --- Reading ---

    def recent(self, last_k: Optional[int] = None) -> list[Message]:
        """The last `last_k` turns, or all of them when `last_k` is None."""
        if last_k is None:
            return list(self.messages)
        if last_k <= 0:
            return []
        return self.messages[-last_k:]

    def get_context_string(self, last_k: Optional[int] = None) -> str:
        """Flatten recent turns into text that can be pasted into a prompt.

        Used when the history has to travel inside the prompt body - a model
        or client that takes no separate history argument. When the client does
        accept one, prefer `as_chat_messages`: a real message list survives the
        model's chat template, whereas a flattened transcript is just more user
        text the model has to be trusted to interpret as history.
        """
        selected = self.recent(last_k)
        if not selected:
            return ""

        lines = [f"{message.role.capitalize()}: {message.content}" for message in selected]
        return "Conversation so far:\n" + "\n\n".join(lines)

    def as_chat_messages(self, last_k: Optional[int] = None) -> list[dict]:
        """Recent turns in the role/content form LLMClient.complete takes."""
        return [message.as_chat_message() for message in self.recent(last_k)]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"AgentMemory(messages={len(self.messages)})"


# --- Shared workflow memory ---
class SharedWorkflowMemory:
    def __init__(
        self,
        artifacts: Optional[dict[str, Any]] = None,
        execution_log: Optional[list[dict]] = None,
    ):
        self.artifacts: dict[str, Any] = dict(artifacts or {})
        self.execution_log: list[dict] = list(execution_log or [])
        self._lock = threading.RLock()

    # --- Artifact storage ------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Publish an intermediate deliverable under `key`, replacing any prior one."""
        with self._lock:
            self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.artifacts.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self.artifacts

    def keys(self) -> list[str]:
        """Artifact names in insertion order - i.e. the order they were produced."""
        with self._lock:
            return list(self.artifacts)

    def update(self, values: dict[str, Any]) -> None:
        """Publish several artifacts at once."""
        with self._lock:
            self.artifacts.update(values)

    def delete(self, key: str) -> bool:
        """Remove one artifact. True if it was there."""
        with self._lock:
            return self.artifacts.pop(key, None) is not None

    # --- Audit log -------------------------------------------------------

    def log_step(
        self,
        agent: str,
        task: str = "",
        success: Optional[bool] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
        workflow: Optional[str] = None,
        **extra: Any,
    ) -> dict:
        """Record one executed step on the global audit trail.

        This is deliberately global rather than per-orchestrator: each pipeline
        already keeps its own `step_results`, but those stop at the pipeline
        boundary. When a router hands a task to a sequential sub-workflow which
        then calls three agents, this log is the only place the whole run reads
        as a single ordered story.
        """
        record = {
            "step": len(self.execution_log) + 1,
            "timestamp": time.time(),
            "workflow": workflow,
            "agent": agent,
            "task": task,
            "success": success,
            "output": output,
            "error": error,
        }
        record.update(extra)
        with self._lock:
            self.execution_log.append(record)
        return record

    def log_for(self, agent: str) -> list[dict]:
        """Every logged step performed by one agent."""
        with self._lock:
            return [record for record in self.execution_log if record.get("agent") == agent]

    # --- Prompt formatting -----------------------------------------------

    def to_prompt_context(
        self,
        keys: Optional[list[str]] = None,
        max_chars: int = MAX_ARTIFACT_CHARS,
    ) -> str:
        """Render the stored artifacts as markdown, ready to drop into a prompt.

        `keys` selects which artifacts to show; None shows all of them. Names
        that hold nothing are skipped rather than rendered as empty sections,
        so an agent asking for an artifact an earlier step never produced sees
        no mention of it at all - better than a heading promising content that
        isn't there.

        Returns "" when there is nothing to show, so callers can write
        `if context:` and avoid pasting an empty header into the prompt.
        """
        with self._lock:
            selected = (
                list(self.artifacts.items())
                if keys is None
                else [(key, self.artifacts[key]) for key in keys if key in self.artifacts]
            )

        if not selected:
            return ""

        blocks = [
            "## Shared workflow context",
            "The following artifacts were produced by earlier steps of this "
            "workflow. Treat them as established input - build on them, and do "
            "not re-derive or contradict them.",
        ]
        for key, value in selected:
            blocks.append(f"### {key}\n{self._render_value(value, max_chars)}")
        return "\n\n".join(blocks)

    @staticmethod
    def _render_value(value: Any, max_chars: int) -> str:
        """Turn one stored value into prompt-safe text.

        Strings go in verbatim - they are usually code or prose the next agent
        must read exactly. Anything else is rendered as JSON when it can be,
        because a dict printed with repr() gives the model Python syntax to
        misread as data.
        """
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, indent=2, default=str)
            except (TypeError, ValueError):
                text = repr(value)

        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
        return text

    # --- Lifecycle ---

    def clear(self) -> None:
        """Wipe both the artifacts and the audit log - a fresh run."""
        with self._lock:
            self.artifacts.clear()
            self.execution_log.clear()

    def snapshot(self) -> dict:
        """A plain-dict copy, for printing a trace or asserting in a test."""
        with self._lock:
            return {
                "artifacts": dict(self.artifacts),
                "execution_log": list(self.execution_log),
            }

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"SharedWorkflowMemory(artifacts={list(self.artifacts)}, "
                f"steps={len(self.execution_log)})"
            )


WorkflowScratchpad = SharedWorkflowMemory


_SHARED_MEMORY_SUPPORT: dict[tuple, bool] = {}


def accepts_shared_memory(method: Callable) -> bool:
    """True when `method` declares a `shared_memory` parameter (or **kwargs)."""
    owner = getattr(method, "__self__", None)
    cache_key = (
        type(owner) if owner is not None else method,
        getattr(method, "__name__", ""),
    )

    if cache_key not in _SHARED_MEMORY_SUPPORT:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return False
        _SHARED_MEMORY_SUPPORT[cache_key] = "shared_memory" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    return _SHARED_MEMORY_SUPPORT[cache_key]


def call_with_shared_memory(
    method: Callable,
    task: Any,
    shared_memory: Optional[SharedWorkflowMemory] = None,
):
    """Call `method(task)`, forwarding `shared_memory` only if it is wanted."""
    if shared_memory is not None and accepts_shared_memory(method):
        return method(task, shared_memory=shared_memory)
    return method(task)
