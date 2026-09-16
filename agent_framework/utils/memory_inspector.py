
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence, TYPE_CHECKING

from agent_framework.core.memory import AgentMemory, SharedWorkflowMemory

if TYPE_CHECKING:  # imported for typing only - keeps this module importable
    from agent_framework.agents.base import BaseAgent  # without the LLM stack

WIDTH = 78

INDENT = "  "


# --- Formatting helpers ---


def _heading(title: str, char: str = "=") -> None:
    """A titled block: rule, title, rule."""
    print(f" {title}")


def _section(number: int, title: str) -> None:
    print()
    print(f" [{number}] {title}")

def _truncate(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters, saying how much was dropped.

    Line breaks are preserved: artifacts are usually code, and code that has
    been flattened onto one line is no longer the thing the agent was handed.
    """
    text = "" if text is None else str(text)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars total]"


def _one_line(text: str, limit: int) -> str:
    """A single-line summary - whitespace collapsed, then truncated.

    Used for the log's task/output columns, where one row per step matters
    more than seeing the full text.
    """
    flat = " ".join(("" if text is None else str(text)).split())
    if not flat:
        return "(empty)"
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def _indent_block(text: str, prefix: str = INDENT * 2) -> str:
    """Indent every line of a multi-line block, blank lines included."""
    if text == "":
        return prefix
    return "\n".join(prefix + line for line in text.splitlines())


def _describe_value(value: Any) -> str:
    """`type` plus whatever size measure fits that type."""
    type_name = type(value).__name__
    if isinstance(value, str):
        return f"{type_name} ({len(value)} chars)"
    if isinstance(value, (list, tuple, set, dict)):
        return f"{type_name} ({len(value)} items)"
    return type_name


def _clock(timestamp: Optional[float]) -> str:
    """A wall-clock string for a float epoch timestamp."""
    if timestamp is None:
        return "(no timestamp)"
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (OverflowError, OSError, ValueError, TypeError):
        return f"(unreadable timestamp: {timestamp!r})"


def _status(success: Optional[bool]) -> str:
    """SUCCESS / FAILED, or UNKNOWN for a step that never recorded an outcome."""
    if success is None:
        return "UNKNOWN"
    return "SUCCESS" if success else "FAILED"


# --- Sections ---

def _print_artifacts(shared_memory: SharedWorkflowMemory, truncate_chars: int) -> None:
    """Section 1: what is on the blackboard right now."""
    artifacts = shared_memory.snapshot()["artifacts"]
    _section(1, f"SHARED BLACKBOARD ARTIFACTS ({len(artifacts)})")

    if not artifacts:
        print(f"{INDENT}(the blackboard is empty - no step published an artifact)")
        return

    for position, (key, value) in enumerate(artifacts.items(), start=1):
        print()
        print(f"{INDENT}{position}. key: {key!r}")
        print(f"{INDENT}   type: {_describe_value(value)}")
        print(f"{INDENT}   value:")
        rendered = value if isinstance(value, str) else repr(value)
        print(_indent_block(_truncate(rendered, truncate_chars), INDENT + "   | "))


def _print_prompt_context(shared_memory: SharedWorkflowMemory, truncate_chars: int) -> None:
    """Section 2: the exact markdown a downstream agent gets prepended.
    """
    _section(2, "PROMPT CONTEXT TRANSFER (to_prompt_context)")

    context = shared_memory.to_prompt_context(max_chars=truncate_chars)

    print(f"{INDENT}Produced by: shared_memory.to_prompt_context(max_chars={truncate_chars})")
    print(f"{INDENT}LLMAgent prepends this block, verbatim, ahead of the task")
    print(f"{INDENT}description in every downstream prompt.")
    print()

    if not context:
        print(f"{INDENT}(renders empty - nothing would be injected into a prompt)")
        return

    print(f"{INDENT}v---- begin injected block ({len(context)} chars) " + "-" * 20)
    print(_indent_block(context, INDENT + "| "))
    print(f"{INDENT}^---- end injected block " + "-" * 32)


def _print_execution_log(shared_memory: SharedWorkflowMemory, truncate_chars: int) -> None:
    """Section 3: every step that ran, in the order it ran.
    """
    log = shared_memory.snapshot()["execution_log"]
    _section(3, f"GLOBAL EXECUTION AUDIT TRAIL ({len(log)} step(s))")

    if not log:
        print(f"{INDENT}(nothing logged - no step executed through execute_agent)")
        return

    # Summaries are half-width so the task and output columns stay one line
    # each even when truncate_chars is generous.
    summary_limit = max(40, truncate_chars // 2)

    for record in log:
        step = record.get("step", "?")
        workflow = record.get("workflow") or "(standalone)"
        agent = record.get("agent", "(unknown agent)")
        role = record.get("role")
        agent_label = f"{agent} ({role})" if role else agent

        print()
        print(
            f"{INDENT}step {step} | {_clock(record.get('timestamp'))} | "
            f"{_status(record.get('success'))}"
        )
        print(f"{INDENT}  workflow: {workflow}")
        print(f"{INDENT}  agent:    {agent_label}")
        print(f"{INDENT}  task:     {_one_line(record.get('task'), summary_limit)}")
        print(f"{INDENT}  output:   {_one_line(record.get('output'), summary_limit)}")
        if record.get("error"):
            print(f"{INDENT}  error:    {_one_line(record.get('error'), summary_limit)}")


def _print_agent_memories(
    agents: Optional[Sequence["BaseAgent"]], truncate_chars: int
) -> None:
    """Section 4: each agent's private conversation.
    """
    _section(4, "AGENT-LOCAL PRIVATE MEMORY")

    if not agents:
        print(f"{INDENT}(no agents passed - call with agents=[...] to see their histories)")
        return

    for agent in agents:
        name = getattr(agent, "name", repr(agent))
        role = getattr(agent, "role", None)
        memory = getattr(agent, "memory", None)

        print()
        print(f"{INDENT}{name}" + (f" (role: {role})" if role else ""))

        if memory is None:
            # A stateless agent, which is the framework's default: it was
            # given everything it needed in the prompt and remembers nothing.
            print(f"{INDENT}  memory: none attached (stateless agent)")
            continue

        if not isinstance(memory, AgentMemory):
            print(f"{INDENT}  memory: {type(memory).__name__} (not an AgentMemory)")
            continue

        print(f"{INDENT}  memory: AgentMemory holding {len(memory)} turn(s)")
        if not memory.messages:
            print(f"{INDENT}    (empty - this agent has not spoken yet)")
            continue

        for position, message in enumerate(memory.messages, start=1):
            label = f"[{message.role.upper()}]"
            print(f"{INDENT}    {position}. {label:<12} {_clock(message.timestamp)}")
            print(_indent_block(_truncate(message.content, truncate_chars), INDENT + "       | "))


# --- Public API ---

def print_full_memory_trace(
    shared_memory: SharedWorkflowMemory,
    agents: Optional[Sequence["BaseAgent"]] = None,
    truncate_chars: int = 300,
) -> None:
    """Print everything a run remembers, shared state and private state apart.
    """
    print()
    _heading("FULL MEMORY TRACE")
    print(
        f"{INDENT}{len(shared_memory.keys())} artifact(s) | "
        f"{len(shared_memory.execution_log)} logged step(s) | "
        f"{len(agents) if agents else 0} agent(s) inspected | "
        f"truncate_chars={truncate_chars}"
    )

    _print_artifacts(shared_memory, truncate_chars)
    _print_prompt_context(shared_memory, truncate_chars)
    _print_execution_log(shared_memory, truncate_chars)
    _print_agent_memories(agents, truncate_chars)

    print()
    _heading("END OF MEMORY TRACE")
