"""Plumbing shared by the end-to-end demos in this directory.

Each demo is a standalone script; this module holds what they have in common:

  * making the `agent_framework` package importable when a demo is run as
    `python examples/<demo>.py` from the repo root;
  * building agents from config/agents.json, optionally against an injected
    LLM client so a demo can run offline (`--mock`) or under test;
  * a scripted client that answers from a Python function instead of a model;
  * small parsers for the shapes the demos ask models for (code blocks, JSON,
    `calculator(...)` calls);
  * exporting a run's trace and printing its summary table.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_framework.agents import BaseAgent, LLMAgent  # noqa: E402
from agent_framework.core.memory import SharedWorkflowMemory  # noqa: E402
from agent_framework.observability import Tracer, print_run_summary  # noqa: E402
from agent_framework.orchestration.sequential import (  # noqa: E402
    MEMORY_PARAMS,
    TOOL_PARAMS,
    build_agent_kwargs,
    load_agent_config,
)
from agent_framework.tools.registry import ToolRegistry  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "agents.json"
REPORTS_DIR = REPO_ROOT / "reports"

# (config key, system prompt, user message) -> reply. Keyed by config entry so
# one function can script every role in a demo.
Responder = Callable[[str, str, str], str]

# (config key, config entry) -> an object with LLMClient's `complete` method.
ClientFactory = Callable[[str, dict], Any]


# --- Scripted client ---


class ScriptedLLMClient:
    """A stand-in for LLMClient that answers from a function, not a model.

    Same `complete` signature as the real client, so an LLMAgent built from
    config cannot tell the difference - its prompt, tools, permissions and
    shared-memory wiring all run exactly as they would against Qwen. Every
    call is kept in `calls`, so a test can check what an agent was shown.
    """

    def __init__(self, key: str, responder: Responder):
        self.key = key
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_message": user_message, "history": history}
        )
        return self.responder(self.key, system_prompt, user_message)


def scripted_client_factory(responder: Responder) -> ClientFactory:
    """A ClientFactory handing each config key its own ScriptedLLMClient.

    The clients it built are kept on the factory as `clients[key]`, so a
    caller can inspect every prompt a given role received.
    """
    clients: dict[str, ScriptedLLMClient] = {}

    def factory(key: str, entry: dict) -> ScriptedLLMClient:
        if key not in clients:
            clients[key] = ScriptedLLMClient(key, responder)
        return clients[key]

    factory.clients = clients  # type: ignore[attr-defined]
    return factory


# --- Building agents from config ---

# Placeholder client for agents that never call a model; see build_agent.
_NO_MODEL = object()


def build_agent(
    config: dict[str, dict],
    key: str,
    *,
    client_factory: Optional[ClientFactory] = None,
    tool_registry: Optional[ToolRegistry] = None,
    agent_cls: type = LLMAgent,
    **overrides: Any,
) -> BaseAgent:
    """Instantiate one config entry as `agent_cls`.

    Grants always come from the entry: `tools` and `permissions` may not be
    overridden here, so what a demo shows an agent doing is exactly what
    agents.json authorizes it to do. Other settings (a numbered name for a
    parallel worker, extra subclass arguments) can be.

    An `agent_cls` that is not an LLMAgent is a pure-Python step (running a
    tool, say); it is still built from its entry, so its grants are real, but
    it is never handed a client and never loads a model.
    """
    if key not in config:
        raise KeyError(f"No agent configured under key '{key}'")
    illegal = set(overrides) & set(TOOL_PARAMS)
    if illegal:
        raise ValueError(f"grants come from config only; cannot override {sorted(illegal)}")

    entry = config[key]
    uses_model = issubclass(agent_cls, LLMAgent)
    if not uses_model:
        client = _NO_MODEL
    elif client_factory is not None:
        client = client_factory(key, entry)
    else:
        client = None  # build_agent_kwargs builds the real one from the entry

    kwargs = build_agent_kwargs(entry, tool_registry=tool_registry, llm_client=client)
    if not uses_model:
        for param in ("llm_client", *MEMORY_PARAMS):
            kwargs.pop(param, None)

    kwargs.setdefault("name", key)
    kwargs.setdefault("role", key)
    kwargs.setdefault("system_prompt", f"You are the '{key}' agent.")
    kwargs.update(overrides)
    return agent_cls(**kwargs)



def load_config(config_path: Union[str, Path, None] = None) -> dict[str, dict]:
    return load_agent_config(str(config_path or DEFAULT_CONFIG))


# --- Parsing model replies ---

_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """The code in a model reply: its first fenced block, or the whole reply.

    An unterminated fence (a reply cut off by max_tokens) still yields what
    was written, so truncation shows up as a failing test rather than as an
    empty file.
    """
    text = text or ""
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def extract_calls(text: str, tool_name: str) -> list[str]:
    """The argument of every `tool_name(...)` in `text`, one level of nesting deep."""
    pattern = re.compile(
        rf"\b{re.escape(tool_name)}\s*\(((?:[^()\n]|\([^()\n]*\))*)\)", re.IGNORECASE
    )
    return [match.group(1).strip() for match in pattern.finditer(text or "") if match.group(1).strip()]


def one_line(text: Any, limit: int = 100) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


# --- Running a demo ---


@dataclass
class DemoResult:
    """What a demo's `run_demo` hands back: enough for a test to assert on."""

    success: bool
    output: str
    shared_memory: SharedWorkflowMemory
    tracer: Tracer
    trace_path: Path
    details: dict[str, Any] = field(default_factory=dict)


def start_run(workflow: str, **metadata: Any) -> tuple[SharedWorkflowMemory, Tracer]:
    """A fresh blackboard with a tracer attached, so every component sees it.

    Attached rather than activated: the research swarm's workers run on pool
    threads, and only an attached tracer reaches them.
    """
    shared_memory = SharedWorkflowMemory()
    tracer = Tracer.for_memory(shared_memory, workflow=workflow, **metadata)
    return shared_memory, tracer


def finish_run(
    tracer: Tracer,
    shared_memory: SharedWorkflowMemory,
    trace_path: Union[str, Path],
    quiet: bool = False,
) -> Path:
    """Snapshot the blackboard, close the trace, export it, print the summary."""
    tracer.record_snapshot(shared_memory, label="final")
    tracer.finish()
    path = Path(trace_path)
    if quiet:
        tracer.export_json(path)
    else:
        print_run_summary(tracer, json_path=path)
    return path


def section(title: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"\n=== {title} ===")


def say(message: str = "", quiet: bool = False) -> None:
    if not quiet:
        print(message)


def demo_arg_parser(description: str, default_trace: Path) -> argparse.ArgumentParser:
    """The flags every demo takes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run offline against scripted responses instead of loading the local model",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="agent definitions (default: config/agents.json)",
    )
    parser.add_argument(
        "--trace",
        default=str(default_trace),
        help=f"where to export the JSON trace (default: {default_trace.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="skip progress output and the summary table"
    )
    return parser
