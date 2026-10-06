from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from agent_framework.tools.base import Tool
from agent_framework.tools.builtin.calculator import (
    CalculatorError,
    build_calculator_tool,
    calculate,
)
from agent_framework.tools.builtin.file_io import (
    DEFAULT_SANDBOX_ROOT,
    FileSandbox,
    SandboxError,
    build_file_tools,
)
from agent_framework.tools.builtin.search import (
    MOCK_CORPUS,
    SEARCH_BACKEND_ENV,
    LiveSearchUnavailable,
    SearchBackend,
    build_search_tool,
    build_web_search_tool,
    live_search,
    mock_search,
    search,
    web_search,
)


def build_builtin_tools(
    sandbox_root: Union[str, Path] = DEFAULT_SANDBOX_ROOT,
    sandbox: Optional[FileSandbox] = None,
    search_backend: Optional[SearchBackend] = None,
) -> list[Tool]:
    """Every built-in tool, freshly constructed.

    Fresh instances rather than module-level singletons: the file tools close
    over a sandbox root, so two registries in one process (a test's and the
    application's) must be able to point at different directories.
    """
    return [
        build_calculator_tool(),
        *build_file_tools(root=sandbox_root, sandbox=sandbox),
        build_search_tool(backend=search_backend),
        build_web_search_tool(backend=search_backend),
    ]


__all__ = [
    "CalculatorError",
    "DEFAULT_SANDBOX_ROOT",
    "FileSandbox",
    "LiveSearchUnavailable",
    "MOCK_CORPUS",
    "SEARCH_BACKEND_ENV",
    "SandboxError",
    "SearchBackend",
    "build_builtin_tools",
    "build_calculator_tool",
    "build_file_tools",
    "build_search_tool",
    "build_web_search_tool",
    "calculate",
    "live_search",
    "mock_search",
    "search",
    "web_search",
]
