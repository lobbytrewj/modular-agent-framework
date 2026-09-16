from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from agent_framework.tools.base import Tool, ToolPermission

# File access, confined to one directory.
#
# Two separate protections are at work here and they are not interchangeable:
#
#   * the SANDBOX (this module) decides *where* a path may point - every
#     request is resolved and checked to be inside the root, so "../../.ssh/id_rsa"
#     and "/etc/passwd" fail the same way;
#   * the PERMISSION (the registry) decides *who* may write at all.
#
# A sandbox without permissions lets any agent scribble in the workspace; a
# permission check without a sandbox lets an authorized agent write anywhere on
# the disk. Both are needed.

# Where the tools operate when no root is given. Relative to the working
# directory and created on first write, not on import - importing the tools
# module should not leave directories behind in a repo.
DEFAULT_SANDBOX_ROOT = Path("workspace")

# Caps, so a runaway agent can't fill the disk or pull a gigabyte into a prompt.
MAX_READ_BYTES = 256_000
MAX_WRITE_BYTES = 256_000


class SandboxError(ValueError):
    """Raised when a path escapes the sandbox or a limit is exceeded."""


class FileSandbox:
    """A directory that file tools are confined to."""

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_SANDBOX_ROOT,
        max_read_bytes: int = MAX_READ_BYTES,
        max_write_bytes: int = MAX_WRITE_BYTES,
    ):
        # Resolved once at construction: every later check compares against a
        # fully-resolved root, so a symlinked root doesn't make every path
        # inside it look like an escape.
        self.root = Path(root).expanduser().resolve()
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def resolve(self, path: Union[str, Path], must_exist: bool = False) -> Path:
        """Turn a caller-supplied path into an absolute one inside the root.

        Absolute paths are refused outright rather than reinterpreted as
        relative: silently rewriting "/etc/passwd" into "<root>/etc/passwd"
        would answer a request nobody made, and the honest failure tells the
        caller what the rule actually is.

        Resolution happens BEFORE the containment check and follows symlinks,
        so a link inside the sandbox pointing outside it is caught too - that
        is the escape route a purely textual ".." check misses.
        """
        if not isinstance(path, (str, Path)) or str(path).strip() == "":
            raise SandboxError("path must be a non-empty string")

        candidate = Path(path)
        if candidate.is_absolute() or str(path).startswith("~"):
            raise SandboxError(
                f"path must be relative to the sandbox root, got '{path}'"
            )

        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise SandboxError(
                f"path '{path}' escapes the sandbox root {self.root}"
            ) from None

        if must_exist and not resolved.is_file():
            raise SandboxError(f"no such file in the sandbox: '{path}'")
        return resolved

    def relative(self, resolved: Path) -> str:
        """The sandbox-relative form, which is all a caller should ever see."""
        return str(resolved.relative_to(self.root))

    # --- Operations ---

    def read(self, path: str, encoding: str = "utf-8") -> dict[str, Any]:
        resolved = self.resolve(path, must_exist=True)

        size = resolved.stat().st_size
        if size > self.max_read_bytes:
            raise SandboxError(
                f"file '{path}' is {size} bytes, over the "
                f"{self.max_read_bytes}-byte read limit"
            )

        content = resolved.read_text(encoding=encoding)
        return {"path": self.relative(resolved), "content": content, "bytes": size}

    def write(
        self,
        path: str,
        content: str,
        append: bool = False,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise SandboxError(
                f"content must be a string, got {type(content).__name__}"
            )

        encoded = len(content.encode(encoding))
        if encoded > self.max_write_bytes:
            raise SandboxError(
                f"content is {encoded} bytes, over the "
                f"{self.max_write_bytes}-byte write limit"
            )

        resolved = self.resolve(path)
        # Parent directories are created inside the sandbox only - `resolve`
        # has already established that `resolved` is under the root.
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved, "a" if append else "w", encoding=encoding) as handle:
            handle.write(content)

        return {
            "path": self.relative(resolved),
            "bytes_written": encoded,
            "mode": "append" if append else "overwrite",
        }

    def list_files(self, path: str = ".") -> dict[str, Any]:
        directory = (
            self.root if path in ("", ".") else self.resolve(path)
        )
        if not directory.is_dir():
            raise SandboxError(f"no such directory in the sandbox: '{path}'")

        entries = sorted(
            (
                self.relative(entry) + ("/" if entry.is_dir() else "")
                for entry in directory.iterdir()
            )
        )
        return {"path": path, "entries": entries}

    def __repr__(self) -> str:
        return f"FileSandbox(root={str(self.root)!r})"


READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path relative to the sandbox root.",
        }
    },
    "required": ["path"],
}

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path relative to the sandbox root.",
        },
        "content": {"type": "string", "description": "Text to write."},
        "append": {
            "type": "boolean",
            "description": "Append instead of overwriting. Defaults to false.",
        },
    },
    "required": ["path", "content"],
}

LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory relative to the sandbox root. Defaults to the root.",
        }
    },
    "required": [],
}


def build_file_tools(
    root: Union[str, Path] = DEFAULT_SANDBOX_ROOT,
    sandbox: Optional[FileSandbox] = None,
) -> list[Tool]:
    """The three sandboxed file tools, all sharing one FileSandbox.

    `file_read` and `file_list` need only READ_ONLY; `file_write` needs WRITE.
    That split is the point of the permission model - an agent can be given
    the ability to consult the workspace without the ability to change it.
    """
    box = sandbox or FileSandbox(root)

    return [
        Tool(
            name="file_read",
            description=(
                f"Read a UTF-8 text file from the sandboxed workspace "
                f"({box.root.name}/). Paths are relative to that directory."
            ),
            func=box.read,
            parameters_schema=READ_SCHEMA,
            required_permission=ToolPermission.READ_ONLY,
        ),
        Tool(
            name="file_list",
            description=(
                "List the files and directories in the sandboxed workspace."
            ),
            func=box.list_files,
            parameters_schema=LIST_SCHEMA,
            required_permission=ToolPermission.READ_ONLY,
        ),
        Tool(
            name="file_write",
            description=(
                "Write UTF-8 text to a file in the sandboxed workspace, "
                "creating parent directories as needed. Overwrites by default; "
                "pass append=true to add to the end instead."
            ),
            func=box.write,
            parameters_schema=WRITE_SCHEMA,
            required_permission=ToolPermission.WRITE,
        ),
    ]
