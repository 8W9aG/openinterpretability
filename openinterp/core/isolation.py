"""The files that the reviewer agent must not read, and the check that enforces this.

``docs/PROTOCOL.md`` §2 forbids the reviewer from seeing probe activations before its
probe set is final. Records, submissions, agent exchange files, logs and the built site
hold them. Two things use :func:`forbidden`:

- The ``PreToolUse`` hook of ``.claude/agents/neuron-reviewer.md`` runs this module
  (``python -m openinterp.core.isolation``) before each tool call of the reviewer. It
  denies a call that reads, searches (``Grep``) or writes a blocked path.
- ``open-interpretability usage`` reads the reviewer's transcript
  (:func:`openinterp.core.transcripts.forbidden_reads`). A blocked read that completed
  is written to the usage file, and ``review --probes`` then refuses the probe set.

Only paths in this checkout are blocked. A copy of the registry in another directory is
not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openinterp.core.paths import (
    DEFAULT_EXCHANGE_DIR,
    DEFAULT_SUBMISSIONS_DIR,
    DEFAULT_WIKI_DIR,
    PROBES_SUFFIX,
    PROMPT_SUFFIX,
    experiments_dir,
)

BLOCKED_DIRS = (
    experiments_dir(DEFAULT_WIKI_DIR),
    DEFAULT_SUBMISSIONS_DIR,
    DEFAULT_EXCHANGE_DIR,
    Path("dist"),  # the built site (DIST_DIR in the Makefile)
    Path("build"),  # sample records and wikis that `make` builds
    Path("logs"),  # logs of driver sessions
)
"""Directories, relative to the project root, that hold probe activations."""

REVIEWER_EXCHANGE_FILES = (PROMPT_SUFFIX, PROBES_SUFFIX)
"""Ends of the names of the files in the exchange directory that the reviewer can use.

These are its prompt, its token-check prompt (``<name>.check.prompt.md``) and its answer.
Prompts and probe sets hold no probe activations. Every other file there can, e.g.
``review.log``.
"""

TOOLS = ("Read", "Write", "Edit", "Grep")
"""The tools that :func:`forbidden` checks. The reviewer has no tool that runs code.

``Glob`` is not checked: it gives file names, and a name holds no activations.
"""


def _resolve(path: str | Path, cwd: str | Path) -> Path:
    """Return ``path`` as an absolute path with symbolic links resolved."""
    return Path(os.path.realpath(Path(cwd) / Path(path).expanduser()))


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def blocked_file(path: Path, root: Path) -> bool:
    """Return True if the reviewer must not read or write the file ``path``.

    ``path`` and ``root`` (the project root) must be resolved.
    """
    for rel in BLOCKED_DIRS:
        blocked = _resolve(rel, root)
        if not _inside(path, blocked):
            continue
        reviewer_file = path.name.endswith(REVIEWER_EXCHANGE_FILES)
        return not (rel == DEFAULT_EXCHANGE_DIR and path != blocked and reviewer_file)
    return False


def blocked_search(path: Path, root: Path) -> bool:
    """Return True if a search from ``path`` can read a blocked file.

    A search is blocked if it starts in a blocked directory, or in a directory that
    contains one. A search of one allowed file is not blocked. That file can be gone,
    e.g. when an old transcript is checked, so its name decides.
    """
    if path.is_file() or (path.name.endswith(REVIEWER_EXCHANGE_FILES) and not path.is_dir()):
        return blocked_file(path, root)
    return any(
        _inside(path, blocked) or _inside(blocked, path)
        for blocked in (_resolve(rel, root) for rel in BLOCKED_DIRS)
    )


def forbidden(tool: str, tool_input: dict, cwd: str | Path, root: str | Path) -> str | None:
    """Return the blocked path that a tool call reads or searches, or None.

    ``tool_input`` is the call's input as Claude Code gives it. Relative paths are
    relative to ``cwd``. ``root`` is the project root that :data:`BLOCKED_DIRS` are
    relative to.
    """
    root = _resolve(root, cwd)
    if tool in ("Read", "Write", "Edit"):
        target = tool_input.get("file_path")
        if target and blocked_file(_resolve(target, cwd), root):
            return str(target)
        return None
    if tool == "Grep":
        start = _resolve(tool_input.get("path") or cwd, cwd)
        if blocked_search(start, root):
            return str(start)
    return None


def _deny(reason: str) -> None:
    decision = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }
    print(json.dumps({"hookSpecificOutput": decision}))


def main() -> int:
    """Run as a Claude Code ``PreToolUse`` hook: read the event and deny a blocked call.

    If the event cannot be checked, the call is denied.
    """
    try:
        event = json.load(sys.stdin)
        cwd = event.get("cwd") or os.getcwd()
        root = os.environ.get("CLAUDE_PROJECT_DIR") or cwd
        hit = forbidden(event.get("tool_name", ""), event.get("tool_input") or {}, cwd, root)
    except Exception as exc:  # noqa: BLE001 - deny when the check itself fails
        _deny(f"The reviewer isolation check failed ({exc!r}), so the call is denied.")
        return 0
    if hit is not None:
        _deny(
            f"Reviewer isolation (docs/PROTOCOL.md §2): {hit} is blocked. Records, "
            "submissions, exchange files other than prompts and probe sets, logs and the "
            "built site hold probe activations. Read only your prompt and the skills."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
