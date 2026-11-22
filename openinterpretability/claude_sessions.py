"""Headless Claude Code sessions, one experiment each (#5831).

``scripts/experiment-loop.sh`` and ``explain --run`` both start ``claude -p`` sessions that
run one experiment through the repository's skills and subagents. This module is the one
launcher for both. It holds the tools that a session may use without asking, the ``claude``
arguments and the provider wrapper. It writes the session's JSON result and its costs
ledger. A session ends its reply with a ``RESULT:`` line, which :func:`result_line` reads.

``open-interpretability claude-session`` runs one session from a prompt on stdin, for the
script.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.explain_backend import claude_bin

DRIVER = "claude-haiku-5-5"  # lands as reliably as larger drivers at a fraction of the cost
MAX_TURNS = 200
# Tools the session may use without asking. Nothing outside the repository's workflow.
ALLOWED_TOOLS = (
    "Read", "Edit", "Write", "Glob", "Grep", "Skill", "Agent",
    "Bash(uv run:*)", "Bash(make:*)", "Bash(git:*)", "Bash(gh pr:*)", "Bash(gh run:*)",
    "Bash(gh repo view:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)",
    "Bash(grep:*)", "Bash(jq:*)",
)  # fmt: skip
COSTS_ENV = "OPENINTERPRETABILITY_COSTS_LOG"


@dataclass(frozen=True)
class Session:
    """How to start one session: the driver model, its turns, and an optional provider."""

    driver: str = DRIVER  # "" for claude's own default
    max_turns: int = MAX_TURNS
    provider: str = ""  # a provider profile (docs/providers.md) whose environment it uses

    def command(self, env: dict[str, str] | None = None) -> list[str]:
        """The command line: the provider wrapper, ``claude`` and its arguments.

        Args:
            env: The environment, for ``$CLAUDE_BIN``; the process's by default.

        Returns:
            The arguments to run, with the allowed tools last, as ``claude`` reads them.

        Examples:
            >>> Session("m", 5).command({})[:4]
            ['claude', '-p', '--output-format', 'json']
            >>> Session("", 5, "deepseek").command({})[:6]
            ['uv', 'run', '--quiet', 'open-interpretability', 'provider', 'exec']
        """
        launch = [claude_bin(env)]
        if self.provider:
            launch = ["uv", "run", "--quiet", "open-interpretability", "provider", "exec",
                      self.provider, "--", *launch]  # fmt: skip
        args = ["-p", "--output-format", "json", "--permission-mode", "acceptEdits",
                "--max-turns", str(self.max_turns)]  # fmt: skip
        if self.driver:
            args += ["--model", self.driver]
        return [*launch, *args, "--allowedTools", *ALLOWED_TOOLS]

    def run(self, prompt: str, cwd: Path, result: Path, log: Path, ledger: Path) -> int:
        """Run one session on ``prompt`` in ``cwd``. Return ``claude``'s exit code.

        Args:
            prompt: What the session does.
            cwd: The worktree it works in.
            result: Where its JSON result goes (stdout).
            log: Where its stderr goes.
            ledger: The costs ledger that its ``open-interpretability`` commands append to.

        Returns:
            The exit code; not 0 when the session failed, for example at a usage limit.
        """
        env = {**os.environ, COSTS_ENV: str(ledger)}
        with open(result, "w") as out, open(log, "a") as err:
            done = subprocess.run(self.command(env), input=prompt, text=True, cwd=cwd,
                                  stdout=out, stderr=err, env=env, check=False)  # fmt: skip
        return done.returncode


def session_text(result: Path) -> str:
    """The session's final message, from its JSON result; "" when it can't be read.

    Args:
        result: The session's JSON result.

    Returns:
        The ``result`` field.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "s.json"
        >>> _ = path.write_text('{"result": "done"}')
        >>> session_text(path), session_text(path.with_name("missing.json"))
        ('done', '')
    """
    try:
        return str(json.loads(result.read_text(encoding="utf-8")).get("result", ""))
    except (OSError, ValueError, AttributeError):
        return ""


def result_line(text: str) -> str | None:
    """The last ``RESULT:`` line of a session's reply, without the prefix.

    Args:
        text: The reply.

    Returns:
        ``<slug> <verdict> <PR url> <merged|open>``, ``none <reason>``, or None.

    Examples:
        >>> result_line("work\\nRESULT: a/L1/N2-positive VERIFIED https://x/pull/3 open\\n")
        'a/L1/N2-positive VERIFIED https://x/pull/3 open'
        >>> result_line("no line") is None
        True
    """
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("RESULT:")]
    return lines[-1].removeprefix("RESULT:").strip() if lines else None
