"""Which agents ``explain`` uses when the user gives none (#5831).

``explain --run`` needs a proposer and a reviewer, and a narrated level needs a narrator.
A LiteLLM model needs an API key. A user with Claude Code has its subagents instead
(``.claude/agents/``), which answer ``external/<name>`` prompts. :func:`backend` is the one
rule that picks how a run's experiments are done:

- **litellm**: a model or a ``--provider`` is given. The experiment loop calls it.
- **claude-sessions**: none is given and ``claude`` is on ``PATH``. Each experiment runs in
  a headless Claude Code session (:mod:`openinterpretability.claude_sessions`), whose
  subagents answer the prompts.
- None, with the reason, when neither can run. Inside Claude Code, the ``explain`` skill
  runs the experiments in the session itself.

:func:`default_narrator` makes the narrator an ``external/`` one inside Claude Code.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass

LITELLM, SESSIONS = "litellm", "claude-sessions"
CLAUDE_BIN_ENV = "CLAUDE_BIN"  # the claude command, as scripts/experiment-loop.sh reads it


@dataclass(frozen=True)
class Backend:
    """How the run's experiments are done: :data:`LITELLM`, :data:`SESSIONS`, or None."""

    kind: str | None
    why: str


def in_claude_code(env: Mapping[str, str] | None = None) -> bool:
    """Whether this process runs inside a Claude Code session, which sets ``CLAUDECODE=1``.

    Args:
        env: The environment; the process's by default.

    Returns:
        True inside Claude Code.

    Examples:
        >>> in_claude_code({"CLAUDECODE": "1"}), in_claude_code({})
        (True, False)
    """
    return (os.environ if env is None else env).get("CLAUDECODE") == "1"


def claude_bin(env: Mapping[str, str] | None = None) -> str:
    """The ``claude`` command: ``$CLAUDE_BIN``, else ``claude``.

    Args:
        env: The environment; the process's by default.

    Returns:
        The command's name or path.

    Examples:
        >>> claude_bin({}), claude_bin({"CLAUDE_BIN": "/opt/claude"})
        ('claude', '/opt/claude')
    """
    return (os.environ if env is None else env).get(CLAUDE_BIN_ENV) or "claude"


def backend(proposer: str, reviewer: str, provider: str, env: Mapping[str, str] | None = None,
            which: Callable[[str], str | None] = shutil.which) -> Backend:  # fmt: skip
    """The backend that runs ``explain --run``'s experiments (module docstring).

    Args:
        proposer: ``--proposer``, or "".
        reviewer: ``--reviewer``, or "".
        provider: ``--provider``, or "".
        env: The environment; the process's by default.
        which: Finds a command on ``PATH``.

    Returns:
        The backend, or a ``kind`` of None with the reason that none can run.

    Examples:
        >>> backend("anthropic/x", "anthropic/y", "", {}, lambda _: None).kind
        'litellm'
        >>> backend("", "", "", {}, lambda _: "/usr/bin/claude").kind
        'claude-sessions'
        >>> backend("", "", "", {"CLAUDECODE": "1"}, lambda _: None).why[:44]
        "no LiteLLM model, and `claude` isn't on PATH"
    """
    if provider or proposer or reviewer:
        return Backend(LITELLM, "a LiteLLM model or a provider is given")
    if which(claude_bin(env)):
        return Backend(SESSIONS, "no LiteLLM model is given, so each experiment runs in a "
                       "headless Claude Code session, whose subagents answer")  # fmt: skip
    if in_claude_code(env):
        return Backend(None, "no LiteLLM model, and `claude` isn't on PATH for headless "
                       "sessions. In this session, run the explain skill "
                       "(.claude/skills/explain), or give --provider.")  # fmt: skip
    return Backend(None, "no LiteLLM model: give --proposer and --reviewer, or --provider, "
                   "or install Claude Code (`claude` on PATH) to run each experiment in a "
                   "headless session.")  # fmt: skip


def default_narrator(env: Mapping[str, str] | None = None) -> str | None:
    """The narrator when ``--narrator-model`` isn't given: ``external/`` inside Claude Code.

    The name is the session's model when ``$ANTHROPIC_MODEL`` gives it, else
    ``claude-code``. The subagent that answers reports its own model ID in its answer, and
    the record names that model (#5966).

    Args:
        env: The environment; the process's by default.

    Returns:
        ``external/<model>`` inside Claude Code, else None for the agents' default.

    Examples:
        >>> default_narrator({"CLAUDECODE": "1", "ANTHROPIC_MODEL": "claude-fable-5-2"})
        'external/claude-fable-5-2'
        >>> default_narrator({"CLAUDECODE": "1"}), default_narrator({})
        ('external/claude-code', None)
    """
    env = os.environ if env is None else env
    if not in_claude_code(env):
        return None
    return f"external/{env.get('ANTHROPIC_MODEL') or 'claude-code'}"


def narrator(model: str | None, echo: Callable[[str], None] = print) -> str | None:
    """The narrator: ``model``, else :func:`default_narrator`, which says what to answer.

    Args:
        model: ``--narrator-model``, or None.
        echo: Where the line about the prompt goes.

    Returns:
        The model to narrate with; None for the agents' default.

    Examples:
        >>> narrator("anthropic/x")
        'anthropic/x'
    """
    if model is not None:
        return model
    found = default_narrator()
    if found is not None:  # inside Claude Code: a subagent answers, with no API key
        from pathlib import Path

        from openinterpretability.core.paths import DEFAULT_EXCHANGE_DIR

        folder = Path(os.environ.get("OPENINTERPRETABILITY_EXCHANGE_DIR", DEFAULT_EXCHANGE_DIR))
        echo(f"The narrator is {found}: answer {folder / 'Narrator.prompt.md'} with the "
             "explanation-narrator agent.")  # fmt: skip
    return found
