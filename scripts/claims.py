"""Claim work with git branches, so that several agents can work at once (#6725).

A claim is a branch ``claim/<key>`` on the remote, with one empty commit that names the
agent. GitHub refuses a push to a branch that holds another commit, so of two agents that
claim one key, only one push succeeds. ``scripts/issue_state.py`` uses these helpers for
``--claim``, ``--claim-release``, ``--unclaim`` and ``--whoami``.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import time

# Claims are branches under this prefix (#6725). CI runs on pushes to main only, so a
# claim starts no workflow.
CLAIM_PREFIX = "claim/"
# The exit code of a claim that another agent holds.
CLAIMED = 3
# Overrides the agent ID that agent_id derives from the checkout.
AGENT_ENV = "OPENINTERPRETABILITY_AGENT"


def agent_id(cwd: str | None = None) -> str:
    """This agent's ID: ``$OPENINTERPRETABILITY_AGENT``, else a hash of the host and the checkout.

    Each agent works in its own checkout (docs/development.md, *Several agents at once*),
    so the checkout names the agent. The hash keeps the host name and path private.

    Args:
        cwd: A directory in the checkout. None means the current directory.

    Returns:
        The ID, for example ``agent-3f9a1c2b``.
    """
    if os.environ.get(AGENT_ENV):
        return os.environ[AGENT_ENV]
    top = git(["rev-parse", "--show-toplevel"], cwd).stdout.strip()
    digest = hashlib.sha256(f"{socket.gethostname()}:{top}".encode()).hexdigest()
    return f"agent-{digest[:8]}"


def git(args: list[str], cwd: str | None = None, check: bool = True):
    """Run ``git`` with ``args`` in ``cwd``, with empty input.

    Args:
        args: The git arguments, without ``git``.
        cwd: The directory to run in. None means the current directory.
        check: Raise when git exits with an error.

    Returns:
        The completed process, with its output as text.
    """
    return subprocess.run(
        ["git", *args], cwd=cwd, input="", capture_output=True, text=True, check=check
    )


def claim_branch(key: str) -> str:
    """The claim branch of ``key``.

    Args:
        key: ``issue-<N>``, ``issue-<N>-reply-<time>`` or ``release-after-<tag>``.

    Returns:
        The branch name, without ``refs/heads/``.
    """
    return f"{CLAIM_PREFIX}{key}"


def claim(key: str, remote: str = "origin", cwd: str | None = None) -> tuple[bool, str]:
    """Claim ``key`` by pushing a new commit to its claim branch on ``remote``.

    The commit has no parent and an empty tree, so the push sends no files. Its message
    names the agent and the time, so no two claims make the same commit. A branch that
    exists already holds another commit, so the push is not a fast-forward and the
    remote refuses it. That refusal is atomic: of two agents, only one wins.

    Args:
        key: What to claim, for example ``issue-3371``.
        remote: The git remote that holds the claims.
        cwd: A directory in the checkout.

    Returns:
        True and the branch when this agent now holds the claim. False and the holder's
        claim message when another agent holds it.

    Raises:
        RuntimeError: When the push fails and no claim branch exists, for example
            because the remote can't be reached.
    """
    branch = claim_branch(key)
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    message = f"Claim {key} by {agent_id(cwd)} at {when} ({time.time_ns()})"
    tree = git(["hash-object", "-t", "tree", "-w", "--stdin"], cwd).stdout.strip()
    commit = git(["commit-tree", tree, "-m", message], cwd).stdout.strip()
    pushed = git(["push", "--quiet", remote, f"{commit}:refs/heads/{branch}"], cwd, check=False)
    if pushed.returncode == 0:
        return True, branch
    holder = claim_holder(key, remote, cwd)
    if holder is None:
        raise RuntimeError(f"could not push {branch}: {pushed.stderr.strip()}")
    return False, holder


def claim_holder(key: str, remote: str = "origin", cwd: str | None = None) -> str | None:
    """The message of the claim on ``key`` that ``remote`` holds.

    Args:
        key: What was claimed, for example ``issue-3371``.
        remote: The git remote that holds the claims.
        cwd: A directory in the checkout.

    Returns:
        The claim commit's message. None when there is no claim, or when ``remote``
        can't be read.
    """
    ref = f"refs/heads/{claim_branch(key)}"
    listed = git(["ls-remote", remote, ref], cwd, check=False)
    if listed.returncode != 0 or not listed.stdout.strip():
        return None
    git(["fetch", "--quiet", remote, ref], cwd)
    return git(["log", "-1", "--format=%s", "FETCH_HEAD"], cwd).stdout.strip()


def unclaim(number: str, remote: str = "origin", cwd: str | None = None) -> list[str]:
    """Delete every claim branch of issue ``number`` on ``remote``, reply claims too.

    Args:
        number: The issue's number.
        remote: The git remote that holds the claims.
        cwd: A directory in the checkout.

    Returns:
        The branches that were deleted.
    """
    base = f"refs/heads/{claim_branch(f'issue-{number}')}"
    listed = git(["ls-remote", remote, base, f"{base}-reply-*"], cwd).stdout.split()
    refs = [r for r in listed[1::2] if r == base or r.startswith(f"{base}-reply-")]
    if refs:
        git(["push", "--quiet", remote, *(f":{r}" for r in refs)], cwd)
    return [r.removeprefix("refs/heads/") for r in refs]


def reply_key(number: str, reply: str | None) -> str:
    """The claim key of issue ``number``, or of the resume after one reply.

    Args:
        number: The issue's number.
        reply: The reply's time, for example ``20261124T194025Z``. None claims the issue.

    Returns:
        ``issue-<number>``, or ``issue-<number>-reply-<reply>``.

    Raises:
        ValueError: When ``reply`` has a character other than a letter or a digit.
    """
    if reply is None:
        return f"issue-{number}"
    if not re.fullmatch(r"[0-9A-Za-z]+", reply):
        raise ValueError(f"a reply time is letters and digits only (20261124T194025Z): {reply!r}")
    return f"issue-{number}-reply-{reply}"
