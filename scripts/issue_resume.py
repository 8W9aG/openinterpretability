"""Find the issues that an agent holds and must resume itself (#7221).

An issue with more than one PR stays In Progress after its first PR merges. Its claim
(``scripts/claims.py``) keeps every other agent away, so the agent that holds it must resume
it. ``scripts/next_issue.py`` offers these issues after an answered issue and before new work.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable


def open_pr_mentions(repo: str, maintainer: str) -> str:
    """The titles and bodies of the maintainer's open pull requests, for :func:`has_open_pr`.

    Only the maintainer's pull requests from ``repo`` itself count. Anyone can open a pull
    request, and a stranger's could otherwise name an issue and stop its agent resuming it.

    Args:
        repo: The repository, as ``owner/name``.
        maintainer: The login whose pull requests count (agents share it).

    Returns:
        The text.
    """
    out = subprocess.run(["gh", "pr", "list", "--repo", repo, "--state", "open", "--limit",
                          "500", "--json", "title,body,author,isCrossRepository"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return own_mentions(json.loads(out), maintainer)


def own_mentions(pulls: list[dict], maintainer: str) -> str:
    """The titles and bodies of the maintainer's pull requests from this repository, joined.

    Args:
        pulls: Pull requests, as ``gh pr list --json title,body,author,isCrossRepository``.
        maintainer: The login whose pull requests count.

    Returns:
        The text.

    Examples:
        >>> own_mentions([{"title": "a", "body": "#1", "author": {"login": "m"},
        ...                "isCrossRepository": False},
        ...               {"title": "b", "body": "#2", "author": {"login": "x"},
        ...                "isCrossRepository": False}], "m")
        'a\\n#1'
    """
    return "\n".join(f"{p['title']}\n{p.get('body') or ''}" for p in pulls
                     if (p.get("author") or {}).get("login") == maintainer
                     and not p.get("isCrossRepository"))  # fmt: skip


def view_issue(repo: str, number: int | str) -> dict:
    """The issue's number, title, labels, creation time, milestone and state.

    Args:
        repo: The repository, as ``owner/name``.
        number: The issue's number.

    Returns:
        The issue, as ``gh issue view --json`` gives it.
    """
    out = subprocess.run(["gh", "issue", "view", str(number), "--repo", repo, "--json",
                          "number,title,labels,createdAt,milestone,state"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return json.loads(out)


def has_open_pr(number: int | str, mentions: str) -> bool:
    """Whether an open pull request names issue ``number`` as ``#<number>``.

    Args:
        number: The issue's number.
        mentions: The open pull requests' text, from :func:`open_pr_mentions`.

    Returns:
        True when ``#<number>`` stands as a whole number in the text.

    Examples:
        >>> has_open_pr(72, "Closes #72."), has_open_pr(72, "Closes #721"), has_open_pr(72, "")
        (True, False, False)
    """
    return re.search(rf"#{number}(?!\d)", mentions) is not None


def own_stalled(
    states: dict,
    claimed: list[int],
    mentions: str,
    view: Callable[[int], dict],
    waits: Callable[[int], bool],
) -> list[dict]:
    """This agent's claimed issues that are open, In Progress and have no open PR.

    An issue that waits on a ``**Question**`` is left out, as ``next_issue.pick`` leaves it
    out. Each issue that is kept gets ``own``, so the pick prints no claim command for it.

    Args:
        states: Linear's state of each issue, by number.
        claimed: The issues that this agent's claims name (``claims.own_claims``).
        mentions: The open pull requests' text, from :func:`open_pr_mentions`.
        view: Reads an issue, with its ``state``.
        waits: Whether an issue waits on the maintainer's answer.

    Returns:
        The issues, oldest number first.

    Examples:
        >>> states = {"1": {"state": "In Progress"}, "2": {"state": "In Review"},
        ...           "3": {"state": "In Progress"}, "4": {"state": "In Progress"}}
        >>> found = own_stalled(states, [1, 2, 3, 4], "Part of #3", lambda n: {
        ...     "number": n, "state": "OPEN"}, lambda n: n == 4)
        >>> [i["number"] for i in found], found[0]["own"]
        ([1], True)
    """
    out = []
    for number in claimed:
        if states.get(str(number), {}).get("state") != "In Progress":
            continue
        if has_open_pr(number, mentions) or waits(number):
            continue
        issue = view(number)
        if issue["state"] == "OPEN":
            out.append(issue | {"own": True})
    return out
