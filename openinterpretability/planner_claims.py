"""What open pull requests claim: the units they work on (:func:`open_pr_claims`).

The planner gives a claimed unit less weight, so that two PRs rarely study one unit.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Sequence

FORK_CLAIM = re.compile(r"^experiment/[A-Za-z0-9._-]+$")
"""The branch name by which a PR from a fork claims one unit."""


class ForkClaim(str):
    """The one unit key that a PR from a fork claims. It matches only an equal key."""

    __slots__ = ()


def claimed(key: str, claims: Sequence[str]) -> bool:
    """Return whether an open PR in ``claims`` claims the unit key or slug ``key``.

    A PR from this repository claims each key that its text contains. A
    :class:`ForkClaim` claims only the key that is equal to it.
    """
    return any(key == c if isinstance(c, ForkClaim) else key in c for c in claims)


def _fork_claim(branch: str) -> ForkClaim | None:
    """Return the unit key of the fork branch ``experiment/<key>[-<kind>[-<time>]]``, or ``None``.

    The key must name one unit, so it holds exactly one layer part (``-L3-``). A follow-up's
    branch ends in the draw's time (``draw_steps.claim_branch``, #6066), which goes first.
    """
    from openinterpretability.planner import SKILLS  # here, because the planner imports this module

    if not FORK_CLAIM.match(branch):
        return None
    key = re.sub(r"-\d{10}$", "", branch.removeprefix("experiment/"))  # the draw's time
    for kind in SKILLS:
        if key.endswith(f"-{kind}"):
            key = key.removesuffix(f"-{kind}")
            break
    return ForkClaim(key) if len(re.findall(r"-L\d+-", key)) == 1 else None


def open_pr_claims() -> list[str]:
    """What open PRs claim: text that names the units they work on.

    A PR from this repository claims by its title and its branch name. A PR from a fork
    claims only by a branch named ``experiment/<key>``, so it claims at most one unit.
    Its claim is a :class:`ForkClaim`, which :func:`claimed` compares by equality.
    Anyone can open a fork PR, and free text could name many units and so bias the draw.
    The list is empty if `gh` is unavailable.
    """
    if not shutil.which("gh"):
        return []
    try:
        out = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "title,headRefName,isCrossRepository",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    claims = []
    for pr in json.loads(out):
        if not pr.get("isCrossRepository", True):
            claims.append(f"{pr['title']} {pr['headRefName']}")
        elif fork_claim := _fork_claim(pr["headRefName"]):
            claims.append(fork_claim)
    return claims
