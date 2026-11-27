"""Pending experiments: the open experiment PRs, for the home and model pages (issue #3283).

An experiment PR's branch names its unit (``experiment/<model>-L<layer>-<unit>[-pole][-kind]``,
``choose-next-experiment``). :func:`open_experiments` lists the open ones with ``gh``.
Only PRs from this repository are listed, as for the planner's claims
(:mod:`openinterpretability.planner_claims`). Anyone can open a fork PR, and its title is free text.
The build lists them only when asked (``open-interpretability build``), so tests and
offline builds never call GitHub.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from labinacan.site.github import open_prs

BRANCH = re.compile(
    r"^experiment/(?P<model>.+?)-L(?P<layer>\d+)-(?P<unit>[NHFS][0-9a-z]+)"
    r"(?:-(?P<pole>positive|negative))?(?:-(?P<kind>[a-z][a-z_-]*))?$"
)


@dataclass(frozen=True)
class Pending:
    """One open experiment PR."""

    number: int
    url: str
    model_slug: str
    layer: int
    unit: str  # N12, H3, F40 or S…
    pole: str | None
    kind: str  # explore, reevaluate, revise, …
    draft: bool
    opened: str  # ISO date

    @property
    def unit_href(self) -> str:
        """The unit's page under ``models/``, as the site names it."""
        return f"models/{self.model_slug}/L{self.layer}/{self.unit}.html"


def parse(pr: dict) -> Pending | None:
    """A PR from ``gh pr list --json``, or None if its branch names no unit."""
    found = BRANCH.match(pr.get("headRefName", ""))
    if found is None or pr.get("isCrossRepository", True):
        return None
    return Pending(
        number=pr["number"], url=pr["url"], model_slug=found["model"],
        layer=int(found["layer"]), unit=found["unit"].upper()[:1] + found["unit"][1:],
        pole=found["pole"], kind=found["kind"] or "explore", draft=bool(pr.get("isDraft")),
        opened=pr.get("createdAt", "")[:10],
    )  # fmt: skip


def open_experiments() -> list[Pending]:
    """The open experiment PRs of this repository, newest first; empty without ``gh``."""
    fields = "number,url,headRefName,isCrossRepository,isDraft,createdAt"
    prs = open_prs(fields)  # labinacan.site.github: none without gh (#6628)
    found = [p for p in map(parse, prs) if p is not None]
    return sorted(found, key=lambda p: (p.opened, p.number), reverse=True)
