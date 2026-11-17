"""Explanation pages: each prompt's newest explanation, and whether it is stale (issue #3744).

``open-interpretability build`` loads the newest explanation record of each prompt
(:mod:`openinterpretability.core.explanation_records`) and the registry's current statuses.
An explanation is stale when a unit it cites has a different status now. For example, a
unit that was unstudied when it was written can have a VERIFIED record now. Its page says
which units changed, so a reader knows that a newer explanation would say more.

Each page shows the paragraph first, then the long form, then the mechanism with a link
to each unit's page. ``explanations/index.html`` lists them, newest first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.core.explanation_records import EXPLANATIONS_WIKI, load_latest

# A pole or head slug, matched in full (the same grammar as explanation_records.UNIT_SLUG).
UNIT_PAGE = re.compile(
    r"^(?P<model>[A-Za-z0-9][A-Za-z0-9._-]*)/L(?P<layer>\d+)/"
    r"(?P<unit>[NH]\d+)(?:-(?P<pole>positive|negative))?$"
)


@dataclass
class ExplanationView:
    """One explanation, with its links and what has changed since it was written."""

    record: object  # ExplanationRecord
    source: str  # its path in the repository, for the source link
    changed: list[dict] = field(default_factory=list)  # {unit, then, now}

    @property
    def href(self) -> str:
        """``explanations/<model slug>/<prompt id>.html``."""
        slug = self.source.split("/")[-3]
        return f"{EXPLANATIONS_WIKI}/{slug}/{self.record.prompt_id}.html"

    @property
    def stale(self) -> bool:
        """True when a cited unit's status has changed since it was written."""
        return bool(self.changed)


def unit_href(slug: str | None) -> str | None:
    """The unit page of a pole or head slug: ``models/<model>/L3/N7.html``."""
    found = UNIT_PAGE.match(slug or "")
    return f"models/{found['model']}/L{found['layer']}/{found['unit']}.html" if found else None


def load_views(wiki_dir: Path, states: dict | None = None) -> list[ExplanationView]:
    """Each prompt's newest explanation, with the units whose status has changed, newest first."""
    from openinterpretability.planner import load_states

    found = load_latest(wiki_dir)
    if not found:
        return []
    states = load_states(wiki_dir) if states is None else states
    views = []
    for path, record in found:
        changed = []
        for u in record.units:
            if u.slug is None:
                continue  # a group: its status isn't in the unit states
            state = states.get(u.slug)
            now = state.record.verdict.status if state is not None else "unstudied"
            if now != u.status and u.status != "stopped":
                changed.append({"unit": u.unit, "then": u.status, "now": now})
        source = path.relative_to(wiki_dir.parent).as_posix() if wiki_dir.parent else str(path)
        views.append(ExplanationView(record, source, changed))
    return sorted(views, key=lambda v: v.record.written, reverse=True)


def render_explanations(render, views: list[ExplanationView]) -> None:
    """Write the index and one page per explanation (none when there is none)."""
    if not views:
        return
    render(f"{EXPLANATIONS_WIKI}/index.html", "explanations.html", views=views)
    for view in views:
        render(view.href, "explanation.html", view=view, unit_href=unit_href)
