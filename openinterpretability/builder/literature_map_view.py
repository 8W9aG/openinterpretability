"""The geometry of the literature map on the review index (issues #3294 and #4577).

The map follows the design that issue #4577 cites, Alexander Coulton's map of cancer
research. Every paper is a dot in the colour of its region. A soft glow makes each region
a cloud. A numbered badge sits at each region's centre, and a numbered legend names
the regions. Callout cards in the margins name the most-cited papers that the repo uses,
with leader lines to their dots (:func:`callouts`, #5798). Every other paper shows its card
on hover or focus.

``scripts/literature_map.py`` lays the papers out offline and writes ``map.json``. This
module only places them. A review's status is its shape (circle, triangle, square), so
colour never carries it alone. Regions are told apart by their numbers and names too.
"""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.builder.cluster_map import Badge, place_badges, touching_slots
from openinterpretability.builder.literature_pages import (
    MAP,
    credits,
    map_text,
    review_rows,
    sections,
)
from openinterpretability.builder.pii import redact_text

WIDTH, HEIGHT = 1000, 720  # the map's viewBox, in SVG units
LEFT, RIGHT, TOP, BOTTOM = 230, 770, 40, 680  # where the dots go; the cards take the margins
CARD_W, CARD_X = 196, 14  # a callout card's width, and its gap from the map's edge
CARD_LINE, CARD_CHARS, CARD_LINES = 14, 30, 3  # its title's line height, wrap width and lines
CARDS_PER_SIDE = 8  # the most callout cards on each side of the map (#5798)
# Index sections of background: architectures, activations, optimisers and datasets. Their
# reviews are cited often, as tools, but the cards are for the work the research builds on.
BACKGROUND = ("Model architecture", "Data")
EDGE, GAP = 8, 10  # a card's least distance from the map's top and bottom, and between cards


@dataclass(frozen=True)
class Point:
    """One review on the map."""

    name: str
    title: str
    status: str
    cx: float
    cy: float
    slot: int = 0  # the region's colour slot, 1 to cluster_map.SLOTS, or 0 for grey
    why: str = ""  # the one-line reason from the index, for the hover card
    credit: str = ""  # "Authors, year", from the index
    neighbours: tuple[str, ...] = ()
    cites: int = 0  # the pages in docs/ and papers/ that link to the review (:func:`citations`)
    section: str = ""  # its section of the literature index


@dataclass(frozen=True)
class Queued:
    """A paper in the review queue (an open literature issue)."""

    number: int
    title: str
    cx: float
    cy: float
    slot: int = 0


@dataclass(frozen=True)
class Callout:
    """A card in the margin that names one paper, with a leader line to its dot."""

    point: Point
    x: float
    y: float
    lines: tuple[str, ...]

    @property
    def credit(self) -> str:
        """The paper's authors and year, short enough for one card line."""
        return short_credit(self.point.credit)

    @property
    def height(self) -> float:
        """The card's height: its title lines, its credit line and padding."""
        return CARD_LINE * (len(self.lines) + 1) + 14

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The card's box (left, top, right, bottom)."""
        return self.x, self.y, self.x + CARD_W, self.y + self.height

    @property
    def anchor(self) -> tuple[float, float]:
        """Where the leader line meets the card: the middle of its side nearest the dot."""
        side = self.x + CARD_W if self.x < self.point.cx else self.x
        return side, self.y + self.height / 2


def short_credit(credit: str, width: int = CARD_CHARS) -> str:
    """``credit`` as "First author et al., year" when it is wider than a card line."""
    if len(credit) <= width or "," not in credit:
        return credit
    authors, year = credit.rsplit(",", 1)
    first = authors.replace(" and ", ", ").split(",")[0].strip()
    return f"{first} et al.,{year}"


def _xy(x: float, y: float) -> tuple[float, float]:
    """A layout coordinate in [0, 1] as SVG units; y grows upwards in the layout."""
    return round(LEFT + x * (RIGHT - LEFT), 1), round(TOP + (1 - y) * (BOTTOM - TOP), 1)


def slots(regions: list[dict]) -> dict[int, int]:
    """Each region's colour slot: its own, apart from its neighbours' (#5798).

    Args:
        regions: The regions of ``map.json``. Each region's ``touches`` gives the regions
            next to it, by number, with the count of neighbour links to each.

    Returns:
        Each region number's slot (:func:`.cluster_map.touching_slots`).

    Examples:
        >>> slots([{"number": 2, "touches": {"1": 4}}, {"number": 1, "touches": {"2": 4}}])
        {1: 1, 2: 2}
    """
    numbers = sorted(r["number"] for r in regions)
    touches = {r["number"]: {int(n): w for n, w in r.get("touches", {}).items()} for r in regions}
    return touching_slots(numbers, touches)


def citations(folder: Path, names: set[str]) -> Counter:
    """How many pages in ``docs/`` and ``papers/`` link to each review in ``names``.

    A page counts once per review. The literature index and the review's own page don't
    count. A page outside ``folder`` counts only a link through ``literature/``.

    Args:
        folder: The reviews' folder, ``docs/literature``.
        names: The reviews' page names.

    Returns:
        The number of citing pages, by review name.
    """
    root = folder.parent.parent
    inside = re.compile(r"\]\((?:\./)?([A-Za-z0-9_-]+)\.md[)#]")
    outside = re.compile(r"literature/([A-Za-z0-9_-]+)\.md")
    count: Counter = Counter()
    for page in sorted([*(root / "docs").rglob("*.md"), *(root / "papers").rglob("*.md")]):
        if page.parent == folder and page.stem == "README":
            continue
        pattern = inside if page.parent == folder else outside
        cited = set(pattern.findall(page.read_text(encoding="utf-8"))) & names
        count.update(cited - {page.stem if page.parent == folder else ""})
    return count


def crosses(a: Callout, b: Callout) -> bool:
    """Whether the leader lines of two cards cross.

    Args:
        a: One card.
        b: Another card.

    Returns:
        True when the two segments, dot to card, cross at a point inside both.
    """
    p, q = (a.point.cx, a.point.cy), a.anchor
    r, t = (b.point.cx, b.point.cy), b.anchor

    def side(o, u, v):
        """Which side of the line from ``o`` to ``u`` the point ``v`` is on, by sign.

        Args:
            o: The line's start.
            u: The line's end.
            v: The point.

        Returns:
            Above 0 for the left, below 0 for the right, 0 on the line.
        """
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    return side(p, q, r) * side(p, q, t) < 0 and side(r, t, p) * side(r, t, q) < 0


def _stack(order: list[Point], x: float) -> list[Callout]:
    """Cards for ``order``, top to bottom, each beside its dot or below the card above.

    Cards that would end below the map move up, from the bottom card, as far as they must.
    The top card can then end above the map, which :func:`_side` checks.

    Args:
        order: The papers, in the order of their cards from top to bottom.
        x: The cards' left edge.

    Returns:
        One card for each paper.
    """
    cards, bottom = [], float(EDGE)
    for p in order:
        lines = tuple(textwrap.wrap(p.title, CARD_CHARS)[:CARD_LINES])
        height = Callout(p, x, 0, lines).height
        y = max(p.cy - height / 2, bottom)
        cards.append(Callout(p, x, y, lines))
        bottom = y + height + GAP
    top = HEIGHT - EDGE + GAP
    for i in reversed(range(len(cards))):
        c = cards[i]
        top = min(c.y, top - GAP - c.height)
        cards[i] = Callout(c.point, x, round(top, 1), c.lines)
    return cards


def _side(chosen: list[Point], x: float) -> list[Callout]:
    """The cards of one side: inside the map, and with no two leader lines crossing.

    The cards follow their dots from top to bottom (:func:`_stack`). While they don't fit
    in the map's height, the least-cited card is dropped. Two cards whose lines cross swap
    places once, and after that the less-cited of them is dropped.

    Args:
        chosen: The side's candidates, most-cited first.
        x: The left edge of the side's cards.

    Returns:
        The side's cards, or none when no card fits.
    """
    rank = {p.name: i for i, p in enumerate(chosen)}  # most-cited first
    order, swapped = sorted(chosen, key=lambda p: p.cy), set()
    while order:
        cards = _stack(order, x)
        if cards[0].y < EDGE:
            order.remove(max(order, key=lambda p: rank[p.name]))
            continue
        pair = next(((a, b) for i, a in enumerate(cards) for b in cards[i + 1 :]
                     if crosses(a, b)), None)  # fmt: skip
        if pair is None:
            return cards
        key = frozenset((pair[0].point.name, pair[1].point.name))
        i, j = order.index(pair[0].point), order.index(pair[1].point)
        if key in swapped:
            order.remove(max((pair[0].point, pair[1].point), key=lambda p: rank[p.name]))
        else:
            swapped.add(key)
            order[i], order[j] = order[j], order[i]
    return []


def callouts(points: list[Point]) -> list[Callout]:
    """The callout cards: the most-cited reviews that the repo uses, at most 8 a side (#5798).

    A review is a candidate when its status is ``used`` and its index section is not in
    :data:`BACKGROUND`. Candidates rank by :attr:`Point.cites`, then by name. Each takes
    the side of the map nearer its dot. The first :data:`CARDS_PER_SIDE` of a side are placed
    by :func:`_side`, so no card is outside the map and no two leader lines cross.

    Args:
        points: The placed reviews.

    Returns:
        The cards of both sides.
    """
    ranked = sorted((p for p in points if p.status == "used" and p.section not in BACKGROUND),
                    key=lambda p: (-p.cites, p.name))  # fmt: skip
    out: list[Callout] = []
    for left in (True, False):
        chosen = [p for p in ranked if (p.cx < WIDTH / 2) == left][:CARDS_PER_SIDE]
        out += _side(chosen, CARD_X if left else WIDTH - CARD_X - CARD_W)
    return out


def badges(regions: list[dict], cards: list[Callout]) -> list[Badge]:
    """Each region's badge at its centre, moved until it overlaps no badge and no card."""
    colour = slots(regions)
    centres = [Badge(r["number"], r["name"], colour[r["number"]], r.get("size", 0),
                     *_xy(r["x"], r["y"]))
               for r in sorted(regions, key=lambda r: r["number"])]  # fmt: skip
    return place_badges(centres, [c.box for c in cards], WIDTH, HEIGHT)


def _read(folder: Path) -> dict:
    path = folder / MAP
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def literature_map(folder: Path, reviews: list) -> tuple[list[Point], list]:
    """The placed reviews, and the ones not placed yet (new, or changed since the map)."""
    data = _read(folder)
    known = {row["name"]: row for row in data.get("reviews", [])}
    colour = slots(data.get("regions", []))
    rows, credit = review_rows(folder), credits(folder)
    cites, section = citations(folder, {r.name for r in reviews}), sections(folder)
    points, missing = [], []
    for review in reviews:
        text = map_text(folder / f"{review.name}.md")
        if text is None:
            continue  # the index and the queue
        row = known.get(review.name)
        if row is None or row["sha256"] != hashlib.sha256(text.encode()).hexdigest():
            missing.append(review)
            continue
        status, why = rows.get(review.name, ("unknown", ""))
        points.append(Point(review.name, review.title, status, *_xy(row["x"], row["y"]),
                            slot=colour.get(row.get("region"), 0), why=why,
                            credit=credit.get(review.name, ""),
                            neighbours=tuple(row.get("neighbours", ())),
                            cites=cites[review.name],
                            section=section.get(review.name) or ""))  # fmt: skip
    return points, missing


def map_view(folder: Path, points: list[Point]) -> dict:
    """The queued papers, the callout cards and the region badges of the map."""
    data = _read(folder)
    colour = slots(data.get("regions", []))
    # Issue titles are text from outside, so they pass the same redaction as the pages.
    queued = [Queued(q["number"], redact_text(q["title"]), *_xy(q["x"], q["y"]),
                     colour.get(q.get("region"), 0))
              for q in data.get("queued", [])]  # fmt: skip
    cards = callouts(points)
    return {"queued": queued, "callouts": cards, "called": {c.point.name for c in cards},
            "badges": badges(data.get("regions", []), cards)}  # fmt: skip
