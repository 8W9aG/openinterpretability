"""The geometry of the literature map on the review index (issues #3294 and #4577).

The map follows the design that issue #4577 cites, Alexander Coulton's map of cancer
research. Every paper is a dot in the colour of its region. A soft glow makes each region
a cloud. A numbered badge sits at each region's centre, and a numbered legend names
the regions. Callout cards in the margins name the papers that the repo uses, with leader
lines to their dots. Every other paper shows its card on hover or focus.

``scripts/literature_map.py`` lays the papers out offline and writes ``map.json``. This
module only places them. A review's status is its shape (circle, triangle, square), so
colour never carries it alone. Regions are told apart by their numbers and names too.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.builder.cluster_map import Badge, colour_slots, place_badges
from openinterpretability.builder.literature_pages import MAP, credits, map_text, review_rows

WIDTH, HEIGHT = 1000, 720  # the map's viewBox, in SVG units
LEFT, RIGHT, TOP, BOTTOM = 230, 770, 40, 680  # where the dots go; the cards take the margins
CARD_W, CARD_X = 196, 14  # a callout card's width, and its gap from the map's edge
CARD_LINE, CARD_CHARS, CARD_LINES = 14, 30, 3  # its title's line height, wrap width and lines


@dataclass(frozen=True)
class Point:
    """One review on the map."""

    name: str
    title: str
    status: str
    cx: float
    cy: float
    slot: int = 0  # the region's colour slot, 1–7, or 0 for grey
    why: str = ""  # the one-line reason from the index, for the hover card
    credit: str = ""  # "Authors, year", from the index
    neighbours: tuple[str, ...] = ()

    @property
    def called_out(self) -> bool:
        """Whether a callout card names the paper: only the reviews the repo uses."""
        return self.status == "used"


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
    """Each region number's colour slot: named regions take 1–7 in order, the rest 0."""
    named = [
        r["number"] for r in sorted(regions, key=lambda r: r["number"]) if r.get("named", True)
    ]
    return {r["number"]: colour_slots(named).get(r["number"], 0) for r in regions}


def callouts(points: list[Point]) -> list[Callout]:
    """A card for each called-out paper, in the margin on its side, stacked without overlap."""
    out: list[Callout] = []
    for left in (True, False):
        side = sorted((p for p in points if p.called_out and (p.cx < WIDTH / 2) == left),
                      key=lambda p: p.cy)  # fmt: skip
        x = CARD_X if left else WIDTH - CARD_X - CARD_W
        bottom = 8.0
        cards = []
        for p in side:
            lines = tuple(textwrap.wrap(p.title, CARD_CHARS)[:CARD_LINES])
            card = Callout(p, x, 0, lines)
            y = max(p.cy - card.height / 2, bottom)
            cards.append(Callout(p, x, round(y, 1), lines))
            bottom = y + card.height + 10
        overflow = bottom - 10 - (HEIGHT - 8)
        out += [Callout(c.point, c.x, round(c.y - max(overflow, 0), 1), c.lines) for c in cards]
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
                            neighbours=tuple(row.get("neighbours", ()))))  # fmt: skip
    return points, missing


def map_view(folder: Path, points: list[Point]) -> dict:
    """The queued papers, the callout cards and the region badges of the map."""
    data = _read(folder)
    colour = slots(data.get("regions", []))
    queued = [Queued(q["number"], q["title"], *_xy(q["x"], q["y"]), colour.get(q.get("region"), 0))
              for q in data.get("queued", [])]  # fmt: skip
    cards = callouts(points)
    return {"queued": queued, "callouts": cards, "badges": badges(data.get("regions", []), cards)}
