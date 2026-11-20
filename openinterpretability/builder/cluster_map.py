"""What the site's two cluster maps share: colours and numbered badges (issue #4577).

The literature map (:mod:`.literature_map_view`) and the concept maps
(:mod:`.concept_map_view`) draw clusters alike. Both follow Alexander Coulton's map of
cancer research. Each dot has the colour of its cluster, on a soft glow. A numbered badge
sits at each cluster's centre, and a numbered legend names the clusters. The colours are the
dataviz palette's, in a fixed order, validated for colour-blind readers. Clusters past
:data:`SLOTS` are grey. The badge's number and the legend tell every cluster apart, so no
cluster rests on colour alone. Templates draw them with ``_map_theme.html``.

A region also gets an outline (:func:`region_outline`, #4578). It is a faint shape with a
dashed edge around the region's points. So a region shows by its edge as well as its colour.
"""

from __future__ import annotations

from dataclasses import dataclass

SLOTS = 8  # coloured clusters, one per palette hue; the rest are grey, slot 0
BADGE_W, BADGE_H = 26, 20
DARK_INK = (3, 4, 5, 8)  # slots light enough to need dark text on their badge


@dataclass(frozen=True)
class Badge:
    """A cluster's numbered badge, centred on (x, y)."""

    number: int
    name: str
    slot: int
    size: int
    x: float
    y: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The badge's box (left, top, right, bottom)."""
        return (self.x - BADGE_W / 2, self.y - BADGE_H / 2,
                self.x + BADGE_W / 2, self.y + BADGE_H / 2)  # fmt: skip

    @property
    def ink(self) -> str:
        """``dark`` or ``light`` text, whichever reads on the badge's colour."""
        return "dark" if self.slot in DARK_INK else "light"


def colour_slots(numbers: list[int]) -> dict[int, int]:
    """Colour slots 1 to :data:`SLOTS` for the first clusters in ``numbers``, in order."""
    return {n: i for i, n in enumerate(numbers[:SLOTS], 1)}


def overlaps(a, b) -> bool:
    """Whether two boxes (left, top, right, bottom) overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


_RING = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1, -1))
_SPOTS = [(dx * r, dy * r) for r in (0, 22, 44, 66) for dx, dy in _RING]


def place_badges(centres: list[Badge], blocked: list, width: float, height: float) -> list[Badge]:
    """Move each badge from its cluster's centre until it overlaps no badge or ``blocked`` box.

    It tries the centre, then rings of spots further out. When none is free, the badge
    stays at the centre.
    """
    from dataclasses import replace

    taken = list(blocked)
    out = []
    for centre in centres:
        for dx, dy in [*_SPOTS, (0, 0)]:
            badge = replace(centre, x=round(centre.x + dx, 1), y=round(centre.y + dy, 1))
            box = badge.box
            inside = box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
            if inside and not any(overlaps(box, o) for o in taken):
                break
        taken.append(badge.box)
        out.append(badge)
    return out


INNER = 0.85  # the share of a region's points, nearest its centre, that its outline holds
PAD_OUT = 6.0  # pixels between the outline and the points it holds


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The convex hull of ``points``, counter-clockwise, by Andrew's monotone chain."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for p in pts:  # each half of the hull keeps only left turns
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def region_outline(points: list[tuple[float, float]], inner: float = INNER) -> str:
    """An SVG path around the ``inner`` share of ``points`` nearest their centre, or "".

    The hull is pushed out from the centre by :data:`PAD_OUT` pixels, so the edge clears the
    marks. A region of fewer than 3 points gets no outline.
    """
    import math

    if len(points) < 3:
        return ""
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    near = sorted(points, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
    shape = hull(near[: max(3, math.ceil(len(near) * inner))])
    if len(shape) < 3:
        return ""
    out = []
    for x, y in shape:
        d = math.hypot(x - cx, y - cy) or 1.0
        out.append((x + (x - cx) / d * PAD_OUT, y + (y - cy) / d * PAD_OUT))
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in out) + " Z"
