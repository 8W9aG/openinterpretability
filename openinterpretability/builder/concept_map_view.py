"""The model page's concept map (#4422), drawn from ``data/concept-maps/<model>.json``.

``open-interpretability map`` writes the file (:mod:`openinterpretability.concept_map`). Here
each point gets its unit page and its current status, from the build's standings. So the
map shows the registry as it stands, and the layout stays the one that was committed.

It is drawn like the literature map (:mod:`.cluster_map`, #4577). Each point has its
cluster's colour on a soft glow. Each cluster has a numbered badge and a legend line. Clusters are
numbered by size. A cluster with no VERIFIED member has no name, so it is grey. A pole's
status is its shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openinterpretability.builder.cluster_map import Badge, colour_slots, place_badges

WIDTH, HEIGHT, PAD = 720, 480, 24  # the SVG's viewBox and margin

# A status's mark, so that no status rests on colour alone. Others get a small dot.
SHAPES = {"VERIFIED": "circle", "NEEDS_REVISION": "triangle", "REJECTED": "square",
          "CONTESTED": "diamond"}  # fmt: skip


@dataclass(frozen=True)
class MapPoint:
    """One studied pole on the map."""

    x: float
    y: float
    href: str
    label: str  # L3/N7+
    status: str
    cluster: int
    slot: int = 0  # its cluster's colour slot (cluster_map.py), 0 for grey
    name: str = ""  # its cluster's name, for the hover card

    @property
    def shape(self) -> str:
        """The mark of its status (:data:`SHAPES`), or ``dot``."""
        return SHAPES.get(self.status, "dot")


def load_maps(folder: Path) -> dict:
    """Each model's map, keyed by model name, from the files in ``folder``."""
    import json

    out = {}
    for path in sorted(Path(folder).glob("*.json")) if Path(folder).is_dir() else []:
        found = json.loads(path.read_text(encoding="utf-8"))
        out[found["model"]] = found
    return out


def map_view(model, found: dict | None) -> dict | None:
    """The points, the lines between a neuron's poles and the cluster labels, or None."""
    from openinterpretability.builder.tree import neuron_href

    if not found or not found.get("points"):
        return None
    standings = {(u.layer, u.idx): u.standings for layer in model.layers for u in layer.neurons}
    order = sorted(found["clusters"], key=lambda c: (-c["size"], c["id"]))
    number = {c["id"]: n for n, c in enumerate(order, 1)}
    slot = colour_slots([number[c["id"]] for c in order if c["verified"]])
    named = {c["id"]: _name(c) for c in order}
    points = []
    for p in found["points"]:
        status = standings.get((p["layer"], p["neuron"]), {}).get(p["pole"])
        mark = "+" if p["pole"] == "positive" else "−"
        points.append(MapPoint(
            PAD + p["x"] * (WIDTH - 2 * PAD), PAD + (1 - p["y"]) * (HEIGHT - 2 * PAD),
            neuron_href(model.slug, p["layer"], p["neuron"]), f"L{p['layer']}/N{p['neuron']}{mark}",
            status.status if status is not None else "unstudied", p["cluster"],
            slot.get(number.get(p["cluster"], 0), 0), named.get(p["cluster"], ""),
        ))  # fmt: skip
    badges = _badges(points, order, number, slot)
    return {"points": points, "lines": _lines(points), "badges": badges,
            "found": found, "size": (WIDTH, HEIGHT)}  # fmt: skip


def _name(cluster: dict) -> str:
    """A cluster's name, or a note that it has none: no VERIFIED member names it."""
    return cluster["name"] if cluster["verified"] else "no VERIFIED member, so no name"


def _badges(points, order: list[dict], number: dict, slot: dict) -> list[Badge]:
    """Each cluster's numbered badge at the centre of its points, kept apart."""
    centres = []
    for c in order:
        inside = [p for p in points if p.cluster == c["id"]]
        if inside:
            x = sum(p.x for p in inside) / len(inside)
            y = sum(p.y for p in inside) / len(inside)
            n = number[c["id"]]
            centres.append(Badge(n, _name(c), slot.get(n, 0), c["size"], x, y))
    return place_badges(centres, [], WIDTH, HEIGHT)


def _lines(points: list[MapPoint]) -> list[tuple[MapPoint, MapPoint]]:
    """The pairs of points that are the two poles of one neuron."""
    by_page: dict[str, list[MapPoint]] = {}
    for p in points:
        by_page.setdefault(p.href, []).append(p)
    return [(a, b) for pair in by_page.values() if len(pair) == 2 for a, b in [pair]]
