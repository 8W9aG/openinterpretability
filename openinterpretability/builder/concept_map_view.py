"""The shared concept map on each model page and on the atlas (#4422, #4586).

``open-interpretability map`` writes ``data/concept-maps/shared.json``
(:mod:`openinterpretability.concept_map`). All models share its one space, so a pole has
the same position on every page. Here each point gets its unit page and its current
status, from the build's standings. So the map shows the registry as it stands, and the
layout stays the one that was committed.

It is drawn like the literature map (:mod:`.cluster_map`, #4577). Each region has a
numbered badge and a legend line, and regions are numbered by size. A region with no
VERIFIED member has no name, so it is grey.

- **A model page** draws its own poles solid, in their region's colour, with a shape for
  each status. The other models' poles are small hollow rings, so "this model" against
  "the others" shows by fill and size as well as colour.
- **The atlas** draws every model's poles, with a shape for each model. The points fill in
  in the order they were first studied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openinterpretability.builder.cluster_map import Badge, colour_slots, place_badges

WIDTH, HEIGHT, PAD = 720, 480, 24  # the SVG's viewBox and margin

# A status's mark, so that no status rests on colour alone. Others get a small dot.
SHAPES = {"VERIFIED": "circle", "NEEDS_REVISION": "triangle", "REJECTED": "square",
          "CONTESTED": "diamond"}  # fmt: skip
MODEL_SHAPES = ("circle", "square", "triangle", "diamond")  # the atlas's mark of each model
SHARED_FILE = "shared.json"


@dataclass(frozen=True)
class MapPoint:
    """One studied pole on the map."""

    x: float
    y: float
    href: str
    label: str  # L3/N7+
    status: str
    cluster: int
    slot: int = 0  # its region's colour slot (cluster_map.py), 0 for grey
    name: str = ""  # its region's name, for the hover card
    model: str = ""
    own: bool = True  # on a model page: the page's model's own pole
    order: int = 0  # its rank by first study, for the atlas's fill-in

    @property
    def shape(self) -> str:
        """The mark of its status (:data:`SHAPES`), or ``dot``."""
        return SHAPES.get(self.status, "dot")


def load_maps(folder: Path) -> dict | None:
    """The shared map in ``folder``, or None when there is none."""
    import json

    path = Path(folder) / SHARED_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _regions(found: dict) -> tuple[list[dict], dict, dict, dict]:
    """The regions by size, each one's number, colour slot and name."""
    order = sorted(found["clusters"], key=lambda c: (-c["size"], c["id"]))
    number = {c["id"]: n for n, c in enumerate(order, 1)}
    slot = colour_slots([number[c["id"]] for c in order if c["verified"]])
    named = {c["id"]: _name(c) for c in order}
    return order, number, slot, named


def _xy(p: dict) -> tuple[float, float]:
    return PAD + p["x"] * (WIDTH - 2 * PAD), PAD + (1 - p["y"]) * (HEIGHT - 2 * PAD)


def _points(found: dict, standings, own_model: str | None) -> list[MapPoint]:
    """Every point of the map, with its page, status, region and rank by first study."""
    from openinterpretability.builder.tree import neuron_href
    from openinterpretability.core.identifiers import model_slug

    _, number, slot, named = _regions(found)
    ranked = sorted(range(len(found["points"])),
                    key=lambda i: (found["points"][i].get("first", ""), i))  # fmt: skip
    rank = {i: r for r, i in enumerate(ranked)}
    out = []
    for i, p in enumerate(found["points"]):
        status = standings.get((p["model"], p["layer"], p["neuron"]), {}).get(p["pole"])
        mark = "+" if p["pole"] == "positive" else "−"
        x, y = _xy(p)
        out.append(MapPoint(
            x, y, neuron_href(model_slug(p["model"]), p["layer"], p["neuron"]),
            f"L{p['layer']}/N{p['neuron']}{mark}",
            status.status if status is not None else "unstudied", p["cluster"],
            slot.get(number.get(p["cluster"], 0), 0), named.get(p["cluster"], ""),
            p["model"], own_model is None or p["model"] == own_model, rank[i],
        ))  # fmt: skip
    return out


def _standings(models) -> dict:
    """Each studied neuron's standings, keyed by (model name, layer, neuron)."""
    return {(m.name, u.layer, u.idx): u.standings
            for m in models for layer in m.layers for u in layer.neurons}  # fmt: skip


def map_view(model, found: dict | None) -> dict | None:
    """The shared map as ``model``'s page draws it, or None when there is none."""
    if not found or not found.get("points"):
        return None
    points = _points(found, _standings([model]), model.name)
    own = [p for p in points if p.own]
    if not own:
        return None
    order, number, slot, _ = _regions(found)
    counts: dict[int, int] = {}  # this model's poles in each region, by the region's number
    for p in own:
        counts[number[p.cluster]] = counts.get(number[p.cluster], 0) + 1
    return {"points": [p for p in points if not p.own] + own, "own": len(own),
            "lines": _lines(own), "badges": _badges(points, order, number, slot),
            "counts": counts, "found": found, "size": (WIDTH, HEIGHT)}  # fmt: skip


def atlas_view(models, found: dict | None) -> dict | None:
    """The shared map with every model's poles, for the atlas page, or None."""
    if not found or not found.get("points"):
        return None
    points = sorted(_points(found, _standings(models), None), key=lambda p: p.order)
    names = list(found["models"])
    shape = {m: MODEL_SHAPES[i % len(MODEL_SHAPES)] for i, m in enumerate(names)}
    order, number, slot, _ = _regions(found)
    reach = found.get("coverage", {})
    rows = []
    for m in names:
        row = reach.get("models", {}).get(m, {"poles": 0, "coverage": 0.0, "regions": {}})
        by_number = {number[int(r)]: c for r, c in row.get("regions", {}).items()}
        rows.append({"model": m, "shape": shape[m], "by_number": by_number, **row})
    return {"points": points, "shape": shape, "badges": _badges(points, order, number, slot),
            "rows": rows, "pairs": reach.get("pairs", []),
            "found": found, "size": (WIDTH, HEIGHT)}  # fmt: skip


def _name(cluster: dict) -> str:
    """A region's name, or a note that it has none: no VERIFIED member names it."""
    return cluster["name"] if cluster["verified"] else "no VERIFIED member, so no name"


def _badges(points, order: list[dict], number: dict, slot: dict) -> list[Badge]:
    """Each region's numbered badge at the centre of its points, kept apart."""
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
