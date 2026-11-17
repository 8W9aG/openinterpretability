"""The model page's concept map (#4422), drawn from ``data/concept-maps/<model>.json``.

``open-interpretability map`` writes the file (:mod:`openinterpretability.concept_map`). Here
each point gets its unit page and its current status, from the build's standings. So the
map shows the registry as it stands, and the layout stays the one that was committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    points = []
    for p in found["points"]:
        status = standings.get((p["layer"], p["neuron"]), {}).get(p["pole"])
        mark = "+" if p["pole"] == "positive" else "−"
        points.append(MapPoint(
            PAD + p["x"] * (WIDTH - 2 * PAD), PAD + (1 - p["y"]) * (HEIGHT - 2 * PAD),
            neuron_href(model.slug, p["layer"], p["neuron"]), f"L{p['layer']}/N{p['neuron']}{mark}",
            status.status if status is not None else "unstudied", p["cluster"],
        ))  # fmt: skip
    return {"points": points, "lines": _lines(points), "labels": _labels(points, found),
            "found": found, "size": (WIDTH, HEIGHT)}  # fmt: skip


def _lines(points: list[MapPoint]) -> list[tuple[MapPoint, MapPoint]]:
    """The pairs of points that are the two poles of one neuron."""
    by_page: dict[str, list[MapPoint]] = {}
    for p in points:
        by_page.setdefault(p.href, []).append(p)
    return [(a, b) for pair in by_page.values() if len(pair) == 2 for a, b in [pair]]


def _labels(points: list[MapPoint], found: dict) -> list[dict]:
    """Each cluster's name at the centre of its points, kept inside the frame and apart.

    A label is moved down, a line at a time, until it overlaps no label placed before it.
    """
    out: list[dict] = []
    for c in sorted(found["clusters"], key=lambda c: -c["size"]):
        inside = [p for p in points if p.cluster == c["id"]]
        if not inside:
            continue
        name = (c["name"] if c["verified"] else "no VERIFIED member") + f" ({c['size']})"
        half = len(name) * 3.1  # half the width of the text, at 11 px
        x = min(max(sum(p.x for p in inside) / len(inside), PAD + half), WIDTH - PAD - half)
        y = min(max(sum(p.y for p in inside) / len(inside), PAD + 12), HEIGHT - PAD)
        for _ in range(HEIGHT // 14):  # at most once down the whole frame
            if not any(abs(y - o["y"]) < 14 and abs(x - o["x"]) < half + o["half"] for o in out):
                break
            y = y + 14 if y + 14 <= HEIGHT - PAD else PAD + 12
        out.append({"x": x, "y": y, "name": name, "verified": c["verified"], "half": half})
    return out
