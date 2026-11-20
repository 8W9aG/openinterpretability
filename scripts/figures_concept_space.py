"""The figures of papers/shared-concept-space, drawn from its results (issue #4586).

Each model has its own shape as well as its own colour, so no model rests on colour alone.
"""

from figure_motion import counter, reveal
from figure_svg import Figure, Panel

CLASSES = ("fc1", "fc2", "fc3", "fneutral")  # each model's fill, in the results' model order
SHAPES = ("circle", "square", "triangle", "diamond")
CYCLE = "9s"
STEPS = 12  # frames of the fill-in
REVEAL_UNTIL = 80.0  # percent of the cycle by which every pole shows; the rest is a hold


def _models(rows: list[dict]) -> list[str]:
    return sorted({r["model"] for r in rows if r["kind"] == "model"})


def _short(model: str) -> str:
    return model.split("/")[-1]


def _mark(fig: Figure, shape: str, x: float, y: float, r: float, cls: str) -> None:
    if shape == "circle":
        fig.dot(x, y, r, cls)
    elif shape == "square":
        fig.bar(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7, cls)
    elif shape == "triangle":
        fig.path([(x, y - r * 1.15), (x + r * 1.1, y + r * 0.85), (x - r * 1.1, y + r * 0.85)],
                 cls, close=True)  # fmt: skip
    else:
        fig.path([(x, y - r * 1.15), (x + r * 1.15, y), (x, y + r * 1.15), (x - r * 1.15, y)],
                 cls, close=True)  # fmt: skip


def _key(fig: Figure, models: list[str], y: float, x: float = 20) -> None:
    """A row of each model's shape and name."""
    for i, model in enumerate(models):
        _mark(fig, SHAPES[i], x + 5, y - 4, 4.5, CLASSES[i])
        fig.text(x + 14, y, _short(model))
        x += 30 + 6.6 * len(_short(model))


def fill(rows: list[dict]) -> Figure:
    """The shared map, filling in as the poles were first studied."""
    models = _models(rows)
    points = sorted((r for r in rows if r["kind"] == "point"), key=lambda r: (r["first"], r["x"]))
    fig = Figure(
        ["Every model's poles fill one shared space, and none keeps a region to itself"],
        [f"The {len(points):,} studied poles of {len(models)} models on the shared concept map,",
         "shown in the order they were first studied. Shape and colour give the model."],
        "A scatter of about two thousand points in one connected cloud. Points of all four "
        "models appear across the whole cloud as the dates advance.",
    )  # fmt: skip
    panel = Panel(fig, (60, fig.body_top + 40, 640, 420), (0, 1), (0, 1))
    _key(fig, models, fig.body_top + 18)
    group = {}
    for i, p in enumerate(points):
        group[id(p)] = min(STEPS - 1, i * STEPS // len(points))
    stamps = []
    for g in range(STEPS):
        at = REVEAL_UNTIL * g / (STEPS - 1)
        fig.motion += reveal(f"g{g}", at, CYCLE)
        members = [p for p in points if group[id(p)] == g]
        stamps.append(members[-1]["first"] if members else "")
    for p in points:
        i = models.index(p["model"])
        _mark(fig, SHAPES[i], panel.x(p["x"]), panel.y(p["y"]), 2.2,
              f"{CLASSES[i]} g{group[id(p)]}")  # fmt: skip
    shown = [REVEAL_UNTIL * g / (STEPS - 1) for g in range(STEPS)]
    for (cls, css), stamp in zip(counter("d", shown, CYCLE), stamps, strict=True):
        fig.motion += css
        fig.text(
            panel.left + panel.width, panel.top - 8, f"studied by {stamp}", f"small {cls}", "end"
        )
    fig.bottom = panel.top + panel.height
    return fig


def _partitions(rows: list[dict]) -> list[tuple[str, int]]:
    found = [(r["partition"], r["regions"]) for r in rows if r["kind"] == "partition"]
    return sorted(found, key=lambda p: p[1])


def coverage(rows: list[dict]) -> Figure:
    """Each model's coverage at each partition of the map."""
    models, parts = _models(rows), _partitions(rows)
    fig = Figure(
        ["No model leaves a region empty until the map is cut into 60"],
        ["Each model's coverage: the share of regions that hold at least one of its studied",
         "poles, at the map's own regions and at finer Leiden partitions of the same graph.",
         "The axis starts at 90%. The labels give each model's coverage at 60 regions."],
        "A line per model over four partitions, on an axis from 90% to 100%. Every model "
        "reaches every region up to 29 regions. At 60, coverage is 95% to 100%.",
    )  # fmt: skip
    top = fig.body_top + 30
    _key(fig, models, top - 10)
    panel = Panel(fig, (80, top + 14, 520, 260), (0, len(parts) - 1), (0.9, 1.0))
    panel.grid_y([0.9, 0.925, 0.95, 0.975, 1.0], 3)
    for j, (_, count) in enumerate(parts):
        fig.text(panel.x(j), panel.top + panel.height + 18, f"{count} regions", "small", "middle")
    for i, model in enumerate(models):
        by = {r["partition"]: r["coverage"] for r in rows
              if r["kind"] == "model" and r["model"] == model}  # fmt: skip
        xy = [(panel.x(j), panel.y(by[name])) for j, (name, _) in enumerate(parts)]
        fig.path(xy, f"line s{CLASSES[i][1:]}")
        for x, y in xy:
            _mark(fig, SHAPES[i], x, y, 4.5, CLASSES[i])
        last = by[parts[-1][0]]
        fig.text(xy[-1][0] + 12, panel.top + 4 + 16 * i, f"{_short(model)}: {last:.0%}", "small")
    fig.bottom = panel.top + panel.height + 24
    return fig


def overlap(rows: list[dict]) -> Figure:
    """Each pair's overlap against its shuffled null, at each partition."""
    parts = _partitions(rows)
    pairs = [(r["a"], r["b"]) for r in rows if r["kind"] == "pair" and r["partition"] == "map"]
    fig = Figure(
        ["Pooled over scan corpora, most pairs of models spread less alike than chance"],
        ["Each pair's overlap (solid dot) against the overlap of shuffled model labels (ring),",
         "at each partition. A line joins the two. A gap means the two models differ by region."],
        "Four panels, one per partition, each with six pairs. In most rows the solid dot "
        "sits left of the ring.",
    )  # fmt: skip
    for k, (name, count) in enumerate(parts):
        col, line = k % 2, k // 2
        panel = Panel(fig, (200 + col * 290, fig.body_top + 34 + line * 190, 250, 140),
                      (0.4, 1.0), (0, len(pairs)))  # fmt: skip
        panel.head(f"{count} regions")
        panel.grid_x([0.4, 0.6, 0.8, 1.0], 1)
        for i, (a, b) in enumerate(pairs):
            row = next(r for r in rows if r["kind"] == "pair" and r["partition"] == name
                       and (r["a"], r["b"]) == (a, b))  # fmt: skip
            y = panel.top + (i + 0.5) * panel.height / len(pairs)
            fig.line(panel.x(row["overlap"]), y, panel.x(row["null"]), y, "axis")
            fig.dot(panel.x(row["null"]), y, 4, "open sneutral")
            fig.dot(panel.x(row["overlap"]), y, 4, "fc1")
            if col == 0:
                fig.text(panel.left - 8, y + 4, f"{_short(a)} · {_short(b)}", "small", "end")
        fig.bottom = panel.top + panel.height + 24
    return fig


def _row(fig: Figure, panel: Panel, y: float, row: dict, label: str) -> None:
    fig.line(panel.x(row["overlap"]), y, panel.x(row["null"]), y, "axis")
    fig.dot(panel.x(row["null"]), y, 4, "open sneutral")
    fig.dot(panel.x(row["overlap"]), y, 4, "fc1")
    fig.text(panel.left - 8, y + 4, label, "small", "end")
    fig.text(panel.left + panel.width + 8, y + 4, f"p = {row['p']:.3f}", "small")


def control(rows: list[dict]) -> Figure:
    """At the map's regions: pairs on all corpora, pairs on one corpus, one model across two."""
    on_map = [r for r in rows if r.get("partition") == "map"]
    groups = [
        ("Two models, every scan corpus", [r for r in on_map if r["kind"] == "pair"]),
        ("Two models, both scanned on pile-10k",
         [r for r in on_map if r["kind"] == "within" and r["corpus"] == "NeelNanda/pile-10k"]),
        ("One model, pile-10k against the mixed corpus",
         [r for r in on_map if r["kind"] == "across"]),
    ]  # fmt: skip
    fig = Figure(
        ["The scan corpus, not the model, sets most of where the poles fall"],
        ["Overlap (solid dot) against the shuffled null (ring), at the map's 7 regions. One",
         "model's poles from two corpora differ more than two models' poles from one corpus."],
        "Three groups of rows. Pairs pooled over corpora sit left of their nulls. Pairs on "
        "pile-10k alone sit on or right of their nulls. Each model against itself across two "
        "corpora sits far left of its null.",
    )  # fmt: skip
    panel = Panel(fig, (250, fig.body_top + 20, 400, 0), (0.5, 1.0), (0, 1))
    y = panel.top
    for head, members in groups:
        fig.text(20, y + 14, head, "head")
        y += 30
        for r in members:
            label = (
                _short(r["model"])
                if r["kind"] == "across"
                else f"{_short(r['a'])} · {_short(r['b'])}"
            )
            _row(fig, panel, y, r, label)
            y += 20
        y += 8
    panel.height = y - panel.top
    panel.grid_x([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 1)
    fig.bottom = y + 20
    return fig


def build(rows: list[dict]) -> dict:
    """Each figure of the paper, by its file name."""
    return {"fill.svg": fill(rows), "coverage.svg": coverage(rows), "overlap.svg": overlap(rows),
            "control.svg": control(rows)}  # fmt: skip
