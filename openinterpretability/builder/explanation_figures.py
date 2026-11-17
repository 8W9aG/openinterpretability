"""The figures of an explanation page, drawn at build time from the record's data (#4479).

They use the paper figures' system (:mod:`openinterpretability.figures`). Each figure is
its own SVG file next to the page. An animated one also has a static copy, which shows its
last frame. The page's button swaps the two, and reduced motion stops the animation too.
No figure rests on colour alone. A status is a marker's shape, a kind of unit is a fill
pattern, and a sign is a line style. Each figure has a text alternative and a table.

1. **Graph:** the prompt's tokens, the top units by layer and the traced edges to the
   answer. It builds one layer at a time.
2. **Waterfall:** the answer's logit built up unit by unit, from the units' mean.
3. **Coverage:** the explained units' coverage against random units of the same count. With
   a ``--run``, the explained share measured before each draw, drawn point by point.
4. **Reading:** each top unit's strength at each token, with the value in each cell.
5. **Ablation:** the answer's top tokens before and after the explained units are ablated.
6. **Groups:** the superpositions, circuits and hubs that hold top units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openinterpretability.figures.motion import reveal
from openinterpretability.figures.svg import WIDTH, Figure, Panel, number

STATIC = "*{animation:none!important}"  # the static copy: every mark at its resting frame
CYCLE = "9s"
DEFS = (
    '<defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="5" height="5" '
    'patternTransform="rotate(45)"><path d="M0 0V5" class="hatchline"/></pattern></defs>'
)
EXTRA = (
    ".hatchline{stroke:var(--ink2);stroke-width:1.4}.fhatch{fill:url(#hatch)}"
    ".dashed{stroke-dasharray:4 3}.outline{fill:var(--surface);stroke:var(--ink2);stroke-width:1.4}"
    ".edge{fill:none;stroke:var(--ink2);stroke-opacity:.55}.cell{stroke:var(--surface)}"
    ".oncell{fill:#ffffff;font-size:11px}"
)
# A status's marker: its shape names it, so colour is never needed.
MARKERS = {"VERIFIED": "circle", "NEEDS_REVISION": "triangle", "REJECTED": "square"}


@dataclass
class Drawn:
    """One figure of a page: its file name, the SVG, its static copy and what it says."""

    name: str
    title: str
    svg: str
    alt: str
    table: list[list[str]] = field(default_factory=list)  # header row first
    static: str | None = None  # the last frame, when the figure moves


def _figure(title: str, subtitle: str, alt: str) -> Figure:
    fig = Figure([title], [subtitle], alt)
    fig.marks.append(DEFS)
    fig.motion = EXTRA
    return fig


def _finish(name: str, fig: Figure, table: list[list[str]], moves: bool) -> Drawn:
    title = fig.alt.split(".")[0]
    svg = fig.render()
    static = None
    if moves:
        fig.motion += STATIC
        static = fig.render()
    return Drawn(name, title, svg, fig.alt, table, static)


def _marker(fig: Figure, x: float, y: float, status: str, cls: str = "fc1") -> None:
    shape = MARKERS.get(status, "dot")
    if shape == "circle":
        fig.dot(x, y, 5, cls)
    elif shape == "triangle":
        fig.path([(x, y - 6), (x + 5.5, y + 4), (x - 5.5, y + 4)], cls, close=True)
    elif shape == "square":
        fig.marks.append(
            f'<rect class="outline" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8"/>'
        )
    else:
        fig.dot(x, y, 2.5, "fneutral")


def figures(data, answer: str = "answer") -> list[Drawn]:
    """Every figure that ``data`` (``ExplanationFigures``) can draw, in page order."""
    out = [
        graph(data, answer),
        waterfall(data),
        coverage(data, None),
        reading(data),
        ablation(data),
    ]
    if data.groups:
        out.append(groups(data))
    return [d for d in out if d is not None]


def with_run(data, run, answer: str = "answer") -> list[Drawn]:
    """:func:`figures`, with the coverage figure showing the run's shares when there was one."""
    out = figures(data, answer)
    if run is not None and run.shares:
        out[2] = coverage(data, run.shares)
    return out


def graph(data, answer: str = "answer") -> Drawn:
    """Figure 1: the tokens, the top units by layer and the edges, built layer by layer."""
    fig = _figure(
        "How the answer is built, layer by layer",
        "Top units by layer. Edge width is weight; a dashed edge pushes the answer down.",
        "Attribution graph. The prompt's tokens on the left, the top units in columns "
        "by layer, and the traced edges between them, to the answer on the right.",
    )
    layers, pos, top, height = _places(data, fig.body_top + 10)
    for i, tok in enumerate(data.tokens):
        fig.text(20, top + (i + 1) * height / (len(data.tokens) + 1), repr(tok)[1:-1], "small")
    fig.text(WIDTH - 20, top + height / 2 + 4, repr(answer)[1:-1], "head", "end")
    weights = [abs(e["weight"]) for e in data.edges] + [abs(u.attribution) for u in data.units]
    peak, layer_of = max(weights + [1e-9]), {u.label: u.layer for u in data.units}
    for k, la in enumerate(layers):
        cls = f"layer{k}"
        fig.motion += reveal(cls, 80 * k / max(1, len(layers)), CYCLE)
        for e in data.edges[:40]:
            a, b = pos.get(e["source"]), pos.get(e["target"])
            if a and b and layer_of[e["target"]] == la:
                _edge(fig, a, b, e["weight"] / peak, cls)
        for u in (u for u in data.units if u.layer == la):
            _edge(fig, pos[u.label], (WIDTH - 90, top + height / 2), u.attribution / peak, cls)
            _node(fig, pos[u.label], u, cls, above=k % 2 == 0)
        fig.text(
            pos[next(u.label for u in data.units if u.layer == la)][0],
            top + height + 14,
            f"L{la}",
            "small",
            "middle",
        )
    fig.bottom = top + height + 4
    _key(fig, fig.bottom + 26)
    fig.bottom += 32
    table = [["from", "to", "weight"]]
    table += [[e["source"], e["target"], number(e["weight"], 3)] for e in data.edges[:40]]
    return _finish("graph", fig, table, moves=True)


def _places(data, top: float):
    """The layers, each unit's (x, y) in its layer's column, and the body's top and height."""
    layers = sorted({u.layer for u in data.units})
    height = max(220, 26 * max(sum(u.layer == la for u in data.units) for la in layers))
    left, right = 120, WIDTH - 110
    pos = {}
    for i, la in enumerate(layers):
        column = [u for u in data.units if u.layer == la]
        x = left + (i + 1) * (right - left) / (len(layers) + 1)
        for j, u in enumerate(column):
            pos[u.label] = (x, top + (j + 1) * height / (len(column) + 1))
    return layers, pos, top, height


def _node(fig: Figure, at: tuple[float, float], unit, cls: str, above: bool) -> None:
    """A unit's marker and short label. Neighbouring columns alternate, so labels don't collide."""
    x, y = at
    fig.marks.append(f'<g class="{cls}">')
    _marker(fig, x, y, unit.status)
    fig.text(x, y - 9 if above else y + 17, unit.label.split("/")[1], "small", "middle")
    fig.marks.append("</g>")


def _key(fig: Figure, y: float) -> None:
    """The markers' key: each status's own shape, then its name."""
    x = 20.0
    for status, name in (
        ("VERIFIED", "VERIFIED"),
        ("NEEDS_REVISION", "NEEDS REVISION"),
        ("REJECTED", "REJECTED"),
        ("", "other"),
    ):
        _marker(fig, x + 5, y - 4, status)
        fig.text(x + 14, y, name, "small")
        x += 30 + 6.2 * len(name)


def _edge(fig: Figure, a, b, weight: float, cls: str) -> None:
    width = 0.6 + 4 * min(1.0, abs(weight))
    style = "edge dashed" if weight < 0 else "edge"
    fig.marks.append(
        f'<path class="{style} {cls}" stroke-width="{width:.1f}" '
        f'd="M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"/>'
    )


def waterfall(data) -> Drawn:
    """Figure 2: the logit built up from the units' mean, unit by unit in rank order."""
    rest = data.net - sum(u.attribution for u in data.units)
    start = data.logit - data.net
    rows = [(u.label, u.attribution, _kind(u)) for u in data.units] + [("the rest", rest, "rest")]
    fig = _figure(
        "The answer's logit, unit by unit",
        "Solid: explained · hatched: blocked · outline: unstudied. From the units' mean.",
        "Attribution waterfall. Each bar adds one unit's attribution, in rank order, "
        "from the logit with every unit at its mean to the answer's logit.",
    )
    level, values = start, [start]
    for _, a, _ in rows:
        level += a
        values.append(level)
    low, high = min(values), max(values)
    panel = Panel(fig, (150, fig.body_top + 10, WIDTH - 190, 16 * len(rows)), (low, high), (0, 1))
    level = start
    for i, (label, a, kind) in enumerate(rows):
        y = panel.top + 16 * i
        cls = {"explained": "fc1", "blocked": "fhatch", "unstudied": "outline", "rest": "fneutral"}[
            kind
        ]
        fig.bar(
            panel.x(min(level, level + a)), y + 2, abs(panel.x(level + a) - panel.x(level)), 11, cls
        )
        fig.text(panel.left - 8, y + 11, f"{label} · {kind}", "small", "end")
        level += a
    panel.grid_x([low, start, high], 1)
    panel.x_title("logit")
    table = [["unit", "attribution", "kind"]] + [[r[0], number(r[1], 3), r[2]] for r in rows]
    return _finish("waterfall", fig, table, moves=False)


def _kind(u) -> str:
    return "explained" if u.explained else "blocked" if u.blocked else "unstudied"


def coverage(data, shares) -> Drawn | None:
    """Figure 3: the explained units' coverage against random units; with a run, its shares."""
    if data.coverage is None:
        return None
    values = [data.coverage, *data.random_draws]
    low, high = min(values + [0.0]), max(values + [0.01])
    fig = _figure(
        "What the explained units bring back, against random ones",
        "Each hollow circle is a random set of the same count; the diamond is the explained units.",
        "Coverage against random units. The share of the answer's log-probability that the "
        "explained units bring back, beside random sets of units of the same count.",
    )
    panel = Panel(fig, (60, fig.body_top + 20, WIDTH - 100, 40), (low, high), (0, 1))
    for v in data.random_draws:
        fig.marks.append(
            f'<circle class="open sneutral" cx="{panel.x(v):.1f}" cy="{panel.y(0.5):.1f}" r="5"/>'
        )
    x = panel.x(data.coverage)
    fig.motion += reveal("found", 45, CYCLE)
    fig.path(
        [
            (x, panel.y(0.5) - 8),
            (x + 8, panel.y(0.5)),
            (x, panel.y(0.5) + 8),
            (x - 8, panel.y(0.5)),
        ],
        "fc1 found",
        close=True,
    )
    fig.text(x, panel.y(0.5) - 12, f"explained {data.coverage:.1%}", "small", "middle")
    panel.grid_x([low, 0.0, high], 2)
    panel.x_title("coverage of the answer's log-probability")
    table = [["set", "coverage"], ["explained units", f"{data.coverage:.2%}"]]
    table += [[f"random {i + 1}", f"{v:.2%}"] for i, v in enumerate(data.random_draws)]
    if shares:
        _shares(fig, shares, table)
    return _finish("coverage", fig, table, moves=True)


def _shares(fig: Figure, shares: list[float], table: list[list[str]]) -> None:
    top = fig.bottom + 60
    panel = Panel(
        fig, (60, top, WIDTH - 100, 120), (0, max(1, len(shares) - 1)), (0, max(shares + [0.01]))
    )
    panel.head("The explained share before each draw of the run")
    points = [(panel.x(i), panel.y(s)) for i, s in enumerate(shares)]
    for i, (x, y) in enumerate(points):
        cls = f"step{i}"
        fig.motion += reveal(cls, 90 * i / max(1, len(points) - 1), CYCLE)
        if i:
            fig.line(*points[i - 1], x, y, f"line sc1 {cls}")
        fig.dot(x, y, 3.5, f"fc1 {cls}")
    panel.grid_y([0, max(shares)], 3)
    panel.x_title("draw")
    table += [[f"draw {i + 1}", f"{s:.2%}"] for i, s in enumerate(shares)]


def reading(data) -> Drawn | None:
    """Figure 4: each top unit's strength at each token, with the value in each cell."""
    if not data.reading:
        return None
    labels, tokens = list(data.reading), data.tokens
    fig = _figure(
        "Where along the prompt each top unit is strong",
        "Darker is stronger, relative to the unit's own peak; the number is the strength.",
        "Reading along the tokens. For each top unit, its strength at each token of the "
        "prompt, as a shaded cell with the value written in it.",
    )
    left, top = 130, fig.body_top + 26
    cell = min(80.0, (WIDTH - left - 30) / len(tokens))
    for j, tok in enumerate(tokens):
        fig.text(left + cell * (j + 0.5), top - 8, repr(tok)[1:-1][:10], "small", "middle")
    for i, label in enumerate(labels):
        values = data.reading[label]
        peak = max(values + [1e-9])
        y = top + 22 * i
        fig.text(left - 8, y + 15, label, "small", "end")
        for j, v in enumerate(values):
            _cell(fig, left + cell * j, y, cell, v, peak)
    fig.bottom = top + 22 * len(labels)
    table = [["unit", *[repr(t)[1:-1] for t in tokens]]]
    table += [[lab, *[number(v, 3) for v in data.reading[lab]]] for lab in labels]
    return _finish("reading", fig, table, moves=False)


def _cell(fig: Figure, x: float, y: float, width: float, value: float, peak: float) -> None:
    """One shaded cell with its value in it; light text on a dark cell."""
    box = [(x, y), (x + width, y), (x + width, y + 20), (x, y + 20)]
    fig.shaded(box, "fc1 cell", 0.08 + 0.8 * value / peak)
    text = "oncell" if value / peak > 0.55 else "small"
    fig.text(x + width / 2, y + 14, number(value, 1), text, "middle")


def ablation(data) -> Drawn | None:
    """Figure 5: the answer's top tokens, before and after the explained units are ablated."""
    if not data.ablation:
        return None
    fig = _figure(
        "The answer's top tokens, before and after ablation",
        "Solid: as the model runs · hatched: with the explained units mean-ablated.",
        "Ablation, before and after. The probability of each of the answer's top tokens, "
        "first as the model runs, then with the explained units set to their means.",
    )
    peak = max(max(r["before"], r["after"]) for r in data.ablation) or 1.0
    panel = Panel(
        fig, (130, fig.body_top + 10, WIDTH - 200, 30 * len(data.ablation)), (0, peak), (0, 1)
    )
    for i, r in enumerate(data.ablation):
        y = panel.top + 30 * i
        fig.text(panel.left - 8, y + 16, repr(r["token"])[1:-1], "small", "end")
        for k, (key, cls) in enumerate((("before", "fc1"), ("after", "fhatch"))):
            fig.bar(panel.left, y + 2 + 12 * k, panel.x(r[key]) - panel.left, 11, cls)
            fig.text(panel.x(r[key]) + 6, y + 11 + 12 * k, f"{r[key]:.1%}", "small")
    panel.grid_x([0, peak], 2)
    panel.x_title("probability")
    table = [["token", "before", "after"]] + [
        [r["token"], f"{r['before']:.2%}", f"{r['after']:.2%}"] for r in data.ablation
    ]
    return _finish("ablation", fig, table, moves=False)


def groups(data) -> Drawn:
    """Figure 6: the superpositions, circuits and hubs, each with the top units it holds."""
    fig = _figure(
        "The groups that hold the top units",
        "A superposition, circuit or hub, and its members among the top units.",
        "Groups. Each superposition, circuit and hub that holds top units, with those "
        "units listed beside it.",
    )
    y = fig.body_top + 20
    for g in data.groups:
        fig.marks.append(
            f'<rect class="outline" x="20" y="{y - 14:.1f}" width="{WIDTH - 40}" '
            f'height="22" rx="6"/>'
        )
        fig.text(30, y + 1, f"{g['kind']} {g['name']} · {g['status']}", "head")
        fig.text(330, y + 1, ", ".join(g["members"]), "small")
        y += 30
    fig.bottom = y
    table = [["group", "kind", "status", "members"]]
    table += [[g["name"], g["kind"], g["status"], ", ".join(g["members"])] for g in data.groups]
    return _finish("groups", fig, table, moves=False)
