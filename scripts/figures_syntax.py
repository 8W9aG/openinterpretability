"""The figures of papers/syntax-positive-controls/README.md, from its results.jsonl."""

from figure_svg import Figure, Panel, columns

DEPTHS = [1, 2, 3, 4, 5]
LAYERS = list(range(12))
TARGET = (2, 676, "positive")
CYCLE = "12s"
PROBE_PANELS = (
    ("Open, closed, partly closed", (("open", "c1"), ("partly_closed", "c3"), ("closed", "c2"))),
    ("Text the parser ignores", (("open", "c1"), ("string", "c3"), ("comment", "c2"))),
    ("Other brackets", (("open", "c1"), ("square", "c3"), ("curly", "c2"))),
)
NAMES = {
    "open": "open",
    "closed": "closed",
    "partly_closed": "one closed",
    "string": "in a string",
    "comment": "in a comment",
    "square": "square",
    "curly": "curly",
}


def _probe(rows: list[dict], condition: str, depth: int) -> float:
    """Return the target pole's mean strength in one probe condition."""
    for row in rows:
        same = (row.get("layer"), row.get("neuron"), row.get("pole")) == TARGET
        at = (row.get("condition"), row.get("depth")) == (condition, depth)
        if row["kind"] == "probe" and same and at:
            return row["mean"]
    raise KeyError((condition, depth))


def _best(rows: list[dict], kind: str, name: str) -> list[float]:
    """Return each layer's best held-out R² for one label set."""
    found = {
        r["layer"]: r["top"][0]["r2"] for r in rows if r["kind"] == kind and r["label_set"] == name
    }
    return [found[layer] for layer in LAYERS]


def _moving_dot(fig: Figure, panel: Panel, values: list[float]) -> None:
    """Add a dot that follows ``values``, token by token, and rests on the last."""
    count = len(values)
    frames = "".join(
        f"{85 * i / (count - 1):.1f}%{{transform:translate({panel.x(i):.1f}px,{panel.y(v):.1f}px)}}"
        for i, v in enumerate(values)
    )
    last = f"transform:translate({panel.x(count - 1):.1f}px,{panel.y(values[-1]):.1f}px)"
    fig.motion += (
        f"@keyframes reader{{{frames}100%{{{last}}}}}"
        f".reader{{animation:reader {CYCLE} linear infinite;{last}}}"
    )
    fig.dot(0, 0, 5, "fc1 ring reader")


def _steps(panel: Panel, values: list[float]) -> list[tuple[float, float]]:
    """Return the outline of a step area over ``values``, one step per token."""
    points = [(panel.x(-0.5), panel.y(0))]
    for i, value in enumerate(values):
        points += [(panel.x(i - 0.5), panel.y(value)), (panel.x(i + 0.5), panel.y(value))]
    return [*points, (panel.x(len(values) - 0.5), panel.y(0))]


def _depth(token: dict) -> float:
    return float(token["bracket_depth"].rstrip("+"))


def _mark(fig: Figure, panel: Panel, tokens: list[dict], word: str, node: str, note: str) -> None:
    """Label the first token ``word`` of syntax node ``node`` on the strength panel."""
    i = next(i for i, t in enumerate(tokens) if t["token"] == word and t["node"] == node)
    value = tokens[i]["strength"][0]
    fig.line(panel.x(i), panel.y(value) - 4, panel.x(i), panel.y(value) - 16, "axis")
    fig.text(panel.x(i), panel.y(value) - 20, note, "small", "middle")


def trace(rows: list[dict]) -> Figure:
    """Draw one written snippet's bracket depth, with the pole's strength above it."""
    tokens = next(r for r in rows if r["kind"] == "trace")["tokens"]
    fig = Figure(
        ["L2 N676+ climbs as brackets open, and drops as soon as they close"],
        [
            "One Python function, written for this paper, read token by token. The dot follows "
            "the pole's strength.",
            "Below is the parser's bracket depth. The pole also fires on brackets in a comment "
            "and in a string.",
        ],
        "Two aligned plots over the tokens of a short function. The pole's strength is zero "
        "outside brackets and rises to about 9 at depth 4. It is also about 3 to 4 inside "
        "brackets in a comment and in a string, where the parser's depth stays at 0.",
    )
    last = len(tokens) - 0.5
    top = Panel(fig, (44, fig.body_top + 28, 692, 150), (-0.5, last), (0, 10))
    top.head("Strength of L2 N676+")
    top.grid_y([0, 5, 10], 0)
    strengths = [t["strength"][0] for t in tokens]
    top.curve(range(len(tokens)), strengths, "line sc1")
    _moving_dot(fig, top, strengths)
    _mark(fig, top, tokens, "per", "comment", "in a comment")
    _mark(fig, top, tokens, "all", "string", "in a string")
    _mark(fig, top, tokens, "0", "number", "depth 4")
    low = Panel(fig, (44, top.top + top.height + 44, 692, 70), (-0.5, last), (0, 4))
    low.head("Bracket depth, from the parser")
    low.grid_y([0, 2, 4], 0)
    fig.path(_steps(low, [_depth(t) for t in tokens]), "fneutral", close=True)
    fig.bottom = low.top + low.height
    return fig


def probes(rows: list[dict]) -> Figure:
    """Draw the pole's strength on written text, at each depth, per condition."""
    fig = Figure(
        ["The same token reads high under open brackets, and near zero once they close"],
        [
            "Mean strength of L2 N676+ at one token after one context token, over 8 items at "
            "each depth.",
            "All the text is written from templates. The open condition is the same in each panel.",
        ],
        "Three plots of strength against depth. Open brackets rise from 4 to 11. Closed "
        "brackets stay near 0. Strings and comments rise as open brackets do. Curly brackets "
        "rise less, to 5.",
    )
    panels = columns(fig, 3, 44, fig.body_top + 22, 210, (0.7, 6.9), (0, 12))
    for panel, (head, lines) in zip(panels, PROBE_PANELS, strict=True):
        panel.head(head)
        panel.grid_x(DEPTHS, 0)
        panel.grid_y([0, 4, 8, 12], 0, labels=panel is panels[0])
        panel.x_title("depth")
        for condition, colour in lines:
            values = [_probe(rows, condition, depth) for depth in DEPTHS]
            panel.curve(DEPTHS, values, f"line s{colour}")
            for depth, value in zip(DEPTHS, values, strict=True):
                fig.dot(panel.x(depth), panel.y(value), 3, f"f{colour}")
    _labels(fig, panels, rows)
    fig.bottom = panels[0].top + panels[0].height + 24
    return fig


def _labels(fig: Figure, panels: list[Panel], rows: list[dict]) -> None:
    """Label each probe line at its right end, and move apart labels that would overlap."""
    for panel, (_, lines) in zip(panels, PROBE_PANELS, strict=True):
        placed: list[float] = []
        ends = sorted(lines, key=lambda line: -_probe(rows, line[0], 5))
        for condition, _ in ends:
            y = panel.y(_probe(rows, condition, 5)) + 4
            while any(abs(y - other) < 13 for other in placed):
                y += 13
            placed.append(y)
            fig.text(panel.x(5) + 7, y, NAMES[condition], "small")


def labels(rows: list[dict]) -> Figure:
    """Draw the best poles' R² per layer, for round and all brackets, with the pole's own."""
    winner = next(r for r in rows if r["kind"] == "winner" and r["neuron"] == TARGET[1])
    round_row = _kind_row(rows, "round depth")
    literal_row = _kind_row(rows, "literal round depth")
    fig = Figure(
        ["Counting round brackets alone puts L2 N676+ first; counting all brackets hides it"],
        [
            "Left: each layer's best pole, in R² on held-out documents, and the highest "
            "R² of a null in that layer.",
            "Right: L2 N676+ alone. Its all-brackets bar is in-sample, so it is an upper "
            "bound for held-out R².",
        ],
        "Left, a plot by layer: round-bracket depth has a peak of 0.28 at layer 2, and the "
        "null stays below 0.01. Right, bars for N676: 0.09 for all brackets, 0.28 for round "
        "brackets and 0.31 for round bracket characters.",
    )
    y0 = fig.body_top + 22
    _best_per_layer(fig, rows, y0, round_row["n676_positive"])
    right = Panel(fig, (530, y0, 200, 210), (0, 3), (0, 0.32))
    right.head("L2 N676+")
    right.grid_y([0, 0.1, 0.2, 0.3], 1, labels=False)
    bars = (
        ("all", winner["labels"]["bracket depth"], "c2"),
        ("round", round_row["n676_positive"], "c1"),
        ("literal", literal_row["n676_positive"], "c3"),
    )
    for i, (name, value, colour) in enumerate(bars):
        x = right.x(i + 0.2)
        fig.bar(x, right.y(0), right.x(0.6) - right.x(0), right.y(value) - right.y(0), f"f{colour}")
        fig.text(x + 18, right.y(value) - 5, f"{value:.2f}", "small", "middle")
        fig.text(x + 18, right.y(0) + 15, name, "small", "middle")
    fig.bottom = y0 + 268
    return fig


def _best_per_layer(fig: Figure, rows: list[dict], y0: float, target_r2: float) -> None:
    """Draw each layer's best pole for round and all brackets, with the round null."""
    left = Panel(fig, (44, y0, 430, 210), (-0.5, 11.5), (0, 0.32))
    left.head("Best pole per layer")
    left.grid_x([0, 2, 4, 6, 8, 10], 0)
    left.grid_y([0, 0.1, 0.2, 0.3], 1)
    left.x_title("layer")
    for name, kind, colour in (("round depth", "kinds", "c1"), ("bracket depth", "layer", "c2")):
        values = _best(rows, kind, name)
        left.curve(LAYERS, values, f"line s{colour}")
        for layer, value in zip(LAYERS, values, strict=True):
            fig.dot(left.x(layer), left.y(value), 3, f"f{colour}")
    nulls = [
        r["null_max_r2"] for r in rows if r["kind"] == "kinds" and r["label_set"] == "round depth"
    ]
    left.curve(LAYERS, nulls, "line sneutral")
    fig.text(left.x(2) + 8, left.y(target_r2) + 2, "L2 N676+", "small")
    fig.legend(
        y0 + 268, (("fc1", "round brackets"), ("fc2", "all brackets"), ("fneutral", "null, round"))
    )


def _kind_row(rows: list[dict], name: str) -> dict:
    """Return the row of layer 2 for one label set of the sweep by bracket kind."""
    return next(
        r for r in rows if r["kind"] == "kinds" and r["layer"] == 2 and r["label_set"] == name
    )


def ablation(rows: list[dict]) -> Figure:
    """Draw how ablating the pole, and control poles, changes P(closing bracket)."""
    found = {r["depth"]: r for r in rows if r["kind"] == "ablation"}
    fig = Figure(
        ["Ablating L2 N676+ cuts the predicted closing bracket; random poles barely move it"],
        [
            "Mean probability that the next token starts with a closing round bracket, over 8 "
            "written items at each depth.",
            "The effect shrinks with depth, and at depth 5 it has a small opposite sign.",
        ],
        "A plot of closing-bracket probability against depth. The model and twenty random "
        "ablated poles overlap, from 0.46 to 0.31. With N676 ablated it is 0.20 at depth 1, "
        "and it rises to 0.32 at depth 5.",
    )
    panel = Panel(fig, (64, fig.body_top + 22, 640, 220), (0.7, 5.3), (0, 0.5))
    panel.grid_x(DEPTHS, 0)
    panel.grid_y([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1)
    panel.x_title("open brackets")
    series = (("controls_mean", "wash", 8), ("clean", "c1", 3), ("ablated", "c2", 3))
    for key, colour, radius in series:
        values = [found[d][key] for d in DEPTHS]
        if radius == 3:
            panel.curve(DEPTHS, values, f"line s{colour}")
        for depth, value in zip(DEPTHS, values, strict=True):
            fig.dot(panel.x(depth), panel.y(value), radius, f"f{colour}")
    bottom = panel.top + panel.height + 62
    fig.legend(
        bottom,
        (
            ("fc1", "model"),
            ("fc2", "L2 N676+ ablated"),
            ("fwash", "mean of 20 random poles ablated"),
        ),
    )
    fig.bottom = bottom
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return each figure by its file name."""
    return {
        "trace.svg": trace(rows),
        "probes.svg": probes(rows),
        "labels.svg": labels(rows),
        "ablation.svg": ablation(rows),
    }
