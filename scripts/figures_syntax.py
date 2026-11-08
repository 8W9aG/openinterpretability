"""The figures of papers/syntax-positive-controls/README.md, from its results.jsonl."""

from figure_svg import Figure, Panel, columns
from figures_syntax_models import models

DEPTHS = [1, 2, 3, 4, 5]
LAYERS = list(range(12))
TARGET = (2, 676, "positive")
CYCLE = "14s"
READ_SHARE = 85  # percent of the cycle spent reading; the rest holds the finished figure
MONOSPACE = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
CODE_SIZE = 13
CHAR_WIDTH = 7.8  # of CODE_SIZE in the common monospace fonts; each token is placed alone
LINE_HEIGHT = 21
LIT_FROM = 0.25  # strengths below this do not show on the code
METER_HEIGHT = 130
FILLERS = [0, 5, 10, 20, 40, 80, 160]
PROBE_PANELS = (
    ("Open, closed, partly closed", (("open", "c1"), ("partly_closed", "c3"), ("closed", "c2"))),
    ("Text the parser ignores", (("open", "c1"), ("string", "c3"), ("comment", "c2"))),
    ("Other brackets", (("open", "c1"), ("square", "c3"), ("curly", "c2"))),
)
JOINT_LEGEND = (
    ("fc1", "model"),
    ("fc2", "L2 N676+ ablated"),
    ("fc3", "the other four ablated"),
    ("fneutral", "all five ablated"),
    ("open sneutral", "mean of 20 random sets ablated"),
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


def _moment(index: int, count: int) -> float:
    """Return the share of the cycle, in percent, at which the reader reaches a token."""
    return READ_SHARE * index / (count - 1)


def _moving_dot(fig: Figure, panel: Panel, values: list[float]) -> None:
    """Add a dot that follows ``values``, token by token, and rests on the last."""
    count = len(values)
    frames = "".join(
        f"{_moment(i, count):.1f}%{{transform:translate({panel.x(i):.1f}px,{panel.y(v):.1f}px)}}"
        for i, v in enumerate(values)
    )
    last = f"transform:translate({panel.x(count - 1):.1f}px,{panel.y(values[-1]):.1f}px)"
    fig.motion += (
        f"@keyframes reader{{{frames}100%{{{last}}}}}"
        f".reader{{animation:reader {CYCLE} linear infinite;{last}}}"
    )
    fig.dot(0, 0, 5, "fc1 ring reader")


def _place_tokens(tokens: list[dict]) -> list[tuple[int, int, str]]:
    """Return each token's line, its column and its text without surrounding spaces."""
    placed, line, column = [], 0, 0
    for token in tokens:
        text = token["token"]
        if text == "\n":
            placed.append((line, column, ""))
            line, column = line + 1, 0
            continue
        word = text.strip(" ")
        placed.append((line, column + len(text) - len(text.lstrip(" ")), word))
        column += len(text)
    return placed


def _code(fig: Figure, tokens: list[dict], x0: float, y0: float) -> list[tuple[float, float]]:
    """Draw the snippet with each firing token lit, and return each token's position.

    A token lights when the reader reaches it, and stays lit until the cycle ends. Text in
    a comment or a string is muted, because the parser does not read brackets there.
    """
    count, where = len(tokens), []
    fig.motion += (
        f".code{{font-family:{MONOSPACE};font-size:{CODE_SIZE}px;fill:var(--ink)}}"
        ".codemuted{fill:var(--muted)}"
    )
    for i, (token, (line, column, word)) in enumerate(
        zip(tokens, _place_tokens(tokens), strict=True)
    ):
        x, y = x0 + column * CHAR_WIDTH, y0 + line * LINE_HEIGHT
        where.append((x, y))
        strength = token["strength"][0]
        if word and strength >= LIT_FROM:
            name = f"lit{i}"
            fig.motion += (
                f"@keyframes {name}{{0%{{opacity:0}}{_moment(i, count):.1f}%{{opacity:1}}}}"
                f".{name}{{animation:{name} {CYCLE} step-end infinite}}"
            )
            box = (x - 1, y - CODE_SIZE, len(word) * CHAR_WIDTH + 2, LINE_HEIGHT - 2)
            fig.marks.append(
                f'<rect class="fc1 {name}" x="{box[0]:.1f}" y="{box[1]:.1f}" width="{box[2]:.1f}" '
                f'height="{box[3]:.1f}" rx="2" fill-opacity="{0.1 + 0.05 * strength:.2f}"/>'
            )
        if word:
            muted = " codemuted" if token["node"] in ("comment", "string") else ""
            fig.text(x, y, word, f"code{muted}")
    return where


def _caret(fig: Figure, where: list[tuple[float, float]]) -> None:
    """Add a mark under the token that the reader is on, hidden at rest."""
    count = len(where)
    frames = "".join(
        f"{_moment(i, count):.1f}%{{opacity:1;transform:translate({x:.1f}px,{y + 4:.1f}px)}}"
        for i, (x, y) in enumerate(where)
    )
    fig.motion += (
        f"@keyframes caret{{{frames}{READ_SHARE + 0.1:.1f}%{{opacity:0}}100%{{opacity:0}}}}"
        f".caret{{animation:caret {CYCLE} step-end infinite;opacity:0}}"
    )
    fig.marks.append(f'<rect class="fc2 caret" x="0" y="0" width="{CHAR_WIDTH}" height="2.5"/>')


def _meter(fig: Figure, x: float, y: float, name: str, values: list[float], limit: float) -> None:
    """Add a bar that shows the reader's current value, and holds at the peak at rest."""
    peak = max(values)
    cls, colour = f"meter{name[0]}", "fc1" if name == "Strength" else "fneutral"
    frames, previous = [], None
    for i, value in enumerate(values):
        if value != previous:
            scale = value / limit
            frames.append(f"{_moment(i, len(values)):.1f}%{{transform:scaleY({scale:.3f})}}")
            previous = value
    rest = f"transform:scaleY({peak / limit:.3f})"
    fig.motion += (
        f"@keyframes {cls}{{{''.join(frames)}{READ_SHARE + 0.1:.1f}%{{{rest}}}100%{{{rest}}}}}"
        f".{cls}{{animation:{cls} {CYCLE} step-end infinite;transform-box:fill-box;"
        f"transform-origin:50% 100%;{rest}}}"
    )
    fig.text(x + 24, y - 22, name, "head", "middle")
    fig.line(x, y + METER_HEIGHT, x + 48, y + METER_HEIGHT, "axis")
    fig.marks.append(
        f'<rect class="{colour} {cls}" x="{x + 8:.1f}" y="{y:.1f}" width="32" '
        f'height="{METER_HEIGHT}" rx="1.5"/>'
    )
    top = y + METER_HEIGHT * (1 - peak / limit)
    fig.line(x, top, x + 48, top, "dip")
    fig.text(x + 24, top - 5, f"peak {peak:.3g}", "small", "middle")
    fig.text(x + 24, y + METER_HEIGHT + 15, f"0 to {limit:g}", "small", "middle")


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
    """Draw one written snippet, its strength and its bracket depth, read token by token."""
    tokens = next(r for r in rows if r["kind"] == "trace")["tokens"]
    fig = Figure(
        ["L2 N676+ climbs as brackets open, and drops as soon as they close"],
        [
            "A function written for this paper, read token by token. Blue shows the pole's "
            "strength at each token.",
            "Grey text is a comment or a string: to the parser, brackets there do not count, "
            "but the pole fires on them.",
        ],
        "A short Python function with its tokens lit by the pole's strength, and meters of "
        "depth and strength. Tokens inside brackets are lit, most at depth 4, where strength "
        "reaches about 9. Tokens outside brackets are not. Tokens inside brackets in a comment "
        "and in a string are also lit, where the parser's depth stays at 0.",
    )
    y0 = fig.body_top + 34
    where = _code(fig, tokens, 24, y0)
    _caret(fig, where)
    strengths = [t["strength"][0] for t in tokens]
    _meter(fig, 548, y0 + 10, "Depth", [_depth(t) for t in tokens], 5)
    _meter(fig, 652, y0 + 10, "Strength", strengths, 10)
    last = len(tokens) - 0.5
    top = Panel(fig, (44, y0 + 8 * LINE_HEIGHT + 56, 692, 100), (-0.5, last), (0, 10))
    top.head("Strength of L2 N676+, token by token")
    top.grid_y([0, 5, 10], 0)
    top.curve(range(len(tokens)), strengths, "line sc1")
    _moving_dot(fig, top, strengths)
    _mark(fig, top, tokens, "per", "comment", "in a comment")
    _mark(fig, top, tokens, "all", "string", "in a string")
    low = Panel(fig, (44, top.top + top.height + 40, 692, 50), (-0.5, last), (0, 4))
    low.head("Bracket depth, from the parser")
    low.grid_y([0, 4], 0)
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
    for depth in DEPTHS:
        fig.dot(panel.x(depth), panel.y(found[depth]["controls_mean"]), 8, "open sneutral")
    for key, colour in (("clean", "c1"), ("ablated", "c2")):
        values = [found[d][key] for d in DEPTHS]
        panel.curve(DEPTHS, values, f"line s{colour}")
        for depth, value in zip(DEPTHS, values, strict=True):
            fig.dot(panel.x(depth), panel.y(value), 3, f"f{colour}")
    bottom = panel.top + panel.height + 62
    fig.legend(
        bottom,
        (
            ("fc1", "model"),
            ("fc2", "L2 N676+ ablated"),
            ("open sneutral", "mean of 20 random poles ablated"),
        ),
    )
    fig.bottom = bottom
    return fig


def joint(rows: list[dict]) -> Figure:
    """Draw how many brackets a predicted closer closes, with round-depth poles ablated."""
    found = {r["depth"]: r for r in rows if r["kind"] == "joint_ablation"}
    fig = Figure(
        ["L2 N676+ and the other round-depth poles push the count of closers apart"],
        [
            "Mean number of round brackets that a predicted closing token closes, over 8 written "
            "items at each depth.",
            "Without L2 N676+ the model closes one. Without the other four it closes nearly four.",
        ],
        "A plot of the number of brackets closed against the number open. The model rises "
        "from 1.0 to 1.95, and twenty random ablated sets overlap it. With L2 N676+ ablated it "
        "stays near 1.0. With the other four ablated it rises to about 3.8 from depth 3. With "
        "all five ablated it is about 2.",
    )
    panel = Panel(fig, (64, fig.body_top + 22, 640, 220), (0.7, 5.3), (0.5, 4))
    panel.grid_x(DEPTHS, 0)
    panel.grid_y([1, 2, 3, 4], 0)
    panel.x_title("open brackets")
    for depth in DEPTHS:
        fig.dot(panel.x(depth), panel.y(found[depth]["controls_mean"]["count"]), 8, "open sneutral")
    for key, colour in (
        ("all five", "neutral"),
        ("clean", "c1"),
        ("L2 N676+", "c2"),
        ("all but L2 N676+", "c3"),
    ):
        values = [found[d][key]["count"] for d in DEPTHS]
        panel.curve(DEPTHS, values, f"line s{colour}")
        for depth, value in zip(DEPTHS, values, strict=True):
            fig.dot(panel.x(depth), panel.y(value), 3, f"f{colour}")
    bottom = panel.top + panel.height + 62
    fig.legend(bottom, JOINT_LEGEND)
    fig.bottom = bottom
    return fig


def reach(rows: list[dict]) -> Figure:
    """Draw how strength falls with distance from an open bracket, and across lines."""
    fig = Figure(
        ["The signal is short-range: it fades within 20 list items, and falls across lines"],
        [
            "Left: strength at one token, some list items after one bracket, open or closed. "
            "The steps double, so the axis is not linear.",
            "Right: strength at one token under open brackets, on one line or with each call "
            "on its own line. Each point has 8 items.",
        ],
        "Two plots. Left, strength under one open bracket falls from 4.2 with no filler to 0.6 "
        "after 10 list items and 0 after 20. A closed bracket stays at 0. Right, on one line "
        "strength rises from 4 to 11 with depth. With each call on its own line it peaks at 3.4 "
        "at depth 2, and falls to 0.5 at depth 5.",
    )
    y0 = fig.body_top + 22
    _distance_panel(fig, rows, y0)
    _lines_panel(fig, rows, y0)
    fig.bottom = y0 + 240
    return fig


def _distance_panel(fig: Figure, rows: list[dict], y0: float) -> None:
    """Draw strength against the list items after one bracket, open or closed."""
    left = Panel(fig, (44, y0, 330, 210), (-0.3, len(FILLERS) - 0.7), (0, 5))
    left.head("Distance from one bracket")
    left.grid_y([0, 1, 2, 3, 4, 5], 0)
    left.x_title("list items between the bracket and the token")
    for position, filler in enumerate(FILLERS):
        fig.text(left.x(position), left.y(0) + 16, str(filler), "small", "middle")
    places = list(range(len(FILLERS)))
    found: dict[int, float] = {}
    for condition, colour in (("closed", "c2"), ("open", "c1")):
        same = (
            r
            for r in rows
            if r["kind"] == "distance"
            and r["condition"] == condition
            and (r.get("layer"), r.get("neuron"), r.get("pole")) == TARGET
        )
        found = {r["filler_items"]: r["mean"] for r in same}
        values = [found[filler] for filler in FILLERS]
        left.curve(places, values, f"line s{colour}")
        for place, value in zip(places, values, strict=True):
            fig.dot(left.x(place), left.y(value), 3, f"f{colour}")
    fig.text(left.x(1) + 8, left.y(found[5]) - 8, "open", "small")
    fig.text(left.x(3), left.y(0) - 8, "closed, and open from 20", "small")


def _lines_panel(fig: Figure, rows: list[dict], y0: float) -> None:
    """Draw strength against open brackets, on one line or with each call on its own."""
    right = Panel(fig, (420, y0, 300, 210), (0.7, 6.1), (0, 12))
    right.head("Line breaks")
    right.grid_x(DEPTHS, 0)
    right.grid_y([0, 4, 8, 12], 0)
    right.x_title("open brackets")
    for condition, colour, name in (("open", "c1", "one line"), ("lines", "c3", "own lines")):
        values = [_probe(rows, condition, depth) for depth in DEPTHS]
        right.curve(DEPTHS, values, f"line s{colour}")
        for depth, value in zip(DEPTHS, values, strict=True):
            fig.dot(right.x(depth), right.y(value), 3, f"f{colour}")
        fig.text(right.x(5) + 7, right.y(values[-1]) + 4, name, "small")


def build(rows: list[dict], others: dict[str, list[dict]] | None = None) -> dict[str, Figure]:
    """Return each figure by its file name. ``others`` holds the sweeps of other models."""
    sweeps = {"GPT-2 small": rows, **(others or {})}
    return {
        "models.svg": models(sweeps),
        "trace.svg": trace(rows),
        "probes.svg": probes(rows),
        "labels.svg": labels(rows),
        "reach.svg": reach(rows),
        "ablation.svg": ablation(rows),
        "joint.svg": joint(rows),
    }
