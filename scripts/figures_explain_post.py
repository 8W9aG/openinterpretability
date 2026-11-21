"""Draw the explain post's figures (posts/explain/media/), from measured data only.

The sources are the sun prompt's explanation record and answer record, the " days"
prompt's explanation record and VERIFIED answer record, and posts/explain/media/data.json,
which scripts/explain_post_data.py measures. No figure draws a value that these don't hold.

A figure moves only when it shows a path: a mean that settles, a series that draws itself,
or a ring that travels. The rest are still. Motion is CSS keyframes inside the SVG
(``openinterpretability.figures``), and the resting state is the final frame, so each
figure makes its point with motion off. No point rests on colour alone: a status, a kind
of unit, a sign and a pass or a fail each have their own shape, fill pattern or line style.

Run it with ``uv run python scripts/figures_explain_post.py``. ``--check`` writes nothing,
and fails if a file differs.
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

from openinterpretability.builder.explanation_figures import DEFS, EXTRA, _marker
from openinterpretability.core.explanation_records import ExplanationRecord
from openinterpretability.figures.motion import animate, counter, reveal, translate
from openinterpretability.figures.svg import WIDTH, Figure, Panel, number

ROOT = Path(__file__).parents[1]
MEDIA = ROOT / "posts/explain/media"
EXPLANATION = ROOT / "wiki/explanations/EleutherAI--pythia-70m/6c78ac114096/20261119T013311Z.json"
ANSWER = ROOT / "wiki/answers/EleutherAI--pythia-70m/6c78ac114096/20261118T182635Z.json"
DAYS = ROOT / "wiki/answers/EleutherAI--pythia-70m/970a91e11d8c"
DAYS_ANSWER = "20261118T234629Z.json"  # the VERIFIED attempt
CYCLE = "10s"
END = 80.0  # percent of the cycle at which the last mark shows; the rest holds the final frame
ROW = 17  # the height of one row of a list figure
STYLE = (
    ".mark{fill:none;stroke:var(--ink);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}"
    ".thin{fill:none;stroke:var(--ink2);stroke-width:1.2}"
    ".rule{stroke:var(--ink);stroke-width:1.4;stroke-dasharray:5 3}"
    ".arrow{fill:none;stroke:var(--ink2);stroke-width:1.6;marker-end:url(#tip)}"
    ".tipfill{fill:var(--ink2)}"
    ".box{fill:var(--surface);stroke:var(--ink2);stroke-width:1.3}"
    ".bigtext{fill:var(--ink);font-size:15px;font-weight:600}"
)
TIP = (
    '<defs><marker id="tip" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse"><path class="tipfill" d="M0 0L8 4L0 8Z"/>'
    "</marker></defs>"
)


def _figure(title: str, subtitle: str, alt: str, drawn: bool = False) -> Figure:
    """A figure with the explanation figures' patterns and this script's styles.

    With ``drawn``, its groups appear in turn, so a series draws itself over the cycle.
    Without it, the figure is still, apart from any mark that :func:`_travel` moves.
    """
    fig = Figure([title], textwrap.wrap(subtitle, 106), alt)
    fig.marks += [DEFS, TIP]
    fig.motion = EXTRA + STYLE
    fig.drawn = drawn
    return fig


def _at(k: int, count: int) -> float:
    """The share of the cycle, in percent, at which step ``k`` of ``count`` shows."""
    return END * k / max(1, count - 1) if count > 1 else 0.0


def _group(fig: Figure, cls: str, at: float, cycle: str = CYCLE) -> None:
    """Open a group that shows from ``at`` percent of the cycle. Close it with ``_end``.

    In a still figure (``_figure`` without ``drawn``) the group always shows.
    """
    if fig.drawn:
        fig.motion += reveal(cls, at, cycle)
    fig.marks.append(f'<g class="{cls}">')


def _end(fig: Figure) -> None:
    fig.marks.append("</g>")


def _travel(fig: Figure, name: str, points: list[tuple[float, tuple[float, float]]]) -> None:
    """A ring that travels through ``points``, each ``(percent of the cycle, (x, y))``.

    It moves in straight lines between the points and rests on the last one.
    """
    stops = [(f"{at:.1f}%", translate(x, y)) for at, (x, y) in points]
    last = translate(*points[-1][1])
    fig.motion += animate(name, stops, CYCLE, timing="ease-in-out", hold=last, rest=last)
    fig.marks.append(f'<circle class="mark {name}" cx="0" cy="0" r="9"/>')


def _tick(fig: Figure, x: float, y: float) -> None:
    """A pass: a check mark."""
    fig.marks.append(f'<path class="mark" d="M{x - 5:.1f} {y:.1f}l3.5 4 7-9"/>')


def _cross(fig: Figure, x: float, y: float) -> None:
    """A fail: a cross."""
    fig.marks.append(
        f'<path class="mark" d="M{x - 4.5:.1f} {y - 4.5:.1f}l9 9'
        f'M{x + 4.5:.1f} {y - 4.5:.1f}l-9 9"/>'
    )


def _arrow(fig: Figure, a: tuple[float, float], b: tuple[float, float], bend: float = 0) -> None:
    """An arrow from ``a`` to ``b``, bent sideways by ``bend`` pixels."""
    if bend:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - bend
        d = f"M{a[0]:.1f} {a[1]:.1f}Q{mx:.1f} {my:.1f} {b[0]:.1f} {b[1]:.1f}"
    else:
        d = f"M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"
    fig.marks.append(f'<path class="arrow" d="{d}"/>')


def _unit(fig: Figure, x: float, y: float, label: str) -> None:
    """A unit of the route: a head is a square, a neuron pole a circle, with its label."""
    if "/H" in label:
        fig.marks.append(
            f'<rect class="fc1" x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2"/>'
        )
    else:
        fig.dot(x, y, 6, "fhatch")
        fig.dot(x, y, 6, "thin")
    fig.text(x + 10, y + 4, label, "small")


def _token(text: str) -> str:
    """A token as the figures print it: quoted, with its leading space shown."""
    return "“" + text.replace("\n", "⏎") + "”"


# The baseline ---------------------------------------------------------------------------


def baseline(data: dict) -> str:
    """Each neuron's mean, window by window, settling on the baseline."""
    base = data["baseline"]
    names = list(base["windows"])
    fig = _figure(
        "A baseline: what each neuron does on ordinary text",
        f"Three pythia-70m neurons over {len(base['windows'][names[0]])} windows of "
        f"{base['window']} corpus tokens. Dots: each window's mean. Line: the running mean.",
        "Three panels, one per neuron. Each window's mean activation appears as a dot, and a "
        "running mean settles on a flat line, the neuron's baseline. A diamond marks the "
        "neuron's value on our prompt, at the token where it is furthest from the baseline, "
        "which is far outside the range it takes on the corpus.",
        drawn=True,
    )
    batches = 8
    width = (WIDTH - 60 - 2 * 24) / 3
    tokens = ["The", " sun", " rises", " in", " the"]
    top = fig.body_top + 26
    for p, name in enumerate(names):
        values = base["windows"][name]
        prompt = base["prompt"][name][1:]  # the first token is left out, as in the means
        mean = sum(values) / len(values)
        low, high = min(values), max(values)
        pad = (high - low) * 0.08 or 0.1
        panel = Panel(fig, (44 + p * (width + 24), top, width - 46, 190), (0, len(values)),
                      (low - pad, high + pad))  # fmt: skip
        panel.head(name)
        panel.grid_y([low, high], 2)
        panel.x_title("corpus window")
        running, total = [], 0.0
        for i, v in enumerate(values):
            total += v
            running.append(total / (i + 1))
        size = math.ceil(len(values) / batches)
        for b in range(batches):
            part = range(b * size, min(len(values), (b + 1) * size))
            if not part:
                continue
            _group(fig, f"b{p}_{b}", _at(b, batches + 1))
            for i in part:
                fig.dot(panel.x(i + 0.5), panel.y(values[i]), 1.8, "fneutral")
            first = max(0, part[0] - 1)
            xs = [j + 0.5 for j in range(first, part[-1] + 1)]
            panel.curve(xs, running[first : part[-1] + 1], "curve")
            _end(fig)
        _group(fig, f"m{p}", END)
        panel.horizontal(mean, "rule")
        fig.text(panel.left + panel.width, panel.y(mean) - 5, f"baseline {number(mean, 3)}",
                 "small", "end")  # fmt: skip
        far = max(range(len(prompt)), key=lambda i: abs(prompt[i] - mean))
        fig.text(panel.left, panel.top + panel.height + 48, f"On {_token(tokens[far + 1])} in our "
                 f"prompt: {number(prompt[far], 2)}", "head")  # fmt: skip
        fig.text(panel.left, panel.top + panel.height + 64,
                 f"on the corpus: {number(low, 2)} to {number(high, 2)}", "small")  # fmt: skip
        _end(fig)
    fig.bottom = top + 190 + 58
    return fig.render()


# Who pushed the answer, and what we know ------------------------------------------------


def _rows(fig: Figure, units, top: float, cls_of=None):
    """The top units as signed bars around a zero line. Returns the panel."""
    peak = max(abs(u.attribution) for u in units)
    panel = Panel(
        fig, (230, top, WIDTH - 290, ROW * len(units)), (-1.25 * peak, 1.25 * peak), (0, 1)
    )
    panel.vertical(0, "axis")
    for i, u in enumerate(units):
        if cls_of:
            _group(fig, cls_of(i), _at(i, len(units)))
        y = top + ROW * i
        x0, x1 = panel.x(0), panel.x(u.attribution)
        fill = "fc1" if u.kind == "head" else "fhatch"
        fig.bar(min(x0, x1), y + 3, abs(x1 - x0), ROW - 6, fill)
        fig.text(panel.left - 10, y + ROW - 5, u.label, "small", "end")
        side = "start" if u.attribution > 0 else "end"
        nudge = 5 if u.attribution > 0 else -5
        fig.text(x1 + nudge, y + ROW - 5, number(u.attribution, 2), "small", side)
        if cls_of:
            _end(fig)
    return panel


def known(record: ExplanationRecord) -> str:
    """The same units, labelled from the registry, and the share that is explained."""
    units = record.figures.units
    share = record.explained_share
    fig = _figure(
        "What the registry already knows about each unit",
        "Large dot: the unit has a tested (VERIFIED) explanation. Small dot: it has none yet. "
        "The bar: the share of all the attribution that explained units hold.",
        f"The top 20 units with their status markers. A bar "
        f"fills to {share:.1%}: the explained units' share of the total attribution, over every "
        f"unit of the model.",
    )
    top = fig.body_top + 14
    _rows(fig, units, top)
    for i, u in enumerate(units):
        _group(fig, f"k{i}", _at(i, len(units)) * 0.75)
        _marker(fig, 30, top + ROW * i + ROW / 2, u.status if u.explained else "")
        _end(fig)
    y = top + ROW * len(units) + 26
    left, width = 230, WIDTH - 290
    fig.marks.append(
        f'<rect class="outline" x="{left}" y="{y}" width="{width}" height="14" rx="2"/>'
    )
    fig.marks.append(
        f'<rect class="fc1" x="{left}" y="{y}" width="{width * share:.1f}" height="14" rx="2"/>'
    )
    fig.text(left - 10, y + 11, "explained share", "small", "end")
    _group(fig, "pct", END)
    fig.text(left + width * share + 8, y + 12, f"{share:.1%} of all the attribution",
             "head")  # fmt: skip
    _end(fig)
    fig.bottom = y + 24
    return fig.render()


# The real test: coverage ----------------------------------------------------------------


def coverage(data: dict) -> str:
    """The answer's log-probability clean, with every unit off, and with the explained kept."""
    c = data["coverage"]
    clean, ablated, kept = c["clean"], c["ablated"], c["kept"]
    draws = [ablated + d * (clean - ablated) for d in c["random_draws"]]
    fig = _figure(
        "Switch everything else off: how much of “sky” comes back?",
        "How likely the model is to say “sky” (log-probability). Every unit not kept is set "
        "to its baseline. The ring follows the answer from one run to the next.",
        f"Four rows on one scale. The model as it is gives “sky” a log-probability of "
        f"{clean:.2f}. With every unit at its baseline it falls to {ablated:.2f}. With only the "
        f"{c['units']} explained units kept it comes back to {kept:.2f}, which is "
        f"{c['coverage']:.1%} of the way. Random sets of the same shape bring back "
        f"{sum(c['random_draws']) / len(c['random_draws']):.1%} on average.",
    )
    values = [clean, ablated, kept] + draws
    low, high = min(values), max(values)
    pad = (high - low) * 0.06
    top = fig.body_top + 16
    panel = Panel(fig, (250, top, WIDTH - 290, 4 * 40), (low - pad, high + pad), (0, 4))
    ticks = [round(v) for v in (low, (low + high) / 2, high)]
    panel.grid_x(sorted(set(ticks)), 0)
    panel.x_title("log-probability of “sky” (higher is more likely)")
    rows = [
        ("the model as it is", [clean], "clean", 0.0),
        ("every unit at its baseline", [ablated], "off", 20.0),
        (f"only the {c['units']} explained units kept", [kept], "kept", 42.0),
        ("random sets of the same shape", draws, "rand", 64.0),
    ]
    for i, (label, xs, cls, at) in enumerate(rows):
        y = top + 40 * i + 20
        _group(fig, cls, at)
        fig.text(panel.left - 12, y + 4, label, "small" if i else "head", "end")
        for x in xs:
            if cls == "rand":
                fig.dot(panel.x(x), y, 5, "open sneutral")
            else:
                fig.dot(panel.x(x), y, 7, "fc1")
        if cls == "kept":
            _arrow(fig, (panel.x(ablated), y), (panel.x(kept) - 9, y))
        _end(fig)
    row_y = [top + 40 * i + 20 for i in range(3)]
    _travel(fig, "travel", [
        (0.0, (panel.x(clean), row_y[0])), (12.0, (panel.x(clean), row_y[0])),
        (35.0, (panel.x(ablated), row_y[1])), (50.0, (panel.x(ablated), row_y[1])),
        (75.0, (panel.x(kept), row_y[2])),
    ])  # fmt: skip
    _group(fig, "share", 52.0)
    span_y = top + 4 * 40 + 62
    fig.line(panel.x(ablated), span_y, panel.x(clean), span_y, "thin")
    fig.line(panel.x(ablated), span_y - 5, panel.x(ablated), span_y + 5, "thin")
    fig.line(panel.x(clean), span_y - 5, panel.x(clean), span_y + 5, "thin")
    fig.line(panel.x(ablated), span_y + 10, panel.x(kept), span_y + 10, "mark")
    fig.text(panel.x(ablated) - 10, span_y + 4, "all the units do for “sky”", "small", "end")
    fig.text(
        panel.x(ablated) - 10, span_y + 18, f"{c['coverage']:.1%} comes back", "bigtext", "end"
    )
    _end(fig)
    _group(fig, "randshare", END)
    mean = sum(c["random_draws"]) / len(c["random_draws"])
    fig.text(20, span_y + 46, f"Random sets with the same number of heads and neuron poles in "
             f"each layer bring back {mean:.1%} on average.", "head")  # fmt: skip
    _end(fig)
    fig.bottom = span_y + 50
    return fig.render()


# The core -------------------------------------------------------------------------------


def core(data: dict, record: ExplanationRecord) -> str:
    """The greedy pruning to the minimal set, then coverage by kind of unit."""
    steps = data["pruning"]["steps"]
    full, floor = data["pruning"]["full"], data["pruning"]["floor"] * data["pruning"]["full"]
    detail = record.coverage_detail
    kept_at_end = steps[-1]["left"]
    fig = _figure(
        "The core: fewer units, nearly the same answer",
        f"Left: each of the {len(steps)} explained units is tried out. ● stays out · ○ put back, "
        f"since coverage fell below the dashed floor (90% of {full:.1%}).",
        f"Left: the coverage as the explained units are taken out one at a time. A unit stays out "
        f"while the coverage holds at 90% of its full value, and goes back in otherwise. "
        f"{kept_at_end} units remain. Right: kept alone, the heads bring back "
        f"{detail.heads_only:.1%} and the neuron poles {detail.poles_only:.1%}; together "
        f"{full:.1%}.",
        drawn=True,
    )
    top = fig.body_top + 22
    values = [s["coverage"] for s in steps if s["coverage"] is not None] + [full, floor]
    low = min(values)
    panel = Panel(fig, (60, top, 420, 200), (0, len(steps)), (min(low, floor) - 0.05, 1.0))
    panel.grid_y([0.0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.x_title("units tried, smallest effect first")
    panel.head("coverage while pruning")
    panel.horizontal(floor, "rule")
    batches = 8
    size = math.ceil(len(steps) / batches)
    starts = []
    for b in range(batches):
        part = range(b * size, min(len(steps), (b + 1) * size))
        if not part:
            continue
        at = _at(b, batches + 2)
        starts.append(at)
        _group(fig, f"p{b}", at)
        for i in part:
            s = steps[i]
            if s["coverage"] is None:
                continue
            cls = "fc1" if s["removed"] else "open sneutral"
            fig.dot(panel.x(i + 0.5), panel.y(s["coverage"]), 3, cls)
        _end(fig)
    for (cls, css), b in zip(counter("left", starts, CYCLE), range(batches), strict=False):
        fig.motion += css
        last = min(len(steps), (b + 1) * size) - 1
        fig.text(panel.left + panel.width, panel.top + 14, f"{steps[last]['left']} units left",
                 f"head {cls}", "end")  # fmt: skip
    right = Panel(fig, (560, top, 160, 200), (0, 3), (0, 1.0))
    right.head("kept alone")
    right.grid_y([0.0, 0.5, 1.0], 1, labels=True)
    bars = [("heads", detail.heads_only, "fc1"), ("poles", detail.poles_only, "fhatch"),
            ("both", full, "fc1")]  # fmt: skip
    for i, (label, value, cls) in enumerate(bars):
        _group(fig, f"k{i}", END - 12 + 6 * i)
        x = right.x(i + 0.15)
        width = right.x(0.7) - right.x(0)
        fig.bar(x, right.y(0), width, right.y(value) - right.y(0), cls)
        if label == "both":
            fig.bar(x, right.y(0), width, right.y(value) - right.y(0), "fhatch")
        fig.text(x + width / 2, right.y(value) - 5, f"{value:.0%}", "small", "middle")
        fig.text(x + width / 2, right.top + right.height + 15, label, "small", "middle")
        _end(fig)
    fig.bottom = top + 200 + 34
    return fig.render()


# The route claim ------------------------------------------------------------------------

STAGES = (
    ("Find the sun", ("L0/N513+", "L0/N1335+", "L0/N353+", "L0/N1418+")),
    ("Set the frame", ("L0/N1669+", "L0/H0", "L0/H1", "L0/H2")),
    ("Copy “sun” forward", ("L1/H6", "L2/H3", "L3/H3")),
    ("Turn it into “sky”", ("L5/H7", "L5/N1677+")),
)


def route(answer: dict) -> str:
    """The proposer's route, stage by stage, on the prompt's positions and the layers."""
    claim = answer["claim"]
    tokens = ["The", " sun", " rises", " in", " the"]
    stated = {u for _, units in STAGES for u in units}
    if stated != set(claim["route"]):
        raise ValueError("the figure's stages don't hold the claim's route")
    fig = _figure(
        "A claim about how: the route from “sun” to “sky”",
        "The proposer's claim, as it states it. Columns: the prompt's tokens. Rows: layers. "
        "Square: a head · hatched circle: a neuron pole. The ring carries “sun” along the route.",
        "A grid of the prompt's tokens across and the model's six layers up. A ring travels from "
        "“sun” up the route to “sky”. The route has four stages: four neurons in layer 0 detect "
        "“sun”; a neuron and three heads in layer 0 "
        "mark the “in the” frame; heads in layers 1 to 3 copy “sun” to the last position; a head "
        "and a neuron in layer 5 turn it into “sky”.",
    )
    top = fig.body_top + 34
    left, col = 80, 116
    row = 44

    def at(token: int, layer: int, dx: float = 0, dy: float = 0) -> tuple[float, float]:
        return left + col * token + dx, top + row * (5 - layer) + dy

    for la in range(6):
        fig.text(28, at(0, la)[1] + 4, f"L{la}", "small")
        fig.line(left - 20, at(0, la)[1], left + col * 5 + 20, at(0, la)[1], "grid")
    base = at(0, 0)[1] + 66
    for i, t in enumerate(tokens):
        fig.text(at(i, 0)[0], base, _token(t), "head", "middle")
    fig.text(at(5, 0)[0] + 10, at(5, 5)[1] + 4, "→ “sky”", "bigtext")
    places = {
        "L0/N513+": at(1, 0, -62, 4), "L0/N1335+": at(1, 0, -62, 26),
        "L0/N353+": at(1, 0, 14, 4), "L0/N1418+": at(1, 0, 14, 26),
        "L0/N1669+": at(4, 0, -62, 4), "L0/H0": at(4, 0, 14, 4),
        "L0/H1": at(4, 0, -62, 26), "L0/H2": at(4, 0, 14, 26),
        "L1/H6": at(4, 1), "L2/H3": at(4, 2), "L3/H3": at(4, 3),
        "L5/H7": at(4, 5, -30), "L5/N1677+": at(4, 5, 40),
    }  # fmt: skip
    for k, (_, units) in enumerate(STAGES):
        _group(fig, f"s{k}", 0.0)
        for u in units:
            _unit(fig, *places[u], u)
        if k == 0:
            _arrow(fig, (at(1, 0)[0], base - 14), (at(1, 0)[0], at(1, 0)[1] + 34))
        if k == 1:
            _arrow(fig, (at(3, 0)[0], base - 14), (at(4, 0)[0] - 70, at(4, 0)[1] + 30), bend=-8)
        if k == 2:
            for la in (1, 2):
                _arrow(
                    fig,
                    (at(1, 0)[0] + 10, at(1, la)[1]),
                    (at(4, la)[0] - 10, at(4, la)[1]),
                    bend=14,
                )
            fig.text(at(2, 1)[0] + 50, at(2, 1)[1] - 14, "attends to “sun”", "small", "middle")
        if k == 3:
            _arrow(fig, (at(4, 3)[0], at(4, 3)[1] - 10), (at(4, 5)[0], at(4, 5)[1] + 12))
        _end(fig)
    _travel(fig, "carry", [
        (0.0, (at(1, 0)[0], base - 5)), (10.0, (at(1, 0)[0], base - 5)),
        (28.0, (at(1, 0)[0], at(1, 0)[1] + 15)), (42.0, (at(1, 1)[0], at(1, 1)[1])),
        (60.0, (at(4, 1)[0], at(4, 1)[1])), (80.0, (at(5, 5)[0] + 40, at(5, 5)[1])),
    ])  # fmt: skip
    fig.bottom = base + 6
    return fig.render()


# Take the route away, or keep only it ---------------------------------------------------


def cuts(answer: dict) -> str:
    """How far each cut lowers the answer, and what the route alone brings back."""
    m, p = answer["verdict"]["measures"], answer["verdict"]["policy"]
    null = sum(m["null_cuts"]) / len(m["null_cuts"])
    necessity = m["route_cut"] / m["all_cut"]
    fig = _figure(
        "Take the route away, or keep only the route",
        "How far “sky” falls (log-probability) when units are set to their baseline.",
        f"Switching off every unit lowers “sky” by {m['all_cut']:.2f}. "
        f"Switching off the 13-unit route lowers it by {m['route_cut']:.2f}, {necessity:.0%} of "
        f"that. Eight random sets of the same shape lower it by {null:.2f} on average, so the "
        f"route's cut is {m['route_cut'] / null:.1f} times theirs. With only the route kept, "
        f"{m['sufficiency']:.0%} of the answer comes back.",
    )
    top = fig.body_top + 16
    panel = Panel(
        fig, (250, top, WIDTH - 310, 2 * 30 + 8 * 9 + 14), (0, m["all_cut"] * 1.05), (0, 1)
    )
    panel.grid_x([0, 2, 4, 6], 0)
    panel.x_title("how far “sky” falls")
    rows = [
        ("every unit off", m["all_cut"], "fc1", 0.0),
        ("only the route off", m["route_cut"], "fhatch", 18.0),
    ]
    y = top
    for k, (label, value, cls, at) in enumerate(rows):
        _group(fig, f"c{k}", at)
        fig.bar(panel.x(0), y + 4, panel.x(value) - panel.x(0), 18, cls)
        if cls == "fhatch":
            fig.bar(panel.x(0), y + 4, panel.x(value) - panel.x(0), 18, "thin")
        fig.text(panel.left - 10, y + 17, label, "small", "end")
        fig.text(panel.x(value) + 6, y + 17, number(value, 2), "small")
        _end(fig)
        y += 30
    _group(fig, "nulls", 36.0)
    fig.text(panel.left - 10, y + 12, "8 random sets off", "small", "end")
    for i, v in enumerate(m["null_cuts"]):
        fig.bar(panel.x(0), y + 2 + 9 * i, panel.x(max(v, 0)) - panel.x(0), 6, "fneutral")
    _end(fig)
    y += 8 * 9 + 14 + 54
    lines = [
        (f"The route's cut is {necessity:.0%} of the whole cut",
         f"rule ≥ {p['min_necessity']:.0%}", 50.0),
        (f"and {m['route_cut'] / null:.1f}× the random sets' mean cut ({null:.2f})",
         f"rule ≥ {p['null_factor']:.0f}×", 60.0),
        (f"With only the route on, {m['sufficiency']:.0%} of “sky” comes back",
         f"rule ≥ {p['min_sufficiency']:.0%}", 70.0),
    ]  # fmt: skip
    for k, (text, rule, at) in enumerate(lines):
        _group(fig, f"l{k}", at)
        _tick(fig, 30, y + 2)
        fig.text(44, y + 6, text, "head")
        fig.text(WIDTH - 24, y + 6, rule, "small", "end")
        _end(fig)
        y += 22
    fig.bottom = y - 10
    return fig.render()


# Swap the subject -----------------------------------------------------------------------


def patching(answer: dict) -> str:
    """The route patched from the corrupted prompt, against random sets patched the same way."""
    claim, m, p = answer["claim"], answer["verdict"]["measures"], answer["verdict"]["policy"]
    null = sum(m["null_patching"]) / len(m["null_patching"])
    fig = _figure(
        "Swap the subject, and move only the route",
        "The route's values are copied from the second prompt into the first. How far does “sky” "
        "fall, as a share of how far it falls on the second prompt?",
        f"Two prompts: “{claim['prompt']}” and “{claim['corrupted']}”. Copying the route's "
        f"values from the second into the first makes “sky” fall {m['patching']:.0%} as far as the "
        f"second prompt does. "
        f"Random sets of the same shape, copied the same way, move it {null:.1%} on average.",
    )
    top = fig.body_top + 18
    fig.text(20, top + 4, "clean prompt", "small")
    fig.text(130, top + 4, claim["prompt"].replace(" sun ", " SUN "), "head")
    fig.text(20, top + 26, "corrupted", "small")
    fig.text(130, top + 26, claim["corrupted"].replace(" price ", " PRICE "), "head")
    track = Panel(fig, (130, top + 70, WIDTH - 200, 70), (-0.1, 1.0), (0, 1))
    track.grid_x([0, p["max_null_patching"], p["min_patching"], 1.0], 1)
    track.x_title("share of the fall in “sky” that the second prompt causes")
    y = track.top + 20
    fig.text(track.left - 10, y + 4, "the route", "small", "end")
    fig.motion += animate(
        "slide",
        [("0%,8%", f"transform:translateX({track.x(0) - track.x(m['patching']):.1f}px)"),
         ("40%", "transform:translateX(0px)")],
        CYCLE, timing="ease-in-out",
        hold="transform:translateX(0px)", rest="transform:translateX(0px)",
    )  # fmt: skip
    fig.marks.append('<g class="slide">')
    fig.dot(track.x(m["patching"]), y, 7, "fc1")
    _end(fig)
    _group(fig, "routepct", 40.0)
    fig.text(track.x(m["patching"]), y - 12, f"{m['patching']:.1%}", "head", "middle")
    _end(fig)
    y2 = track.top + 52
    _group(fig, "nulls", 55.0)
    fig.text(track.left - 10, y2 + 4, "8 random sets", "small", "end")
    for v in m["null_patching"]:
        fig.dot(track.x(v), y2, 4, "open sneutral")
    fig.line(track.x(null), y2 - 9, track.x(null), y2 + 9, "mark")
    _end(fig)
    y = track.top + track.height + 62
    lines = [
        (f"The route moves the answer {m['patching']:.0%} of the way",
         f"rule ≥ {p['min_patching']:.0%}", 50.0),
        (f"Random sets move it {null:.1%} on average",
         f"rule ≤ {p['max_null_patching']:.0%}", 68.0),
    ]  # fmt: skip
    for k, (text, rule, at) in enumerate(lines):
        _group(fig, f"l{k}", at)
        _tick(fig, 30, y + 2)
        fig.text(44, y + 6, text, "head")
        fig.text(WIDTH - 24, y + 6, rule, "small", "end")
        _end(fig)
        y += 22
    fig.bottom = y - 10
    return fig.render()


# The reviewer's probes ------------------------------------------------------------------


def probes(
    answer: dict, title: str = "Try to break it: a second agent writes the test prompts"
) -> str:
    """Each of the reviewer's prompts, what the claim predicts and what the model said."""
    m = answer["verdict"]["measures"]
    rows = answer["probes"]["prompts"]
    word = answer["claim"]["answer"].strip()
    cycle = "14s"
    fig = _figure(
        title,
        f"The reviewer sees only the claim, never a measurement. “keeps” or “leaves”: what the "
        f"claim predicts for “{word}”. ✓ as predicted · ✗ not.",
        f"A list of the reviewer's {len(rows)} prompts, each with what the claim "
        f"predicts, the model's top token and a check mark when it came out as predicted. "
        f"{sum(m['probes'].values())} of {len(m['probes'])} did. The proposer's own "
        f"{len(m['controls'])} control prompts all did too.",
    )
    top = fig.body_top + 14
    fig.text(48, top, "prompt", "small")
    fig.text(520, top, "claim", "small")
    fig.text(600, top, "model says", "small")
    top += 8
    for i, row in enumerate(rows):
        at = END * i / len(rows)
        y = top + ROW * i
        _group(fig, f"q{i}", at, cycle)
        ok = m["probes"][row["prompt"]]
        (_tick if ok else _cross)(fig, 28, y + ROW / 2)
        text = row["prompt"] if len(row["prompt"]) < 68 else row["prompt"][:65] + "…"
        fig.text(48, y + ROW - 4, text, "small")
        fig.text(520, y + ROW - 4, "keeps" if row["expect"] == "kept" else "leaves", "small")
        fig.text(600, y + ROW - 4, _token(m["top_tokens"].get(row["prompt"], "?")), "small")
        _end(fig)
    y = top + ROW * len(rows) + 22
    passed = sum(m["probes"].values())
    fig.text(20, y, f"{passed} of {len(rows)} as predicted", "bigtext")
    controls = sum(m["controls"].values())
    _group(fig, "ctl", END, cycle)
    fig.text(WIDTH - 24, y, f"and the proposer's controls: {controls} of {len(m['controls'])}",
             "head", "end")  # fmt: skip
    _end(fig)
    fig.bottom = y - 4
    return fig.render()


# The " days" mistake --------------------------------------------------------------------

DAYS_EXPLANATION = (
    ROOT / "wiki/explanations/EleutherAI--pythia-70m/970a91e11d8c/20261118T235000Z.json"
)
COUNT_WORD = ("L0/N1877+", "L0/N1277+", "L0/N1108+", "L0/N141+", "L0/N1654+", "L0/N883+")
TIME_UNIT = ("L0/N1101+", "L0/N912+", "L0/N2008+", "L0/N780+")
RIGHT = " months"  # the right answer, which the records don't name


def days_mistake(explanation: dict) -> str:
    """The model's top next tokens for the " days" prompt, with the right answer marked."""
    rows = explanation["figures"]["ablation"]
    fig = _figure(
        "The mistake: the model is sure, and wrong",
        f"pythia-70m's most likely next tokens after “{explanation['prompt']}”.",
        "Bar chart of the five most likely next tokens. “days” has "
        f"{rows[0]['before']:.0%}. The right answer, “months”, has "
        f"{next(r['before'] for r in rows if r['token'] == RIGHT):.0%}.",
    )
    top = fig.body_top + 16
    panel = Panel(fig, (150, top, WIDTH - 260, 26 * len(rows)), (0, 1), (0, 1))
    panel.grid_x([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.x_title("probability")
    for i, r in enumerate(rows):
        y = top + 26 * i
        right = r["token"] == RIGHT
        fig.bar(panel.x(0), y + 4, panel.x(r["before"]) - panel.x(0), 18,
                "fhatch" if right else "fc1")  # fmt: skip
        if right:
            fig.bar(panel.x(0), y + 4, panel.x(r["before"]) - panel.x(0), 18, "thin")
        fig.text(panel.left - 10, y + 17, _token(r["token"]), "head" if i == 0 else "", "end")
        note = f"{r['before']:.1%}" + ("  ← the right answer" if right else "")
        fig.text(panel.x(r["before"]) + 8, y + 17, note, "head" if right else "small")
    fig.bottom = top + 26 * len(rows) + 4
    return fig.render()


def days_copy(answer: dict) -> str:
    """Where the count-word and time-unit neurons fire, and the copy the claim describes."""
    claim, m = answer["claim"], answer["verdict"]["measures"]
    reading = m["readings"][claim["prompt"]]
    tokens, strength = reading["tokens"], reading["strength"]
    fig = _figure(
        "How it goes wrong: the model copies the pattern",
        "Bars: how strongly the claim's layer-0 neurons fire at each token (measured). The ring: "
        "the copy step, as the claim states it.",
        "The prompt's ten tokens. Count-word neurons fire on “seven” and “twelve”; time-unit "
        "neurons fire on “week”, “days” and “year”. A ring starts at “twelve”, jumps back to the "
        "earlier count word “seven”, steps to the word after it, “days”, and carries “days” to "
        "the end, where the model says it.",
    )
    top = fig.body_top + 30
    left, col = 40, (WIDTH - 140) / len(tokens)

    def x(i: int) -> float:
        return left + col * (i + 0.5)

    lanes = [("count-word neurons", COUNT_WORD, "fc1"), ("time-unit neurons", TIME_UNIT, "fhatch")]
    peak = max(
        sum(strength[u][i] for u in units) for _, units, _ in lanes for i in range(len(tokens))
    )
    lane_h, copy_y = 46, top + 20
    base = copy_y + 56 + len(lanes) * (lane_h + 18)
    fig.text(20, copy_y - 22, "copy heads L3/H3, L4/H0 and L4/H7, as the claim states it", "small")
    fig.line(x(0) - 20, copy_y, x(len(tokens) - 1) + 20, copy_y, "grid")
    for k, (name, units, cls) in enumerate(lanes):
        floor = copy_y + 56 + k * (lane_h + 18) + lane_h
        fig.text(20, floor - lane_h - 4, name, "small")
        fig.line(x(0) - 20, floor, x(len(tokens) - 1) + 20, floor, "axis")
        for i in range(len(tokens)):
            value = sum(strength[u][i] for u in units)
            if value > 0.05:
                h = lane_h * value / peak
                fig.bar(x(i) - 11, floor, 22, -h, cls)
                if cls == "fhatch":
                    fig.bar(x(i) - 11, floor, 22, -h, "thin")
    for i, t in enumerate(tokens):
        fig.text(x(i), base, _token(t), "head", "middle")
    end = x(len(tokens) - 1) + 54
    fig.text(end - 8, copy_y + 5, "→ “days”", "bigtext")
    seven, days, twelve = (tokens.index(" seven"), tokens.index(" days"),
                           len(tokens) - 1)  # fmt: skip
    _arrow(fig, (x(twelve), copy_y - 4), (x(seven) + 6, copy_y - 4), bend=34)
    fig.text((x(twelve) + x(seven)) / 2, copy_y - 32, "look back for the last count word",
             "small", "middle")  # fmt: skip
    _travel(fig, "copy", [
        (0.0, (x(twelve), copy_y)), (10.0, (x(twelve), copy_y)),
        (32.0, (x(seven), copy_y)), (42.0, (x(seven), copy_y)),
        (52.0, (x(days), copy_y)), (60.0, (x(days), copy_y)),
        (80.0, (end - 20, copy_y)),
    ])  # fmt: skip
    fig.bottom = base + 4
    return fig.render()


def build() -> dict[str, str]:
    """Each figure's SVG text by its file name."""
    record = ExplanationRecord.model_validate_json(EXPLANATION.read_text(encoding="utf-8"))
    answer = json.loads(ANSWER.read_text(encoding="utf-8"))
    days = json.loads((DAYS / DAYS_ANSWER).read_text(encoding="utf-8"))
    data = json.loads((MEDIA / "data.json").read_text(encoding="utf-8"))
    return {
        "coverage.svg": coverage(data),
        "baseline.svg": baseline(data),
        "known.svg": known(record),
        "core.svg": core(data, record),
        "route.svg": route(answer),
        "cuts.svg": cuts(answer),
        "patching.svg": patching(answer),
        "days-mistake.svg": days_mistake(json.loads(DAYS_EXPLANATION.read_text(encoding="utf-8"))),
        "days-copy.svg": days_copy(days),
        "days-probes.svg": probes(days, "What fools the model, and where the claim misses"),
    }


def main() -> int:
    """Write the figures, or with ``--check`` fail if one differs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    stale = []
    for name, svg in build().items():
        path = MEDIA / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != svg:
                stale.append(name)
        else:
            path.write_text(svg, encoding="utf-8")
    if stale:
        print("stale: " + ", ".join(stale), file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
