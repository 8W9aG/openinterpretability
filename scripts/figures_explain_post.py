"""Draw the explain post's figures (posts/explain/media/), from measured data only.

There are four sources. Two are the sun prompt's explanation record and answer record.
One is the " days" prompt's explanation record with its VERIFIED answer record. The last is
posts/explain/media/data.json, which scripts/explain_post_data.py measures. No figure
draws a value that these don't hold.

A figure moves only when it shows a path: a mean that settles, a series that draws itself,
or a ring that travels. The rest are still. Motion is CSS keyframes inside the SVG
(``openinterpretability.figures``). The resting state is the final frame, so each figure
makes its point with motion off. No point rests on colour alone. A status, a kind of unit,
a sign, a pass and a fail each have their own shape, fill pattern or line style.

The marks are in scripts/explain_post_marks.py, and the answer figures are in
scripts/explain_post_answer_figures.py.

Run it with ``uv run python scripts/figures_explain_post.py``. ``--check`` writes nothing,
and fails if a file differs.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from explain_post_answer_figures import (
    DAYS_EXPLANATION,
    cuts,
    days_copy,
    days_mistake,
    patching,
    probes,
    route,
)
from explain_post_marks import CYCLE, END, ROW, _arrow, _at, _end, _figure, _group, _token, _travel

from openinterpretability.builder.explanation_figures import _marker
from openinterpretability.core.explanation_records import ExplanationRecord
from openinterpretability.figures.motion import counter
from openinterpretability.figures.svg import WIDTH, Figure, Panel, number

ROOT = Path(__file__).parents[1]
MEDIA = ROOT / "posts/explain/media"
EXPLANATION = ROOT / "wiki/explanations/EleutherAI--pythia-70m/6c78ac114096/20261119T013311Z.json"
ANSWER = ROOT / "wiki/answers/EleutherAI--pythia-70m/6c78ac114096/20261118T182635Z.json"
DAYS = ROOT / "wiki/answers/EleutherAI--pythia-70m/970a91e11d8c"
DAYS_ANSWER = "20261118T234629Z.json"  # the VERIFIED attempt


def baseline(data: dict) -> str:  # long-function: one figure, drawn mark by mark
    """Each neuron's mean, window by window, settling on the baseline.

    Args:
        data: The measured data of the post, from data.json.
    """
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
    """The top units as signed bars around a zero line. Returns the panel.

    Args:
        fig: The figure to draw in.
        units: The units to draw, each with an attribution, a kind and a label.
        top: The y position of the first row.
        cls_of: A function that gives the group class of row i. None draws the rows still.
    """
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


def known(record: ExplanationRecord) -> str:  # long-function: one figure, drawn mark by mark
    """The same units, labelled from the registry, and the share that is explained.

    Args:
        record: The explanation record of the sun prompt.
    """
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


def coverage(data: dict) -> str:  # long-function: one figure, drawn mark by mark
    """The answer's log-probability clean, with every unit off, and with the explained kept.

    Args:
        data: The measured data of the post, from data.json.
    """
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


def core(  # long-function: one figure, drawn mark by mark
    data: dict, record: ExplanationRecord
) -> str:
    """The greedy pruning to the minimal set, then coverage by kind of unit.

    Args:
        data: The measured data of the post, with the pruning steps.
        record: The explanation record of the sun prompt.
    """
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
