"""The figures of papers/copy-first-unit, drawn from its results (issue #5015).

Every mark is told apart by its shape or line style and a label, never by colour alone.
"""

import math
import statistics

from figure_motion import animate, counter, even_stops, translate
from figure_svg import Figure, Panel

DASH = ".dash{stroke-dasharray:6 4}"
OPEN = ".soft2{fill:none}"  # a ring with no fill, so a dot under it shows
CYCLE = "7s"
HOLD_FROM = 80  # percent of the cycle after which the moving dots rest on the last layer
RANKS = (1, 10, 100, 1000, 10_000)  # rank gridlines, on a log scale
TOP_RANK = 50_304  # pythia-70m's vocabulary


def _wrong(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("wrong")]


def _short(row: dict) -> str:
    """A prompt's label: its right answer against the unit that it copies."""
    return f"{row['right'].strip()}, not {row['copied'].strip()}"


def _square(fig: Figure, x: float, y: float, cls: str) -> None:
    fig.bar(x - 4, y - 4, 8, 8, cls)


def layers(rows: list[dict]) -> Figure:  # long-function: one panel per prompt, then motion
    """Each wrong prompt's right and copied ranks at the answer position, by layer."""
    wrong = _wrong(rows)
    fig = Figure(
        ["By layer 4 the right unit is near the top, but the copied unit stays first"],
        [
            "The rank of the right answer (solid) and of the copied unit (dashed) at the",
            "answer position, under the logit lens after each layer. Rank 1 is at the top.",
            "A moving dot steps along the right answer's line, layer by layer.",
        ],
        "Six small panels, one per wrong prompt. In each, both ranks climb from the thousands "
        "toward the top; the copied unit reaches rank 1 by layer 3 or 4, and the right unit "
        "stays at rank 2 to 6.",
    )
    fig.motion += DASH
    span_y = (-math.log10(TOP_RANK), 0.0)
    frames = []
    for i, row in enumerate(wrong):
        col, line = i % 3, i // 3
        panel = Panel(fig, (64 + col * 232, fig.body_top + 34 + line * 200, 190, 140),
                      (0, 5), span_y)  # fmt: skip
        panel.head(_short(row))
        for rank in RANKS:
            y = panel.y(-math.log10(rank))
            fig.line(panel.left, y, panel.left + panel.width, y, "grid")
            if col == 0:
                fig.text(panel.left - 6, y + 4, f"{rank:,}", "small", "end")
        for layer in range(6):
            fig.text(panel.x(layer), panel.top + panel.height + 15, str(layer), "small", "middle")
        layers_ = row["enrichment"]["layers"]
        for key, cls in (("last_copied", "line sneutral dash"), ("last_right", "curve")):
            panel.curve([x["layer"] for x in layers_],
                        [-math.log10(x[key]) for x in layers_], cls)  # fmt: skip
        frames.append((panel, [(x["layer"], -math.log10(x["last_right"])) for x in layers_]))
        if line == 1:
            panel.x_title("layer")
    _animate(fig, frames)
    fig.legend(
        fig.bottom + 46, [("ink", "right answer, solid"), ("fneutral", "copied unit, dashed")]
    )
    fig.bottom += 50
    return fig


def _animate(fig: Figure, frames: list) -> None:
    """A dot on each panel's right-answer line, stepping through the measured layers."""
    for i, (panel, points) in enumerate(frames):
        stops = even_stops([translate(panel.x(x), panel.y(y)) for x, y in points], HOLD_FROM)
        last = stops[-1][1]
        fig.motion += animate(f"rank{i}", stops, CYCLE, timing="step-end", hold=last, rest=last)
        fig.dot(0, 0, 4, f"ink ring rank{i}")
    starts = [float(at[:-1]) for at, _ in even_stops([""] * 6, HOLD_FROM)]
    first = frames[0][0]
    for (cls, css), layer in zip(counter("layernow", starts, CYCLE), range(6), strict=True):
        fig.motion += css
        fig.text(first.left + 3 * 232 - 42, first.top - 26, f"layer {layer}", f"small {cls}", "end")


def heads(rows: list[dict]) -> Figure:
    """The gap logit(right) − logit(copied) with heads switched off, per wrong prompt."""
    wrong = _wrong(rows)
    fig = Figure(
        ["Without the copy heads, the copied unit still wins on five of the six prompts"],
        [
            "logit(right) − logit(copied) at the answer position. Right of 0, the right",
            "answer beats the copied unit. Line: 20 sets of 3 random heads of layers 3–4",
            "switched off, from lowest to highest, with its median as a tick.",
        ],
        "Six rows, one per wrong prompt. The model's own gap is negative in each. Switching "
        "off the three copy heads, or them and L4/H1, moves it toward 0 on most prompts, "
        "and past 0 only for the eggs prompt. Random heads move it about as much.",
    )
    panel = Panel(fig, (200, fig.body_top + 30, 520, 44 * len(wrong)), (-6.0, 6.0), (0, len(wrong)))
    for tick in (-6, -4, -2, 0, 2, 4, 6):
        fig.line(panel.x(tick), panel.top, panel.x(tick), panel.top + panel.height,
                 "axis" if tick == 0 else "grid")  # fmt: skip
        fig.text(panel.x(tick), panel.top + panel.height + 15, f"{tick:+d}" if tick else "0",
                 "small", "middle")  # fmt: skip
    panel.x_title("logit(right) − logit(copied)")
    for i, row in enumerate(wrong):
        y = panel.top + 22 + 44 * i
        fig.text(panel.left - 12, y + 4, _short(row), "", "end")
        gaps = sorted(x["gap"] for x in row["scaling"]["random_off"])
        fig.line(panel.x(gaps[0]), y, panel.x(gaps[-1]), y, "range sneutral")
        median = statistics.median(gaps)
        fig.line(panel.x(median), y - 6, panel.x(median), y + 6, "range sneutral")
        off = next(x for x in row["scaling"]["copy_heads"] if x["alpha"] == 0.0)
        _square(fig, panel.x(row["scaling"]["four_off"]["gap"]), y, "fneutral")
        fig.dot(panel.x(off["gap"]), y, 5, "open sneutral")
        fig.dot(panel.x(row["answer"]["gap"]), y, 5, "ink ring")
    fig.legend(panel.top + panel.height + 46, [("ink", "the model"), ("open sneutral",
               "copy heads off")], x=200)  # fmt: skip
    _square(fig, 418, panel.top + panel.height + 42, "fneutral")
    fig.text(428, panel.top + panel.height + 46, "copy heads and L4/H1 off")
    fig.bottom = panel.top + panel.height + 52
    return fig


def single(rows: list[dict]) -> Figure:
    """The right answer's rank in the prompt, and in its second clause alone."""
    wrong = _wrong(rows)
    fig = Figure(
        ["Alone, the second clause rarely gets its right unit either"],
        [
            "The right answer's rank after the full prompt, after its second clause alone",
            '("A year has twelve"), and after that clause behind four unrelated facts.',
            "Rank 1, at the left, is the model's answer.",
        ],
        "Six rows, one per wrong prompt. With the full prompt the right answer is rank 2 to "
        "6. With the clause alone it is rank 1 to 15, and with the four facts before it "
        "rank 3 to 254.",
    )
    fig.motion += OPEN
    panel = Panel(fig, (200, fig.body_top + 30, 520, 44 * len(wrong)), (0.0, 3.0),
                  (0, len(wrong)))  # fmt: skip
    for rank in (1, 10, 100, 1000):
        x = panel.x(math.log10(rank))
        fig.line(x, panel.top, x, panel.top + panel.height, "grid")
        fig.text(x, panel.top + panel.height + 15, f"{rank:,}", "small", "middle")
    panel.x_title("rank of the right answer (log scale)")
    for i, row in enumerate(wrong):
        y = panel.top + 22 + 44 * i
        fig.text(panel.left - 12, y + 4, _short(row), "", "end")
        forms = {s["form"]: s["right_rank"] for s in row["single"]}
        _square(fig, panel.x(math.log10(forms["few-shot"])), y, "fneutral")
        fig.dot(panel.x(math.log10(row["answer"]["right_rank"])), y, 4, "ink")
        # A larger ring, drawn last, so that it stays visible on the same rank as the dot.
        fig.dot(panel.x(math.log10(forms["zero-shot"])), y, 7, "open sneutral soft2")
    fig.legend(panel.top + panel.height + 46, [("ink", "full prompt"), ("open sneutral",
               "clause alone")], x=200)  # fmt: skip
    _square(fig, 418, panel.top + panel.height + 42, "fneutral")
    fig.text(428, panel.top + panel.height + 46, "clause after four facts")
    fig.bottom = panel.top + panel.height + 52
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """The paper's figures by file name."""
    return {"layers.svg": layers(rows), "heads.svg": heads(rows), "single.svg": single(rows)}
