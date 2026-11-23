"""The figures of papers/template-or-tuning/README.md, from its results.jsonl (#5126)."""

from figure_motion import counter
from figure_svg import Figure, Panel, columns, number

# One colour and one shape per model through the paper. The shape keeps them apart without
# colour: circles are the base model, triangles the instruct model.
MODELS = (("base", "c1", "Qwen2.5-0.5B (base)"), ("instruct", "c2", "Qwen2.5-0.5B-Instruct"))
# The four conditions in the order of the animation, each with its model and its text.
FRAMES = (("base_plain", "base", "base model, plain text"),
          ("base_chat", "base", "base model, inside the chat template"),
          ("instruct_plain", "instruct", "instruct model, plain text"),
          ("instruct_chat", "instruct", "instruct model, inside the chat template"))  # fmt: skip
CYCLE = "8s"
HALF = ".half{opacity:.6}"  # dense dots, see-through so that overlaps show


def _mark(fig: Figure, model: str, x: float, y: float, size: float, style: str) -> None:
    """One mark in the model's shape: a circle for the base model, a triangle for instruct.

    Args:
        fig: The figure.
        model: ``base`` or ``instruct``.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
        size: The mark's radius, in pixels.
        style: The mark's CSS classes.
    """
    if model == "base":
        fig.dot(x, y, size, style)
    else:
        h = size * 1.2
        fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], style, close=True)


def _legend(fig: Figure, y: float) -> None:
    """Each model's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for model, colour, label in MODELS:
        _mark(fig, model, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label)
        x += 30 + 6.4 * len(label)


def _poles(rows: list[dict]) -> list[dict]:
    """The pole rows that keep on the base model as plain text: the poles that the paper counts.

    Args:
        rows: The paper's results.

    Returns:
        The rows, in the order of the file.
    """
    return [r for r in rows if r["table"] == "pole" and r["base_plain"]["passes"]]


def _square(panel: Panel, values: dict[str, float], places: int) -> None:
    """One model's line from plain text to chat text, for each model, with the values.

    Args:
        panel: The panel. Its x axis is 0 for plain text and 1 for chat text.
        values: The value of each condition.
        places: The decimal places of the labels.
    """
    fig = panel.figure
    for model, colour, _ in MODELS:
        ys = [values[f"{model}_plain"], values[f"{model}_chat"]]
        panel.curve([0, 1], ys, f"line s{colour}")
        for x, y in zip((0, 1), ys, strict=True):
            _mark(fig, model, panel.x(x), panel.y(y), 5, f"f{colour} ring")
            side, anchor = (-12, "end") if x == 0 else (12, "start")
            lift = -6 if model == "base" else 12  # base above, instruct below: no overlap
            fig.text(panel.x(x) + side, panel.y(y) + (4 if x == 0 else lift),
                     number(y, places), "ink", anchor)  # fmt: skip
    for x, label in ((0, "plain text"), (1, "chat template")):
        fig.text(panel.x(x), panel.top + panel.height + 18, label, "small", "middle")


def square(rows: list[dict]) -> Figure:
    """The keeps and the mean sensitivity in the four conditions: the paper's head figure.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    poles = _poles(rows)
    keeps = {c: sum(r[c]["passes"] for r in poles) for c, _, _ in FRAMES}
    sens = {c: sum(r[c]["sensitivity"] for r in poles) / len(poles) for c, _, _ in FRAMES}
    fig = Figure(
        ["The chat template costs the base model more",
         "explanations than instruction tuning does"],
        [f"{len(poles)} VERIFIED poles of Qwen2.5-0.5B, each read on its own probes in four "
         "conditions.", "Left: the poles that still pass both gates. Right: the mean "
         "sensitivity of the probes."],
        f"Two line charts, plain text against the chat template, one line per model. Keeps: "
        f"base {keeps['base_plain']} to {keeps['base_chat']}, instruct {keeps['instruct_plain']} "
        f"to {keeps['instruct_chat']}. Mean sensitivity: base {number(sens['base_plain'], 3)} "
        f"to {number(sens['base_chat'], 3)}, instruct {number(sens['instruct_plain'], 3)} to "
        f"{number(sens['instruct_chat'], 3)}.",
    )  # fmt: skip
    top = fig.body_top + 30
    left, right = columns(fig, 2, 80, top, 220, (-0.35, 1.35), (0, 1))
    for panel, values, lo, hi, places, head in (
            (left, keeps, 250, 370, 0, "poles that keep"),
            (right, sens, 0.72, 0.84, 3, "mean sensitivity")):  # fmt: skip
        panel.y_range = (lo, hi)
        panel.grid_y([lo + (hi - lo) * i / 4 for i in range(5)], max(places - 1, 0))
        panel.head(head)
        _square(panel, values, places)
    bottom = top + 220 + 50
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def _grid_frame(fig: Figure, poles: list[dict], condition: str, model: str, frame: str,
                spot) -> None:  # fmt: skip
    """One frame of the grid: every pole, filled when it keeps and open when it fails.

    Args:
        fig: The figure.
        poles: The poles, in their grid order.
        condition: The condition of this frame.
        model: The condition's model, which gives the shape and the colour.
        frame: The frame's class from :func:`counter`.
        spot: A function from a pole's place in ``poles`` to its centre, in pixels.
    """
    colour = dict((m, c) for m, c, _ in MODELS)[model]
    for i, row in enumerate(poles):
        x, y = spot(i)
        style = f"f{colour}" if row[condition]["passes"] else f"open s{colour}"
        _mark(fig, model, x, y, 3.2, f"{style} {frame}")


def grid(rows: list[dict]) -> Figure:
    """Every pole, by layer, in each of the four conditions in turn.

    Args:
        rows: The paper's results.

    Returns:
        The figure. It steps through the conditions and rests on the instruct model inside
        the template.
    """
    poles = sorted(_poles(rows), key=lambda r: (r["layer"], r["slug"]))
    fig = Figure(
        ["Most of the explanations that fail inside the",
         "chat template fail on the base model already"],
        ["Each mark is one VERIFIED pole of Qwen2.5-0.5B, in the column of its layer. A filled",
         "mark passes its own probes, and an open mark fails. The frames step through the "
         "four conditions."],
        f"A grid of {len(poles)} marks in 24 columns, one per layer. Four frames: the base "
        "model on plain text, the base model inside the chat template, the instruct model on "
        "plain text and the instruct model inside the chat template. Open marks appear in "
        "every layer from the second frame on.",
    )  # fmt: skip
    fig.motion += HALF
    left, top, step = 70.0, fig.body_top + 40, 8.6
    seen: dict[int, int] = {}
    places = []
    for row in poles:
        places.append((row["layer"], seen.get(row["layer"], 0)))
        seen[row["layer"]] = places[-1][1] + 1
    spot = lambda i: (left + places[i][0] * 27.5, top + places[i][1] * step)  # noqa: E731
    starts = [100 * i / len(FRAMES) * 0.85 for i in range(len(FRAMES))]
    for (cls, css), (condition, model, label) in zip(counter("cond", starts, CYCLE), FRAMES,
                                                     strict=True):  # fmt: skip
        fig.motion += css
        kept = sum(r[condition]["passes"] for r in poles)
        fig.text(left - 6, top - 18, f"{label}: {kept} of {len(poles)} keep", f"head {cls}")
        _grid_frame(fig, poles, condition, model, cls, spot)
    for layer in range(0, 24, 4):
        fig.text(spot(0)[0] + layer * 27.5, top + max(seen.values()) * step + 12,
                 f"L{layer}", "small", "middle")  # fmt: skip
    bottom = top + max(seen.values()) * step + 44
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig


def _spread(i: int) -> float:
    """A fixed spread for marks that share a value, so that they do not hide each other.

    Args:
        i: The mark's place in its list.

    Returns:
        An offset from −0.5 to 0.5.

    Examples:
        >>> _spread(0), _spread(3)
        (-0.5, 0.5)
    """
    return (i * 37) % 11 / 10 - 0.5


def tuning(rows: list[dict]) -> Figure:
    """Each pole's change in sensitivity under the tuning, on plain text against chat text.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    poles = _poles(rows)
    pairs = [(r["instruct_plain"]["sensitivity"] - r["base_plain"]["sensitivity"],
              r["instruct_chat"]["sensitivity"] - r["base_chat"]["sensitivity"],
              r["base_chat"]["passes"]) for r in poles]  # fmt: skip
    fig = Figure(
        ["Inside the template, the tuning moves sensitivity",
         "further, and it moves it up as often as down"],
        ["Each mark is one pole: the instruct model's sensitivity minus the base model's, on "
         "plain text", "(across) and inside the chat template (up). Open marks failed inside "
         "the template on the base model."],
        f"A scatter plot of {len(pairs)} marks. The marks spread further up and down than "
        "across. Many of the open marks, the poles that the template cost on the base model, "
        "sit above the zero line.",
    )  # fmt: skip
    fig.motion += HALF
    panel = Panel(fig, (90, fig.body_top + 24, 560, 300), (-0.85, 0.5), (-0.65, 0.5))
    panel.grid_x([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4], 1)
    panel.grid_y([-0.6, -0.4, -0.2, 0, 0.2, 0.4], 1)
    panel.horizontal(0, "axis")
    panel.vertical(0, "axis")
    panel.x_title("change in sensitivity on plain text")
    fig.text(panel.left - 50, panel.top - 8, "change inside the chat template", "small")
    for i, (x, y, kept) in enumerate(pairs):
        nudge = _spread(i) * 0.02
        style = "fc1 half" if kept else "open sc1"
        fig.dot(panel.x(x + nudge), panel.y(y + _spread(i + 5) * 0.02), 3, style)
    bottom = panel.top + panel.height + 50
    fig.dot(25, bottom - 4, 4, "fc1")
    fig.text(35, bottom, "kept inside the template on the base model")
    fig.dot(345, bottom - 4, 4, "open sc1")
    fig.text(355, bottom, "failed inside the template on the base model")
    fig.bottom = bottom
    return fig


def pearson(rows: list[dict]) -> Figure:
    """Each pole's base-to-instruct correlation on plain text against chat text.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    poles = _poles(rows)
    lower = sum(r["pearson_chat"] < r["pearson_plain"] for r in poles)
    fig = Figure(
        [f"On chat text, {lower} of {len(poles)} poles correlate less",
         "between the base and instruct models"],
        ["Each mark is one pole: Pearson's r between its activations in the two models, on "
         "the plain", "corpus sample (across) and on the same documents as user turns (up). "
         "The line is equal r."],
        f"A scatter plot of {len(poles)} marks between r = 0.86 and 1. {lower} marks sit "
        "below the line of equal correlation.",
    )  # fmt: skip
    fig.motion += HALF
    panel = Panel(fig, (90, fig.body_top + 24, 400, 300), (0.92, 1.0), (0.86, 1.0))
    panel.grid_x([0.92, 0.94, 0.96, 0.98, 1.0], 2)
    panel.grid_y([0.86, 0.9, 0.94, 0.98], 2)
    fig.line(panel.x(0.92), panel.y(0.92), panel.x(1.0), panel.y(1.0), "dip")
    panel.x_title("r on plain text")
    fig.text(panel.left - 50, panel.top - 8, "r on chat text", "small")
    for r in poles:
        fig.dot(panel.x(r["pearson_plain"]), panel.y(r["pearson_chat"]), 3, "fc1 half")
    fig.bottom = panel.top + panel.height + 24
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"square.svg": square(rows), "grid.svg": grid(rows), "tuning.svg": tuning(rows),
            "pearson.svg": pearson(rows)}  # fmt: skip
