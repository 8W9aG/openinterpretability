"""The figures of papers/route-keep/README.md, from its results.jsonl (#5164).

The base model is a circle in the first colour, and the instruct model a triangle in the
second, as in papers/random-change-keep.
"""

from figure_motion import reveal
from figure_svg import Figure, Panel
from figures_random_change_keep import _mark

CYCLE = "8s"
MODELS = (("Qwen/Qwen2.5-0.5B", "base model", "instruct", "c1"),
          ("Qwen/Qwen2.5-0.5B-Instruct", "instruct model", "random", "c2"))  # fmt: skip
GATES = (("necessity", "necessity", 0.5), ("sufficiency", "sufficiency", 0.5),
         ("patching", "patching, the route's own prompt", 0.5))  # fmt: skip


def _row(rows: list[dict], model: str) -> dict:
    """The ``gate`` row of ``model`` with the record's own seed, the first one written.

    Args:
        rows: The paper's results.
        model: The model's name.
    """
    return next(r for r in rows if r["table"] == "gate" and r["model"] == model)


def _legend(fig: Figure, y: float) -> None:
    """Each model's shape and name, in one row.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
    """
    x = 20.0
    for _, label, which, colour in MODELS:
        _mark(fig, which, x + 5, y - 4, 5, f"f{colour}")
        fig.text(x + 15, y, label, "small")
        x += 30 + 6.0 * len(label)
    fig.bottom = y


def gates(rows: list[dict]) -> Figure:
    """Each gate's measure on the base and the instruct model, against its threshold.

    Args:
        rows: The paper's results.

    Returns:
        The figure. The base model's marks appear first, then the instruct model's.
    """
    base, tuned = (_row(rows, m) for m, *_ in MODELS)
    fig = Figure(
        ["The route stays sufficient on the instruct model,",
         "but it is no longer necessary"],
        ["Each mark is one gate's measure for the 33-unit route, on each model. The grey "
         "line is",
         "the gate's threshold. The route and its prompts are the same on both models."],
        f"Necessity falls from {base['necessity']:.2f} to {tuned['necessity']:.2f}, and "
        f"patching from {base['patching']:.2f} to {tuned['patching']:.2f}. Sufficiency stays "
        f"at {base['sufficiency']:.2f} and {tuned['sufficiency']:.2f}. The threshold is 0.5.",
    )  # fmt: skip
    panel = Panel(fig, (260, fig.body_top + 30, 440, 150), (0, 1), (0, len(GATES)))
    panel.grid_x([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.vertical(0.5, "dip")
    for i, (key, label, _) in enumerate(GATES):
        y = panel.y(len(GATES) - i - 0.5)
        fig.text(20, y + 4, label, "small")
        fig.line(panel.x(tuned[key]), y, panel.x(base[key]), y, "grid")
        for j, (_, _, which, colour) in enumerate(MODELS):
            shown = f"m{j}"
            row = base if j == 0 else tuned
            _mark(fig, which, panel.x(row[key]), y, 6, f"f{colour} ring {shown}")
    fig.motion += reveal("m0", 10, CYCLE) + reveal("m1", 50, CYCLE)
    panel.x_title("the gate's measure")
    _legend(fig, panel.top + panel.height + 52)
    return fig


def patching(rows: list[dict]) -> Figure:
    """The route's patching share on each corrupted prompt, on each model.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    base, tuned = (_row(rows, m) for m, *_ in MODELS)
    prompts = [("route's own: book", "patching")]
    prompts += [(f"reviewer's: {p.split(' has a red ')[1].split(' and')[0]}", p)
                for p in base["reviewer_patching"]]  # fmt: skip

    def value(row, key):
        return row["patching"] if key == "patching" else row["reviewer_patching"][key]

    falls = sum(value(tuned, k) < 0.5 for _, k in prompts)
    fig = Figure(
        [f"On the instruct model, patching falls below the gate on {falls} of the "
         f"{len(prompts)} corrupted prompts"],
        ["Each mark is the share of the gap between the clean and the corrupted prompt that "
         "the route's",
         "activations from the corrupted prompt move. The prompts swap ' ball' for "
         "another noun."],
        "; ".join(f"{name}: {value(base, k):.2f} on the base model, {value(tuned, k):.2f} "
                  "on the instruct model" for name, k in prompts),
    )  # fmt: skip
    panel = Panel(fig, (260, fig.body_top + 30, 440, 40 * len(prompts)), (0, 1),
                  (0, len(prompts)))  # fmt: skip
    panel.grid_x([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.vertical(0.5, "dip")
    for i, (name, key) in enumerate(prompts):
        y = panel.y(len(prompts) - i - 0.5)
        fig.text(20, y + 4, name, "small")
        for row, (_, _, which, colour) in zip((base, tuned), MODELS, strict=True):
            _mark(fig, which, panel.x(min(value(row, key), 1.0)), y, 6, f"f{colour} ring")
    panel.x_title("patching share (the gate is 0.5)")
    _legend(fig, panel.top + panel.height + 52)
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"gates.svg": gates(rows), "patching.svg": patching(rows)}
