"""The figures of papers/coverage-planner/README.md, from its results.jsonl (#7061).

The old plan is grey, the lazy plan is the first colour and the full scan the second, in
every figure.
"""

from figure_motion import reveal
from figure_svg import Figure, Panel

CYCLE = "9s"
PLANS = (("attribution", "old plan", "sneutral"), ("lazy", "lazy plan", "sc1"),
         ("full", "full scan", "sc2"))  # fmt: skip


def _prompts(rows: list[dict]) -> list[dict]:
    """The rows of the ``prompt`` table, in the sample's order.

    Args:
        rows: The paper's results.

    Returns:
        The prompt rows.
    """
    return [r for r in rows if r["table"] == "prompt"]


def short(prompt: str) -> str:
    """The prompt's two nouns, as a label.

    Args:
        prompt: The prompt's text.

    Returns:
        The label.

    Examples:
        >>> short("A square has four sides and a foot has five")
        'square / foot'
    """
    words = prompt.split()
    return f"{words[1]} / {words[words.index('and') + 2]}"


def _line_legend(fig: Figure, y: float, entries) -> None:
    """A row of short lines, each with its label.

    Args:
        fig: The figure.
        y: The row's baseline, in pixels.
        entries: Each line's class and its label.
    """
    x = 20.0
    for cls, label in entries:
        fig.line(x, y - 4, x + 18, y - 4, f"curve {cls}")
        fig.text(x + 24, y, label, "small")
        x += 40 + 6.0 * len(label)
    fig.bottom = y


def trajectory(rows: list[dict]) -> Figure:
    """One prompt's coverage after each pick, for the old plan, the lazy plan and the full scan.

    The three paths draw in together, one pick at a time.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    row = next(r for r in _prompts(rows) if r["prompt"].startswith("A banjo"))
    paths = {name: row[name]["path"] for name, *_ in PLANS}
    most = max(len(p) for p in paths.values()) - 1
    fig = Figure(
        ["Scoring every candidate once lets the greedy plan reach full coverage,",
         "whereas the old plan stalls"],
        [f"Coverage after each pick on “{row['prompt']}” (pythia-70m). The old plan",
         "scores only the first 120 candidates by attribution. The grey line is the target."],
        f"The old plan stops at {max(paths['attribution']):.2f} after "
        f"{len(paths['attribution']) - 1} picks. The lazy plan reaches 1.0 after "
        f"{row['lazy']['to_1.0']} picks, and the full scan after {row['full']['to_1.0']}.",
    )  # fmt: skip
    panel = Panel(fig, (70, fig.body_top + 20, 650, 260), (0, most), (-0.1, 1.05))
    panel.grid_y([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.grid_x([0, 10, 20, 30, 40, 50, 60, 70], 0)
    panel.horizontal(1.0, "dip")
    for name, _, cls in PLANS:
        path = paths[name]
        for i in range(1, len(path)):  # one segment per pick, shown when the pick is made
            fig.line(panel.x(i - 1), panel.y(path[i - 1]), panel.x(i), panel.y(path[i]),
                     f"curve {cls} p{i}")  # fmt: skip
    for i in range(1, most + 1):
        fig.motion += reveal(f"p{i}", 5 + 85 * i / most, CYCLE)
    panel.x_title("picks (unit experiments)")
    _line_legend(fig, panel.top + panel.height + 52, [(c, label) for _, label, c in PLANS])
    return fig


def needed(row: dict, name: str) -> int:
    """The experiments that plan ``name`` needs for a coverage of 1.0.

    Args:
        row: The prompt's row of results.
        name: The plan: ``lazy``, ``full`` or ``prune``.

    Returns:
        The number of experiments.

    Examples:
        >>> needed({"prune": {"1.0": 9}, "lazy": {"to_1.0": 3}}, "prune")
        9
    """
    return row["prune"]["1.0"] if name == "prune" else row[name]["to_1.0"]


def picks(rows: list[dict]) -> Figure:
    """For each prompt, the experiments that each plan needs for full coverage.

    Args:
        rows: The paper's results.

    Returns:
        The figure. The axis is logarithmic, because the pruned sets run to a thousand.
    """
    import math

    prompts = _prompts(rows)
    marks = (("lazy", "lazy plan", "fc1"), ("full", "full scan", "fc2"),
             ("prune", "every unit, pruned", "fc3"))  # fmt: skip

    lazy = [needed(r, "lazy") for r in prompts]
    fig = Figure(
        ["The lazy plan needs a few more picks than the full scan,",
         "and far fewer experiments than pruning every unit"],
        ["Experiments to reach a coverage of 1.0, for each of the 8 prompts (log scale).",
         "The old plan reaches it on none of them, so it has no mark."],
        "; ".join(f"{short(r['prompt'])}: lazy {needed(r, 'lazy')}, full {needed(r, 'full')},"
                  f" pruned {needed(r, 'prune')}" for r in prompts),
    )  # fmt: skip
    panel = Panel(fig, (170, fig.body_top + 24, 540, 30 * len(prompts)), (math.log10(20),
                  math.log10(1500)), (0, len(prompts)))  # fmt: skip
    for tick in (20, 50, 100, 200, 500, 1000):
        x = panel.x(math.log10(tick))
        fig.line(x, panel.top, x, panel.top + panel.height, "grid")
        fig.text(x, panel.top + panel.height + 16, str(tick), "small", "middle")
    for i, row in enumerate(prompts):
        y = panel.y(len(prompts) - i - 0.5)
        fig.text(20, y + 4, short(row["prompt"]), "small")
        values = [needed(row, name) for name, *_ in marks]
        fig.line(panel.x(math.log10(min(values))), y, panel.x(math.log10(max(values))), y,
                 "grid")  # fmt: skip
        for value, (_, _, cls) in zip(values, marks, strict=True):
            fig.dot(panel.x(math.log10(value)), y, 5, f"{cls} ring")
    panel.x_title(f"experiments (the lazy plan needs {min(lazy)} to {max(lazy)})")
    fig.legend(panel.top + panel.height + 56, [(c, label) for _, label, c in marks])
    fig.bottom = panel.top + panel.height + 56
    return fig


def joint(rows: list[dict]) -> Figure:
    """The experiments that k prompts need when planned together or one at a time.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    found = [r for r in rows if r["table"] == "joint"]
    bars = (("picks", "planned together", "fc1"),
            ("separate_union", "planned apart, each unit once", "fc3"),
            ("separate_sum", "planned apart, summed", "fneutral"))  # fmt: skip
    last = found[-1]
    fig = Figure(
        [f"Planned together, {last['prompts']} prompts need {last['picks']} experiments,",
         f"against {last['separate_union']} when each prompt is planned alone"],
        [f"Experiments to bring every prompt of the first k to a coverage of "
         f"{last['target']}.",
         "A unit that two prompts share counts once in the middle bar."],
        "; ".join(f"{r['prompts']} prompts: {r['picks']} together, {r['separate_union']} "
                  f"apart, {r['separate_sum']} summed" for r in found),
    )  # fmt: skip
    top = max(r["separate_sum"] for r in found)
    panel = Panel(fig, (70, fig.body_top + 24, 640, 220), (0, len(found)), (0, top * 1.1))
    panel.grid_y([0, 100, 200, 300], 0)
    width = 0.8 / len(bars)
    for i, row in enumerate(found):
        for j, (key, _, cls) in enumerate(bars):
            x = panel.x(i + 0.1 + j * width)
            fig.bar(x, panel.y(0), panel.x(width) - panel.x(0) - 4,
                    panel.y(row[key]) - panel.y(0), cls)  # fmt: skip
            fig.text(x + (panel.x(width) - panel.x(0) - 4) / 2, panel.y(row[key]) - 5,
                     str(row[key]), "small", "middle")  # fmt: skip
        fig.text(panel.x(i + 0.5), panel.top + panel.height + 18, f"{row['prompts']} prompts",
                 "small", "middle")  # fmt: skip
    fig.legend(panel.top + panel.height + 56, [(c, label) for _, label, c in bars])
    fig.bottom = panel.top + panel.height + 56
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"trajectory.svg": trajectory(rows), "picks.svg": picks(rows),
            "joint.svg": joint(rows)}  # fmt: skip
