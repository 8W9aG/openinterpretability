"""Figure 3 of papers/neuron-development: explanations that fail after step 16,000 (#5119).

Some explanations hold at step 16,000 and fail at a later checkpoint. For each one, the
figure shows whether it holds at each checkpoint from 16,000 on. A failure gives its
shortfall: the number of extra probes that had to pass to reach the nearer failed gate.
"""

import math

from figure_svg import Figure

from openinterpretability.core.schemas import MAX_PROBES, ReviewPolicy

POLICY = ReviewPolicy()  # each of the paper's records uses the default gates
HARD_NEGATIVES = MAX_PROBES * 2 // 3  # 20: the harness asks for two thirds hard negatives
HOLDOUTS = MAX_PROBES - HARD_NEGATIVES  # 10 positive holdouts
FROM = 16_000  # the step at which the loss is within 1% of its final value (the paper)


def shortfall(specificity: float, sensitivity: float) -> int:
    """The probes that would have had to pass more for both gates to hold; 0 when they hold.

    Args:
        specificity: The share of hard negatives that stayed quiet.
        sensitivity: The share of holdouts that fired.

    Returns:
        The larger of the two gates' shortfalls, in probes.

    Examples:
        >>> shortfall(0.75, 1.0), shortfall(0.05, 1.0), shortfall(0.8, 0.3), shortfall(0.9, 0.6)
        (1, 15, 3, 0)
    """
    spec = (POLICY.min_specificity - specificity) * HARD_NEGATIVES
    sens = (POLICY.min_sensitivity - sensitivity) * HOLDOUTS
    return max(0, *(math.ceil(round(x, 6)) for x in (spec, sens)))


def savings(rows: list[dict]) -> list[dict]:
    """Each pole that holds at :data:`FROM` and fails later, with its later checkpoints.

    Args:
        rows: The paper's results.

    Returns:
        One dict per pole: ``slug``, ``steps`` (each ``(step, shortfall)`` from :data:`FROM`
        on, 0 when it holds), and ``back``, the first intermediate step at which it holds
        again after its first failure, or None. Poles that come back first, then by slug.
    """
    by_pole: dict[str, list[dict]] = {}
    for r in rows:
        if r["table"] == "pole" and r["step"] >= FROM:
            by_pole.setdefault(r["slug"], []).append(r)
    out = []
    for slug, own in by_pole.items():
        own.sort(key=lambda r: r["step"])
        steps = [(r["step"], shortfall(r["specificity"], r["sensitivity"])) for r in own]
        if steps[0][1] or not any(s for _, s in steps):
            continue  # not held at FROM, or never failed afterwards
        first_fail = next(i for i, (_, s) in enumerate(steps) if s)
        back = next((st for st, s in steps[first_fail + 1 : -1] if not s), None)
        out.append({"slug": slug, "steps": steps, "back": back})
    return sorted(out, key=lambda p: (p["back"] is None, p["back"] or 0, p["slug"]))


def _label(slug: str) -> str:
    """``L4/N216+`` from ``EleutherAI--pythia-70m/L4/N216-positive``.

    Args:
        slug: The pole's slug.

    Returns:
        The short name.
    """
    layer, neuron = slug.split("/")[1:3]
    return f"{layer}/{neuron.replace('-positive', '+').replace('-negative', '−')}"


def _fail(fig: Figure, x: float, y: float) -> None:
    """A failed checkpoint's mark: a triangle, so that colour is not the only cue.

    Args:
        fig: The figure.
        x: The mark's centre, in pixels.
        y: The mark's centre, in pixels.
    """
    h = 6
    fig.path([(x, y - h), (x + h, y + h * 0.8), (x - h, y + h * 0.8)], "fc2", close=True)


def _grid(fig: Figure, poles: list[dict], checkpoints: list[int]) -> float:
    """One row for each pole, one column for each checkpoint, and the line under the returns.

    Args:
        fig: The figure.
        poles: The poles of :func:`savings`.
        checkpoints: The steps of the columns.

    Returns:
        The baseline of the legend, in pixels.
    """
    left, top, col, row = 130, fig.body_top + 40, 120, 22
    for j, st in enumerate(checkpoints):
        fig.text(left + j * col, top - 14, f"{st // 1000}k", "head", "middle")
    fig.text(left + (len(checkpoints) - 1) * col, top - 30, "written for this step", "small",
             "middle")  # fmt: skip
    for i, pole in enumerate(poles):
        y = top + i * row
        fig.text(left - 50, y + 4, _label(pole["slug"]), "small", "end")
        for j, (_, s) in enumerate(pole["steps"]):
            if not s:
                fig.dot(left + j * col, y, 5, "fc1")
            else:
                _fail(fig, left + j * col, y)
                fig.text(left + j * col + 10, y + 4, str(s), "small")
        if i + 1 < len(poles) and pole["back"] is not None and poles[i + 1]["back"] is None:
            fig.line(20, y + row / 2, 740, y + row / 2, "axis")  # above: back before the end
    return top + len(poles) * row + 20


def _legend(fig: Figure, bottom: float) -> None:
    """The legend: the two marks and the line.

    Args:
        fig: The figure.
        bottom: The legend's baseline, in pixels.
    """
    fig.dot(25, bottom - 4, 5, "fc1")
    fig.text(35, bottom, "holds")
    _fail(fig, 95, bottom - 4)
    fig.text(105, bottom, "fails, with the probes it fell short by")
    fig.line(400, bottom - 4, 430, bottom - 4, "axis")
    fig.text(438, bottom, "rows above the line pass again before the final step")


def figure(rows: list[dict]) -> Figure:
    """Figure 3: the grid of the poles that fail after step 16,000, by checkpoint.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    poles = savings(rows)
    checkpoints = [st for st, _ in poles[0]["steps"]]
    fails = [s for p in poles for _, s in p["steps"] if s]
    near = sum(s <= 2 for s in fails)
    fig = Figure(
        ["Most explanations that fail after step 16,000",
         "miss by a probe or two, then pass again"],
        ["Each row is a pole whose explanation holds at step 16,000 and fails later.",
         "A circle holds. A triangle fails, with the number of probes it fell short by.",
         f"{near} of the {len(fails)} failures are within two probes. Every explanation holds "
         f"at step {checkpoints[-1]:,} by selection."],
        f"A grid of {len(poles)} poles against the checkpoints from step 16,000 to "
        f"{checkpoints[-1]:,}. {sum(p['back'] is not None for p in poles)} poles pass again "
        f"before the final step. {near} of {len(fails)} failures are one or two probes short. "
        f"The deepest are 15 and 14 probes short.",
    )  # fmt: skip
    bottom = _grid(fig, poles, checkpoints)
    _legend(fig, bottom)
    fig.bottom = bottom
    return fig
