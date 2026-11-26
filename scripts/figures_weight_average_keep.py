"""The figures of papers/weight-average-keep/README.md, from its results.jsonl (#5404).

The steps along the line to the instruct model are circles in the first colour. The random
changes of each step's size are triangles in the second, as in papers/random-change-keep.
"""

import math

from figure_motion import reveal
from figure_svg import Figure, Panel
from figures_random_change_keep import _legend, _mark

CYCLE = "10s"


def _steps(rows: list[dict], condition: str) -> list[dict]:
    """One condition's ``step`` rows, by α, then by seed.

    Args:
        rows: The paper's results.
        condition: ``interpolate`` or ``random``.
    """
    return sorted((r for r in rows if r["table"] == "step" and r["condition"] == condition),
                  key=lambda r: (r["alpha"], r["seed"] or 0))  # fmt: skip


def keeps(rows: list[dict]) -> Figure:
    """The explanations that keep at each α, and under the random change of each α's size.

    Args:
        rows: The paper's results.

    Returns:
        The figure. The steps appear one α at a time, from the base model outward.
    """
    tuned, noise = _steps(rows, "interpolate"), _steps(rows, "random")
    total = tuned[0]["reproduced"]
    at = {r["alpha"]: r for r in tuned}
    fig = Figure(
        ["Explanations fail step by step along the line to the instruct model,",
         "and more often than under random changes of the same size"],
        [f"Each mark is the number of the {total} explanations that still pass at a step "
         "θ₀ + α·Δ, with",
         "the random changes of the same norm beside it. α = 0.5 is the weight average, and "
         "α = 1 the instruct model."],
        f"Along the line, {at[0.25]['keeps']}, {at[0.5]['keeps']}, {at[0.75]['keeps']}, "
        f"{at[1.0]['keeps']} and {at[1.5]['keeps']} of {total} explanations keep at α = 0.25, "
        f"0.5, 0.75, 1 and 1.5, and {at[-0.5]['keeps']} at α = −0.5. The random changes keep "
        f"{min(r['keeps'] for r in noise)} to {max(r['keeps'] for r in noise)}.",
    )  # fmt: skip
    low = min(r["keeps"] for r in tuned + noise)
    panel = Panel(fig, (80, fig.body_top + 30, 600, 230), (-0.7, 1.7), (low - 3, total + 1))
    panel.grid_x([-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.5], 2)
    panel.grid_y(list(range(5 * math.ceil((low - 3) / 5), total + 1, 5)), 0)
    panel.x_title("step α along θ₀ + α·Δ (0: the base model, 1: the instruct model)")
    panel.horizontal(total, "dip")
    line = [(0.0, total)] + [(r["alpha"], r["keeps"]) for r in tuned if r["alpha"] > 0]
    panel.curve([a for a, _ in line], [k for _, k in line], "line sc1 soft")
    for i, step in enumerate(sorted({r["alpha"] for r in tuned}, key=abs)):
        shown = f"s{i}"
        fig.motion += reveal(shown, 8 + 80 * i / len(tuned), CYCLE)
        for r in (r for r in noise if r["alpha"] == step):
            dx = 0.035 * (1 if r["seed"] else -1)
            _mark(fig, "random", panel.x(step + dx), panel.y(r["keeps"]), 4.5, f"fc2 ring {shown}")
        _mark(fig, "instruct", panel.x(step), panel.y(at[step]["keeps"]), 5, f"fc1 ring {shown}")
    _legend(fig, panel.top + panel.height + 52)
    fig.bottom = panel.top + panel.height + 52
    return fig


def path(rows: list[dict]) -> Figure:
    """Each explanation that fails at some step, and the steps at which it fails.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    tuned = _steps(rows, "interpolate")
    alphas = [r["alpha"] for r in tuned]
    fails = {r["alpha"]: set(r["flipped"]) for r in tuned}
    paths = {r["slug"]: r for r in rows if r["table"] == "path"}
    first = {s: paths[s]["first_fail"] if s in paths else 9.0 for s in set().union(*fails.values())}
    slugs = sorted(first, key=lambda s: (first[s], s))  # by first failing step; −0.5 only last
    steady = sum(p["one_run"] for p in paths.values())
    fig = Figure(
        ["Most explanations that fail stay failed as the step grows,",
         "and the negated tuning breaks others"],
        ["Each row is one explanation that fails at some step, by the step where it first fails. "
         "A filled cell",
         "is a step at which it fails. The left column, α = −0.5, subtracts half the tuning."],
        f"A grid of {len(slugs)} explanations by {len(alphas)} steps. {steady} of the "
        f"{len(paths)} that fail at a positive step stay failed at every larger step.",
    )  # fmt: skip
    size, left, top = 15, 300, fig.body_top + 40
    for j, alpha in enumerate(alphas):
        fig.text(left + j * (size + 30) + size / 2, top - 10, f"α = {alpha:g}", "small", "middle")
    for i, slug in enumerate(slugs):
        y = top + i * (size + 4)
        fig.text(left - 10, y + size - 3, slug.split("/", 1)[1], "small", "end")
        for j, alpha in enumerate(alphas):
            fill = "fc1" if slug in fails[alpha] else "fwash"
            fig.bar(left + j * (size + 30), y, size, size, fill)
    fig.bottom = top + len(slugs) * (size + 4) + 6
    return fig


def kl(rows: list[dict]) -> Figure:
    """The token KL from the base model at each α, and under the random change of its size.

    Args:
        rows: The paper's results.

    Returns:
        The figure.
    """
    tuned, noise = _steps(rows, "interpolate"), _steps(rows, "random")
    at = {r["alpha"]: r["token_kl"] for r in tuned}
    one = [r["token_kl"] for r in noise if r["alpha"] == 1.0]
    ratio = at[1.0] / (sum(one) / len(one))
    fig = Figure(
        [f"The tuning moves the output about {ratio:.0f} times as far as a random change of its "
         "size"],
        ["The mean KL of the changed model's next-token distribution from the base model's, "
         "on 2,048",
         "corpus tokens, at each step and under the random changes of each step's norm "
         "(log scale)."],
        f"At α = 1 the token KL is {at[1.0]:.3f} nats, against {min(one):.4f} and "
        f"{max(one):.4f} for the random changes of the same norm. It grows faster than α.",
    )  # fmt: skip
    values = [r["token_kl"] for r in tuned + noise]
    lo, hi = math.log10(min(values)) - 0.2, math.log10(max(values)) + 0.2
    panel = Panel(fig, (80, fig.body_top + 30, 600, 220), (-0.7, 1.7), (lo, hi))
    panel.grid_x([-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.5], 2)
    for e in range(math.ceil(lo), math.floor(hi) + 1):
        panel.horizontal(e, "grid")
        fig.text(panel.left - 7, panel.y(e) + 4, f"{10.0**e:g}", "small", "end")
    panel.x_title("step α along θ₀ + α·Δ")
    for r in noise:
        dx = 0.035 * (1 if r["seed"] else -1)
        _mark(fig, "random", panel.x(r["alpha"] + dx), panel.y(math.log10(r["token_kl"])), 4.5,
              "fc2 ring")  # fmt: skip
    for r in tuned:
        _mark(fig, "instruct", panel.x(r["alpha"]), panel.y(math.log10(r["token_kl"])), 5,
              "fc1 ring")  # fmt: skip
    _legend(fig, panel.top + panel.height + 52)
    fig.bottom = panel.top + panel.height + 52
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name.

    Args:
        rows: The paper's results.

    Returns:
        Each figure by its file name in ``media/``.
    """
    return {"keeps.svg": keeps(rows), "path.svg": path(rows), "kl.svg": kl(rows)}
