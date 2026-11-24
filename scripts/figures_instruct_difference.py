"""The figure and summary of papers/instruct-difference, drawn from its results (#5140)."""

import math
import random

from figure_svg import Figure, Panel

DESIGNS = {"into_instruct": "base", "into_base": "none"}  # each design's reference condition
BOOT, SEED = 1000, 0


def _kl(rows: list[dict], design: str) -> dict:
    """``(layer, condition) -> {conversation: (all, early)}`` for one design.

    Args:
        rows: The results.
        design: ``into_instruct`` or ``into_base``.

    Returns:
        The KL of each conversation, by layer and condition.
    """
    out: dict = {}
    for r in rows:
        if r.get("table") == "kl" and r["design"] == design:
            out.setdefault((r["layer"], r["condition"]), {})[r["conversation"]] = r["kl"]
    return out


def share(kl: dict, ref: dict, convs: list[int], which: int) -> float:
    """The share of the reference KL that a condition removes, pooled over ``convs``.

    Args:
        kl: The condition's KL by conversation.
        ref: The reference condition's KL by conversation.
        convs: The conversations to pool.
        which: 0 for every response token, 1 for response tokens 2 to 10.

    Returns:
        ``1 - mean(kl) / mean(ref)``.

    Examples:
        >>> share({0: (1.0, 0), 1: (1.0, 0)}, {0: (2.0, 0), 1: (2.0, 0)}, [0, 1], 0)
        0.5
    """
    return 1 - sum(kl[c][which] for c in convs) / sum(ref[c][which] for c in convs)


def summary(rows: list[dict], which: int = 0) -> list[dict]:
    """For each design and layer, the share that each condition removes, with a 95% interval.

    The intervals resample the conversations (:data:`BOOT` draws, seed :data:`SEED`). The
    random poles' share is the mean of their draws.

    Args:
        rows: The results.
        which: 0 for every response token, 1 for response tokens 2 to 10.

    Returns:
        One row per design, layer and condition.
    """
    out = []
    for design, reference in DESIGNS.items():
        kl = _kl(rows, design)
        layers = sorted({layer for layer, _ in kl if layer is not None})
        for layer in layers:
            ref = kl[(None if design == "into_base" else layer, reference)]
            convs = sorted(ref)
            names = sorted({c for la, c in kl if la == layer and c != reference})
            draws = [n for n in names if n.startswith("random")]
            for name in [n for n in names if not n.startswith("random")] + ["random"]:
                group = draws if name == "random" else [name]

                def value(cs, group=group, ref=ref, kl=kl, layer=layer):
                    return sum(share(kl[(layer, g)], ref, cs, which) for g in group) / len(group)

                rng = random.Random(SEED)
                boots = sorted(value([rng.choice(convs) for _ in convs]) for _ in range(BOOT))
                out.append({"design": design, "layer": layer, "condition": name,
                            "share": value(convs), "low": boots[int(0.025 * BOOT)],
                            "high": boots[int(0.975 * BOOT) - 1]})  # fmt: skip
    return out


def _lines(panel: Panel, rows: list[dict], design: str) -> None:
    """The VERIFIED and random shares by layer, each labelled at its end.

    Args:
        panel: The panel.
        rows: The summary rows.
        design: The design.
    """
    styles = {"verified": "curve", "random": "line sneutral dash"}
    labels = {"verified": "VERIFIED poles", "random": "random poles"}
    ends = []
    for name, cls in styles.items():
        own = sorted((r for r in rows if r["design"] == design and r["condition"] == name),
                     key=lambda r: r["layer"])  # fmt: skip
        if name == "verified":
            upper = [(panel.x(r["layer"]), panel.y(r["high"])) for r in own]
            lower = [(panel.x(r["layer"]), panel.y(r["low"])) for r in reversed(own)]
            panel.figure.shaded(upper + lower, "fwash", 0.9)
        panel.curve([r["layer"] for r in own], [r["share"] for r in own], cls)
        ends.append((panel.y(own[-1]["share"]), labels[name]))
    last = -1e9
    for y, label in sorted(ends):  # top to bottom, 13 px apart so labels don't overlap
        y = max(y + 4, last + 13)
        panel.figure.text(panel.x(23) + 6, y, label, "small")
        last = y


def figure(rows: list[dict]) -> Figure:
    """The share of the base-to-instruct KL that each layer's poles carry, in each design.

    Args:
        rows: The results.

    Returns:
        The figure.
    """
    found = summary(rows)
    fig = Figure(
        ["Each layer's VERIFIED poles carry little of the base-to-instruct difference"],
        [
            "The share of the KL from Qwen2.5-0.5B-Instruct that patching one layer's poles",
            "removes, over every response token, by layer. Left: the instruct model with that",
            "layer at its base activation, except the patched poles. Right: the base model with",
            "the patched poles at their instruct activation. The band is the VERIFIED poles'",
            "95% interval. The panels' scales differ.",
        ],
        "Two line charts over 24 layers: the share of the KL that VERIFIED poles and as many "
        "random poles remove. Both stay within about 2% of zero at every layer.",
    )
    fig.motion += ".dash{stroke-dasharray:6 4}"
    for left, design, head in ((64, "into_instruct", "Into the instruct model"),
                               (440, "into_base", "Into the base model")):  # fmt: skip
        shown = [
            r for r in found if r["design"] == design and r["condition"] in ("verified", "random")
        ]
        low, high = min(r["low"] for r in shown), max(r["high"] for r in shown)
        step = 0.01 if high - low > 0.01 else 0.002  # a tick every 1 or 0.2 points of share
        low, high = step * math.floor(low / step), step * math.ceil(high / step)
        panel = Panel(fig, (left, fig.body_top + 28, 220, 230), (0, 23), (low, high))
        panel.head(head)
        panel.grid_x((0, 6, 12, 18, 23), places=0)
        ticks = [round(low + k * step, 4) for k in range(round((high - low) / step) + 1)]
        panel.grid_y(ticks, places=3 if step < 0.01 else 2)
        panel.horizontal(0.0, "dip")
        panel.x_title("layer")
        _lines(panel, found, design)
    fig.bottom = fig.body_top + 28 + 230 + 36
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """The paper's figures by file name.

    Args:
        rows: The results.

    Returns:
        The figures.
    """
    return {"layers.svg": figure(rows)}
