"""The figures of papers/native-vs-learned/README.md, from its results.jsonl."""

from figure_svg import WIDTH, Figure, Panel, columns, number

# One colour per basis through the paper: learned latents, native neurons, rotated units.
COLOURS = {
    "transcoder latents": "c1",
    "transcoder latents, dead draws counted": "c1",
    "positive neuron poles": "c2",
    "native positive poles": "c2",
    "rotated directions": "c3",
}
LABELS = {
    "transcoder latents": "Transcoder latents that fire",
    "transcoder latents, dead draws counted": "Transcoder latents, dead draws counted",
    "positive neuron poles": "Positive neuron poles",
    "native positive poles": "Native positive poles",
    "rotated directions": "The same layers, rotated at random",
}
ROW = 34  # pixels per arm
LABEL_ROOM = 280


def shares(rows: list[dict]) -> Figure:
    """Each arm's share VERIFIED, with its 95% interval, grouped by model."""
    arms = [r for r in rows if r["kind"] == "share"]
    fig = Figure(
        [
            "Per unit audited, learned latents verify most often,",
            "and the native basis beats a random rotation of it",
        ],
        [
            "Share VERIFIED with its 95% Wilson interval. One audit per unit, one agent pair "
            "and one protocol in each model.",
        ],
        "Five dots with intervals. GPT-2 small: transcoder latents that fire 0.93, latents "
        "with dead draws counted 0.61, positive neuron poles 0.50. Qwen2.5-0.5B: native "
        "positive poles 0.60, the same layers rotated at random 0.03.",
    )
    top = fig.body_top + 34
    height = ROW * (len(arms) + 1)
    panel = Panel(
        fig, (LABEL_ROOM, top, WIDTH - LABEL_ROOM - 80, height), (0, 1), (len(arms) + 1, 0)
    )
    panel.grid_x([0, 0.25, 0.5, 0.75, 1], 2)
    panel.x_title("share VERIFIED")
    model, slot = None, 0
    for arm in arms:
        if arm["model"] != model:
            model = arm["model"]
            fig.text(20, panel.y(slot + 0.62), model, "head")
            slot += 0.6
        y = panel.y(slot + 0.5)
        colour = COLOURS[arm["arm"]]
        lo, hi = arm["ci95"]
        fig.line(panel.x(lo), y, panel.x(hi), y, f"line s{colour}")
        fig.dot(panel.x(arm["share"]), y, 6, f"f{colour} ring")
        fig.text(32, y + 4, LABELS[arm["arm"]])
        fig.text(panel.x(hi) + 8, y + 4, f"{arm['verified']} of {arm['units']}", "small")
        slot += 1
    return fig


def layer(rows: list[dict]) -> Figure:
    """Audits, and VERIFIED units to expect, to audit every unit of one GPT-2 layer once."""
    budget = next(r for r in rows if r["kind"] == "layer")
    fig = Figure(
        [
            "To audit a whole GPT-2 layer, latents need five times the audits",
            "and give ten times the VERIFIED units",
        ],
        [
            "An extrapolation of the measured shares to one layer: 24,576 latents, of which "
            "66% fire, against 3,072 positive poles.",
        ],
        f"Two bar charts. Audits per layer: latents {budget['latent_audits']}, positive poles "
        f"{budget['pole_audits']}. VERIFIED units to expect: latents {budget['latent_verified']}, "
        f"positive poles {budget['pole_verified']}.",
    )
    panels = columns(fig, 2, 70, fig.body_top + 40, 200, (0, 2), (0, 18000), gap=60)
    pairs = (
        ("audits", budget["latent_audits"], budget["pole_audits"]),
        ("VERIFIED units to expect", budget["latent_verified"], budget["pole_verified"]),
    )
    for panel, (name, latents, poles) in zip(panels, pairs, strict=True):
        panel.head(name)
        panel.grid_y([0, 4000, 8000, 12000, 16000], 0, labels=panel is panels[0])
        for slot, (value, colour) in enumerate(((latents, "c1"), (poles, "c2"))):
            x = panel.x(slot + 0.2)
            width = panel.x(slot + 0.8) - x
            fig.bar(x, panel.y(0), width, panel.y(value) - panel.y(0), f"f{colour}")
            fig.text(x + width / 2, panel.y(value) - 6, number(value, 0), "small", "middle")
    bottom = panels[0].top + panels[0].height + 40
    fig.legend(bottom, [("fc1", "transcoder latents"), ("fc2", "positive neuron poles")])
    fig.bottom = bottom
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name."""
    return {"shares.svg": shares(rows), "layer.svg": layer(rows)}
