"""The figures of papers/spd-input-projections/README.md, from its results.jsonl."""

from figure_svg import Figure, Panel, columns

LAYERS = list(range(12))
RANDOM_COSINE = 0.07  # the closest-neuron cosine of a random direction in 3,072 dimensions
FROM_RANDOM, FROM_NEURONS = "fc2", "fc1"
START_STEP = 5_000  # p has reached its floor here, so the active counts mean the same
FRAME_EVERY = 2_000
CYCLE = "12s"
HOLD_FROM = 80  # percent of the cycle after which the moving dots rest
MATRICES = (("up", "Input projections"), ("down", "Down projections"))


def _sites(rows: list[dict], run: str, matrix: str) -> list[dict]:
    """Return one run's site rows for one matrix, in layer order."""
    found = [r for r in rows if r["kind"] == "site" and r["run"] == run and r["matrix"] == matrix]
    return sorted(found, key=lambda r: r["layer"])


def _trace(rows: list[dict], run: str) -> list[dict]:
    """Return one run's evaluations, in step order."""
    return sorted((r for r in rows if r["kind"] == "trace" and r["run"] == run),
                  key=lambda r: r["step"])  # fmt: skip


def _layer_series(panel: Panel, sites: list[dict], field: str, dot: str, line: str) -> None:
    """Add one run's value per layer as a line with dots."""
    values = [site[field] for site in sites]
    panel.curve(LAYERS, values, f"line {line}")
    for layer, value in zip(LAYERS, values, strict=True):
        panel.figure.dot(panel.x(layer), panel.y(value), 4, f"{dot} ring")


def alignment(rows: list[dict]) -> Figure:
    """Show that each run's subcomponents stay close to the basis they start in."""
    fig = Figure(
        ["Neither start crosses over: neurons stay neurons, random stays random"],
        ["Median cosine of each subcomponent with its closest neuron, after 50,000 steps."],
        "Per layer, the closest-neuron cosine is 0.89 to 0.99 from the neurons and 0.07 to "
        "0.12 from random, near a random direction's 0.07.",
    )
    fig.legend(fig.body_top + 12, [(FROM_NEURONS, "Run D, from the neurons"),
                                   (FROM_RANDOM, "Run C, from random")])  # fmt: skip
    panels = columns(fig, 2, 48, fig.body_top + 46, 210, (-0.5, 11.5), (0, 1), gap=44)
    for panel, (matrix, name) in zip(panels, MATRICES, strict=True):
        panel.grid_y([0, 0.25, 0.5, 0.75, 1], 2)
        panel.grid_x([0, 2, 4, 6, 8, 10], 0)
        panel.horizontal(RANDOM_COSINE, "dip")
        panel.head(name)
        panel.x_title("layer")
        _layer_series(panel, _sites(rows, "D", matrix), "best_neuron_cosine", FROM_NEURONS, "sc1")
        _layer_series(panel, _sites(rows, "C", matrix), "best_neuron_cosine", FROM_RANDOM, "sc2")
        fig.text(panel.x(11.4), panel.y(0.17), "random direction: 0.07", "small", "end")
    return fig


def whole_model(rows: list[dict]) -> Figure:
    """Show the KL from each start, with and without the input projections."""
    fig = Figure(
        ["The start from the neurons keeps its lead once the input projections are in"],
        ["KL of the output with every mask at its causal importance, on held-out text.",
         "Lower is better. The first evaluation, at step 0, is off the scale."],
        "With both projections the KL ends at 0.093 from the neurons and 0.158 from random. "
        "With the down projections alone it ends at 0.064 and 0.063.",
    )  # fmt: skip
    fig.legend(fig.body_top + 12, [(FROM_NEURONS, "From the neurons"),
                                   (FROM_RANDOM, "From random")])  # fmt: skip
    panels = columns(fig, 2, 48, fig.body_top + 46, 220, (0, 50_000), (0, 0.6), gap=44)
    pairs = (("Both projections: runs C and D", "C", "D"),
             ("Down projections only: runs A and B", "A", "B"))  # fmt: skip
    for panel, (name, random, neurons) in zip(panels, pairs, strict=True):
        panel.grid_y([0, 0.2, 0.4, 0.6], 1)
        panel.grid_x([0, 10_000, 20_000, 30_000, 40_000, 50_000], 0, thousands=True)
        panel.head(name)
        panel.x_title("training step")
        for run, line in ((random, "sc2"), (neurons, "sc1")):
            trace = [r for r in _trace(rows, run) if r["step"] > 0]
            panel.curve([r["step"] for r in trace], [r["kl"] for r in trace], f"line {line}")
    return fig


def yardstick(rows: list[dict]) -> Figure:
    """Show the neurons that each active subcomponent is worth, per site and run."""
    fig = Figure(
        ["At every site, an active subcomponent does more work from the neurons"],
        ["Neurons per token needed to reach a run's fit, per subcomponent it keeps active.",
         "Above 1, SPD beats the neuron baseline. The input baseline picks the best neurons",
         "for this error. The down baseline ignores cancellation, so it is easier to beat."],
        "From the neurons the ratio is higher than from random at every site: 0.79 to 2.92 "
        "against 0.45 to 1.60 on input projections, and 1.77 to 6.50 against 0.32 to 4.86 on "
        "down projections.",
    )  # fmt: skip
    fig.legend(fig.body_top + 12, [(FROM_NEURONS, "Run D, from the neurons"),
                                   (FROM_RANDOM, "Run C, from random")])  # fmt: skip
    panels = columns(fig, 2, 48, fig.body_top + 46, 220, (-0.5, 11.5), (0, 7), gap=44)
    for panel, (matrix, name) in zip(panels, MATRICES, strict=True):
        panel.grid_y([0, 1, 2, 3, 4, 5, 6, 7], 0)
        panel.grid_x([0, 2, 4, 6, 8, 10], 0)
        panel.horizontal(1, "dip")
        panel.head(name)
        panel.x_title("layer")
        pairs = zip(_sites(rows, "C", matrix), _sites(rows, "D", matrix), strict=True)
        for random, neurons in pairs:
            x = panel.x(random["layer"])
            low, high = random["neurons_per_active"], neurons["neurons_per_active"]
            fig.line(x, panel.y(low), x, panel.y(high), "range sneutral")
            fig.dot(x, panel.y(low), 4, f"{FROM_RANDOM} ring")
            fig.dot(x, panel.y(high), 4, f"{FROM_NEURONS} ring")
    return fig


def _frames(trace: list[dict], site: str) -> list[tuple[float, float]]:
    """Return a site's active count and fit at each frame, always with the last evaluation."""
    chosen = [r for r in trace if r["step"] >= START_STEP and r["step"] % FRAME_EVERY == 0]
    if chosen[-1]["step"] != trace[-1]["step"]:
        chosen.append(trace[-1])
    return [(r["active"][site], r["explained_variance"][site]) for r in chosen]


def _moving_site(fig: Figure, panel: Panel, name: str, points: list, cls: str, stroke: str) -> None:
    """Add a site's faint path and a dot that follows it and rests on its last point."""
    panel.curve([p[0] for p in points], [p[1] for p in points], f"line {stroke} soft")
    where = [(panel.x(x), panel.y(y)) for x, y in points]
    last = f"transform:translate({where[-1][0]:.1f}px,{where[-1][1]:.1f}px)"
    stops = "".join(
        f"{HOLD_FROM * i / (len(where) - 1):.1f}%{{transform:translate({x:.1f}px,{y:.1f}px)}}"
        for i, (x, y) in enumerate(where)
    )
    fig.motion += (
        f"@keyframes {name}{{{stops}100%{{{last}}}}}"
        f".{name}{{animation:{name} {CYCLE} linear infinite;{last}}}"
    )
    fig.dot(0, 0, 4, f"{cls} ring {name}")


def training(rows: list[dict]) -> Figure:
    """Animate each input projection's fit and active count through training."""
    fig = Figure(
        ["From the neurons, input projections keep more active and fit far better"],
        ["Each dot is one layer's input projection, moving through training from step 5,000",
         "to 49,999. Measured every 2,000 steps on the training run's 8 held-out pieces."],
        "From random the input projections end at 93 to 221 active subcomponents with fits of "
        "0.40 to 0.87. From the neurons they end at 125 to 248 active with fits of 0.71 to 0.88.",
    )  # fmt: skip
    fig.legend(fig.body_top + 12, [(FROM_NEURONS, "Run D, from the neurons"),
                                   (FROM_RANDOM, "Run C, from random")])  # fmt: skip
    panel = Panel(fig, (48, fig.body_top + 48, 620, 280), (50, 450), (0, 1))
    panel.grid_y([0, 0.25, 0.5, 0.75, 1], 2)
    panel.grid_x([100, 200, 300, 400], 0)
    panel.x_title("active subcomponents per token")
    fig.text(48, panel.top - 8, "explained variance", "small")
    for run, cls, stroke in (("C", FROM_RANDOM, "sc2"), ("D", FROM_NEURONS, "sc1")):
        trace = _trace(rows, run)
        for layer in LAYERS:
            points = _frames(trace, f"{layer}.up")
            _moving_site(fig, panel, f"m{run}{layer}", points, cls, stroke)
    return fig


def build(rows: list[dict]) -> dict[str, Figure]:
    """Return the paper's figures by file name."""
    return {
        "training.svg": training(rows),
        "whole-model.svg": whole_model(rows),
        "yardstick.svg": yardstick(rows),
        "alignment.svg": alignment(rows),
    }
