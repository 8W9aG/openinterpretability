"""The figure of papers/syntax-positive-controls/README.md that compares models' sweeps."""

from figure_svg import Figure, columns

MODEL_COLOURS = ("c1", "c2", "c3")


def _layer_best(rows: list[dict], name: str) -> list[float]:
    """Return each layer's best held-out R² for one label set, in a sweep of any model."""
    found = {
        r["layer"]: r["top"][0]["r2"]
        for r in rows
        if r["kind"] in ("layer", "kinds") and r["label_set"] == name
    }
    return [found[layer] for layer in sorted(found)]


def models(sweeps: dict[str, list[dict]]) -> Figure:
    """Draw each model's best pole per layer, for round depth and for syntax construct."""
    fig = Figure(
        ["Neither Qwen model has a round-depth pole as strong as L2 N676+ of GPT-2"],
        [
            "The best held-out R² in each layer, with layers placed by their share of the "
            "model's depth.",
            "Qwen's best construct poles are stronger than GPT-2's, and its best round-depth "
            "poles are weaker.",
        ],
        "Two plots of the best R² per layer against relative depth, one line per model. For "
        "round depth GPT-2 peaks at 0.28 in layer 2, and Qwen2.5-0.5B and 1.5B stay at or under "
        "0.16 and 0.18. For syntax construct Qwen2.5-0.5B reaches 0.47, 1.5B 0.55 and GPT-2 0.27.",
    )
    panels = columns(fig, 2, 60, fig.body_top + 30, 200, (0, 1), (0, 0.6), gap=40)
    for panel, name in zip(panels, ("round depth", "syntax construct"), strict=True):
        panel.head(name)
        panel.grid_x([0, 0.5, 1], 1)
        panel.grid_y([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 1, labels=panel is panels[0])
        panel.x_title("share of the model's layers")
        for rows, colour in zip(sweeps.values(), MODEL_COLOURS, strict=False):
            values = _layer_best(rows, name)
            where = [i / (len(values) - 1) for i in range(len(values))]
            panel.curve(where, values, f"line s{colour}")
            for x, value in zip(where, values, strict=True):
                fig.dot(panel.x(x), panel.y(value), 2.5, f"f{colour}")
    bottom = panels[0].top + panels[0].height + 62
    fig.legend(
        bottom,
        [(f"f{colour}", model) for model, colour in zip(sweeps, MODEL_COLOURS, strict=False)],
    )
    fig.bottom = bottom
    return fig
