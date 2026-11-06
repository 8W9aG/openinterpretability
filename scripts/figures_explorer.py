"""The data of the interactive page in papers/below-the-dip/, from the two papers' results."""

import json
from statistics import mean

from figures_default_on import _conditions, _part, _rest_rows
from figures_why_rest import LONG_STEPS, _rest, _run

from openinterp.builder.pii import redact_text
from openinterp.sensitive import is_offensive

QWEN = "Qwen/Qwen2.5-0.5B"
CONDITIONS = {
    "clamp": "clamp_negative",
    "zero": "zero_negative",
    "mean": "mean",
    "null": "null_negative",
}


def _tail(row: dict, real: bool) -> dict:
    """Return one low tail for the quiz. The site must not show personal data or masked words."""
    tokens = [
        {"token": redact_text(t["token"]), "rate": round(t["rate"], 4)}
        for t in row["top_tokens"]
        if not is_offensive(t["token"])
    ]
    return {"real": real, "tokens": tokens}


def _costs(rows: list[dict]) -> dict:
    """Return the measured cost of each intervention, for both models."""

    def by_condition(model: str) -> dict:
        return {
            r["condition"]: {"nats": r["added_ce"], "same": r["same_top"]}
            for r in _part(rows, "all", model)
        }

    gpt2, qwen = by_condition("openai-community/gpt2"), by_condition(QWEN)
    return {k: {"gpt2": gpt2[v], "qwen": qwen[v]} for k, v in CONDITIONS.items()}


def data_js(poles: list[dict], resting: list[dict], dip: float) -> str:
    """Return the script that gives the page its data. It is stable text."""
    trained, fresh = _rest_rows(poles)
    population = _part(poles, "band_population")[0]
    neuron = next(r for r in _part(poles, "band_neuron") if r["neuron"] == 1046)
    tails = [_tail(r, False) for r in _part(poles, "band_random_direction")]
    tails.insert(3, _tail(neuron, True))
    long_runs = [
        [
            [
                _rest(_run(resting, condition, step=step, part="training_long")["layers"][layer])
                for step in [0, *LONG_STEPS]
            ]
            for layer in range(4)
        ]
        for condition in ("gelu", "rest_learned_fast")
    ]
    data = {
        "dip": dip,
        "init": [
            {
                "h": mean(s["median_h"] for s in fresh[r["layer"]]),
                "sd": mean(s["median_h_sd"] for s in fresh[r["layer"]]),
            }
            for r in trained
        ],
        "trained": [{"h": r["median_h"], "sd": r["median_h_sd"]} for r in trained],
        "long": long_runs,
        "illustration": _conditions(),
        "costs": _costs(poles),
        "tails": tails,
        "reliability": {
            "neuron": neuron["reliability"],
            "random": population["random_direction_at_studied_share"]["q50"],
        },
    }
    return f"window.EXPLORER_DATA = {json.dumps(data, sort_keys=True)};\n"
