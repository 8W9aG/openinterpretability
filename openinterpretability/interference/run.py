"""Run the interference report over a set of sides, and collect its rows (#5230).

``sides`` maps a name to a side (:mod:`.floors`): ``base``, ``changed``, and the floors,
``floor:self`` and ``floor:random``. Every measure runs on each non-base side against the
base, and :mod:`.report` turns the readings into rows, a footprint and a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from openinterpretability.interference import behaviour, keep, report, routes, units


def _columns(side, layer: int) -> torch.Tensor:
    """The side's output columns. A gated weight edit is read with its gates forced open."""
    from openinterpretability.patches.schema import PatchError

    try:
        return units.output_columns(side, layer)
    except PatchError:
        return units.output_columns(side.forced_open(), layer)


def _unembedding(side) -> torch.Tensor:
    """The side's unembedding, ``[vocab, d_model]``, as float32 on the CPU."""
    with side.entered() as runner:
        return runner.model.get_output_embeddings().weight.detach().float().cpu()


def _logit_cosine(w_a: torch.Tensor, w_b: torch.Tensor, col_a, col_b) -> float:
    """The cosine between two output columns, each read through its side's unembedding."""
    a, b = w_a @ col_a, w_b @ col_b
    return round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 6)


def _output_side(sides: dict, n_layers: int, records: list) -> tuple[dict, dict]:
    """Each side's per-neuron output change by layer, and each VERIFIED pole's logit cosine."""
    change: dict[str, dict[int, torch.Tensor]] = {k: {} for k in sides if k != "base"}
    cosine: dict[str, float] = {}
    wanted = {}
    for r in records:
        wanted.setdefault(r.submission.identifier.layer, []).append(r.submission.identifier)
    unembed = {k: _unembedding(sides[k]) for k in ("base", "changed")} if wanted else {}
    for layer in range(n_layers):
        report.say(f"output columns of layer {layer + 1}/{n_layers}")
        base = _columns(sides["base"], layer)
        for name in change:
            cols = _columns(sides[name], layer)
            change[name][layer] = units.column_change(base, cols)
            if name == "changed":
                for ident in wanted.get(layer, []):
                    j = ident.neuron_idx
                    cosine[ident.slug] = _logit_cosine(unembed["base"], unembed["changed"],
                                                       base[:, j], cols[:, j])  # fmt: skip
    return change, cosine


def unit_rows(readings: dict, stats: dict, cosine: dict, change: dict, records: list) -> list:
    """One row per VERIFIED pole: its keep readings, flips, drift and output change."""
    rows = []
    for record in records:
        ident = record.submission.identifier
        flips = report._flips(readings[ident.slug])
        r = stats["changed"].pearson(ident.layer)[ident.neuron_idx]
        moved = change["changed"][ident.layer][ident.neuron_idx]
        rows.append({"table": "unit", "slug": ident.slug, "layer": ident.layer,
                     "neuron": ident.neuron_idx, "pole": ident.pole,
                     "readings": readings[ident.slug], **flips,
                     "pearson": None if torch.isnan(r) else round(float(r), 6),
                     "logit_cosine": cosine.get(ident.slug),
                     "output_change": round(float(moved), 6)})  # fmt: skip
    return rows


@dataclass
class Inputs:
    """What the report reads: the registry's units and routes, and the corpus sample."""

    records: list  # the VERIFIED poles' standing records
    batches: list  # the corpus sample, as batches of token windows
    prompts: list[dict]  # the explained prompts (behaviour.explained_prompts)
    n_layers: int
    known: set  # (layer, neuron) with an experiment in the registry
    direct: set = field(default_factory=set)  # neurons a patch edits directly
    answers: list = field(default_factory=list)  # VERIFIED answer records (routes.py)
    explanations: list = field(default_factory=list)  # explanation records (routes.py)
    texts: list[str] = field(default_factory=list)  # the corpus texts, for the means
    means_tokens: int = 20_000
    key: str = ""  # the base model's name, for each side's means cache


def _routes(sides: dict, given: Inputs) -> list[dict]:
    """The route rows: answer routes and explanations, each on every side."""
    if not (given.answers or given.explanations):
        return []
    means = routes.side_means(sides, given.key, given.texts, given.means_tokens)
    return (routes.answer_routes(sides, given.answers, means)
            + routes.explanation_routes(sides, given.explanations, means))  # fmt: skip


def run(sides: dict, given: Inputs) -> list[dict]:
    """Every row of the report: the summary first, then units, routes, answers, footprint."""
    readings = report._keep(sides, given.records, given.batches)
    others = {k: v for k, v in sides.items() if k != "base"}
    stats = {k: units.activation_stats(sides["base"], s, given.batches, given.n_layers)
             for k, s in others.items()}  # fmt: skip
    change, cosine = _output_side(sides, given.n_layers, given.records)
    rows = unit_rows(readings, stats, cosine, change, given.records)
    report.say(f"routes: {len(given.answers)} answers, {len(given.explanations)} explanations")
    route_rows = _routes(sides, given)
    feet = report.footprint(stats, change, given.n_layers, given.direct, given.known)
    report.say("behaviour: answers and token KL")
    answers = {k: behaviour.answer_probabilities(s, given.prompts) for k, s in sides.items()}
    kl = {k: round(behaviour.token_kl(sides["base"], s, given.batches), 8)
          for k, s in others.items()}  # fmt: skip
    changed = sides["changed"]
    local = (
        behaviour.gated_kl(sides["base"], changed, given.batches)
        if getattr(changed, "gated", False)
        else {}
    )
    flipped = [r["slug"] for r in rows if r["counts"]]
    broken = [r.get("id") or r["prompt"] for r in route_rows if r["counts"]]
    unstudied = [f for f in feet if not f["studied"]]
    summary = {"table": "summary", "verdict": report.verdict(flipped + broken, unstudied),
               "flipped": flipped, "routes_broken": broken, "routes": len(route_rows),
               "footprint": len(feet), "unstudied": len(unstudied), "token_kl": kl,
               "gated_kl": local,
               "coverage": report.coverage(stats["changed"], given.records, given.n_layers),
               "keeps": sum(r["readings"]["changed"]["passes"] for r in rows),
               "reproduced_on_base": sum(r["readings"]["base"]["passes"] for r in rows),
               "poles_tested": len(rows),
               "traffic_map": "not yet added: #5231 is open"}  # fmt: skip
    prompt_rows = [{"table": "answer", **p, **{k: v[i] for k, v in answers.items()}}
                   for i, p in enumerate(given.prompts)]  # fmt: skip
    return [summary, *rows, *route_rows, *prompt_rows, *feet]


FOOTPRINT_ROWS = 1_000  # past this, the footprint is written per layer, as counts


def compact(rows: list[dict]) -> list[dict]:
    """``rows`` with a large footprint folded into one row per layer.

    A change such as instruction tuning moves nearly every neuron, which would be a row
    each. Past :data:`FOOTPRINT_ROWS`, each layer gets a count of its moved neurons, and
    only the studied neurons keep a row of their own. The summary keeps the totals.
    """
    feet = [r for r in rows if r["table"] == "footprint"]
    if len(feet) <= FOOTPRINT_ROWS:
        return rows
    layers: dict[int, dict] = {}
    for f in feet:
        layer = layers.setdefault(f["layer"], {"table": "footprint_layer", "layer": f["layer"],
                                               "moved": 0, "studied": 0, "edited": 0})  # fmt: skip
        layer["moved"] += 1
        layer["studied"] += f["studied"]
        layer["edited"] += f["edited"]
    kept = [r for r in rows if r["table"] != "footprint" or r["studied"]]
    return kept + [layers[k] for k in sorted(layers)]


__all__ = ["compact", "keep", "run"]
