"""Run the interference report over a set of sides, and collect its rows (#5230).

``sides`` maps a name to a side (:mod:`.floors`): ``base``, ``changed``, and the floors,
``floor:self`` and ``floor:random``. Every measure runs on each non-base side against the
base, and :mod:`.report` turns the readings into rows, a footprint and a verdict.
"""

from __future__ import annotations

import torch

from openinterpretability.interference import behaviour, keep, report, units


def _columns(side, layer: int) -> torch.Tensor:
    """The side's output columns. A gated weight edit is read with its gates forced open."""
    from openinterpretability.patches.schema import PatchError

    try:
        return units.output_columns(side, layer)
    except PatchError:
        return units.output_columns(side.forced_open(), layer)


def _logit_cosine(side_a, side_b, col_a, col_b) -> float:
    """The cosine between two output columns read through each side's unembedding."""
    with side_a.entered() as runner:
        w_a = runner.model.get_output_embeddings().weight.detach().float().cpu()
    with side_b.entered() as runner:
        w_b = runner.model.get_output_embeddings().weight.detach().float().cpu()
    a, b = w_a @ col_a, w_b @ col_b
    return round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 6)


def _output_side(sides: dict, n_layers: int, records: list) -> tuple[dict, dict]:
    """Each side's per-neuron output change by layer, and each VERIFIED pole's logit cosine."""
    change: dict[str, dict[int, torch.Tensor]] = {k: {} for k in sides if k != "base"}
    cosine: dict[str, float] = {}
    wanted = {}
    for r in records:
        wanted.setdefault(r.submission.identifier.layer, []).append(r.submission.identifier)
    for layer in range(n_layers):
        base = _columns(sides["base"], layer)
        for name in change:
            cols = _columns(sides[name], layer)
            change[name][layer] = units.column_change(base, cols)
            if name == "changed":
                for ident in wanted.get(layer, []):
                    j = ident.neuron_idx
                    cosine[ident.slug] = _logit_cosine(sides["base"], sides[name],
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


def run(sides: dict, records: list, batches, prompts: list[dict], n_layers: int,
        known: set, direct: set) -> list[dict]:  # fmt: skip
    """Every row of the report: units, footprint, behaviour and the summary first."""
    readings = report._keep(sides, records, batches)
    others = {k: v for k, v in sides.items() if k != "base"}
    stats = {k: units.activation_stats(sides["base"], s, batches, n_layers)
             for k, s in others.items()}  # fmt: skip
    change, cosine = _output_side(sides, n_layers, records)
    rows = unit_rows(readings, stats, cosine, change, records)
    feet = report.footprint(stats, change, n_layers, direct, known)
    answers = {k: behaviour.answer_probabilities(s, prompts) for k, s in sides.items()}
    kl = {k: round(behaviour.token_kl(sides["base"], s, batches), 8) for k, s in others.items()}
    flipped = [r["slug"] for r in rows if r["counts"]]
    unstudied = [f for f in feet if not f["studied"]]
    summary = {"table": "summary", "verdict": report.verdict(flipped, unstudied),
               "flipped": flipped, "footprint": len(feet), "unstudied": len(unstudied),
               "token_kl": kl, "coverage": report.coverage(stats["changed"], records, n_layers),
               "keeps": sum(r["readings"]["changed"]["passes"] for r in rows),
               "reproduced_on_base": sum(r["readings"]["base"]["passes"] for r in rows),
               "poles_tested": len(rows),
               "traffic_map": "not yet added: #5231 is open"}  # fmt: skip
    prompt_rows = [{"table": "answer", **p, **{k: v[i] for k, v in answers.items()}}
                   for i, p in enumerate(prompts)]  # fmt: skip
    return [summary, *rows, *prompt_rows, *feet]


__all__ = ["keep", "run"]
