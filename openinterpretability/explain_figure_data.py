"""The measured data of an explanation page's figures, taken when the record is written (#4479).

The site draws the figures at build time with no model, so ``explain`` stores what they
need in the record (:class:`~openinterpretability.core.explanation_records.ExplanationFigures`):

- the prompt's tokens, the top units with their signed attribution, and the traced edges;
- the sum of every unit's signed attribution, for the waterfall's last bar;
- the explained units' coverage and each random draw's, for the coverage figure;
- each of the top :data:`READ` units' strength at each token, for the reading figure;
- the answer's top :data:`TOP_TOKENS` tokens, with the explained units in place and then
  mean-ablated, for the ablation figure;
- the superpositions, circuits and hubs that hold top units, for the groups figure.
"""

from __future__ import annotations

import torch

READ = 8  # top units whose strength along the prompt is stored
TOP_TOKENS = 5  # the answer's top tokens, before and after ablation
GROUPS = 10  # groups whose members are stored


@torch.no_grad()
def figure_data(runner, means, result, coverage: dict, planned, groups=()):
    """The :class:`ExplanationFigures` of ``result`` (module docstring)."""
    from openinterpretability.core.explanation_records import ExplanationFigures, FigureUnit
    from openinterpretability.explain import HEAD

    blocked = {b["target"] for b in planned.blocked_units}
    explained = {unit for unit, _, _, done in result.order if done}
    units = [FigureUnit(label=u.label, layer=u.layer, kind="head" if u.pole == HEAD else "pole",
                        attribution=round(u.attribution, 4), status=u.status,
                        explained=u.key in explained, blocked=u.label in blocked)
             for u in result.poles]  # fmt: skip
    encoded = runner.encode(result.prompt)
    ids = encoded["input_ids"][0].tolist()
    return ExplanationFigures(
        tokens=[runner.tokenizer.decode([i]) for i in ids], logit=round(result.logit, 4),
        units=units,
        net=round(result.net, 4), edges=[_edge(e) for e in result.edges],
        coverage=coverage.get("coverage"),
        random_draws=[round(d, 4) for d in coverage.get("random_draws", [])],
        reading=_reading(runner, means, result, encoded),
        ablation=_ablation(runner, means, result, sorted(explained)),
        groups=_groups(result, groups),
    )  # fmt: skip


def _edge(e: dict) -> dict:
    return {"source": e["source"], "target": e["target"], "weight": round(e["weight"], 4)}


def _reading(runner, means, result, encoded) -> dict[str, list[float]]:
    """Each top unit's strength at each token.

    A pole's strength is its magnitude. A head's is its output's distance from its mean.
    """
    from openinterpretability.explain import HEAD, _captured, has_heads

    _, neurons, outputs = _captured(runner, encoded, has_heads(runner))
    out = {}
    for u in result.poles[:READ]:
        if u.pole == HEAD:  # the outputs are flat: each head is a slice of the columns
            out_layer = outputs[u.layer][0].float()
            width = out_layer.shape[-1] // runner.n_heads
            cols = slice(u.neuron * width, (u.neuron + 1) * width)
            mean = means.heads.get(u.layer)
            centre = mean[cols].float() if mean is not None else 0.0
            values = (out_layer[:, cols] - centre).norm(dim=-1)
        else:
            act = neurons[u.layer][0, :, u.neuron].float()
            values = act.clamp(min=0) if u.pole == "positive" else (-act).clamp(min=0)
        out[u.label] = [round(float(v), 4) for v in values.cpu()]
    return out


def _ablation(runner, means, result, explained: list) -> list[dict]:
    """The answer's top tokens' probabilities, with the explained units, then mean-ablated."""
    from openinterpretability.explain import logits_with, mean_ablation

    before = torch.softmax(logits_with(runner, result.prompt, {}, result.position), -1)
    after = before
    if explained:
        ablated = mean_ablation(explained, means, runner)
        after = torch.softmax(logits_with(runner, result.prompt, ablated, result.position), -1)
    top = torch.topk(before, TOP_TOKENS).indices.tolist()
    return [{"token": runner.tokenizer.decode([i]), "before": round(float(before[i]), 4),
             "after": round(float(after[i]), 4)} for i in top]  # fmt: skip


def _groups(result, groups) -> list[dict]:
    """The groups that carry the answer, with the top units they hold, and the hubs."""
    from openinterpretability.explain import PoleAttribution

    defs = {(g.kind, g.name): g for g in groups}
    top = {u.key for u in result.poles}
    out = []
    for g in result.groups[:GROUPS]:
        d = defs.get((g.kind, g.name))
        held = [PoleAttribution(*k, 0.0, 0.0).label for k in (d.units if d else []) if k in top]
        if held:
            out.append({"name": g.name, "kind": g.kind, "status": g.status, "members": held})
    for u in result.poles:
        if getattr(u, "hub", None):
            out.append({"name": u.hub["hub"], "kind": "hub", "status": u.status,
                        "members": [u.label]})  # fmt: skip
    return out
