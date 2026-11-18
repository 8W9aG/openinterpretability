"""The route on each prompt of an answer experiment, for its page's figures (issue #4917).

For each prompt, :func:`read` measures what the claim's route does there:

- each route unit's strength at each token. A neuron pole's strength is its activation
  past zero on its side. A head's strength is ``|W_O (z − z̄)|``, as ``explain`` measures it.
- each unit's first-order attribution to the claim's answer, and the edges between the
  route's units (:func:`openinterpretability.explain.paths`).
- the route's share: the cut from mean-ablating the route, over the cut from every unit.
- "patched back": the answer's log-probability with the original prompt's route values
  put into this prompt's run. It needs the same token count.

The readings are reported, not gated (the maintainer decided on 2026-11-18).
"""

from __future__ import annotations

import torch

from openinterpretability.core.answer_records import (
    AnswerClaim,
    LayerView,
    RouteReading,
    parse_unit,
)
from openinterpretability.explain import (
    Graph,
    PoleAttribution,
    _captured,
    mean_ablation,
    paths,
)
from openinterpretability.explain_coverage import _log_prob, keep_hooks
from openinterpretability.explain_groups import HEAD


def _strength(runner, means, prompt: str, route) -> dict[tuple, list[float]]:
    """Each route unit's strength at each token of ``prompt``; the first token is 0."""
    with torch.no_grad():
        _, neurons, heads = _captured(runner, runner.encode(prompt), bool(means.heads))
    out = {}
    for layer, index, pole in route:
        if pole == HEAD:
            d = runner.head_dim(layer)
            cols = slice(index * d, (index + 1) * d)
            offset = heads[layer][0, :, cols].float().cpu() - means.heads[layer][cols]
            w_o = runner.head_output_weights(layer, index).float().cpu()
            values = (offset @ w_o.T).norm(dim=-1)
        else:
            act = neurons[layer][0, :, index].float().cpu()
            values = act.clamp(min=0) if pole == "positive" else (-act).clamp(min=0)
        values[0] = 0.0  # the first token is an attention sink, left out as in explain
        out[(layer, index, pole)] = [round(float(v), 4) for v in values]
    return out


def read(runner, means, claim: AnswerClaim, token: int, prompt: str) -> RouteReading:
    """The route on ``prompt`` (module docstring)."""
    route = [parse_unit(u) for u in claim.route]
    label = dict(zip(route, claim.route, strict=True))

    def lp(p=prompt, hooks=None) -> float:
        return _log_prob(runner, p, token, -1, hooks)

    clean, ablated = lp(), lp(hooks=keep_hooks(runner, means, []))
    cut = clean - lp(hooks=mean_ablation(route, means, runner))
    ids = runner.tokenizer.encode(prompt)
    scores, edges = {}, []
    if len(ids) > 1:  # attribution leaves out the first token, so one token has none
        graph = Graph(runner, prompt, means, token, -1)
        scores = graph.scores(graph.grads)
        pole = {PoleAttribution(*u, 0.0, 0.0).label: label[u] for u in route}
        edges = [e | {"source": pole[e["source"]], "target": pole[e["target"]]}
                 for e in paths(graph, route, len(route))]  # fmt: skip
        del graph
    with torch.no_grad():
        top = int(runner.forward(runner.encode(prompt))[0, -1].argmax())
    return RouteReading(
        tokens=[runner.tokenizer.decode([i]) for i in ids],
        top=runner.tokenizer.decode([top]),
        answer_log_prob=clean,
        route_share=cut / (clean - ablated) if clean - ablated > 1e-9 else None,
        strength={label[u]: v for u, v in _strength(runner, means, prompt, route).items()},
        attribution={label[u]: round(scores.get(u, (0.0, 0.0))[0], 4) for u in route},
        edges=[e | {"weight": round(e["weight"], 4)} for e in edges],
        patched_log_prob=_patched(runner, claim, route, prompt, lp) if prompt != claim.prompt
        else None,
    )  # fmt: skip


def _patched(runner, claim, route, prompt: str, lp) -> float | None:
    """The answer's log-probability here, with the original prompt's route values patched in."""
    from openinterpretability.answer_review import patch_hooks

    tok = runner.tokenizer
    if len(tok.encode(prompt)) != len(tok.encode(claim.prompt)):
        return None
    return lp(hooks=patch_hooks(runner, claim.prompt, route))


def read_all(runner, means, claim: AnswerClaim, probes, token: int) -> dict[str, RouteReading]:
    """The readings of the original prompt, the proposer's and the reviewer's prompts."""
    from openinterpretability.answer_lens import measures

    prompts = list(dict.fromkeys([claim.prompt, *(c.prompt for c in claim.counterfactuals),
                                  *(c.prompt for c in probes.prompts)]))  # fmt: skip
    route = [parse_unit(u) for u in claim.route]
    layers = measures(runner, means, prompts, token, route, claim.route)  # on shared axes
    out = {}
    for p in prompts:
        v = layers[p]
        out[p] = read(runner, means, claim, token, p).model_copy(update={
            "lens": v["clean"]["lens"], "spectrum": v["clean"]["spectrum"],
            "route_off": LayerView(**v["route_off"]),
            "unit_off": {u: LayerView(**x) for u, x in v.get("unit_off", {}).items()},
        })  # fmt: skip
    return out
