"""Measure and decide an answer claim (issue #4778, ``core/answer_records.py``).

:func:`review` runs the claim's route on the model. It mean-ablates the route, keeps only
the route live, and patches the route from the corrupted prompt. It does the same for
matched random sets (:func:`~openinterpretability.explain_coverage.matched_draw`). Then it
checks each counterfactual prompt. The baseline for ablation is each unit's mean on the
corpus, as ``explain`` uses (:func:`~openinterpretability.explain.unit_means`).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import torch

from openinterpretability.core.answer_records import (
    AnswerClaim,
    AnswerMeasures,
    AnswerPolicy,
    AnswerProbes,
    AnswerRecord,
    AnswerVerdict,
    decide,
    parse_unit,
)
from openinterpretability.explain import mean_ablation
from openinterpretability.explain_coverage import _log_prob, _share, keep_hooks, matched_draw
from openinterpretability.explain_groups import HEAD


def answer_token(runner, answer: str) -> int:
    """The answer's token id. The answer must be one token."""
    ids = runner.tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"the answer {answer!r} is {len(ids)} tokens, not one")
    return ids[0]


def check_claim(tokenizer, claim: AnswerClaim) -> list[str]:
    """What is wrong with ``claim`` before any measure: token counts and unit labels."""
    problems = []
    if len(tokenizer.encode(claim.answer, add_special_tokens=False)) != 1:
        problems.append(f"the answer {claim.answer!r} is not one token")
    clean, corrupted = (len(tokenizer.encode(t)) for t in (claim.prompt, claim.corrupted))
    if clean != corrupted:
        problems.append(f"the corrupted prompt is {corrupted} tokens, the prompt {clean}")
    for label in claim.route:
        try:
            parse_unit(label)
        except ValueError as error:
            problems.append(str(error))
    return problems


def every_unit(runner, means) -> list[tuple[int, int, str]]:
    """Every neuron pole and every head of the model, for the matched random draws."""
    units = []
    for layer, mean in sorted(means.neurons.items()):
        poles = ("positive", "negative")
        units += [(layer, n, pole) for n in range(mean.shape[0]) for pole in poles]
        if means.heads:
            units += [(layer, h, HEAD) for h in range(runner.n_heads)]
    return units


def _sites(units) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """The neurons and the heads of ``units``, by layer. A pole patches its whole neuron."""
    neurons: dict[int, list[int]] = {}
    heads: dict[int, list[int]] = {}
    for layer, index, pole in units:
        (heads if pole == HEAD else neurons).setdefault(layer, []).append(index)
    return neurons, heads


@torch.no_grad()
def patch_hooks(runner, corrupted: str, units):
    """Hooks that set ``units`` to their values on ``corrupted``, past the first token."""
    neurons, heads = _sites(units)
    seen_n: dict[int, torch.Tensor] = {}
    seen_h: dict[int, torch.Tensor] = {}

    def keep(store, layer):
        def hook(act):
            store[layer] = act.detach().clone()

        return hook

    with runner.hooks({la: keep(seen_n, la) for la in neurons}, None,
                      {la: keep(seen_h, la) for la in heads}):  # fmt: skip
        runner.forward(runner.encode(corrupted))

    def set_neurons(layer):
        def hook(act):
            act = act.clone()
            for n in neurons[layer]:
                act[..., 1:, n] = seen_n[layer][..., 1:, n].to(act)
            return act

        return hook

    def set_heads(layer):
        def hook(z):
            z, d = z.clone(), runner.head_dim(layer)
            for h in heads[layer]:
                cols = slice(h * d, (h + 1) * d)
                z[..., 1:, cols] = seen_h[layer][..., 1:, cols].to(z)
            return z

        return hook

    return {la: set_neurons(la) for la in neurons}, {la: set_heads(la) for la in heads}


@torch.no_grad()
def top_token(runner, prompt: str) -> int:
    """The model's top token after ``prompt``."""
    return int(runner.forward(runner.encode(prompt))[0, -1].argmax())


def _came_out(runner, token: int, prompts, tops: dict[str, str]) -> dict[str, bool]:
    """Each prompt → whether the answer stayed or changed as it predicts."""
    out = {}
    for c in prompts:
        top = top_token(runner, c.prompt)
        tops[c.prompt] = runner.tokenizer.decode([top])
        out[c.prompt] = (top == token) == (c.expect == "kept")
    return out


def measure(runner, means, claim: AnswerClaim, probes: AnswerProbes, policy: AnswerPolicy,
            seed: int = 0) -> AnswerMeasures:  # fmt: skip
    """Every measure of the module docstring, for ``claim`` on ``probes``."""
    token, position = answer_token(runner, claim.answer), claim.position
    route = [parse_unit(u) for u in claim.route]

    def lp(prompt=claim.prompt, hooks=None) -> float:
        return _log_prob(runner, prompt, token, position, hooks)

    clean, ablated = lp(), lp(hooks=keep_hooks(runner, means, []))
    corrupt = lp(claim.corrupted)
    gap = clean - corrupt

    def cut(units) -> float:
        return clean - lp(hooks=mean_ablation(units, means, runner))

    def patched(units) -> float | None:
        moved = clean - lp(hooks=patch_hooks(runner, claim.corrupted, units))
        return moved / gap if abs(gap) > 1e-9 else None

    rng, units = random.Random(seed), every_unit(runner, means)
    nulls = [matched_draw(rng, units, route) for _ in range(policy.null_sets)]
    tops: dict[str, str] = {}
    return AnswerMeasures(
        clean=clean, all_cut=clean - ablated, route_cut=cut(route),
        null_cuts=[cut(n) for n in nulls],
        sufficiency=_share(clean, ablated, lp(hooks=keep_hooks(runner, means, route))),
        corrupted_gap=gap, patching=patched(route), null_patching=[patched(n) for n in nulls],
        controls=_came_out(runner, token, claim.counterfactuals, tops),
        probes=_came_out(runner, token, probes.prompts, tops), top_tokens=tops,
    )  # fmt: skip


def review(runner, means, claim: AnswerClaim, probes: AnswerProbes,
           policy: AnswerPolicy | None = None, seed: int = 0,
           remeasures: str | None = None) -> AnswerRecord:  # fmt: skip
    """Measure ``claim`` on ``probes``, decide it, and return the record.

    The record holds the route's reading on every prompt too (#4917). ``remeasures`` names
    an earlier record of the same claim and probes that this one measures again.
    """
    from openinterpretability.answer_readings import read_all

    policy = policy or AnswerPolicy()
    problems = check_claim(runner.tokenizer, claim)
    if problems:
        raise ValueError("; ".join(problems))
    measured = measure(runner, means, claim, probes, policy, seed)
    token = answer_token(runner, claim.answer)
    readings = read_all(runner, means, claim, probes, token)
    measured = measured.model_copy(update={"readings": readings})
    status, gates = decide(measured, policy)
    verdict = AnswerVerdict(status=status, gates=gates, policy=policy, measures=measured,
                            reviewer_agent_model=probes.reviewer_agent_model,
                            timestamp=datetime.now(UTC))  # fmt: skip
    return AnswerRecord(claim=claim, probes=probes, verdict=verdict, remeasures=remeasures)
