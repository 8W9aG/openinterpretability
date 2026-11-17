"""Prompt coverage: how much of one answer the explained units bring back (issue #3742).

The registry's model coverage (:mod:`openinterpretability.coverage`) is the share of a
layer's loss rise that the VERIFIED poles bring back. Prompt coverage has the same form,
on the answer to one prompt:

    coverage = (kept − ablated) / (clean − ablated)

Each term is the answer's log-probability at its position. *clean* is the model as it is.
*ablated* mean-ablates every neuron pole and every attention head, past the first token.
*kept* does the same, except that the explained units keep their real values. A unit is
explained when it is VERIFIED, or in a VERIFIED superposition or circuit
(:func:`openinterpretability.explain._ranked`).

The same number of units, drawn at random, gives a baseline. It is the mean over a few
draws. Explained units that bring back no more than random ones explain no more of this
answer than any units would. The measure is causal, not first-order, so the long tail of
small attributions doesn't inflate it. It is reported, not gated.

The credited coverage is capped at 1 (#4507). On some prompts, mean-ablating the other
units raises the answer above the clean model, because they held it down. Then *kept*
exceeds *clean* and the raw ratio exceeds 1: 536% on one Pythia-70m prompt. The units still
bring the whole answer back, so they get full credit, and no more. The raw value is kept
and reported as an overshoot, so that a reader sees it.

:func:`add_gains` measures how much coverage each planned experiment would add, if its
units were explained too. That ranks the experiments by what they would explain of this
answer, not by first-order attribution. Each gain costs one forward pass, so it is measured
for a pool of the top planned experiments and for every superposition and circuit.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import torch

from openinterpretability.explain_groups import HEAD

DRAWS = 4  # random sets for the baseline


def keep_hooks(runner, means, keep: Iterable[tuple[int, int, str]]):
    """``(neuron hooks, head hooks)`` that mean-ablate every unit but ``keep``, past token 0."""
    poles: dict[int, list[tuple[int, str]]] = {}
    heads: dict[int, set[int]] = {}
    for layer, index, pole in keep:
        if pole == HEAD:
            heads.setdefault(layer, set()).add(index)
        else:
            poles.setdefault(layer, []).append((index, pole))

    def neurons_at(layer: int):
        def intervene(act: torch.Tensor) -> torch.Tensor:
            out = means.neurons[layer].to(act).expand_as(act).clone()
            for n, pole in poles.get(layer, []):
                a = act[..., n]
                active = a > 0 if pole == "positive" else a < 0
                out[..., n] = torch.where(active, a, out[..., n])
            out[..., 0, :] = act[..., 0, :]  # the first token stays, as in the attribution
            return out

        return intervene

    def heads_at(layer: int):
        def intervene(z: torch.Tensor) -> torch.Tensor:
            z, d = z.clone(), runner.head_dim(layer)
            mean = means.heads[layer].to(z)
            for h in range(runner.n_heads):
                if h not in heads.get(layer, set()):
                    cols = slice(h * d, (h + 1) * d)
                    z[..., 1:, cols] = mean[cols]
            return z

        return intervene

    layers = range(runner.n_layers)
    return {la: neurons_at(la) for la in layers}, (
        {la: heads_at(la) for la in layers} if means.heads else {}
    )


@torch.no_grad()
def _log_prob(runner, prompt: str, token: int, position: int, hooks=None) -> float:
    neurons, heads = hooks or ({}, {})
    with runner.hooks(neurons, None, heads):
        logits = runner.forward(runner.encode(prompt))[0, position].float()
    return float(torch.log_softmax(logits, -1)[token])


def _raw(clean: float, ablated: float, kept: float) -> float | None:
    """(kept − ablated) / (clean − ablated), or None when ablating changes nothing."""
    span = clean - ablated
    return (kept - ablated) / span if abs(span) > 1e-9 else None


def _share(clean: float, ablated: float, kept: float) -> float | None:
    """The credited coverage: :func:`_raw`, at most 1 (module docstring)."""
    raw = _raw(clean, ablated, kept)
    return None if raw is None else min(raw, 1.0)


def prompt_coverage(runner, prompt, means, token, position, order, *, draws=DRAWS, seed=0):
    """The explained units' coverage of the answer, and the random baseline (module docstring).

    ``order`` is ``Explanation.order``: every unit with its status and whether it is explained.
    """
    keep = [unit for unit, _, _, explained in order if explained]
    clean = _log_prob(runner, prompt, token, position)
    ablated = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, []))
    kept = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, keep))
    rng, units = random.Random(seed), [unit for unit, *_ in order]
    randoms = []
    for _ in range(draws if keep else 0):
        drawn = rng.sample(units, len(keep))
        value = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, drawn))
        randoms.append(_share(clean, ablated, value))
    baseline = [r for r in randoms if r is not None]
    raw = _raw(clean, ablated, kept)
    return {
        "units": len(keep), "clean": clean, "ablated": ablated, "kept": kept,
        "coverage": _share(clean, ablated, kept), "raw": raw,
        "overshoot": raw is not None and raw > 1.0,  # the kept units push past the clean run
        "random": sum(baseline) / len(baseline) if baseline else None,
        "random_draws": baseline,  # each draw's coverage, for the page's figure (#4479)
    }  # fmt: skip


POOL = 40  # planned unit experiments whose gain is measured, in attribution order


def add_gains(runner, prompt, means, token, position, order, planned, *, pool=POOL):
    """Set ``gain`` on the first ``pool`` unit experiments and every group experiment.

    A gain is the coverage with the experiment's units kept too, minus the coverage now.
    The experiments with a gain are put first, highest gain first, in each list.
    """
    keep = [unit for unit, _, _, explained in order if explained]
    clean = _log_prob(runner, prompt, token, position)
    ablated = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, []))
    now = _share(clean, ablated, _log_prob(runner, prompt, token, position,
                                           keep_hooks(runner, means, keep)))  # fmt: skip
    if now is None:
        return
    from openinterpretability.harness.progress import Progress

    measured = [*planned.experiments[:pool], *planned.groups]
    progress = Progress("experiments measured for their gain", len(measured))
    for e in measured:
        hooks = keep_hooks(runner, means, [*keep, *e.units])
        after = _share(clean, ablated, _log_prob(runner, prompt, token, position, hooks))
        e.gain = None if after is None else after - now
        progress.step()

    def by_gain(e):
        return (e.gain is None, -(e.gain or 0.0))

    planned.experiments.sort(key=by_gain)  # stable: unmeasured ones keep attribution order
    planned.groups.sort(key=by_gain)


GREEDY_POOL = 120  # candidate unit experiments for the greedy plan, in attribution order
GREEDY_STEPS = 60  # the most experiments the greedy plan picks


def coverage_plan(runner, prompt, means, token, position, order, planned, target: float,
                  *, pool: int = GREEDY_POOL, steps: int = GREEDY_STEPS) -> dict:  # fmt: skip
    """The fewest planned unit experiments that bring the coverage to ``target`` (#4476, #4558).

    The target is causal coverage, not a share of |attribution|. Coverage doesn't add up:
    units push the answer both ways, so a big set can bring back less than a small one.
    So the plan is greedy. Each step keeps the explained units and the picks so far. It tries
    each of the first ``pool`` candidates, and adds the one that raises coverage most. Its
    ``gain`` becomes that marginal gain. It stops at the target, when no
    candidate helps, or after ``steps`` picks. ``planned.experiments`` is reordered with the
    picks first, in pick order. ``reach`` is the best coverage found: the picks', or with
    every planned unit kept, whichever is higher.
    """
    explained = [unit for unit, _, _, done in order if done]
    clean = _log_prob(runner, prompt, token, position)
    ablated = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, []))

    def raw(units: list) -> float:
        kept = _log_prob(runner, prompt, token, position, keep_hooks(runner, means, units))
        found = _raw(clean, ablated, kept)
        return -1.0 if found is None else found

    every = raw(explained + [u for e in planned.experiments for u in e.units])
    now = min(raw(explained), 1.0)
    chosen, units, cur = [], list(explained), now
    candidates = list(planned.experiments[:pool])
    while cur < target and candidates and len(chosen) < steps:
        scored = [(min(raw(units + e.units), 1.0), e) for e in candidates]
        best, pick = max(scored, key=lambda s: s[0])
        if best <= cur:
            break  # no candidate raises coverage
        pick.gain, cur = best - cur, best
        chosen.append(pick)
        candidates.remove(pick)
        units += pick.units
    rest = [e for e in planned.experiments if e not in chosen]
    planned.experiments[:] = chosen + rest
    reach = max(cur, min(every, 1.0))
    out = {"target": target, "now": now, "reach": reach, "overshoot": every > 1.0,
           "chosen": [e.target for e in chosen]}  # fmt: skip
    return out | {"experiments": len(chosen) if cur >= target else None}
