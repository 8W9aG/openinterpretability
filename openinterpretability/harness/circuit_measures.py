"""Measure a circuit on a task: neurons and heads, ablated together (issue #4284, part B).

A circuit's members are neurons and attention heads (``docs/circuits-and-hubs.md``). A
task is a set of clean and corrupted prompt pairs with a metric (:class:`PairTask`). Each
member is ablated to its mean over the task's clean and corrupted prompts, at each
position, as in ``scripts/sva_circuit.py``. Every measure is relative to ablating all
neurons and heads: (m − m(none)) / (m(all) − m(none)).

* :func:`measures` gives *faithfulness* (the share that the members keep alone) and
  *necessity* (the share lost when they are ablated).
* :func:`null_necessity` gives the necessity of random member sets with the same layers
  and kinds, the null that the pilot (#4283) proposed as the gate.
* :func:`attribution` ranks every neuron and head by attribution patching: the clean
  minus corrupted activation, times the metric's gradient on the corrupted prompts. A
  task hypothesis grows its members from a wiring cluster with :func:`grow`.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch

MEMBER = re.compile(r"L(\d+)/([NH])(\d+)")


@dataclass(frozen=True, order=True)
class Member:
    """One neuron (``N``) or attention head (``H``) of a circuit."""

    kind: str
    layer: int
    index: int

    @classmethod
    def parse(cls, text: str) -> Member:
        """From ``L4/N2999``, ``L7/H2`` or a unit page such as ``models/m/L4/N2999.html``."""
        found = MEMBER.search(text)
        if found is None:
            raise ValueError(f"not a neuron or head: {text}")
        return cls(found.group(2), int(found.group(1)), int(found.group(3)))

    def __str__(self) -> str:
        return f"L{self.layer}/{self.kind}{self.index}"


class PairTask:
    """Clean and corrupted prompts of equal length, and the target each one predicts.

    The metric is logit(the clean prompt's target) − logit(the corrupted prompt's) at
    the last position.
    """

    def __init__(self, clean, corrupted, good, bad):
        self.clean, self.corrupted = torch.tensor(clean), torch.tensor(corrupted)
        self.good, self.bad = torch.tensor(good), torch.tensor(bad)
        if self.clean.shape != self.corrupted.shape:
            raise ValueError("a clean prompt and its pair have different token counts")

    def metric(self, logits: torch.Tensor) -> torch.Tensor:
        """The metric of each prompt."""
        last = logits[:, -1].float()
        rows = torch.arange(last.shape[0], device=last.device)
        return last[rows, self.good.to(last.device)] - last[rows, self.bad.to(last.device)]


def _by_layer(members: Iterable[Member], kind: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for m in members:
        if m.kind == kind:
            out.setdefault(m.layer, []).append(m.index)
    return out


def _encode(runner, ids: torch.Tensor) -> dict:
    return {"input_ids": ids.to(runner.device)}


@torch.no_grad()
def task_means(runner, task: PairTask) -> tuple[dict, dict]:
    """``(neurons, heads)``: each layer's mean activation over the task's prompts, by position."""
    neurons: dict[int, torch.Tensor] = {}
    heads: dict[int, torch.Tensor] = {}
    both = torch.cat([task.clean, task.corrupted])
    layers = range(runner.n_layers)

    def keep(store, layer):
        def record(act):
            store[layer] = act.float().mean(0)

        return record

    with runner.hooks({n: keep(neurons, n) for n in layers}, None,
                      {n: keep(heads, n) for n in layers}):  # fmt: skip
        runner.forward(_encode(runner, both))
    return neurons, heads


def _masks(runner, members, kind: str, keep: bool, layer: int) -> torch.Tensor | None:
    """The positions to replace with the mean in one layer, or None for none."""
    head_dim = runner.head_dim(layer) if kind == "H" else 1
    width = runner.n_heads * head_dim if kind == "H" else runner.mlp_width(layer)
    chosen = _by_layer(members, kind).get(layer, [])
    mask = torch.full((width,), keep, dtype=torch.bool)  # True: replace
    for index in chosen:
        mask[index * head_dim : (index + 1) * head_dim] = not keep
    return mask if mask.any() else None


@torch.no_grad()
def ablated(runner, task: PairTask, means, members=(), keep: bool = False) -> float:
    """The mean metric on the clean prompts with members ablated.

    With ``keep``, every neuron and head *but* ``members`` is ablated; otherwise only
    ``members`` are. ``keep`` with no members ablates everything.
    """
    neuron_means, head_means = means
    hooks: dict[str, dict[int, object]] = {"N": {}, "H": {}}
    for kind, store in (("N", neuron_means), ("H", head_means)):
        for layer in range(runner.n_layers):
            mask = _masks(runner, members, kind, keep, layer)
            if mask is None:
                continue
            m, mean = mask.to(runner.device), store[layer]

            def replace(act, m=m, mean=mean):
                return torch.where(m, mean.to(act.dtype), act)

            hooks[kind][layer] = replace
    with runner.hooks(hooks["N"], None, hooks["H"]):
        logits = runner.forward(_encode(runner, task.clean))
    return float(task.metric(logits).mean())


def measures(runner, task: PairTask, means, members) -> dict:
    """Faithfulness and necessity of ``members``, relative to ablating everything."""
    full = ablated(runner, task, means)
    empty = ablated(runner, task, means, keep=True)
    span = full - empty
    if abs(span) < 1e-9:
        raise ValueError("the task's metric doesn't change when every unit is ablated")
    alone = ablated(runner, task, means, members, keep=True)
    without = ablated(runner, task, means, members)
    return {"faithfulness": (alone - empty) / span, "necessity": 1 - (without - empty) / span,
            "full": full, "empty": empty}  # fmt: skip


def random_like(runner, members, rng: random.Random) -> list[Member]:
    """Random members with the same layers and kinds as ``members``."""
    out = []
    for m in members:
        size = runner.n_heads if m.kind == "H" else runner.mlp_width(m.layer)
        out.append(Member(m.kind, m.layer, rng.randrange(size)))
    return out


def null_necessity(runner, task, means, members, sets: int = 64, seed: int = 0) -> list[float]:
    """Necessity of ``sets`` random member sets like ``members``, sorted.

    Each set is a pass over every pair, so a :class:`~.progress.Progress` reports the count.
    """
    from openinterpretability.harness.progress import Progress

    rng, progress, found = random.Random(seed), Progress("random member sets", sets), []
    for _ in range(sets):
        found.append(measures(runner, task, means, random_like(runner, members, rng))["necessity"])
        progress.step()
    return sorted(found)


def attribution(runner, task: PairTask) -> dict[Member, float]:
    """Each neuron's and head's attribution patching score on ``task``, summed over prompts.

    The score is (clean − corrupted activation) · ∂metric/∂activation, at the corrupted
    prompts. A head's score sums over its slice of the attention output.
    """
    clean_n: dict[int, torch.Tensor] = {}
    clean_h: dict[int, torch.Tensor] = {}
    layers = range(runner.n_layers)

    def record(store, layer):
        def fn(act):
            store[layer] = act.detach()

        return fn

    with torch.no_grad(), runner.hooks({n: record(clean_n, n) for n in layers}, None,
                                       {n: record(clean_h, n) for n in layers}):  # fmt: skip
        runner.forward(_encode(runner, task.clean))
    held_n: dict[int, torch.Tensor] = {}
    held_h: dict[int, torch.Tensor] = {}

    def hold(store, layer):
        # A leaf in place of the activation: the model's own weights may not track grad.
        def fn(act):
            leaf = act.detach().requires_grad_(True)
            store[layer] = leaf
            return leaf

        return fn

    with torch.enable_grad(), runner.hooks({n: hold(held_n, n) for n in layers}, None,
                                           {n: hold(held_h, n) for n in layers}):  # fmt: skip
        task.metric(runner.forward(_encode(runner, task.corrupted))).sum().backward()
    scores: dict[Member, float] = {}
    for layer in layers:
        n = ((clean_n[layer] - held_n[layer].detach()) * held_n[layer].grad).sum((0, 1))
        for i, v in enumerate(n.float().cpu().tolist()):
            scores[Member("N", layer, i)] = v
        h = ((clean_h[layer] - held_h[layer].detach()) * held_h[layer].grad).sum((0, 1))
        per_head = h.float().reshape(runner.n_heads, -1).sum(-1).cpu().tolist()
        for i, v in enumerate(per_head):
            scores[Member("H", layer, i)] = v
    return scores


def grow(seed: Iterable[Member], scores: dict[Member, float], size: int) -> list[Member]:
    """``seed`` plus the members of highest absolute attribution, up to ``size`` in all."""
    chosen = list(dict.fromkeys(seed))
    for m in sorted(scores, key=lambda m: -abs(scores[m])):
        if len(chosen) >= size:
            break
        if m not in chosen:
            chosen.append(m)
    return chosen
