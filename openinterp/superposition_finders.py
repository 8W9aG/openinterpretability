"""Ways to find superpositions: sparse directions over one layer's MLP neurons (issue #3686).

Two finders give definitions that ``open-interpretability audit --superposition`` reads.

- :func:`from_circuit` groups a traced circuit's neurons by layer. The weights are the
  neurons' attributions.
- :func:`grammatical_number` fits a sparse probe for the number of the noun that the
  verb agrees with, at one layer's last token.

``open-interpretability superpositions`` runs them (``openinterp/cli_superpositions.py``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import torch

from openinterp.agreement import TEMPLATES, pairs
from openinterp.core.profiles import SuperpositionMember, SuperpositionProfile

NUMBER_SOURCE = "grammatical number from the agreement templates (#3686)"
TRAIN_SHARE = 0.8  # the share of examples the probe trains on. The rest is held out.
L1_DECAY = 0.7  # each step of the L1 path multiplies the strength by this
ISTA_STEPS = 200  # proximal-gradient steps at each L1 strength
L2_STRENGTH = 1e-2  # the ridge penalty of the refit on the kept neurons
BATCH = 64  # prompts in one forward pass


def from_circuit(
    rows: Iterable[dict], *, top: int = 100, min_members: int = 2, source: str
) -> list[tuple[int, SuperpositionProfile]]:
    """One (layer, profile) for each layer with ``min_members`` of the circuit's ``top`` neurons.

    ``rows`` are a circuit's rows, as ``scripts/sva_circuit.py`` writes them. Only rows
    with ``"table": "neuron"`` count, in ``rank`` order. A member's weight is its
    attribution. Members with a zero attribution are not used.
    """
    if min_members < 2:
        raise ValueError("a superposition has at least 2 members, so min_members must be ≥ 2")
    neurons = sorted((r for r in rows if r.get("table") == "neuron"), key=lambda r: r["rank"])
    by_layer: dict[int, list[SuperpositionMember]] = {}
    for row in neurons[:top]:
        if row["attribution"] != 0:
            member = SuperpositionMember(neuron_idx=row["neuron_idx"], weight=row["attribution"])
            by_layer.setdefault(row["layer"], []).append(member)
    return [
        (layer, _profile(members, "circuit", source))
        for layer, members in sorted(by_layer.items())
        if len(members) >= min_members
    ]


def _profile(members: list[SuperpositionMember], method: str, source: str):
    ordered = sorted(members, key=lambda m: m.neuron_idx)
    return SuperpositionProfile(members=ordered, method=method, source=source)


def split_by_group(groups: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Train and held-out row indices. All rows of one group are on the same side.

    A fixed shuffle of the groups puts the first 80% of them in training.
    """
    ids = groups.unique()
    order = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))]
    in_train = torch.isin(groups, order[: int(TRAIN_SHARE * len(ids))])
    return in_train.nonzero().flatten(), (~in_train).nonzero().flatten()


def sparse_probe(
    acts: torch.Tensor,
    labels: torch.Tensor,
    *,
    k: int,
    seed: int = 0,
    groups: torch.Tensor | None = None,
) -> tuple[list[int], list[float], float]:
    """The ``k`` neurons that best predict ``labels``, their weights and the held-out accuracy.

    ``acts`` is ``[n, width]`` and ``labels`` is ``[n]`` of 0 and 1. ``groups`` gives
    each row a group id, and :func:`split_by_group` holds out 20% of the groups. With no
    ``groups``, each row is a group. Each column is standardised on the training part.
    An L1 logistic regression gets weaker until at least ``k`` weights are not zero. The
    ``k`` largest weights choose the neurons. Then an L2 logistic regression is fitted
    again on those neurons only. The weights are in the raw activation space.
    """
    x, y = acts.float().cpu(), labels.float().cpu()
    groups = torch.arange(len(y)) if groups is None else groups.cpu()
    train, test = split_by_group(groups, seed)
    mean, std = x[train].mean(0), x[train].std(0)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))  # a constant column stays 0
    z = (x - mean) / std
    weights = _l1_path(z[train], y[train], k)
    keep = weights.abs().topk(k).indices.sort().values
    w, b = _ridge_logistic(z[train][:, keep], y[train])
    predicted = (z[test][:, keep] @ w + b > 0).float()
    accuracy = float((predicted == y[test]).float().mean()) if len(test) else math.nan
    return keep.tolist(), (w / std[keep]).tolist(), accuracy


def _l1_path(z: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """L1 logistic weights at the strongest strength on the path with at least ``k`` non-zero."""
    n = len(y)
    strength = float((z.T @ (y - y.mean())).abs().max()) / n  # all weights are 0 above this
    step = 1.0 / (0.25 * _top_eigenvalue(z) / n)  # 1 / the gradient's Lipschitz constant
    w, b = torch.zeros(z.shape[1]), torch.zeros(())
    while int((w != 0).sum()) < k and strength > 1e-8:
        strength *= L1_DECAY
        for _ in range(ISTA_STEPS):
            residual = torch.sigmoid(z @ w + b) - y
            w = _shrink(w - step * (z.T @ residual) / n, step * strength)
            b = b - step * residual.mean()
    return w


def _shrink(w: torch.Tensor, by: float) -> torch.Tensor:
    """Soft thresholding: the proximal step of the L1 penalty."""
    return torch.sign(w) * torch.clamp(w.abs() - by, min=0.0)


def _top_eigenvalue(z: torch.Tensor, steps: int = 50) -> float:
    """The largest eigenvalue of ``zᵀz``, by power iteration."""
    v = torch.ones(z.shape[1]) / math.sqrt(z.shape[1])
    for _ in range(steps):
        v = z.T @ (z @ v)
        v = v / v.norm().clamp_min(1e-12)
    return float((z @ v).pow(2).sum()) or 1.0


def _ridge_logistic(z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights and bias of an L2 logistic regression, fitted with L-BFGS."""
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y)
        loss = loss + L2_STRENGTH * w.pow(2).sum() / 2
        loss.backward()
        return loss

    with torch.enable_grad():
        optimiser.step(closure)
    return w.detach(), b.detach()


def agreement_examples(n: int, seed: int) -> list[tuple[str, int]]:
    """``2n`` (prompt, label) examples from ``n`` agreement pairs, spread over the templates.

    The label is 1 when the noun that the verb agrees with is plural. Each pair gives
    both of its prompts, so it gives one singular and one plural example.
    """
    rng = random.Random(seed)
    out = []
    for i, template in enumerate(TEMPLATES):
        count = n // len(TEMPLATES) + (i < n % len(TEMPLATES))
        for row in pairs(template, count, rng):
            clean_plural = int(row["sign"] < 0)
            out += [(row["clean"], clean_plural), (row["corrupted"], 1 - clean_plural)]
    return out


def last_token_activations(runner, texts: Sequence[str], layers: Sequence[int]):
    """``{layer: [len(texts), width]}``: each layer's MLP activations at each text's last token.

    Texts with the same number of tokens run together, so no text needs padding.
    """
    ids = [runner.tokenizer.encode(t) for t in texts]
    by_length: dict[int, list[int]] = {}
    for i, row in enumerate(ids):
        by_length.setdefault(len(row), []).append(i)
    out = {layer: torch.zeros(len(texts), runner.mlp_width(layer)) for layer in layers}
    for group in by_length.values():
        for start in range(0, len(group), BATCH):
            part = group[start : start + BATCH]
            batch = torch.tensor([ids[i] for i in part])
            cache = runner.layer_activations({"input_ids": batch}, list(layers))
            for layer in layers:
                out[layer][part] = cache[layer][:, -1].float().cpu()
    return out


def grammatical_number(
    runner, layers: Sequence[int], *, k: int, n: int, seed: int
) -> list[tuple[int, SuperpositionProfile, float]]:
    """One (layer, profile, held-out accuracy) per layer: a ``k``-neuron probe for noun number.

    It uses ``n`` agreement pairs (:func:`agreement_examples`). The probe reads the
    layer's MLP activations at the last token, where the verb comes next. The two
    prompts of a pair are always on the same side of the held-out split.
    """
    examples = agreement_examples(n, seed)
    labels = torch.tensor([label for _, label in examples])
    pair_ids = torch.arange(len(examples)) // 2  # agreement_examples puts a pair's prompts together
    acts = last_token_activations(runner, [text for text, _ in examples], layers)
    out = []
    for layer in layers:
        neurons, weights, accuracy = sparse_probe(
            acts[layer], labels, k=k, seed=seed, groups=pair_ids
        )
        members = [
            SuperpositionMember(neuron_idx=i, weight=w)
            for i, w in zip(neurons, weights, strict=True)
            if w != 0
        ]
        out.append((layer, _profile(members, "sparse-probe", NUMBER_SOURCE), accuracy))
    return out


def definition(model: str, layer: int, profile: SuperpositionProfile) -> dict:
    """The definition that ``open-interpretability audit --superposition`` reads."""
    return {"model": model, "layer": layer, **profile.model_dump()}


def compare(a: SuperpositionProfile, b: SuperpositionProfile) -> tuple[list[int], float]:
    """The neurons that ``a`` and ``b`` share, and the cosine of their two directions."""
    wa = {m.neuron_idx: m.weight for m in a.members}
    wb = {m.neuron_idx: m.weight for m in b.members}
    shared = sorted(wa.keys() & wb.keys())
    dot = sum(wa[i] * wb[i] for i in shared)
    norms = math.sqrt(sum(v * v for v in wa.values())) * math.sqrt(sum(v * v for v in wb.values()))
    return shared, dot / norms
