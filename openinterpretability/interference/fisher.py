"""The diagonal Fisher importance of each MLP neuron, and the tests that use it (#5118).

A neuron's input weights are the rows of the MLP's input projections that write its
pre-activation. For Qwen2.5, these are ``gate_proj`` and ``up_proj``. Its output weights
are its column of the down projection. The diagonal Fisher of one weight is the mean, over
samples, of the squared gradient of the sample's log-likelihood. A neuron's importance is
the sum of that value over its own weights. Kirkpatrick et al. (2017) use this value in
elastic weight consolidation to decide which weights a new task may move.

Two kinds of label are read in one pass:

* ``model``: each next token is sampled from the model's own distribution. This is the
  Fisher of the model.
* ``data``: each next token is the corpus's next token. This is the *empirical* Fisher.

One sample is one window of tokens, so a window's gradient is the gradient of the sum
of its tokens' log-likelihoods.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

LABELS = ("model", "data")  # the two kinds of label, see the module docstring
SIDES = ("input", "output")  # the neuron's input weights and its output weights
FISHER_WINDOW = 128  # tokens in one Fisher sample
SEED = 5118  # the seed of the sampled labels and of the permutations
PERMUTATIONS = 10_000


def neuron_weights(runner, layer: int) -> list[tuple[str, nn.Parameter, int]]:
    """Each weight matrix of ``layer``'s MLP that has one neuron per slice.

    Args:
        runner: The model's ``NeuronHookRunner``.
        layer: The layer.

    Returns:
        ``(side, weight, axis)`` for each matrix. ``axis`` is the axis of the weight
        that indexes the neurons. ``side`` is ``"output"`` for the down projection and
        ``"input"`` for each projection that writes the neurons' pre-activations.
    """
    width = runner.mlp_width(layer)
    down = runner.down_proj(layer)
    out = []
    for module in runner.mlp(layer).modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter) or weight.dim() != 2:
            continue
        linear = hasattr(module, "in_features")  # nn.Linear is [out, in]; Conv1D is [in, out]
        if module is down:
            out.append(("output", weight, 1 if linear else 0))
        elif weight.shape[0 if linear else 1] == width:
            out.append(("input", weight, 0 if linear else 1))
    return out


def fisher_windows(batches: Sequence[torch.Tensor], length: int = FISHER_WINDOW) -> list:
    """The tokens of ``batches`` cut again into rows of ``length`` tokens.

    Args:
        batches: Token ids, as ``sample_windows`` gives them.
        length: The tokens in one row.

    Returns:
        The rows, each a 1-D tensor of ``length`` token ids. The last tokens that don't
        fill a row are left out.

    Examples:
        >>> [r.tolist() for r in fisher_windows([torch.arange(5).view(1, 5)], 2)]
        [[0, 1], [2, 3]]

    Raises:
        ValueError: The batches hold less than one row of ``length`` tokens.
    """
    ids = torch.cat([b.reshape(-1) for b in batches])
    whole = len(ids) // length * length
    if not whole:
        raise ValueError(f"the sample holds {len(ids)} tokens, less than one window of {length}")
    return list(ids[:whole].view(-1, length))


def diagonal_fisher(runner, rows: Sequence[torch.Tensor], seed: int = SEED) -> dict:
    """The diagonal Fisher of every MLP neuron, for each kind of label and each side.

    Only the MLP weights take a gradient while it runs. Their ``requires_grad`` is set
    back afterwards.

    Args:
        runner: The model's ``NeuronHookRunner``.
        rows: The samples, each a 1-D tensor of token ids.
        seed: The seed of the labels that are sampled from the model.

    Returns:
        ``{label: {side: tensor [n_layers, width]}}``, each the mean over ``rows``.
    """
    model = runner.model
    weights = [
        (layer, *w) for layer in range(runner.n_layers) for w in neuron_weights(runner, layer)
    ]
    kept = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for _, _, weight, _ in weights:
        weight.requires_grad_(True)
    width = runner.mlp_width(0)
    sums = {k: {s: torch.zeros(runner.n_layers, width) for s in SIDES} for k in LABELS}
    generator = torch.Generator().manual_seed(seed)
    try:
        for row in rows:
            row = row.view(1, -1).to(runner.device)
            logits = model(row).logits[0, :-1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            sampled = torch.multinomial(log_probs.detach().exp().cpu(), 1, generator=generator)
            targets = {"model": sampled.squeeze(1).to(row.device), "data": row[0, 1:]}
            for i, label in enumerate(LABELS):
                loglik = log_probs.gather(1, targets[label].unsqueeze(1)).sum()
                grads = torch.autograd.grad(loglik, [w for _, _, w, _ in weights],
                                            retain_graph=i < len(LABELS) - 1)  # fmt: skip
                for (layer, side, _, axis), grad in zip(weights, grads, strict=True):
                    sums[label][side][layer] += grad.detach().float().pow(2).sum(1 - axis).cpu()
    finally:
        for p, flag in kept.items():
            p.requires_grad_(flag)
    return {k: {s: t / len(rows) for s, t in v.items()} for k, v in sums.items()}


@torch.no_grad()
def activation_rms(runner, rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Each neuron's root-mean-square activation over ``rows``.

    The activation is the input of the down projection. The output Fisher of a neuron
    grows with the square of this value, so it is the control for that Fisher.

    Args:
        runner: The model's ``NeuronHookRunner``.
        rows: The samples, each a 1-D tensor of token ids.

    Returns:
        A tensor ``[n_layers, width]`` of the RMS over every token of ``rows``.
    """
    sums = torch.zeros(runner.n_layers, runner.mlp_width(0))
    count = 0

    def reader(layer):
        """A hook that adds the squares of ``layer``'s activations to ``sums``.

        Args:
            layer: The layer that the hook reads.

        Returns:
            The forward pre-hook.
        """

        def hook(_module, args):
            """Add the squares of the down projection's input to ``sums``.

            Args:
                _module: The down projection, not used.
                args: The down projection's inputs. The first is the activation.
            """
            sums[layer] += args[0].detach().float().pow(2).sum((0, 1)).cpu()

        return hook

    handles = [runner.down_proj(layer).register_forward_pre_hook(reader(layer))
               for layer in range(runner.n_layers)]  # fmt: skip
    try:
        for row in rows:
            runner.model(row.view(1, -1).to(runner.device))
            count += row.numel()
    finally:
        for handle in handles:
            handle.remove()
    return (sums / count).sqrt()


def layer_percentile(values: torch.Tensor, layer: int, neuron: int) -> float:
    """The share of ``layer``'s other neurons whose value is below the neuron's.

    Args:
        values: A tensor ``[n_layers, width]`` of one value for each neuron.
        layer: The neuron's layer.
        neuron: The neuron's index in its layer.

    Returns:
        A share from 0 to 1. Ties count half.

    Examples:
        >>> layer_percentile(torch.tensor([[1.0, 2.0, 3.0]]), 0, 1)
        0.5
    """
    row = values[layer]
    below = (row < row[neuron]).sum().item()
    ties = (row == row[neuron]).sum().item() - 1
    return (below + ties / 2) / (len(row) - 1)


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """The area under the ROC curve of ``scores`` for ``labels``.

    Args:
        scores: One score for each item.
        labels: One label for each item, in the same order.

    Returns:
        The probability that a True item's score is above a False item's, ties counted half.

    Examples:
        >>> auc([0.1, 0.9, 0.4], [False, True, True])
        1.0

    Raises:
        ValueError: The labels are all True or all False.
    """
    ranks = _ranks(scores)
    n_pos = sum(bool(y) for y in labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("auc needs both labels")
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y)  # Mann-Whitney U
    return (rank_sum - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)


def _ranks(values: Sequence[float]) -> list[float]:
    """The rank of each value from 0, with tied values given their mean rank.

    Args:
        values: The values.

    Returns:
        One rank for each value, in the same order.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """The Spearman rank correlation of ``a`` and ``b``.

    Args:
        a: The first values.
        b: The second values, in the same order.

    Returns:
        The correlation, or ``None`` when either side is constant.

    Examples:
        >>> round(spearman([1, 2, 3], [10, 30, 20]), 6)
        0.5
    """
    ra, rb = _ranks(a), _ranks(b)
    ra_t, rb_t = torch.tensor(ra), torch.tensor(rb)
    ra_t, rb_t = ra_t - ra_t.mean(), rb_t - rb_t.mean()
    den = (ra_t.norm() * rb_t.norm()).item()
    return (ra_t @ rb_t).item() / den if den else None


def partial_spearman(
    a: Sequence[float], b: Sequence[float], control: Sequence[float]
) -> float | None:
    """The Spearman correlation of ``a`` and ``b`` with ``control`` taken out of both.

    Each side's ranks are regressed on the control's ranks, and the residuals are
    correlated.

    Args:
        a: The first values.
        b: The second values, in the same order.
        control: The values to take out, in the same order.

    Returns:
        The correlation of the residuals, or ``None`` when a residual is constant.

    Examples:
        A constant control takes nothing out, so this is the plain Spearman correlation:

        >>> round(partial_spearman([1, 2, 3], [1, 3, 2], [0, 0, 0]), 6)
        0.5
    """
    rc = torch.tensor(_ranks(control))
    rc = rc - rc.mean()

    def residual(values):
        """The centred ranks of ``values``, less their regression on the control's ranks.

        Args:
            values: The values.

        Returns:
            The residuals, as a tensor.
        """
        r = torch.tensor(_ranks(values))
        r = r - r.mean()
        return r - (r @ rc) / (rc @ rc) * rc if (rc @ rc) else r

    ea, eb = residual(a), residual(b)
    den = (ea.norm() * eb.norm()).item()
    return (ea @ eb).item() / den if den > 1e-9 else None


def stratified_null(statistic: Callable[[list], float], outcomes: Sequence, strata: Sequence,
                    n: int = PERMUTATIONS, seed: int = SEED) -> dict:  # fmt: skip
    """The statistic against a null that shuffles the outcomes within each stratum.

    The p-value is two-sided about the null's mean, with the observed value counted in
    the null.

    Args:
        statistic: Takes the outcomes, in their given order, and gives one number.
        outcomes: One outcome for each unit.
        strata: One stratum for each unit, in the same order. A shuffle moves an outcome
            only between units of one stratum.
        n: The number of shuffles.
        seed: The seed of the shuffles.

    Returns:
        ``{"observed", "null_mean", "p", "null"}``, where ``null`` holds every shuffled value.

    Examples:
        Each stratum holds one unit, so no shuffle moves an outcome:

        >>> out = stratified_null(sum, [1, 0], ["a", "b"], n=5)
        >>> out["observed"], out["p"]
        (1, 1.0)
    """
    groups: dict = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    generator = torch.Generator().manual_seed(seed)
    observed = statistic(list(outcomes))
    null = []
    for _ in range(n):
        shuffled = list(outcomes)
        for idx in groups.values():
            perm = torch.randperm(len(idx), generator=generator).tolist()
            for dst, src in zip(idx, perm, strict=True):
                shuffled[dst] = outcomes[idx[src]]
        null.append(statistic(shuffled))
    mean = sum(null) / n
    extreme = sum(abs(v - mean) >= abs(observed - mean) - 1e-12 for v in null)
    return {"observed": observed, "null_mean": mean, "p": (extreme + 1) / (n + 1), "null": null}
