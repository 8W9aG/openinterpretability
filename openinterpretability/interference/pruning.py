"""Zero part of a neuron's input weights, as PackNet prunes a network (#5122).

A neuron's input weights are its rows of the MLP's input projections. These are
``gate_proj`` and ``up_proj`` for Qwen2.5, ``dense_h_to_4h`` for GPT-NeoX and ``c_fc`` for
GPT-2. They are found with
:func:`openinterpretability.interference.fisher.neuron_weights`. Each row is pruned on its
own, by one of two methods:

* ``magnitude``: zero the weights with the smallest absolute value first, as PackNet does.
* ``random``: zero the same number of weights, chosen at random. This is the null.

The bias stays. Pruning a neuron's own input weights changes only that neuron at its
layer. So the neurons of one layer can be pruned together and read in one pass.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

from openinterpretability.interference.fisher import neuron_weights

METHODS = ("magnitude", "random")
LEVELS = (0.5, 0.75, 0.9)  # the shares of a row that are zeroed (PackNet prunes 50-75%)
SEED = 5122  # the seed of the random null


def zeroed(row: torch.Tensor, fraction: float, method: str,
           generator: torch.Generator | None = None) -> torch.Tensor:  # fmt: skip
    """The positions of ``row`` to zero: a boolean mask with ``round(fraction * len)`` set.

    Args:
        row: One neuron's weights in one matrix.
        fraction: The share of the row to zero, from 0 to 1.
        method: ``magnitude`` or ``random``.
        generator: The random source, for ``random`` only.

    Returns:
        The mask, on the CPU.

    Raises:
        ValueError: The method is not one of :data:`METHODS`.

    Examples:
        >>> zeroed(torch.tensor([0.1, -3.0, 0.5, 2.0]), 0.5, "magnitude").tolist()
        [True, False, True, False]
    """
    count = round(fraction * row.numel())
    mask = torch.zeros(row.numel(), dtype=torch.bool)
    if method == "magnitude":
        order = row.detach().abs().float().cpu().argsort(stable=True)
    elif method == "random":
        order = torch.randperm(row.numel(), generator=generator)
    else:
        raise ValueError(f"unknown method {method!r}: use one of {METHODS}")
    mask[order[:count]] = True
    return mask


@contextmanager
def pruned(runner, layer: int, neurons: Sequence[int], fraction: float, method: str,
           seed: int = SEED) -> Iterator[None]:  # fmt: skip
    """Zero a share of each neuron's input weights in ``layer``, then restore them.

    Args:
        runner: The model's ``NeuronHookRunner``.
        layer: The layer.
        neurons: The neurons to prune.
        fraction: The share of each row to zero.
        method: ``magnitude`` or ``random``.
        seed: The seed of the ``random`` method. Each neuron and matrix gets its own draw.

    Yields:
        Nothing. The weights are pruned inside the ``with`` block.
    """
    kept = []
    with torch.no_grad():
        for m, (side, weight, axis) in enumerate(neuron_weights(runner, layer)):
            if side != "input":
                continue
            for neuron in neurons:
                row = weight.select(axis, neuron)
                draw = seed * 1_000_003 + layer * 100_003 + neuron * 7 + m  # one draw per row
                generator = torch.Generator().manual_seed(draw)
                mask = zeroed(row, fraction, method, generator).to(row.device)
                kept.append((row, row.clone()))
                row[mask] = 0
    try:
        yield
    finally:
        with torch.no_grad():
            for row, original in kept:
                row.copy_(original)
