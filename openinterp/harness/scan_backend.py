"""Where a scan's forward pass runs, and the exact measurement of its top candidates.

A runner can carry a ``scan_backend``, an :class:`~openinterp.harness.ane.backend.AneScan`.
It runs the scan's forward pass on Apple's Neural Engine (ANE). The device planner
(``harness/devices.py``) adds it when the GPU is busy. The ANE computes in fp16, about 1%
from fp32. So after the pass, :func:`rescore_scan` measures each pole's strongest
candidates again on the runner's own device:

- the :data:`RESCORE_TOP` strongest pieces;
- and any piece within :data:`RESCORE_MARGIN` of the 20th strongest, up to
  :data:`RESCORE_CAP` pieces.

On ``corpus.jsonl`` that rule found the exact top 20 for every neuron and both poles. The
layers were GPT-2 L5, Qwen2.5-0.5B L12 and Qwen2.5-1.5B L8 (docs/analyses/ane-feasibility.md).
Then it makes sure that every example the scan will keep was measured again. The
numbers that ``openinterp reproduce`` checks are then the device's numbers. The
histogram, partners and confounds keep the ANE's values, which correlate with fp32 at
r ≥ 0.9998 for every neuron.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import torch

from openinterp.harness.hooks import pole_strength

if TYPE_CHECKING:
    from openinterp.core.schemas import Pole
    from openinterp.harness.hooks import NeuronHookRunner
    from openinterp.harness.scan import Chunk, Piece, TopExamples

RESCORE_TOP = 60
RESCORE_MARGIN = 0.06
RESCORE_CAP = 120
RESCORE_RANK = 20  # the margin is below this rank's peak
RESCORE_BATCH = 16  # pieces per forward pass when measuring again
RESCORE_ROUNDS = 5  # rounds that add examples the selection would keep unmeasured


def ane_scan(runner: NeuronHookRunner, pieces: Sequence[Piece]):
    """The runner's ANE backend if it can run all of ``pieces``, else None."""
    backend = getattr(runner, "scan_backend", None)
    if backend is None or not pieces:
        return None
    every = runner.tokenizer([p.text for p in pieces], return_tensors="pt", padding=True)
    mask = every.get("attention_mask", torch.ones_like(every["input_ids"]))
    return backend if backend.why_not(mask.sum(1).tolist()) is None else None


def scan_device(runner: NeuronHookRunner, pieces: Sequence[Piece]) -> str:
    """The device a scan of ``pieces`` runs on: the backend's name, or the runner's device."""
    backend = ane_scan(runner, pieces)
    return backend.name if backend is not None else runner.device.type


def forward_chunks(
    runner: NeuronHookRunner, pieces: Sequence[Piece], layer: int, batch_tokens: int
) -> Iterator[Chunk]:
    """``layer``'s chunks over ``pieces``, from the ANE if it can run them, else torch."""
    from openinterp.harness.scan import layer_chunks

    backend = ane_scan(runner, pieces)
    if backend is None:
        return layer_chunks(runner, pieces, layer, batch_tokens)
    backend.used = True
    return backend.chunks(runner, pieces, layer, batch_tokens)


def rescore_scan(runner, pieces, layer: int, neuron_idx: int, poles, tops, ks) -> None:
    """Measure again the top candidates of each pole, if the scan's values are the ANE's.

    ``poles``, ``tops`` and ``ks`` are each pole, its :class:`TopExamples` and the number
    of examples it keeps.
    """
    if ane_scan(runner, pieces) is None:
        return
    for pole, top, k in zip(poles, tops, ks, strict=True):
        rescore(top, _measure(runner, layer, neuron_idx, pole), k)


def _measure(runner, layer: int, neuron_idx: int, pole: Pole) -> Callable:
    def measure(texts: list[str]) -> list[list[float]]:
        encoded = runner.encode(texts)
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
        with torch.no_grad():
            acts = runner.layer_activations(encoded, [layer])[layer][..., neuron_idx]
        values = pole_strength(acts.float(), pole).cpu()
        return [values[r][mask[r].cpu()].tolist() for r in range(len(texts))]

    return measure


def rescore(top: TopExamples, measure: Callable[[list[str]], list[list[float]]], k: int) -> int:
    """Replace the values of ``top``'s strongest candidates with ``measure``'s.

    Return how many candidates it measured again.
    """
    done: set[int] = set()
    todo = chosen(top.candidates)
    for _ in range(RESCORE_ROUNDS):
        todo -= done
        if not todo:
            break
        order = sorted(todo)
        for start in range(0, len(order), RESCORE_BATCH):
            part = order[start : start + RESCORE_BATCH]
            new = measure([top.candidates[i][0].text for i in part])
            for i, values in zip(part, new, strict=True):
                piece, ids, old = top.candidates[i]
                if len(values) != len(old):
                    raise ValueError(f"measured {len(values)} tokens of a {len(old)}-token piece")
                top.candidates[i] = (piece, ids, values)
        done |= todo
        todo = kept(top.candidates, k)
    return len(done)


def chosen(candidates: list) -> set[int]:
    """Indices of the candidates to measure again, by their peak with and without token 0."""
    picked: set[int] = set()
    for start in (0, 1):
        peaks = torch.tensor(
            [max(v[start:]) if len(v) > start else -torch.inf for *_, v in candidates]
        )
        order = peaks.argsort(descending=True)
        ranked = peaks[order]
        if not len(ranked):
            continue
        nth = ranked[min(RESCORE_RANK, len(ranked)) - 1]
        line = nth - RESCORE_MARGIN * nth.abs() - 1e-3 * ranked[0].abs()
        for rank, index in enumerate(order.tolist()):
            if rank < RESCORE_TOP or (ranked[rank] >= line and rank < RESCORE_CAP):
                picked.add(index)
    return picked


def kept(candidates: list, k: int) -> set[int]:
    """Indices of the candidates that the top ``k`` would keep, with and without token 0.

    As ``TopExamples.top``, it skips a piece that overlaps one it kept.
    """
    keep: set[int] = set()
    for start in (0, 1):
        ranked = sorted(
            (i for i, (*_, v) in enumerate(candidates) if len(v) > start),
            key=lambda i: -max(candidates[i][2][start:]),
        )
        taken: list[int] = []
        for i in ranked:
            if len(taken) == k:
                break
            if not any(candidates[i][0].overlaps(candidates[j][0]) for j in taken):
                taken.append(i)
        keep.update(taken)
    return keep
