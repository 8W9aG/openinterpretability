"""One pass over a corpus that gathers everything an audit needs about a neuron pole.

For every non-padding token it records the pole strength (top examples and
histogram), the whole layer's activations (partner neuron poles: co-activation lift,
Spearman, Pearson) and the token id and position (confound profile).

Documents are scanned whole by default. With ``window_tokens``, each is cut into
overlapping windows that start every ``stride`` tokens, so a long document is read in
full without one huge forward pass. Every token still counts once in the statistics,
in the first window that reaches it after the first, where it has at least
``window_tokens - stride`` tokens of context (the strided-perplexity convention).
Top examples are windows, decoded back to text so a record re-measures them as the
scan did, and no two overlap.

Each forward pass yields a :class:`Chunk`: the whole layer's activations on every real
token of its pieces. Given a :class:`~openinterp.harness.layer_cache.LayerCache`, a scan
stores its chunks, and a later scan of any neuron in that layer replays them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from openinterp.core.schemas import (
    ActivatingExample,
    ConfoundProfile,
    CorrelatedNeuron,
    Pole,
    ReviewPolicy,
)
from openinterp.harness.confounds import ConfoundAccumulator
from openinterp.harness.covariance import PartnerSearch
from openinterp.harness.hooks import NeuronHookRunner, pole_strength

if TYPE_CHECKING:
    from openinterp.harness.layer_cache import LayerCache


# Padded tokens per forward pass (pieces × the longest). Part of a cached scan's key, as
# it sets padding. On Qwen2.5-0.5B over corpus.jsonl the model reads 27% fewer tokens,
# padding included, than with 8 pieces a pass, and 2048 is near the fastest budget on
# Apple's GPU.
BATCH_TOKENS = 2048


@dataclass
class ScanResult:
    examples: list[ActivatingExample]
    values: list[float]  # every scanned token's pole strength, each token once
    correlated: list[CorrelatedNeuron]
    confounds: ConfoundProfile
    windows: int = 0  # forward-pass inputs: documents, or windows of them


@dataclass(frozen=True)
class Piece:
    """What one forward pass reads: a whole document, or tokens [start, end) of it."""

    doc: int
    start: int
    end: int
    text: str
    fresh: int | None = None  # trailing tokens new to the scan; None: every token is

    def overlaps(self, other: Piece) -> bool:
        return self.doc == other.doc and self.start < other.end and other.start < self.end


def default_stride(window_tokens: int) -> int:
    """Half a window, so every counted token has at least half a window of context."""
    return max(window_tokens // 2, 1)


def cut_pieces(
    tokenizer, texts: Sequence[str], window_tokens: int | None, stride: int | None
) -> Iterator[Piece]:
    """Whole documents, or each document's windows (see the module docstring)."""
    if window_tokens is None:
        yield from (Piece(doc, 0, 1, text) for doc, text in enumerate(texts))
        return
    stride = stride or default_stride(window_tokens)
    if not 0 < stride <= window_tokens:
        raise ValueError(f"stride must be in 1..{window_tokens}, got {stride}")
    for doc, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        start = prev_end = 0
        while start < len(ids):
            end = min(start + window_tokens, len(ids))
            fresh = None if start == 0 else end - prev_end
            yield Piece(doc, start, end, tokenizer.decode(ids[start:end]), fresh)
            if end == len(ids):
                break
            start, prev_end = start + stride, end


@dataclass
class Batch:
    """One forward pass of pieces, with which of its tokens the scan counts."""

    pieces: list[Piece]
    encoded: dict[str, torch.Tensor]
    ids: torch.Tensor  # [b, seq], on the CPU
    mask: torch.Tensor  # real (non-padding) tokens
    counted: torch.Tensor  # real tokens new to the scan
    positions: torch.Tensor


def plan_batches(lengths: Sequence[int], batch_tokens: int) -> list[int]:
    """How many consecutive pieces each forward pass takes: as many as fit in
    ``batch_tokens`` once padded to the longest, and at least one."""
    sizes: list[int] = []
    size = longest = 0
    for n in lengths:
        if size and max(longest, n) * (size + 1) > batch_tokens:
            sizes.append(size)
            size = longest = 0
        size, longest = size + 1, max(longest, n)
    return [*sizes, size] if size else sizes


def batches(
    runner: NeuronHookRunner, pieces: Sequence[Piece], batch_tokens: int = BATCH_TOKENS
) -> Iterator[Batch]:
    """``pieces`` in order, in forward passes of at most ``batch_tokens`` padded tokens."""
    if not pieces:
        return
    every = runner.tokenizer([p.text for p in pieces], return_tensors="pt", padding=True)
    lengths = every.get("attention_mask", torch.ones_like(every["input_ids"])).sum(1).tolist()
    start = 0
    for size in plan_batches(lengths, batch_tokens):
        batch = list(pieces[start : start + size])
        start += size
        encoded = runner.encode([p.text for p in batch])
        ids = encoded["input_ids"].cpu()
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool().cpu()
        # Count only each piece's fresh tail: real tokens at or after it ≤ fresh.
        fresh = torch.tensor([p.fresh or ids.shape[1] for p in batch]).unsqueeze(1)
        counted = mask & (mask.flip(1).cumsum(1).flip(1) <= fresh)
        positions = torch.arange(ids.shape[1]).expand_as(ids)
        yield Batch(batch, encoded, ids, mask, counted, positions)


@dataclass
class Chunk:
    """One forward pass, flattened to its pieces' real tokens in scan order."""

    pieces: list[Piece]
    lengths: list[int]  # real tokens of each piece
    ids: torch.Tensor  # [n]
    positions: torch.Tensor  # [n]
    counted: torch.Tensor  # [n] bool: tokens new to the scan
    acts: torch.Tensor  # [n, width] the whole layer, in the model's dtype, on the CPU

    def rows(self) -> Iterator[tuple[Piece, slice]]:
        """Each piece and its rows."""
        start = 0
        for piece, n in zip(self.pieces, self.lengths, strict=True):
            yield piece, slice(start, start + n)
            start += n


def layer_chunks(
    runner: NeuronHookRunner, pieces: Sequence[Piece], layer: int, batch_tokens: int
) -> Iterator[Chunk]:
    """Run the model over ``pieces`` and yield ``layer``'s activations batch by batch."""
    for batch in batches(runner, pieces, batch_tokens):
        _, cache = runner.run_with_cache(batch.encoded, [layer])
        yield Chunk(
            pieces=batch.pieces,
            lengths=batch.mask.sum(1).tolist(),
            ids=batch.ids[batch.mask],
            positions=batch.positions[batch.mask],
            counted=batch.counted[batch.mask],
            acts=cache[layer].cpu()[batch.mask],
        )


def scan_chunks(
    runner: NeuronHookRunner,
    pieces: Sequence[Piece],
    layer: int,
    batch_tokens: int = BATCH_TOKENS,
    cache: LayerCache | None = None,
) -> Iterator[Chunk]:
    """``layer``'s chunks: replayed from ``cache`` if it holds them, else from the model
    (and stored in ``cache``)."""
    chunks = layer_chunks(runner, pieces, layer, batch_tokens)
    return chunks if cache is None else cache.through(runner, layer, pieces, batch_tokens, chunks)


class TopExamples:
    """Collects every piece's strengths and keeps the ``k`` strongest that don't overlap."""

    def __init__(self) -> None:
        self.candidates: list[tuple[float, Piece, list[int], list[float]]] = []

    def update(self, batch: Batch, strengths: torch.Tensor) -> None:
        for row, piece in enumerate(batch.pieces):
            mask = batch.mask[row]
            self.add(piece, batch.ids[row][mask].tolist(), strengths[row][mask].tolist())

    def update_chunk(self, chunk: Chunk, strengths: torch.Tensor) -> None:
        for piece, rows in chunk.rows():
            self.add(piece, chunk.ids[rows].tolist(), strengths[rows].tolist())

    def add(self, piece: Piece, ids: list[int], values: list[float]) -> None:
        if values:
            self.candidates.append((max(values), piece, ids, values))

    def top(self, tokenizer, k: int) -> list[ActivatingExample]:
        examples: list[ActivatingExample] = []
        chosen: list[Piece] = []
        for peak_value, piece, row_ids, row_vals in sorted(self.candidates, key=lambda c: -c[0]):
            if len(examples) == k:
                break
            if any(piece.overlaps(c) for c in chosen):
                continue
            chosen.append(piece)
            examples.append(
                ActivatingExample(
                    text=piece.text,
                    tokens=[tokenizer.decode([i]) for i in row_ids],
                    activations=[round(v, 5) for v in row_vals],
                    max_activation=peak_value,
                    max_token_index=row_vals.index(peak_value),
                )
            )
        return examples


def scan_neuron(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    k: int = 20,
    batch_tokens: int = BATCH_TOKENS,
    n_correlated: int = 10,
    min_count: int = 3,
    window_tokens: int | None = None,
    stride: int | None = None,
    cache: LayerCache | None = None,
) -> ScanResult:
    """Scan ``texts`` for one neuron pole. With ``cache``, replay the layer's stored
    activations if it holds this scan, and store them if not."""
    if runner.tokenizer is None:
        raise ValueError("scan_neuron requires a tokenizer")
    partners = PartnerSearch(runner.mlp_width(layer))
    confounds = ConfoundAccumulator()
    top = TopExamples()
    values: list[float] = []
    pieces = list(cut_pieces(runner.tokenizer, texts, window_tokens, stride))

    for chunk in scan_chunks(runner, pieces, layer, batch_tokens, cache):
        acts = chunk.acts.float()  # [n, d_mlp]
        strengths = pole_strength(acts[:, neuron_idx], pole)
        counted = chunk.counted
        partners.update(strengths[counted], acts[counted])
        confounds.update(chunk.ids[counted], chunk.positions[counted], strengths[counted])
        values.extend(strengths[counted].tolist())
        top.update_chunk(chunk, strengths)

    # "Active" for co-activation uses the protocol's own definition for the target.
    active = ReviewPolicy().activation_threshold * max(values, default=0.0)
    return ScanResult(
        examples=top.top(runner.tokenizer, k),
        values=values,
        correlated=partners.partners(neuron_idx, active, k=n_correlated) if active > 0 else [],
        confounds=confounds.finalize(runner.tokenizer, min_count),
        windows=len(pieces),
    )
