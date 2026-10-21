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
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

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


def scan_neuron(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    k: int = 20,
    batch_size: int = 8,
    n_correlated: int = 10,
    min_count: int = 3,
    window_tokens: int | None = None,
    stride: int | None = None,
) -> ScanResult:
    if runner.tokenizer is None:
        raise ValueError("scan_neuron requires a tokenizer")
    partners = PartnerSearch(runner.mlp_width(layer))
    confounds = ConfoundAccumulator()
    candidates: list[tuple[float, Piece, list[int], list[float]]] = []
    values: list[float] = []
    pieces = list(cut_pieces(runner.tokenizer, texts, window_tokens, stride))

    for start in range(0, len(pieces), batch_size):
        batch = pieces[start : start + batch_size]
        encoded = runner.encode([p.text for p in batch])
        _, cache = runner.run_with_cache(encoded, [layer])
        acts = cache[layer].float().cpu()  # [b, seq, d_mlp]
        strengths = pole_strength(acts[..., neuron_idx], pole)
        ids = encoded["input_ids"].cpu()
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool().cpu()
        positions = torch.arange(ids.shape[1]).expand_as(ids)
        # Count only each piece's fresh tail: real tokens at or after it ≤ fresh.
        fresh = torch.tensor([p.fresh or ids.shape[1] for p in batch]).unsqueeze(1)
        counted = mask & (mask.flip(1).cumsum(1).flip(1) <= fresh)

        partners.update(strengths[counted], acts[counted])
        confounds.update(ids[counted], positions[counted], strengths[counted])
        values.extend(strengths[counted].tolist())
        for row, piece in enumerate(batch):
            row_vals = strengths[row][mask[row]].tolist()
            if row_vals:
                candidates.append((max(row_vals), piece, ids[row][mask[row]].tolist(), row_vals))

    examples: list[ActivatingExample] = []
    chosen: list[Piece] = []
    for peak_value, piece, row_ids, row_vals in sorted(candidates, key=lambda c: -c[0]):
        if len(examples) == k:
            break
        if any(piece.overlaps(c) for c in chosen):
            continue
        chosen.append(piece)
        examples.append(
            ActivatingExample(
                text=piece.text,
                tokens=[runner.tokenizer.decode([i]) for i in row_ids],
                activations=[round(v, 5) for v in row_vals],
                max_activation=peak_value,
                max_token_index=row_vals.index(peak_value),
            )
        )
    # "Active" for co-activation uses the protocol's own definition for the target.
    active = ReviewPolicy().activation_threshold * max(values, default=0.0)
    return ScanResult(
        examples=examples,
        values=values,
        correlated=partners.partners(neuron_idx, active, k=n_correlated) if active > 0 else [],
        confounds=confounds.finalize(runner.tokenizer, min_count),
        windows=len(pieces),
    )
