"""One pass over a corpus that gathers everything an audit needs about a neuron pole.

For every non-padding token it records the pole strength (top examples and
histogram), the whole layer's activations (partner neuron poles: co-activation lift,
Spearman, Pearson) and the token id and position (confound profile).

Documents are scanned whole by default. With ``window_tokens``, each is cut into
overlapping windows that start every ``stride`` tokens, so a long document is read in
full without one huge forward pass. Every token still counts once in the statistics,
in the first window that reaches it after the first, where it has at least
``window_tokens - stride`` tokens of context (the strided-perplexity convention).
A unit with a first-token floor (protocol 0.8) is above its line at the first token
whatever the token, so its scan leaves first tokens out of its scale, histogram and
top-example ranking (:func:`settle`). That is decided after the pass, so every scan
keeps whether each counted token is its piece's first.
Top examples are windows, decoded back to text so a record re-measures them as the
scan did, and no two overlap.

Each forward pass yields a :class:`Chunk`: the whole layer's activations on every real
token of its pieces. Given a :class:`~openinterp.harness.layer_cache.LayerCache`, a scan
stores its chunks, and a later scan of any neuron in that layer replays them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from openinterp.core.schemas import (
    ActivatingExample,
    ConfoundProfile,
    CorrelatedNeuron,
    Pole,
    ReviewPolicy,
)
from openinterp.harness.annotations import LabelAccumulator, Tagger, TokenLabels
from openinterp.harness.confounds import ConfoundAccumulator
from openinterp.harness.covariance import PartnerSearch
from openinterp.harness.hooks import Inputs, NeuronHookRunner, pole_strength

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
    opposite: OppositeScan | None = None  # the neuron's other pole, from the same pass
    # Protocol 0.8: first tokens are left out of examples and values (see ``settle``).
    first_token_floor: bool = False


@dataclass
class OppositeScan:
    pole: Pole
    examples: list[ActivatingExample]
    values: list[float]
    first_token_floor: bool = False


Strengths = Callable[[Inputs], torch.Tensor]


@torch.no_grad()
def first_position_share(
    runner: NeuronHookRunner, strengths: Strengths, line: float, batch_size: int = 8192
) -> float:
    """The share of the vocabulary (special tokens left out) on which ``strengths`` is at
    or above ``line`` at a text's first position. Attention there sees only the token
    itself, so this is a property of the weights, the same for every corpus."""
    tokenizer = runner.tokenizer
    try:
        size = len(tokenizer)
    except TypeError:  # a tokenizer without a length: one id per embedding row
        size = runner.model.get_input_embeddings().num_embeddings
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    vocab = [i for i in range(size) if i not in special]
    above = 0
    for start in range(0, len(vocab), batch_size):
        ids = torch.tensor(vocab[start : start + batch_size], device=runner.device).unsqueeze(1)
        out = strengths({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        above += int((out[:, 0] >= line).sum())
    return above / max(len(vocab), 1)


def has_first_token_floor(
    runner: NeuronHookRunner, strengths: Strengths, scale: float, policy: ReviewPolicy
) -> bool:
    """Protocol 0.8: is the unit active at the first position on at least
    ``policy.max_active_share`` of the vocabulary, against its line ``scale`` sets?"""
    if scale <= 0:
        return False
    line = policy.activation_threshold * scale
    return first_position_share(runner, strengths, line) >= policy.max_active_share


def settle(
    runner: NeuronHookRunner,
    strengths: Strengths,
    values: Sequence[float],
    first: Sequence[bool],
    top: TopExamples,
    k: int,
) -> tuple[list[ActivatingExample], list[float], bool]:
    """A scan's top examples, counted values and whether the unit has a first-token floor.
    The floor is judged against the scale the unit would have without first tokens (the
    top example's peak after its first token), which is the scale it records if it has the
    floor; with it, first tokens are left out of the examples' ranking and of the values."""
    best = top.top(runner.tokenizer, 1, skip_first=True)
    scale = best[0].max_activation if best else 0.0
    floor = has_first_token_floor(runner, strengths, scale, ReviewPolicy())
    kept = [v for v, f in zip(values, first, strict=True) if not f] if floor else list(values)
    return top.top(runner.tokenizer, k, skip_first=floor), kept, floor


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

    @property
    def first(self) -> torch.Tensor:
        """Each piece's first real token."""
        return self.mask & (self.mask.cumsum(1) == 1)


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

    @property
    def first(self) -> torch.Tensor:
        """``[n]`` bool: each piece's first token (pieces are padded on the right)."""
        return self.positions == 0

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
        self.candidates: list[tuple[Piece, list[int], list[float]]] = []

    def update(self, batch: Batch, strengths: torch.Tensor) -> None:
        for row, piece in enumerate(batch.pieces):
            mask = batch.mask[row]
            self.add(piece, batch.ids[row][mask].tolist(), strengths[row][mask].tolist())

    def update_chunk(self, chunk: Chunk, strengths: torch.Tensor) -> None:
        for piece, rows in chunk.rows():
            self.add(piece, chunk.ids[rows].tolist(), strengths[rows].tolist())

    def add(self, piece: Piece, ids: list[int], values: list[float]) -> None:
        if values:
            self.candidates.append((piece, ids, values))

    def top(self, tokenizer, k: int, skip_first: bool = False) -> list[ActivatingExample]:
        """The ``k`` strongest pieces by their peak, after their first token if
        ``skip_first``."""
        start = int(skip_first)
        ranked = [(max(v[start:]), p, i, v) for p, i, v in self.candidates if len(v) > start]
        examples: list[ActivatingExample] = []
        chosen: list[Piece] = []
        for peak_value, piece, row_ids, row_vals in sorted(ranked, key=lambda c: -c[0]):
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
                    max_token_index=row_vals.index(peak_value, start),
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
    opposite_k: int = 5,
    taggers: Sequence[Tagger] = (),
) -> ScanResult:
    """Scan ``texts`` for one neuron pole, and for its other pole's top ``opposite_k``
    examples and histogram (free: the pass reads the whole activation). With ``cache``,
    replay the layer's stored activations if it holds this scan, and store them if not.
    Each of ``taggers`` labels the counted tokens, and its confound joins the profile's."""
    if runner.tokenizer is None:
        raise ValueError("scan_neuron requires a tokenizer")
    other: Pole = "negative" if pole == "positive" else "positive"
    pieces = list(cut_pieces(runner.tokenizer, texts, window_tokens, stride))
    labels = TokenLabels(runner.tokenizer, texts, taggers)
    scan = _pole_pass(runner, pieces, layer, neuron_idx, (pole, other), batch_tokens, cache, labels)
    read, read_other = (_reader(runner, layer, neuron_idx, p) for p in (pole, other))
    examples, values, floor = settle(runner, read, scan.values, scan.first, scan.top, k)
    other_examples, other_values, other_floor = settle(
        runner, read_other, scan.other_values, scan.first, scan.other_top, opposite_k
    )
    # "Active" for co-activation uses the protocol's own definition for the target.
    active = ReviewPolicy().activation_threshold * max(values, default=0.0)
    confounds = scan.confounds.finalize(runner.tokenizer, min_count)
    if taggers:
        by_label = {n: a.finalize(labels.names[n], active) for n, a in scan.labels.items()}
        confounds = confounds.model_copy(update={"labels": by_label})
    return ScanResult(
        examples=examples,
        values=values,
        correlated=scan.partners.partners(neuron_idx, active, k=n_correlated) if active > 0 else [],
        confounds=confounds,
        windows=len(pieces),
        opposite=OppositeScan(other, other_examples, other_values, other_floor),
        first_token_floor=floor,
    )


@dataclass
class _PolePass:
    """What one pass over the pieces gathers for a neuron's pole and its other pole."""

    partners: PartnerSearch
    confounds: ConfoundAccumulator = field(default_factory=ConfoundAccumulator)
    top: TopExamples = field(default_factory=TopExamples)
    other_top: TopExamples = field(default_factory=TopExamples)
    values: list[float] = field(default_factory=list)
    other_values: list[float] = field(default_factory=list)
    first: list[bool] = field(default_factory=list)
    labels: dict[str, LabelAccumulator] = field(default_factory=dict)


def _pole_pass(runner, pieces, layer: int, neuron_idx: int, poles, batch_tokens, cache, labels):
    pole, other = poles
    scan = _PolePass(PartnerSearch(runner.mlp_width(layer)))
    scan.labels = {t.name: LabelAccumulator() for t in labels.taggers}
    for chunk in scan_chunks(runner, pieces, layer, batch_tokens, cache):
        acts = chunk.acts.float()  # [n, d_mlp]
        strengths = pole_strength(acts[:, neuron_idx], pole)
        counted = chunk.counted
        scan.partners.update(strengths[counted], acts[counted])
        scan.confounds.update(chunk.ids[counted], chunk.positions[counted], strengths[counted])
        _update_labels(scan.labels, labels, chunk, strengths)
        scan.values.extend(strengths[counted].tolist())
        scan.first.extend(chunk.first[counted].tolist())
        scan.top.update_chunk(chunk, strengths)
        other_strengths = pole_strength(acts[:, neuron_idx], other)
        scan.other_values.extend(other_strengths[counted].tolist())
        scan.other_top.update_chunk(chunk, other_strengths)
    return scan


def _update_labels(accumulators, labels: TokenLabels, chunk: Chunk, strengths) -> None:
    """Add the chunk's counted tokens to each label set's accumulator."""
    if not accumulators:
        return
    rows = [labels.of(piece, chunk.ids[r]) for piece, r in chunk.rows()]
    for name, accumulator in accumulators.items():
        every = torch.cat([row[name] for row in rows])
        accumulator.update(every[chunk.counted], strengths[chunk.counted])


def _reader(runner, layer: int, neuron_idx: int, pole: Pole) -> Strengths:
    return lambda inputs: runner.neuron_activations(inputs, layer, neuron_idx, pole)
