"""Per-token labels for the scan, and the confound any label set gives.

A :class:`Tagger` labels character spans of a document. A document label (its language)
is one span over the whole text. A word tagger (part of speech) gives one span per word.
The scan aligns each tagger's spans to the model's tokens once per piece, through the
tokenizer's character offsets. It keeps the labels next to the activations it counts.

For every label set, :class:`LabelAccumulator` reports what ``token_r2`` reports for
token identity, and in information units:

* ``r2``: variance explained by a per-label mean model, over labelled tokens;
* ``mutual_information_bits``: I(activation; label) in bits. Pole strengths are sparse
  and heavy-tailed, so the estimate is binned, not Gaussian. Zero is a bin of its own,
  and positive strengths fall in ``ACTIVATION_BINS`` equal-count bins. Each entropy has
  the Miller–Madow correction. The plug-in estimate is biased upwards by about
  (labels − 1)(bins − 1) / 2N nats when the two are independent.

Some tokens are left out of both numbers: those that a tagger leaves unlabelled. They
are special tokens, words that it doesn't cover, and pieces whose text is not in their
document.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Protocol

import torch

from openinterp.core.confounds import NO_MATCH
from openinterp.core.schemas import ConfoundProfile, LabelConfound, LabelStat, ReviewPolicy

if TYPE_CHECKING:
    from openinterp.harness.scan import Batch, Chunk

ACTIVATION_BINS = 8
MAX_LEVELS = 50  # labels listed in a record, most frequent first
UNLABELLED = -1


@dataclass(frozen=True)
class Span:
    """Characters [start, end) of a document carry ``label``."""

    start: int
    end: int
    label: str


class Tagger(Protocol):
    """Anything that labels character spans of a document, under one label set's name."""

    name: str  # the label set's name in the record (``"language"``)

    def spans(self, doc: int, text: str) -> list[Span]:
        """The labelled spans of document ``doc``, whose text is ``text``."""
        ...


@dataclass(frozen=True)
class DocumentLabels:
    """One label per document, known in advance (a corpus field such as ``lang``)."""

    name: str
    labels: Sequence[str | None]

    def spans(self, doc: int, text: str) -> list[Span]:
        """One span over the whole text, or none when the document has no label."""
        label = self.labels[doc]
        return [] if label is None else [Span(0, len(text), label)]


@dataclass(frozen=True)
class MatchTagger:
    """Label what ``find`` matches in a document, and every other character ``NO_MATCH``."""

    name: str
    find: Callable[[str], Sequence[tuple[int, int, str]]]

    def spans(self, doc: int, text: str) -> list[Span]:
        """The matches, then the gaps, so a token half in a match takes the match's label."""
        matches = self.find(text)
        spans = [Span(start, end, label) for start, end, label in matches]
        position = 0
        for start, end, _ in matches:
            if start > position:
                spans.append(Span(position, start, NO_MATCH))
            position = max(position, end)
        if position < len(text):
            spans.append(Span(position, len(text), NO_MATCH))
        return spans


def align(offsets: Sequence[tuple[int, int]], spans: Sequence[Span]) -> list[str | None]:
    """Return each token's label: that of the span that covers most of its characters.

    The first span wins a tie. ``None`` when no span covers the token. Tokens with no
    characters (special tokens) are unlabelled.
    """
    # A document has about as many syntax spans as tokens, so reading every span for every
    # token is too slow. The spans are sorted by start. A walk back from the token's end
    # stops when no earlier span reaches past the token's start.
    order = sorted(range(len(spans)), key=lambda i: spans[i].start)
    starts = [spans[i].start for i in order]
    reach = list(accumulate((spans[i].end for i in order), max))
    out: list[str | None] = []
    for start, end in offsets:
        best, most, first = None, 0, len(spans)
        i = bisect_left(starts, end) - 1
        while i >= 0 and reach[i] > start:
            span = spans[order[i]]
            overlap = min(end, span.end) - max(start, span.start)
            if overlap > most or (overlap == most and overlap > 0 and order[i] < first):
                best, most, first = span.label, overlap, order[i]
            i -= 1
        out.append(best)
    return out


class TokenLabels:
    """The labels of every piece's tokens for each tagger, aligned once per piece."""

    def __init__(self, tokenizer, texts: Sequence[str], taggers: Sequence[Tagger]) -> None:
        self.tokenizer, self.texts, self.taggers = tokenizer, texts, list(taggers)
        self.names: dict[str, list[str]] = {t.name: [] for t in self.taggers}
        self._index: dict[str, dict[str, int]] = {t.name: {} for t in self.taggers}
        self._spans: dict[tuple[str, int], list[Span]] = {}
        self._pieces: dict = {}

    def of(self, piece, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{tagger: [n] label index or UNLABELLED}`` for the piece's real tokens ``ids``."""
        if piece not in self._pieces:
            self._pieces[piece] = {t.name: self._align(t, piece, ids) for t in self.taggers}
        return self._pieces[piece]

    def at(self, piece, ids: Sequence[int], index: int) -> dict[str, str] | None:
        """Return each tagger's label of token ``index`` of the piece's real tokens ``ids``.

        Only the taggers that label it are included. ``None`` if none does.
        """
        row = self.of(piece, torch.as_tensor(ids, dtype=torch.long))
        found = {name: int(labels[index]) for name, labels in row.items()}
        out = {name: self.names[name][i] for name, i in found.items() if i != UNLABELLED}
        return out or None

    def _align(self, tagger: Tagger, piece, ids: torch.Tensor) -> torch.Tensor:
        key = (tagger.name, piece.doc)
        if key not in self._spans:
            self._spans[key] = tagger.spans(piece.doc, self.texts[piece.doc])
        labels = self._piece_labels(self._spans[key], piece, ids)
        index = self._index[tagger.name]
        for label in labels:
            if label is not None and label not in index:
                index[label] = len(index)
                self.names[tagger.name].append(label)
        ids = [UNLABELLED if lab is None else index[lab] for lab in labels]
        return torch.tensor(ids, dtype=torch.long)

    def _piece_labels(self, spans: list[Span], piece, ids: torch.Tensor) -> list[str | None]:
        if not spans:  # no tokenising for a document with no labels (code with no grammar)
            return [None] * len(ids)
        special = set(getattr(self.tokenizer, "all_special_ids", None) or [])
        text = self.texts[piece.doc]
        if len(spans) == 1 and spans[0].start == 0 and spans[0].end >= len(text):
            # A document label covers every token: no offsets needed.
            return [None if int(i) in special else spans[0].label for i in ids]
        shift = self._shift(piece, text)
        offsets = self._offsets(piece.text) if shift is not None else None
        if offsets is None or len(offsets) != len(ids):
            return [None] * len(ids)
        return align([(a + shift, b + shift) for a, b in offsets], spans)

    def _shift(self, piece, text: str) -> int | None:
        """Return where the piece's text starts in its document.

        ``None`` if it isn't there. A window's decoded text need not match the document's
        characters.
        """
        if piece.start == 0:
            return 0 if text.startswith(piece.text) else None
        doc_offsets = self._offsets(text, special=False)
        if doc_offsets is None or piece.start >= len(doc_offsets):
            return None
        shift = doc_offsets[piece.start][0]
        return shift if text[shift : shift + len(piece.text)] == piece.text else None

    def _offsets(self, text: str, special: bool = True) -> list[tuple[int, int]] | None:
        """Return the character offsets of ``text``'s tokens as the scan encodes them.

        ``None`` if the tokenizer can't give them. Only fast tokenizers can.
        """
        try:
            enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=special)
        except (TypeError, NotImplementedError):
            return None
        return [tuple(o) for o in enc["offset_mapping"]]


class LabelAccumulator:
    """Pole strengths and label indices of every counted token, for one label set."""

    def __init__(self) -> None:
        self.labels: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def update(self, labels: torch.Tensor, strengths: torch.Tensor) -> None:
        """Add a batch of label indices and strengths."""
        self.labels.append(labels.long())
        self.values.append(strengths.double())

    def finalize(self, names: Sequence[str], line: float) -> LabelConfound:
        """The confound, counting a token active at or above ``line``."""
        labels = torch.cat(self.labels) if self.labels else torch.zeros(0, dtype=torch.long)
        values = torch.cat(self.values) if self.values else torch.zeros(0, dtype=torch.float64)
        seen = labels != UNLABELLED
        labels, values = labels[seen], values[seen]
        size = len(names)
        n = torch.bincount(labels, minlength=size).double()
        s = torch.bincount(labels, weights=values, minlength=size)
        active = torch.bincount(labels[values >= line], minlength=size) if line > 0 else n * 0
        order = n.argsort(descending=True)[:MAX_LEVELS]
        return LabelConfound(
            r2=round(_r2(values, n, s), 6),
            mutual_information_bits=round(mutual_information(values, labels, size), 6),
            labelled_tokens=int(labels.numel()),
            unlabelled_tokens=int((~seen).sum()),
            levels=[
                LabelStat(
                    label=names[i],
                    count=int(n[i]),
                    mean_strength=round(max(0.0, float(s[i] / n[i])), 6),
                    active=int(active[i]),
                )
                for i in order.tolist()
                if n[i] > 0
            ],
        )


def _r2(values: torch.Tensor, n: torch.Tensor, s: torch.Tensor) -> float:
    """Between-label sum of squares over total sum of squares."""
    count = values.numel()
    if count == 0:
        return 0.0
    total = float((values * values).sum()) - float(values.sum()) ** 2 / count
    seen = n > 0
    between = float((s[seen] ** 2 / n[seen]).sum()) - float(values.sum()) ** 2 / count
    return min(1.0, max(0.0, between / total)) if total > 1e-12 else 0.0


def activation_bins(values: torch.Tensor, bins: int = ACTIVATION_BINS) -> torch.Tensor:
    """Return each value's bin: 0 for zero strength, then equal-count bins of the rest.

    There are ``bins`` bins of the positive strengths, or fewer where ties make edges
    coincide.
    """
    out = torch.zeros(values.numel(), dtype=torch.long)
    positive = values > 0
    if positive.any():
        q = torch.linspace(0, 1, bins + 1, dtype=torch.float64)[1:-1]
        edges = torch.quantile(values[positive].double(), q).unique()
        out[positive] = 1 + torch.bucketize(values[positive], edges, right=True)
    return out


def mutual_information(values: torch.Tensor, labels: torch.Tensor, n_labels: int) -> float:
    """I(binned strength; label) in bits, each entropy Miller–Madow corrected; at least 0."""
    count = values.numel()
    if count == 0 or n_labels < 2:
        return 0.0
    bins = activation_bins(values)
    joint = torch.bincount(bins * n_labels + labels).double()
    mi = _entropy(torch.bincount(bins).double(), count)
    mi += _entropy(torch.bincount(labels).double(), count) - _entropy(joint, count)
    return max(0.0, mi / math.log(2))


def _entropy(counts: torch.Tensor, total: int) -> float:
    """Miller–Madow entropy in nats."""
    counts = counts[counts > 0]
    p = counts / total
    return float(-(p * p.log()).sum()) + (counts.numel() - 1) / (2 * total)


class LabelPass:
    """Each tagger's labels of a scan's counted tokens, with their strengths.

    They are gathered pass by pass. They give the label confounds (see
    ``harness/annotations.py``).
    """

    def __init__(self, tokenizer, texts: Sequence[str], taggers: Sequence[Tagger]) -> None:
        self.tokens = TokenLabels(tokenizer, texts, taggers)
        self.accumulators = {t.name: LabelAccumulator() for t in self.tokens.taggers}

    def update_chunk(self, chunk: Chunk, strengths: torch.Tensor) -> None:
        """Add a chunk's counted tokens; ``strengths`` is ``[n]``, one per real token."""
        if self.accumulators:
            rows = [self.tokens.of(piece, chunk.ids[r]) for piece, r in chunk.rows()]
            self._add(rows, chunk.counted, strengths)

    def update(self, batch: Batch, strengths: torch.Tensor) -> None:
        """Add a batch's counted tokens; ``strengths`` is ``[b, seq]``, padding included."""
        if self.accumulators:
            mask = batch.mask
            rows = [self.tokens.of(p, batch.ids[i][mask[i]]) for i, p in enumerate(batch.pieces)]
            self._add(rows, batch.counted[mask], strengths.cpu()[mask])

    def _add(self, rows, counted: torch.Tensor, strengths: torch.Tensor) -> None:
        for name, accumulator in self.accumulators.items():
            every = torch.cat([row[name] for row in rows])
            accumulator.update(every[counted], strengths[counted])

    def finalize(self, confounds: ConfoundProfile, values: Sequence[float]) -> ConfoundProfile:
        """Return ``confounds`` with each label set's confound added.

        A token counts as active at or above the protocol's line for the scan's settled
        ``values``.
        """
        if not self.accumulators:
            return confounds
        line = ReviewPolicy().activation_threshold * max(values, default=0.0)
        names = self.tokens.names
        by_label = {n: a.finalize(names[n], line) for n, a in self.accumulators.items()}
        return confounds.model_copy(update={"labels": by_label})
