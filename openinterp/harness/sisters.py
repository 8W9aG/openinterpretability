"""Sister neurons: the poles of another model whose activity tracks a pole's.

Both models read the same corpus, and are compared where both have just read the same
text: at *shared boundaries*, character offsets where a token ends in both models'
tokenizations. Models that share a tokenizer share every boundary; across tokenizers
(GPT-2 and Qwen2.5) most word ends are shared. Since both models are causal, each sees
exactly the same text up to a shared boundary, so a pole of one and a pole of the other
give two series over the same positions. Their Pearson correlation says how far they
fire together; a pole whose best match correlates strongly, and is in
turn that match's best partner in the first model (a *mutual* best match), is a
candidate for the same feature in both models (Gurnee et al., 2024, "Universal Neurons
in GPT2 Language Models", match neurons across seeds this way).

Each series is a pole strength (:func:`~openinterp.harness.hooks.pole_strength`), so a
neuron's two poles are matched separately. A text's first token is left out: attention
there sees only the token itself, and many units take extreme values on it that would
dominate the correlation.

``rotated`` puts a match in context. It is the best correlation the pole reaches with any
unit of a Haar-random rotation of the match's layer (as ``docs/analyses/
null-calibration.md`` rotates a basis): the same number of candidates, mixing the same
activations, but not the model's own neurons. A sister well above it is a fact about the
other model's neuron basis, not only about what the layer can represent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch

from openinterp.core.schemas import Pole
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.scan import BATCH_TOKENS, plan_batches

POLES: tuple[Pole, Pole] = ("positive", "negative")

# A unit: (layer, neuron, pole).
Unit = tuple[int, int, Pole]


@dataclass(frozen=True)
class Match:
    layer: int
    neuron_idx: int
    pole: Pole
    pearson: float


def pole_columns(acts: torch.Tensor) -> torch.Tensor:
    """``[n, w]`` signed activations → ``[n, 2w]`` pole strengths, positive poles first."""
    return torch.cat([acts.clamp(min=0), (-acts).clamp(min=0)], dim=1)


def _unit(layer: int, column: int, width: int) -> Unit:
    return layer, column % width, POLES[column // width]


class Correlations:
    """Streaming Pearson correlation between ``P`` source series and every column of
        some target matrices (one per layer). Each batch's sums are taken on the device in
    the inputs' dtype (MPS has no float64); the running totals are float64 on the CPU."""

    def __init__(self, n_sources: int) -> None:
        self.n = 0
        self.sx = torch.zeros(n_sources, dtype=torch.float64)
        self.sxx = torch.zeros(n_sources, dtype=torch.float64)
        self.sy: dict[int, torch.Tensor] = {}
        self.syy: dict[int, torch.Tensor] = {}
        self.sxy: dict[int, torch.Tensor] = {}

    def add_sources(self, x: torch.Tensor) -> None:
        """Count one batch's ``[n, P]`` source rows; call once per batch."""
        self.n += x.shape[0]
        self.sx += x.sum(0).cpu().double()
        self.sxx += (x * x).sum(0).cpu().double()

    def add_targets(self, key: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """Pair the batch's ``[n, P]`` sources with target ``key``'s ``[n, C]`` columns."""
        xy = (x.to(y.dtype).T @ y).cpu().double()
        if key not in self.sxy:
            self.sxy[key] = torch.zeros_like(xy)
            self.sy[key] = torch.zeros(y.shape[1], dtype=torch.float64)
            self.syy[key] = torch.zeros(y.shape[1], dtype=torch.float64)
        self.sxy[key] += xy
        self.sy[key] += y.sum(0).cpu().double()
        self.syy[key] += (y * y).sum(0).cpu().double()

    def pearson(self, key: int) -> torch.Tensor:
        """``[P, C]`` correlations with target ``key``; 0 where either side is constant."""
        n = max(self.n, 1)
        mx, my = self.sx / n, self.sy[key] / n
        cov = self.sxy[key] / n - mx[:, None] * my[None, :]
        var = (self.sxx / n - mx**2)[:, None] * (self.syy[key] / n - my**2)[None, :]
        denom = var.clamp(min=0).sqrt()
        return torch.where(denom > 1e-12, cov / denom, torch.zeros_like(cov)).clamp(-1, 1)


def _seed(model_name: str, layer: int) -> int:
    digest = hashlib.sha256(f"openinterp sisters {model_name} {layer}".encode()).digest()
    return int.from_bytes(digest[:8])


def rotation(model_name: str, layer: int, width: int) -> torch.Tensor:
    """A Haar-random orthogonal ``[width, width]`` matrix (QR of a Gaussian, signs
    fixed), seeded by model and layer so a rerun draws the same one."""
    generator = torch.Generator().manual_seed(_seed(model_name, layer))
    q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


@dataclass(frozen=True)
class Tokens:
    """One text as one model reads it: token ids, and the character offset each token
    ends at (-1 for a token that covers no characters, such as a special token)."""

    ids: list[int]
    ends: list[int]


def _context(runner: NeuronHookRunner) -> int | None:
    config = getattr(runner.model, "config", None)
    return getattr(config, "n_positions", None) or getattr(config, "max_position_embeddings", None)


def tokenize(runner: NeuronHookRunner, text: str) -> Tokens:
    """``text`` as ``runner`` reads it, cut at the model's context length."""
    enc = runner.tokenizer(text, return_offsets_mapping=True)
    ends = [end if end > start else -1 for start, end in enc["offset_mapping"]]
    limit = _context(runner)
    return Tokens(list(enc["input_ids"])[:limit], ends[:limit])


def shared_boundaries(a: Sequence[int], b: Sequence[int]) -> tuple[list[int], list[int]]:
    """The positions in two tokenizations (token end offsets ``a`` and ``b``) where both
    have just read the same text: tokens that end at the same character. Where several
    tokens end at one character, the last counts. Each text's first token is left out."""
    last_a = {end: i for i, end in enumerate(a) if end >= 0 and i > 0}
    last_b = {end: i for i, end in enumerate(b) if end >= 0 and i > 0}
    shared = sorted(last_a.keys() & last_b.keys())
    return [last_a[e] for e in shared], [last_b[e] for e in shared]


def paired_rows(
    a: NeuronHookRunner,
    b: NeuronHookRunner,
    texts: Sequence[str],
    layers_a: Iterable[int],
    layers_b: Iterable[int],
    batch_tokens: int = BATCH_TOKENS,
) -> Iterator[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]]:
    """Both models' activations ``{layer: [n, width]}`` at the shared boundaries of each
    group of documents, row for row. Each document is one forward pass of each model."""
    docs = [(tokenize(a, t), tokenize(b, t)) for t in texts]
    docs = [(x, y) for x, y in docs if x.ids and y.ids]
    lengths = [max(len(x.ids), len(y.ids)) for x, y in docs]
    layers_a, layers_b, start = sorted(set(layers_a)), sorted(set(layers_b)), 0
    for size in plan_batches(lengths, batch_tokens):
        group, start = docs[start : start + size], start + size
        pairs = [shared_boundaries(x.ends, y.ends) for x, y in group]
        yield (
            _gather(a, [x.ids for x, _ in group], [p[0] for p in pairs], layers_a),
            _gather(b, [y.ids for _, y in group], [p[1] for p in pairs], layers_b),
        )


def _gather(runner: NeuronHookRunner, seqs, positions, layers) -> dict[int, torch.Tensor]:
    """Run ``seqs`` (right-padded) and take ``layers``' activations at ``positions``."""
    width = max(map(len, seqs))
    pad = getattr(runner.tokenizer, "pad_token_id", None) or 0
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs])
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
    _, cache = runner.run_with_cache({"input_ids": ids, "attention_mask": mask}, layers)
    rows = [(row, p) for row, pos in enumerate(positions) for p in pos]
    index = torch.tensor(rows, dtype=torch.long).reshape(-1, 2).to(runner.device)
    return {layer: cache[layer][index[:, 0], index[:, 1]].float() for layer in layers}


def _source_rows(rows: dict[int, torch.Tensor], units: Sequence[Unit]) -> torch.Tensor:
    """``[n, len(units)]`` pole strengths of ``units``."""
    cols = []
    for layer, idx, pole in units:
        act = rows[layer][:, idx]
        cols.append(act.clamp(min=0) if pole == "positive" else (-act).clamp(min=0))
    return torch.stack(cols, dim=1)


def _top(corr: Correlations, keys: Iterable[int], widths: dict[int, int], k: int):
    """Each source's ``k`` best target units over ``keys``, by Pearson."""
    best: list[list[Match]] = [[] for _ in range(corr.sx.shape[0])]
    for key in keys:
        r = corr.pearson(key)
        values, cols = r.topk(min(k, r.shape[1]), dim=1)
        for i in range(r.shape[0]):
            for value, col in zip(values[i].tolist(), cols[i].tolist(), strict=True):
                layer, idx, pole = _unit(key, col, widths[key])
                best[i].append(Match(layer, idx, pole, value))
    return [sorted(m, key=lambda m: -m.pearson)[:k] for m in best]


@dataclass(frozen=True)
class Sisters:
    """One source pole's best matches in the other model, with each match's own best partner
    back in the source model (``back``) and the rotated-basis reference."""

    matches: list[Match]
    back: list[Match]  # per match: its best partner among the source model's units
    rotated: float  # best Pearson with a unit of a rotated basis of matches[0]'s layer


def find_sisters(
    source: NeuronHookRunner,
    other: NeuronHookRunner,
    units: Sequence[Unit],
    texts: Sequence[str],
    *,
    other_name: str,
    k: int = 5,
    batch_tokens: int = BATCH_TOKENS,
    progress=None,
) -> tuple[dict[Unit, Sisters], int]:
    """Each of ``units`` (in ``source``) with its sisters in ``other``, over ``texts``,
    and how many token positions were paired. Two passes over the corpus: the first finds
    each unit's ``k`` best poles over all of ``other``'s layers; the second, each of those
    poles' best partner over all of ``source``'s layers, and the rotated reference."""
    units = list(dict.fromkeys(units))
    pair = _Pair(source, other, texts, batch_tokens, progress or (lambda *_: None))
    matches = _forward(pair, units, k)
    back_of, rotated, tokens = _backward(pair, units, matches, other_name)
    out = {}
    for i, unit in enumerate(units):
        back = [back_of[(m.layer, m.neuron_idx, m.pole)] for m in matches[i]]
        out[unit] = Sisters(matches[i], back, rotated.get(i, 0.0))
    return out, tokens


@dataclass(frozen=True)
class _Pair:
    source: NeuronHookRunner
    other: NeuronHookRunner
    texts: Sequence[str]
    batch_tokens: int
    tick: Callable[[str, int], None]

    def rows(self, source_layers: Iterable[int], other_layers: Iterable[int]):
        return paired_rows(
            self.source, self.other, self.texts, source_layers, other_layers, self.batch_tokens
        )

    @staticmethod
    def widths(runner: NeuronHookRunner) -> dict[int, int]:
        return {layer: runner.mlp_width(layer) for layer in range(runner.n_layers)}


def _forward(pair: _Pair, units: Sequence[Unit], k: int) -> list[list[Match]]:
    """Pass 1: each unit's ``k`` best poles over every layer of the other model."""
    every = range(pair.other.n_layers)
    corr = Correlations(len(units))
    for src, oth in pair.rows({layer for layer, _, _ in units}, every):
        x = _source_rows(src, units)
        corr.add_sources(x)
        for layer, acts in oth.items():
            corr.add_targets(layer, x.to(acts.device), pole_columns(acts))
        pair.tick("forward", corr.n)
    return _top(corr, every, _Pair.widths(pair.other), k)


def _backward(pair: _Pair, units: Sequence[Unit], matches: list[list[Match]], name: str):
    """Pass 2: each match's best pole over every source layer (``{unit: Match}``), each
    unit's rotated reference at its top match's layer (``{unit index: r}``), and how many
    positions were paired."""
    found = list(dict.fromkeys((m.layer, m.neuron_idx, m.pole) for ms in matches for m in ms))
    by_layer: dict[int, list[int]] = {}
    for i, ms in enumerate(matches):
        if ms:
            by_layer.setdefault(ms[0].layer, []).append(i)
    widths = _Pair.widths(pair.other)
    device = pair.other.device
    turns = {layer: rotation(name, layer, widths[layer]).to(device) for layer in by_layer}
    back = Correlations(len(found))
    rotated = {layer: Correlations(len(idx)) for layer, idx in by_layer.items()}
    every = range(pair.source.n_layers)
    other_layers = {layer for layer, _, _ in found} | set(by_layer)
    for src, oth in pair.rows(every, other_layers):
        x = _source_rows(oth, found)
        back.add_sources(x)
        for layer, acts in src.items():
            back.add_targets(layer, x.to(acts.device), pole_columns(acts))
        own = _source_rows(src, units)
        for layer, idx in by_layer.items():
            xs = own[:, idx]
            rotated[layer].add_sources(xs)
            rotated[layer].add_targets(
                layer, xs.to(device), pole_columns(oth[layer] @ turns[layer])
            )
        pair.tick("backward", back.n)
    best = _top(back, every, _Pair.widths(pair.source), 1)
    reference = {}
    for layer, idx in by_layer.items():
        r = rotated[layer].pearson(layer).amax(dim=1)
        reference.update({i: float(v) for i, v in zip(idx, r.tolist(), strict=True)})
    return {unit: m[0] for unit, m in zip(found, best, strict=True)}, reference, back.n
