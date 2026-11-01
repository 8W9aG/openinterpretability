"""Scanning and patching one attention head.

A head's strength is ``|W_O (z_t − z̄)|`` (see :mod:`openinterp.harness.units`), so the
scan makes two passes over the corpus: the first averages the head's output z over every
counted token to get z̄, the second measures strengths against it. Each pass stops at the
head's layer. Both count the same tokens as a neuron scan (see :mod:`scan`), so the top
examples, histogram and confound profile mean the same thing for both kinds of unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import torch

from openinterp.core.schemas import ActivatingExample, HeadIdentifier, HeadProfile
from openinterp.harness.annotations import Tagger
from openinterp.harness.confounds import ConfoundAccumulator
from openinterp.harness.hooks import Inputs, NeuronHookRunner
from openinterp.harness.patcher import Effect, Positions
from openinterp.harness.scan import (
    BATCH_TOKENS,
    LabelPass,
    ScanResult,
    TopExamples,
    batches,
    cut_pieces,
    settle,
)
from openinterp.harness.units import HeadUnit, head_strength


@dataclass
class HeadScanResult(ScanResult):
    profile: HeadProfile | None = None


@torch.no_grad()
def pattern_scores(
    runner: NeuronHookRunner, layer: int, head: int, *, sequences: int, length: int, seed: int
) -> dict[str, float]:
    """Previous-token, duplicate-token and induction scores on repeated random tokens."""
    vocab = runner.model.get_input_embeddings().num_embeddings
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (sequences, length), generator=generator)
    att = runner.attention_patterns(torch.cat([ids, ids], 1), layer)[:, head].float().cpu()
    query = torch.arange(1, 2 * length)
    second = torch.arange(length, 2 * length)
    return {
        "previous_token_score": float(att[:, query, query - 1].mean()),
        "duplicate_token_score": float(att[:, second, second - length].mean()),
        "induction_score": float(att[:, second, second - length + 1].mean()),
    }


@torch.no_grad()
def peak_attention(
    runner: NeuronHookRunner, example: ActivatingExample, layer: int, head: int, k: int = 3
) -> ActivatingExample:
    """``example`` with the ``k`` tokens its peak token attends to most."""
    att = runner.attention_patterns(example.text, layer)[0, head, example.max_token_index]
    top = att[: example.max_token_index + 1].float().topk(min(k, example.max_token_index + 1))
    return example.model_copy(
        update={
            "attends_to": top.indices.tolist(),
            "attention_weights": [min(1.0, round(float(w), 4)) for w in top.values],
        }
    )


def scan_head(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    ident: HeadIdentifier,
    *,
    k: int = 20,
    batch_tokens: int = BATCH_TOKENS,
    min_count: int = 3,
    window_tokens: int | None = None,
    stride: int | None = None,
    pattern_sequences: int = 16,
    pattern_length: int = 32,
    pattern_seed: int = 0,
    taggers: Sequence[Tagger] = (),
) -> HeadScanResult:
    """Top examples, strengths, confounds and the :class:`HeadProfile` of one head. Each
    of ``taggers`` labels the counted tokens, and its confound joins the profile's.

    Needs eager attention (``attn_implementation="eager"``) for the attention patterns.
    """
    if runner.tokenizer is None:
        raise ValueError("scan_head requires a tokenizer")
    layer, head = ident.layer, ident.head_idx
    pieces = list(cut_pieces(runner.tokenizer, texts, window_tokens, stride))

    mean, count = _mean_output(runner, pieces, layer, head, batch_tokens)
    confounds = ConfoundAccumulator()
    labels = LabelPass(runner.tokenizer, texts, taggers)
    top = TopExamples()
    values: list[float] = []
    first: list[bool] = []
    for batch in batches(runner, pieces, batch_tokens):
        z = runner.head_outputs(batch.encoded, layer)[..., head, :]
        strengths = head_strength(runner, ident, z, mean).cpu()
        counted = batch.counted
        confounds.update(batch.ids[counted], batch.positions[counted], strengths[counted])
        labels.update(batch, strengths)
        values.extend(strengths[counted].tolist())
        first.extend(batch.first[counted].tolist())
        top.update(batch, strengths)
    unit = HeadUnit(ident, tuple(mean))
    read = partial(unit.strengths, runner)
    examples, values, floor = settle(runner, read, values, first, top, k, labels.tokens)

    patterns = (pattern_sequences, pattern_length, pattern_seed)
    profile = _head_profile(runner, layer, head, mean, count, *patterns)
    return HeadScanResult(
        examples=[peak_attention(runner, e, layer, head) for e in examples],
        values=values,
        correlated=[],
        confounds=labels.finalize(confounds.finalize(runner.tokenizer, min_count), values),
        windows=len(pieces),
        first_token_floor=floor,
        profile=profile,
    )


def _mean_output(runner, pieces, layer: int, head: int, batch_tokens: int):
    """The head's mean output over every counted token (one pass), and how many there were."""
    total, count = None, 0
    for batch in batches(runner, pieces, batch_tokens):
        z = runner.head_outputs(batch.encoded, layer)[..., head, :].float().cpu()
        chunk = z[batch.counted].sum(0)
        total = chunk if total is None else total + chunk
        count += int(batch.counted.sum())
    if not count:
        raise ValueError("the corpus has no tokens to scan")
    return [round(float(v), 6) for v in total / count], count


def _head_profile(runner, layer, head, mean, count, sequences, length, seed) -> HeadProfile:
    scores = pattern_scores(runner, layer, head, sequences=sequences, length=length, seed=seed)
    return HeadProfile(
        mean_output=mean,
        mean_tokens=count,
        **{name: round(min(1.0, max(0.0, v)), 6) for name, v in scores.items()},
        pattern_sequences=sequences,
        pattern_length=length,
        pattern_seed=seed,
    )


def head_write_norms(runner: NeuronHookRunner, layer: int, z: torch.Tensor) -> torch.Tensor:
    """``|W_O,h z_h|`` for every head h of ``z`` ``[..., n_heads, head_dim]``."""
    return torch.stack(
        [
            (z[..., h, :].float() @ runner.head_output_weights(layer, h).float().T).norm(dim=-1)
            for h in range(runner.n_heads)
        ],
        dim=-1,
    )


@torch.no_grad()
def patch_heads(
    runner: NeuronHookRunner,
    source: Inputs,
    target: Inputs,
    layer: int,
    heads: Sequence[int],
    *,
    effect: Effect,
    positions: Positions = "all",
    freeze_downstream: bool = False,
    batch_size: int = 8,
) -> list[float]:
    """``effect`` of patching each head's output, one at a time, from source into target.

    The attention-head counterpart of :meth:`ActivationPatcher.patch_many`: each head gets
    its own row of a batch, and later MLP outputs can be frozen against self-repair.
    """
    from openinterp.harness.patcher import ActivationPatcher

    d = runner.head_dim(layer)
    z_src = runner.head_outputs(source, layer).flatten(-2)  # [1, seq, n_heads * d]
    target_enc = runner.encode(target)
    target_logits = runner.run_with_interventions(target_enc)
    downstream = list(range(layer + 1, runner.n_layers)) if freeze_downstream else []
    downstream_out = ActivationPatcher(runner)._cache_outputs(target_enc, downstream)
    if z_src.shape[0] != 1 or target_logits.shape[0] != 1:
        raise ValueError("patch_heads takes one source and one target prompt")
    if positions == "all" and z_src.shape[1] != target_logits.shape[1]:
        raise ValueError("source and target differ in length; use positions='last'")
    span = slice(None) if positions == "all" else slice(-1, None)

    effects: list[float] = []
    for start in range(0, len(heads), batch_size):
        chunk = list(heads[start : start + batch_size])
        batch = {k: v.repeat(len(chunk), *[1] * (v.dim() - 1)) for k, v in target_enc.items()}

        def inject(z: torch.Tensor, chunk=chunk) -> torch.Tensor:
            z = z.clone()
            for row, h in enumerate(chunk):
                cols = slice(h * d, (h + 1) * d)
                z[row, span, cols] = z_src[0, span, cols].to(z.dtype)
            return z

        n = len(chunk)
        frozen = {i: _repeat(cached, n) for i, cached in downstream_out.items()}
        patched = runner.run_with_interventions(batch, None, frozen, {layer: inject})
        effects += [effect(patched[row : row + 1], target_logits) for row in range(n)]
    return effects


def _repeat(cached: torch.Tensor, n: int):
    """An output intervention returning ``cached`` for each of ``n`` batch rows."""
    value = cached.repeat(n, 1, 1)
    return lambda _out: value
