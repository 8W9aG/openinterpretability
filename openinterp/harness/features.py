"""Scanning and patching one latent of a sparse dictionary (a transcoder or SAE).

A latent's strength is its activation (see :mod:`openinterp.harness.units`). The scan
reads the dictionary's input at its site in one pass that stops at the feature's layer,
and counts the same tokens as a neuron scan (see :mod:`scan`), so the top examples,
histogram and confound profile mean the same thing for every kind of unit. The same
pass measures how well the whole dictionary reconstructs its output there, leaving
out each piece's first token, so a record says whether the baseline it belongs to was a
faithful one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from openinterp.core.schemas import DictionaryProfile, FeatureIdentifier
from openinterp.harness.annotations import Tagger
from openinterp.harness.confounds import ConfoundAccumulator
from openinterp.harness.dictionaries import Dictionary, DictionarySpec, load_spec, spec_profile
from openinterp.harness.heads import _repeat
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
from openinterp.harness.sites import dictionary_io

ENCODE_CHUNK = 1024  # tokens encoded through the whole dictionary at once


@dataclass
class FeatureScanResult(ScanResult):
    profile: DictionaryProfile | None = None


class _Fit:
    """Running reconstruction error, output variance and L0 of a dictionary."""

    def __init__(self) -> None:
        self.tokens = 0
        self.error = self.sum_sq = self.l0 = 0.0
        self.total: torch.Tensor | None = None
        self.fired: torch.Tensor | None = None  # [n] latents active on some token

    def update(self, dictionary: Dictionary, x: torch.Tensor, out: torch.Tensor) -> None:
        for start in range(0, x.shape[0], ENCODE_CHUNK):
            xs, target = x[start : start + ENCODE_CHUNK], out[start : start + ENCODE_CHUNK]
            target = target.float().to(dictionary.w_dec.device)
            acts = dictionary.encode(xs)
            self.error += float(((dictionary.reconstruct(xs, acts) - target) ** 2).sum())
            self.l0 += float((acts > 0).sum())
            fired = (acts > 0).any(0)
            self.fired = fired if self.fired is None else self.fired | fired
            exact = target.cpu().double()  # MPS has no float64
            self.sum_sq += float((exact**2).sum())
            chunk = exact.sum(0)
            self.total = chunk if self.total is None else self.total + chunk
        self.tokens += x.shape[0]

    def stats(self) -> dict[str, float]:
        variance = self.sum_sq - float((self.total**2).sum()) / self.tokens
        explained = 1 - self.error / variance if variance > 0 else None
        return {
            "explained_variance": None if explained is None else round(explained, 6),
            "mean_l0": round(self.l0 / self.tokens, 4),
            "dead_share": round(1 - float(self.fired.float().mean()), 6),
        }


def scan_feature(
    runner: NeuronHookRunner,
    texts: Sequence[str],
    ident: FeatureIdentifier,
    spec: DictionarySpec,
    *,
    k: int = 20,
    batch_tokens: int = BATCH_TOKENS,
    min_count: int = 3,
    window_tokens: int | None = None,
    stride: int | None = None,
    taggers: Sequence[Tagger] = (),
) -> FeatureScanResult:
    """Top examples, strengths, confounds and the :class:`DictionaryProfile` of one latent.
    Each of ``taggers`` labels the counted tokens, and its confound joins the profile's."""
    if runner.tokenizer is None:
        raise ValueError("scan_feature requires a tokenizer")
    layer, f = ident.layer, ident.feature_idx
    dictionary = load_spec(spec, layer, runner.device)
    if f >= dictionary.n_features:
        raise ValueError(f"{ident}: the dictionary has {dictionary.n_features} latents")
    pieces = list(cut_pieces(runner.tokenizer, texts, window_tokens, stride))
    labels = LabelPass(runner.tokenizer, texts, taggers)
    confounds, top, fit, values, first = _feature_pass(
        runner, dictionary, spec, layer, f, pieces, batch_tokens, labels
    )
    if not values:
        raise ValueError("the corpus has no tokens to scan")

    def read(inputs):
        x, _ = dictionary_io(runner, inputs, layer, spec.input_site)
        return dictionary.feature(x, f)

    density = round(sum(v > 0 for v in values) / len(values), 6)
    tokens = len(values)
    examples, values, floor = settle(runner, read, values, first, top, k, labels.tokens)
    profile = spec_profile(
        spec,
        layer,
        n_features=dictionary.n_features,
        density=density,
        tokens=tokens,
        **fit.stats(),
    )
    return FeatureScanResult(
        examples=examples,
        values=values,
        correlated=[],
        confounds=labels.finalize(confounds.finalize(runner.tokenizer, min_count), values),
        windows=len(pieces),
        first_token_floor=floor,
        profile=profile,
    )


def _feature_pass(
    runner, dictionary, spec, layer: int, f: int, pieces, batch_tokens: int, labels: LabelPass
):
    """One pass over ``pieces``: confounds (and ``labels``' counts), top examples, the
    dictionary's fit, and each counted token's strength and whether it is a piece's first
    token."""
    confounds = ConfoundAccumulator()
    top = TopExamples()
    fit = _Fit()
    values: list[float] = []
    first: list[bool] = []
    for batch in batches(runner, pieces, batch_tokens):
        x, out = dictionary_io(runner, batch.encoded, layer, spec.input_site)
        strengths = dictionary.feature(x, f).cpu()
        counted = batch.counted
        confounds.update(batch.ids[counted], batch.positions[counted], strengths[counted])
        labels.update(batch, strengths)
        values.extend(strengths[counted].tolist())
        first.extend(batch.first[counted].tolist())
        top.update(batch, strengths)
        # Not a piece's first token: dictionaries aren't trained on that attention sink,
        # whose error can be 10^8 times the output's variance and would swamp the fit.
        fitted = (counted & ~batch.first).to(x.device)
        fit.update(dictionary, x[fitted], out[fitted])
    return confounds, top, fit, values, first


@torch.no_grad()
def latent_activations(
    runner: NeuronHookRunner, dictionary: Dictionary, inputs: Inputs, layer: int, site: str
) -> torch.Tensor:
    """Every latent's activation on one prompt: ``[seq, n]``."""
    x, _ = dictionary_io(runner, inputs, layer, site)
    return dictionary.encode(x[0])


@torch.no_grad()
def write_changes(
    runner: NeuronHookRunner,
    dictionary: Dictionary,
    ident: FeatureIdentifier,
    site: str,
    base: str,
    cf: str,
) -> tuple[Positions, int, torch.Tensor]:
    """Where to patch and score, and every latent's write change ``|Δa| · |W_dec[f]|`` up
    to there, in units of the audited latent's activation (so its own entry is ``|Δa|``).

    Latents' decoder norms differ, so comparing writes, as heads do, matches the null
    latents to the audited one by how much each moves the output they write to.
    """
    a_b, a_c = (latent_activations(runner, dictionary, t, ident.layer, site) for t in (base, cf))
    norms = dictionary.w_dec.norm(dim=-1)
    scale = norms / norms[ident.feature_idx].clamp(min=1e-12)
    if a_b.shape != a_c.shape:
        return "last", -1, (a_b[-1] - a_c[-1]).abs() * scale
    position = int((a_b[:, ident.feature_idx] - a_c[:, ident.feature_idx]).abs().argmax())
    return "all", position, (a_b[: position + 1] - a_c[: position + 1]).abs().amax(0) * scale


@torch.no_grad()
def patch_features(
    runner: NeuronHookRunner,
    dictionary: Dictionary,
    site: str,
    source: Inputs,
    target: Inputs,
    layer: int,
    features: Sequence[int],
    *,
    effect: Effect,
    positions: Positions = "all",
    freeze_downstream: bool = False,
    batch_size: int = 8,
) -> list[float]:
    """``effect`` of patching each latent's activation, one at a time, from source into
    target: the dictionary's output (the MLP's, or the residual stream) gains
    ``(a_source − a_target) · W_dec[f]``.

    The dictionary counterpart of :meth:`ActivationPatcher.patch_many`: each latent gets
    its own row of a batch, and later MLP outputs can be frozen against self-repair.
    """
    from openinterp.harness.patcher import ActivationPatcher

    a_src = latent_activations(runner, dictionary, source, layer, site)  # [seq, n]
    target_enc = runner.encode(target)
    target_logits = runner.run_with_interventions(target_enc)
    if target_logits.shape[0] != 1:
        raise ValueError("patch_features takes one source and one target prompt")
    if positions == "all" and a_src.shape[0] != target_logits.shape[1]:
        raise ValueError("source and target differ in length; use positions='last'")
    downstream = list(range(layer + 1, runner.n_layers)) if freeze_downstream else []
    downstream_out = ActivationPatcher(runner)._cache_outputs(target_enc, downstream)
    span = slice(None) if positions == "all" else slice(-1, None)

    effects: list[float] = []
    for start in range(0, len(features), batch_size):
        chunk = torch.tensor(list(features[start : start + batch_size]))
        n = len(chunk)
        batch = {k: v.repeat(n, *[1] * (v.dim() - 1)) for k, v in target_enc.items()}
        idx = chunk.to(dictionary.w_dec.device)

        def inject(x: torch.Tensor, out: torch.Tensor, idx=idx, n=n) -> torch.Tensor:
            a_tgt = dictionary.feature(x[0, span], idx)  # [span, n]; every row is the target
            delta = (a_src[span][:, idx] - a_tgt).T  # [n, span]
            out = out.clone()
            write = delta.unsqueeze(-1) * dictionary.w_dec[idx].unsqueeze(1)  # [n, span, d]
            out[:n, span] += write.to(out.dtype)
            return out

        frozen = {i: _repeat(cached, n) for i, cached in downstream_out.items()}
        patched = runner.run_with_interventions(
            batch, None, frozen, dictionary_interventions={layer: (site, inject)}
        )
        effects += [effect(patched[row : row + 1], target_logits) for row in range(n)]
    return effects
