"""A superposition as an auditable unit (issue #3215, ``docs/PROTOCOL.md`` §1).

A superposition is a sparse direction over one layer's MLP hidden activations, the input
to the down projection where neurons live. Its strength is the positive part of the unit
direction ŵ: ``max(0, ŵ · a)``. That is one latent of a ``basis`` dictionary at
``mlp_hidden``. So this module builds such a dictionary in memory, and the feature code
scans, measures, ablates and patches it.

The dictionary's latent 0 is the superposition. Latents 1 to :data:`NULLS` are null
directions for the causal metrics. Each has as many members as the superposition, on
neurons drawn at random. Its weights are the superposition's magnitudes, in a shuffled
order and with random signs. The draws are seeded by the superposition's id, so a record always
measures against the same nulls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from openinterp.core.profiles import SuperpositionProfile
from openinterp.core.schemas import SuperpositionIdentifier
from openinterp.harness.dictionaries import Dictionary
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.units import FeatureUnit

NULLS = 64  # null directions beside the superposition, for the causal metrics


def directions(ident: SuperpositionIdentifier, profile: SuperpositionProfile, width: int):
    """``[width, 1 + NULLS]``: the unit direction ŵ, then the null directions."""
    rng = random.Random(int(ident.superposition_id, 16))
    weights = [m.weight for m in profile.members]
    columns = [list(zip(profile.neurons, weights, strict=True))]
    magnitudes = [abs(w) for w in weights]
    for _ in range(NULLS):
        neurons = rng.sample(range(width), len(weights))
        shuffled = rng.sample(magnitudes, len(magnitudes))
        columns.append(
            [(n, w * rng.choice((1, -1))) for n, w in zip(neurons, shuffled, strict=True)]
        )
    q = torch.zeros(width, len(columns))
    for j, column in enumerate(columns):
        for neuron, weight in column:
            q[neuron, j] = weight
    return q / q.norm(dim=0, keepdim=True)


def build_dictionary(runner: NeuronHookRunner, ident, profile) -> Dictionary:
    """The in-memory ``basis`` dictionary whose latent 0 is the superposition.

    A latent's write is its activation times ``W_down q`` for its direction q. So
    ablating it removes exactly its direction's positive part from the hidden activations.
    """
    width = runner.mlp_width(ident.layer)
    if max(profile.neurons) >= width:
        raise ValueError(f"{ident}: layer {ident.layer} has {width} neurons")
    q = directions(ident, profile, width)
    used = q.abs().sum(dim=1).nonzero().flatten().tolist()  # neurons some direction reads
    out = torch.stack([runner.neuron_output_direction(ident.layer, i).float().cpu() for i in used])
    w_dec = q[used].T @ out  # [n, d_model]
    device = runner.device
    return Dictionary(
        w_enc=q.to(device),
        b_enc=torch.zeros(q.shape[1], device=device),
        w_dec=w_dec.to(device),
        b_dec=torch.zeros(w_dec.shape[1], device=device),
        threshold=None,
        activation="relu",
        subtract_decoder_bias=False,
    )


@dataclass(frozen=True)
class SuperpositionUnit(FeatureUnit):
    """A superposition as an auditable unit: latent 0 of its in-memory dictionary.

    It is a :class:`~openinterp.harness.units.FeatureUnit`, so every measurement that
    reads a feature reads it the same way. ``profile`` is its
    :class:`~openinterp.core.profiles.SuperpositionProfile`.
    """

    _cache: dict[Any, Dictionary] = field(default_factory=dict, compare=False, repr=False)

    def dictionary(self, runner: NeuronHookRunner) -> Dictionary:
        """The superposition's dictionary, built once per runner."""
        key = id(runner)
        if key not in self._cache:
            self._cache[key] = build_dictionary(runner, self.ident, self.profile)
        return self._cache[key]


def scan_superposition(runner: NeuronHookRunner, texts, ident, profile, **options):
    """Scan a superposition as a latent; the result's profile is ``profile`` itself."""
    from openinterp.harness.features import scan_latent

    dictionary = SuperpositionUnit(ident, profile).dictionary(runner)
    return scan_latent(
        runner, texts, ident, dictionary, profile.input_site, lambda **_: profile, **options
    )


def superposition_identifier(model_name: str, layer: int, profile) -> SuperpositionIdentifier:
    """The identifier of the superposition that ``profile`` defines on ``layer``."""
    from openinterp.core.profiles import superposition_id

    sid = superposition_id(profile.members)
    return SuperpositionIdentifier(model_name=model_name, layer=layer, superposition_id=sid)


def superposition_write(runner: NeuronHookRunner, ident, profile) -> torch.Tensor:
    """The residual-stream direction that the superposition writes: ``W_down ŵ``."""
    return SuperpositionUnit(ident, profile).dictionary(runner).w_dec[0]
