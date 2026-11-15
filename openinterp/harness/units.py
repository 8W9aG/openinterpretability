"""What "strength" means for each kind of auditable unit, so one protocol covers them all.

Every measurement in an audit reads one non-negative strength per token: the scan's
top examples, probes, counterfactual Δ, held-out windows. A unit supplies it, and the
ablation the lookahead profile uses:

* An MLP neuron pole: its pole strength. Ablation zeroes the pole.
* An attention head: ``|W_O (z_t − z̄)|``. This is the distance between its write to the
  residual stream and its mean write over the scan. z̄ is stored in
  :class:`~openinterp.core.schemas.HeadProfile`. Ablation replaces z with z̄ (mean
  ablation). So strength is exactly how far ablating the head moves the residual.
* A superposition: the positive part of its unit direction over one layer's neurons. It is
  scanned and ablated as latent 0 of an in-memory dictionary
  (:mod:`openinterp.harness.superpositions`).
* A dictionary feature: its latent's activation ``a_f`` (non-negative). Ablation removes
  its write ``a_f · W_dec[f]`` from the MLP output, or from the residual stream for a
  residual-stream SAE. The dictionary's error and every other latent stay as they were.

A unit with a *first-token floor* (protocol 0.8) is not measured at a text's first
token. The first token has no context and is the attention sink. Some early-layer units
sit above their line there whatever the token, so every probe would fire. A unit
has the floor if it is active at the first position on at least half the vocabulary
(:func:`openinterp.harness.scan.first_position_share`). A prompt's peak is then over
its other tokens (``skip_first``). Every other unit is measured at every token, so one
that detects *which* token opens a text can still be audited.

The raw write norm ``|W_O z_t|`` would not do. Most heads write a sizeable resting
vector, often while attending to the first token. So in GPT-2 small, an induction head's
median token sits at a quarter of its maximum. Half of all text would count as "active".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from openinterp.core.schemas import (
    ComponentIdentifier,
    DictionaryProfile,
    FeatureIdentifier,
    HeadIdentifier,
    NeuronIdentifier,
    SuperpositionIdentifier,
)
from openinterp.harness.dictionaries import Dictionary, load_dictionary
from openinterp.harness.hooks import Inputs, NeuronHookRunner, ablate_head, ablate_pole
from openinterp.harness.sites import dictionary_io


@dataclass(frozen=True)
class NeuronUnit:
    """An MLP neuron pole as an auditable unit."""

    ident: NeuronIdentifier
    skip_first: bool = False

    def strengths(self, runner: NeuronHookRunner, inputs: Inputs) -> torch.Tensor:
        """Per-token pole strength ``[batch, seq]``."""
        ident = self.ident
        return runner.neuron_activations(inputs, ident.layer, ident.neuron_idx, ident.pole)

    def ablation(self, runner: NeuronHookRunner) -> dict[str, Mapping[int, Any]]:
        """``run_with_interventions`` keyword arguments that zero the pole."""
        ident = self.ident
        return {"interventions": {ident.layer: ablate_pole(ident.neuron_idx, ident.pole)}}


@dataclass(frozen=True)
class HeadUnit:
    """An attention head as an auditable unit, with the mean output that defines its strength."""

    ident: HeadIdentifier
    mean_output: tuple[float, ...]
    skip_first: bool = False

    def strengths(self, runner: NeuronHookRunner, inputs: Inputs) -> torch.Tensor:
        """Per-token ``|W_O (z − z̄)|`` ``[batch, seq]``."""
        z = runner.head_outputs(inputs, self.ident.layer)[..., self.ident.head_idx, :]
        return head_strength(runner, self.ident, z, self.mean_output)

    def ablation(self, runner: NeuronHookRunner) -> dict[str, Mapping[int, Any]]:
        """``run_with_interventions`` keyword arguments that mean-ablate the head."""
        ident = self.ident
        mean = torch.tensor(self.mean_output, device=runner.device)
        fn = ablate_head(ident.head_idx, runner.head_dim(ident.layer), mean)
        return {"head_interventions": {ident.layer: fn}}


@dataclass(frozen=True)
class FeatureUnit:
    """A dictionary feature as an auditable unit, with its pinned dictionary."""

    ident: FeatureIdentifier
    profile: DictionaryProfile
    skip_first: bool = False

    def dictionary(self, runner: NeuronHookRunner) -> Dictionary:
        """The feature's dictionary, loaded on the runner's device."""
        return load_dictionary(self.profile, runner.device)

    def strengths(self, runner: NeuronHookRunner, inputs: Inputs) -> torch.Tensor:
        """Per-token latent activation ``[batch, seq]``."""
        x, _ = dictionary_io(runner, inputs, self.ident.layer, self.profile.input_site)
        return self.dictionary(runner).feature(x, self.ident.feature_idx)

    def ablation(self, runner: NeuronHookRunner) -> dict[str, Mapping[int, Any]]:
        """``run_with_interventions`` keyword arguments that remove the latent's write."""
        dictionary, f = self.dictionary(runner), self.ident.feature_idx
        direction = dictionary.w_dec[f]

        def ablate(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            write = dictionary.feature(x, f).unsqueeze(-1) * direction
            return out - write.to(out.dtype)

        return {"dictionary_interventions": {self.ident.layer: (self.profile.input_site, ablate)}}


Unit = NeuronUnit | HeadUnit | FeatureUnit


def head_strength(
    runner: NeuronHookRunner, ident: HeadIdentifier, z: torch.Tensor, mean_output
) -> torch.Tensor:
    """``|W_O (z − z̄)|`` over the last dimension of ``z`` (one head's outputs)."""
    w_o = runner.head_output_weights(ident.layer, ident.head_idx).float()
    mean = torch.as_tensor(mean_output, dtype=torch.float32, device=z.device)
    return ((z.float() - mean) @ w_o.T.to(z.device)).norm(dim=-1)


def unit_of(
    ident: ComponentIdentifier,
    head_profile=None,
    dictionary_profile=None,
    skip_first=False,
    superposition_profile=None,
) -> Unit:
    """Return the unit that ``ident`` names.

    A head, a feature or a superposition needs the profile that defines its strength.
    """
    if isinstance(ident, SuperpositionIdentifier):
        from openinterp.harness.superpositions import SuperpositionUnit

        if superposition_profile is None:
            raise ValueError(f"{ident} has no superposition profile, so its strength is undefined")
        return SuperpositionUnit(ident, superposition_profile, skip_first)
    if isinstance(ident, HeadIdentifier):
        if head_profile is None:
            raise ValueError(f"{ident} has no head profile, so its strength is undefined")
        return HeadUnit(ident, tuple(head_profile.mean_output), skip_first)
    if isinstance(ident, FeatureIdentifier):
        if dictionary_profile is None:
            raise ValueError(f"{ident} has no dictionary profile, so its strength is undefined")
        return FeatureUnit(ident, dictionary_profile, skip_first)
    return NeuronUnit(ident, skip_first)


def submission_unit(submission) -> Unit:
    """Return the unit that a submission, scan evidence or scoring task is about.

    A record from before protocol 0.8 has no ``first_token_floor``, and is measured at
    every token.
    """
    floor = bool(getattr(submission, "first_token_floor", None))
    return unit_of(
        submission.identifier,
        submission.head_profile,
        submission.dictionary_profile,
        floor,
        getattr(submission, "superposition_profile", None),
    )


def peak_strength(runner: NeuronHookRunner, unit: Unit, text: str) -> float | None:
    """Return the peak strength of ``text`` over its measured tokens.

    ``None`` when it has none: one token, for a unit that skips the first.
    """
    strengths = unit.strengths(runner, text)[0][int(unit.skip_first) :]
    return float(strengths.max()) if strengths.numel() else None


def max_strength(runner: NeuronHookRunner, unit: Unit, text: str) -> float:
    """:func:`peak_strength`, for a text that must have a measured token."""
    peak = peak_strength(runner, unit, text)
    if peak is None:
        raise ValueError(
            f"{text!r} is one token, and {unit.ident} is not measured at the first token "
            "(it has a first-token floor), so use two or more"
        )
    return peak


def attn_implementation(ident: ComponentIdentifier) -> str | None:
    """Return the attention implementation to load the model with: ``"eager"`` for a head.

    SDPA and flash attention return no patterns. Every measurement of a head experiment
    should run the same attention kernel.
    """
    return "eager" if isinstance(ident, HeadIdentifier) else None
