"""What "strength" means for each kind of auditable unit, so one protocol covers both.

Every measurement in an audit reads one non-negative strength per token: the scan's
top examples, probes, counterfactual Δ, held-out windows. A unit supplies it, and the
ablation the lookahead profile uses:

* an MLP neuron pole: its pole strength; ablation zeroes the pole;
* an attention head: ``|W_O (z_t − z̄)|``, the distance between its write to the
  residual stream and its mean write over the scan (z̄, stored in
  :class:`~openinterp.core.schemas.HeadProfile`); ablation replaces z with z̄ (mean
  ablation), so strength is exactly how far ablating the head moves the residual.

The raw write norm ``|W_O z_t|`` would not do: most heads write a sizeable resting
vector (often while attending to the first token), so in GPT-2 small an induction
head's median token sits at a quarter of its maximum and half of all text would count
as "active".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from openinterp.core.schemas import ComponentIdentifier, HeadIdentifier, NeuronIdentifier
from openinterp.harness.hooks import Inputs, NeuronHookRunner, ablate_head, ablate_pole


@dataclass(frozen=True)
class NeuronUnit:
    ident: NeuronIdentifier

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
    ident: HeadIdentifier
    mean_output: tuple[float, ...]

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


Unit = NeuronUnit | HeadUnit


def head_strength(
    runner: NeuronHookRunner, ident: HeadIdentifier, z: torch.Tensor, mean_output
) -> torch.Tensor:
    """``|W_O (z − z̄)|`` over the last dimension of ``z`` (one head's outputs)."""
    w_o = runner.head_output_weights(ident.layer, ident.head_idx).float()
    mean = torch.as_tensor(mean_output, dtype=torch.float32, device=z.device)
    return ((z.float() - mean) @ w_o.T.to(z.device)).norm(dim=-1)


def unit_of(ident: ComponentIdentifier, head_profile=None) -> Unit:
    """The unit ``ident`` names; a head needs the profile its strength is defined by."""
    if isinstance(ident, HeadIdentifier):
        if head_profile is None:
            raise ValueError(f"{ident} has no head profile, so its strength is undefined")
        return HeadUnit(ident, tuple(head_profile.mean_output))
    return NeuronUnit(ident)


def submission_unit(submission) -> Unit:
    """The unit a submission (or scan evidence) is about."""
    return unit_of(submission.identifier, submission.head_profile)


def max_strength(runner: NeuronHookRunner, unit: Unit, text: str) -> float:
    return float(unit.strengths(runner, text).max())


def attn_implementation(ident: ComponentIdentifier) -> str | None:
    """Heads load with eager attention: SDPA and flash attention return no patterns, and
    every measurement of a head experiment should run the same attention kernel."""
    return "eager" if isinstance(ident, HeadIdentifier) else None
