"""Prove from the weights that a neuron pole can't fire on any input.

A scan that never sees a pole fire shows only that the corpus doesn't make it fire. This
module gives a proof. The norm before the MLP puts the MLP input on a ball:

- LayerNorm: ``γ⊙ẑ + β``, where ``ẑ`` has mean 0 and ``‖ẑ‖ < √d``.
- RMSNorm: ``γ⊙x̂``, where ``‖x̂‖ < √d``.

So a neuron's pre-activation ``h = w·input + b`` lies in ``(c − R, c + R)``.

- LayerNorm: ``c = w·β + b`` and ``R = √d·‖centred(w⊙γ)‖``.
- RMSNorm: ``c = b`` and ``R = √d·‖w⊙γ‖``.

Take an ungated MLP with a sign-preserving activation (GELU, SiLU). Its positive pole
fires only when ``h > 0``, and its negative pole only when ``h < 0``. So
``c + R ≤ 0`` proves the positive pole dead, and ``c − R ≥ 0`` proves the negative pole
dead. With ReLU, the negative pole is always dead.

A gated MLP (SwiGLU) multiplies two reads, so the sign of the product is free on a ball.
Its pole is proven dead only when a read is exactly zero and the MLP has no bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from openinterp.core.identifiers import NeuronIdentifier
from openinterp.harness.hooks import NeuronHookRunner, _try_path
from openinterp.harness.wiring import MLP_READ_PATHS, _matrix

# Activations whose output has the sign of their input: f(h) > 0 iff h > 0, f(h) < 0 iff
# h < 0. ReLU is apart: it is never negative.
SIGN_PRESERVING = {
    "gelu",
    "gelu_new",
    "gelu_fast",
    "gelu_pytorch_tanh",
    "quick_gelu",
    "silu",
    "swish",
}
RELU = "relu"
# The bound must clear 0 by this share of its scale, so float64 rounding can't prove it.
MARGIN = 1e-9


@dataclass(frozen=True)
class DeadProof:
    """The result of the weight check on one neuron pole."""

    proven: bool
    # The largest value of h (positive pole) or −h (negative pole) that any input can
    # give, with the margin added. None when the proof doesn't use a bound.
    bound: float | None
    reason: str


def _activation(runner: NeuronHookRunner) -> str | None:
    config = getattr(runner.model, "config", None)
    name = getattr(config, "activation_function", None) or getattr(config, "hidden_act", None)
    return name if isinstance(name, str) else None


def _is_rms(norm: nn.Module) -> bool:
    """An RMSNorm that scales by ``weight`` (Gemma's scales by ``1 + weight``: not this)."""
    name = type(norm).__name__
    return name.endswith("RMSNorm") and not name.startswith("Gemma")


def _reads(runner: NeuronHookRunner, layer: int) -> dict[str, nn.Module]:
    block = runner.layers[layer]
    for paths in MLP_READ_PATHS:
        modules = {c: _try_path(block, p) for c, p in paths.items()}
        if all(isinstance(m, nn.Module) for m in modules.values()):
            return modules
    return {}


def _range(norm: nn.Module, proj: nn.Module, neuron: int) -> tuple[float, float] | None:
    """``(c, R)``: the pre-activation lies in ``(c − R, c + R)`` for every input."""
    w = _matrix(proj)[neuron].double().cpu()
    bias = getattr(proj, "bias", None)
    b = float(bias.detach()[neuron]) if bias is not None else 0.0
    gain = getattr(norm, "weight", None)
    v = w * gain.detach().double().cpu() if gain is not None else w
    shift = getattr(norm, "bias", None)
    if isinstance(norm, nn.LayerNorm):
        c = b + (float(w @ shift.detach().double().cpu()) if shift is not None else 0.0)
        return c, math.sqrt(v.numel()) * float((v - v.mean()).norm())
    if _is_rms(norm):
        return b, math.sqrt(v.numel()) * float(v.norm())
    return None


def prove_dead(runner: NeuronHookRunner, ident: NeuronIdentifier) -> DeadProof:
    """Check whether no input can make ``ident``'s pole fire."""
    act, reads = _activation(runner), _reads(runner, ident.layer)
    norm = runner.mlp_norm(ident.layer)
    if set(reads) == {"in"}:
        if act == RELU and ident.pole == "negative":
            return DeadProof(True, None, "ReLU is never negative")
        if act not in SIGN_PRESERVING and act != RELU:
            return DeadProof(False, None, f"no bound for the activation {act!r}")
        span = _range(norm, reads["in"], ident.neuron_idx)
        if span is None:
            return DeadProof(False, None, f"no bound for the norm {type(norm).__name__}")
        c, r = span
        bound = (c + r if ident.pole == "positive" else r - c) + MARGIN * (abs(c) + r)
        return DeadProof(bound <= 0, bound, f"pre-activation in ({c - r:.4g}, {c + r:.4g})")
    if set(reads) == {"gate", "up"}:
        if any(getattr(m, "bias", None) is not None for m in reads.values()):
            return DeadProof(False, None, "no bound for a gated MLP with biases")
        rows = [_matrix(m)[ident.neuron_idx] for m in reads.values()]
        if any(not torch.any(row) for row in rows):
            return DeadProof(True, 0.0, "a read of the gated MLP is zero")
        return DeadProof(False, None, "a gated MLP's pole is proven dead only by a zero read")
    return DeadProof(False, None, "MLP input projection not found")
