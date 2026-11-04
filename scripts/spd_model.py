"""One layer's Stochastic Parameter Decomposition (``train_spd.py``), and its export."""

from __future__ import annotations

import math

import torch
from torch import nn

from openinterp.harness.dictionaries import gate_logits, layer_gate_logits

GATE_NAMES = {
    "component": ("gate_w1", "gate_b1", "gate_w2", "gate_b2"),
    "layer": ("gate_in_w", "gate_in_b", "gate_out_w", "gate_out_b"),
}


def linear_weight(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A projection's weight as ``[d_in, d_out]``, and its bias.

    GPT-2's ``Conv1D`` stores the weight that way already. ``nn.Linear`` stores its
    transpose.
    """
    w = module.weight.T if isinstance(module, nn.Linear) else module.weight
    return w.detach().float().clone(), None if module.bias is None else module.bias.detach().float()


class LayerDecomposition(nn.Module):
    """The subcomponents of one down projection and their causal-importance gate.

    With ``gate="component"`` (the SPD paper's), each subcomponent has a small GELU MLP
    that reads only its own inner activation. With ``gate="layer"``, one GELU MLP reads
    the whole centred hidden vector and gives every subcomponent's importance.
    """

    def __init__(
        self, w: torch.Tensor, bias, c: int, d_gate: int, gate: str, generator, init="random"
    ) -> None:
        super().__init__()
        d_in, d_out = w.shape
        self.gate = gate
        self.register_buffer("w_target", w)
        self.register_buffer("bias", torch.zeros(d_out) if bias is None else bias)
        self.v = nn.Parameter(torch.randn(d_in, c, generator=generator) / math.sqrt(d_in))
        self.u = nn.Parameter(torch.randn(c, d_out, generator=generator) / math.sqrt(c))
        if init == "neurons":  # subcomponent i is neuron i: V = I, U = W
            if c != d_in:
                raise ValueError(f"init='neurons' needs C = d_mlp = {d_in}, not {c}")
            self.v.data, self.u.data = torch.eye(d_in), w.clone()
        if gate == "component":
            self.gate_w1 = nn.Parameter(torch.randn(c, d_gate, generator=generator))
            self.gate_b1 = nn.Parameter(torch.randn(c, d_gate, generator=generator))
            self.gate_w2 = nn.Parameter(torch.randn(c, d_gate, generator=generator) / d_gate)
            self.gate_b2 = nn.Parameter(torch.zeros(c))
        else:
            self.gate_in_w = nn.Parameter(torch.randn(d_in, d_gate, generator=generator))
            self.gate_in_b = nn.Parameter(torch.zeros(d_gate))
            scale = 1 / math.sqrt(d_gate)
            self.gate_out_w = nn.Parameter(torch.randn(d_gate, c, generator=generator) * scale)
            self.gate_out_b = nn.Parameter(torch.zeros(c))
        # Divides the gate's input; set from the first batch. A component gate reads the
        # inner activations, a layer gate the centred hidden vector (scaled by its norm).
        self.register_buffer("gate_scale", torch.tensor(0.0))
        # Subcomponents read h - mean; the mean's write is exact, through w_target.
        self.register_buffer("mean", torch.zeros(d_in))

    def gate_parameters(self) -> list[nn.Parameter]:
        """The gate's parameters."""
        return [getattr(self, name) for name in GATE_NAMES[self.gate]]

    def offset(self) -> torch.Tensor:
        """What the projection writes with every subcomponent and the delta ablated."""
        return self.mean @ self.w_target + self.bias

    def gate_logits(self, hidden: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
        """The importance before the hard sigmoid, ``[..., C]``, from centred activations."""
        x = inner if self.gate == "component" else hidden
        if float(self.gate_scale) == 0:
            norm = x[:, 1:].detach().pow(2).mean().sqrt()
            self.gate_scale.fill_(
                float(norm if self.gate == "component" else norm * x.shape[-1] ** 0.5)
            )
        if self.gate == "component":
            return gate_logits(x / self.gate_scale, *self.gate_parameters())
        return layer_gate_logits(x / self.gate_scale, *self.gate_parameters())

    def delta(self) -> torch.Tensor:
        """What the subcomponents leave out of the weight, ``[d_in, d_out]``."""
        return self.w_target - self.v @ self.u

    def masked(self, h: torch.Tensor, mask: torch.Tensor, delta_mask: torch.Tensor) -> torch.Tensor:
        """The projection of ``h``, subcomponents scaled by ``mask``, delta by ``delta_mask``."""
        x = h - self.mean
        out = ((x @ self.v) * mask) @ self.u + self.offset()
        return out + delta_mask.unsqueeze(-1) * (x @ self.delta())


def _pair(t: torch.Tensor, keep: torch.Tensor, sign: bool, dim: int = 0) -> torch.Tensor:
    """``t`` at ``keep`` along ``dim``, each slice followed by itself, negated if ``sign``."""
    t = t.index_select(dim, keep)
    return torch.stack([t, -t if sign else t], dim + 1).flatten(dim, dim + 1).contiguous()


def latent_pairs(part: LayerDecomposition, keep: torch.Tensor) -> dict[str, torch.Tensor]:
    """The ``keep`` subcomponents as the harness's latents, each followed by its negation.

    The harness reads the raw hidden activations h. The centring goes into ``b_enc``
    and, for a layer gate, into its input bias.
    """
    weights = {
        "W_enc": _pair(part.v, keep, True, dim=1),  # [d_mlp, 2k]
        "b_enc": _pair(-(part.mean @ part.v), keep, True),  # reads h - mean
        "W_dec": _pair(part.u, keep, True),  # [2k, d_model]
        "b_dec": part.offset(),
    }
    s = part.gate_scale
    if part.gate == "component":
        # The negative latent's pre-activation is -h·V, so its gate reads it negated.
        weights |= {
            "ci_w1": _pair(part.gate_w1 / s, keep, True),
            "ci_b1": _pair(part.gate_b1, keep, False),
            "ci_w2": _pair(part.gate_w2, keep, False),
            "ci_b2": _pair(part.gate_b2, keep, False),
        }
    else:  # the gate read (h - mean) / s; both latents of a pair share its importance
        weights |= {
            "ci_in_w": part.gate_in_w / s,
            "ci_in_b": part.gate_in_b - (part.mean / s) @ part.gate_in_w,
            "ci_out_w": _pair(part.gate_out_w, keep, False, dim=1),
            "ci_out_b": _pair(part.gate_out_b, keep, False),
        }
    return {k: v.detach().float().cpu() for k, v in weights.items()}
