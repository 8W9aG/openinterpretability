"""One projection's Stochastic Parameter Decomposition (``train_spd.py``), and its export.

A site is ``(layer, matrix)``. The matrix is ``"down"``, the MLP's down projection, which
reads the neurons, or ``"up"``, its input projection, which writes them.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from openinterp.harness.dictionaries import gate_logits, layer_gate_logits
from openinterp.harness.hooks import _try_path

# The matrices that ``--matrices`` decomposes in each layer, in the order they are built.
MATRICES = {"down": ("down",), "up": ("up",), "both": ("down", "up")}
# An ungated MLP's input projection, and its activation function. A gated MLP (SwiGLU)
# has two input projections and is not supported.
UP_PATHS = ("mlp.c_fc", "mlp.dense_h_to_4h", "mlp.fc_in", "mlp.fc1", "fc1")
ACT_PATHS = ("mlp.act", "mlp.act_fn", "mlp.activation_fn")

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


def _first_path(block: nn.Module, paths: tuple[str, ...], what: str):
    for path in paths:
        if (found := _try_path(block, path)) is not None:
            return found
    raise ValueError(f"no {what} at any of {paths}")


def projection(runner, layer: int, matrix: str) -> nn.Module:
    """The projection that ``matrix`` names in ``layer``."""
    if matrix == "down":
        return runner.down_proj(layer)
    return _first_path(runner.layers[layer], UP_PATHS, "ungated MLP input projection")


def activation(runner, layer: int):
    """The MLP's activation function in ``layer``."""
    return _first_path(runner.layers[layer], ACT_PATHS, "MLP activation")


def site_label(site: tuple[int, str]) -> str:
    """``"4"`` for layer 4's down projection, ``"4.up"`` for its input projection."""
    layer, matrix = site
    return str(layer) if matrix == "down" else f"{layer}.{matrix}"


def load_parts(parts: dict) -> dict:
    """A checkpoint's parts keyed by site. Older checkpoints keyed down projections by layer."""
    return {(k, "down") if isinstance(k, int) else tuple(k): v for k, v in parts.items()}


def make_parts(runner, sites: list, components: int, d_gate: int, gate: str, seed: int,
               init: str) -> dict:  # fmt: skip
    """A new ``LayerDecomposition`` per site, all drawn from one generator in site order."""
    gen, parts = torch.Generator().manual_seed(seed), {}
    for layer, matrix in sites:
        w, bias = linear_weight(projection(runner, layer, matrix))
        parts[(layer, matrix)] = LayerDecomposition(
            w.cpu(), None if bias is None else bias.cpu(), components, d_gate, gate, gen, init,
            neurons="in" if matrix == "down" else "out")  # fmt: skip
    return parts


class LayerDecomposition(nn.Module):
    """The subcomponents of one projection and their causal-importance gate.

    The neurons are the projection's inputs (``neurons="in"``, the down projection) or
    its outputs (``neurons="out"``, the input projection). ``init="neurons"`` starts with
    one subcomponent per neuron.

    With ``gate="component"`` (the SPD paper's), each subcomponent has a small GELU MLP
    that reads only its own inner activation. With ``gate="layer"``, one GELU MLP reads
    the whole centred hidden vector and gives every subcomponent's importance.
    """

    def __init__(
        self,
        w: torch.Tensor,
        bias,
        c: int,
        d_gate: int,
        gate: str,
        generator,
        init="random",
        neurons="in",
    ) -> None:
        super().__init__()
        d_in, d_out = w.shape
        self.gate = gate
        self.register_buffer("w_target", w)
        self.register_buffer("bias", torch.zeros(d_out) if bias is None else bias)
        self.v = nn.Parameter(torch.randn(d_in, c, generator=generator) / math.sqrt(d_in))
        self.u = nn.Parameter(torch.randn(c, d_out, generator=generator) / math.sqrt(c))
        if init == "neurons":  # subcomponent i is neuron i
            d_mlp = d_in if neurons == "in" else d_out
            if c != d_mlp:
                raise ValueError(f"init='neurons' needs C = d_mlp = {d_mlp}, not {c}")
            if neurons == "in":  # V = I, U = W
                self.v.data, self.u.data = torch.eye(d_in), w.clone()
            else:  # V = W, U = I
                self.v.data, self.u.data = w.clone(), torch.eye(d_out)
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


class Decomposer:
    """Every decomposed site of one model, with hooks that swap in masked projections.

    A site is ``(layer, matrix)``. ``cfg`` is ``train_spd.Settings``.
    """

    def __init__(self, runner, layers: list[int], cfg) -> None:
        self.runner, self.layers, self.cfg = runner, layers, cfg
        self.sites = [(layer, m) for layer in layers for m in MATRICES[cfg.matrices]]
        parts = make_parts(runner, self.sites, cfg.components, cfg.d_gate, cfg.gate,
                           cfg.seed, cfg.init)  # fmt: skip
        self.parts: dict[tuple, LayerDecomposition] = {
            site: part.to(runner.device) for site, part in parts.items()
        }
        # Each input projection's activation function, for its output loss.
        self.acts = {(layer, m): activation(runner, layer) for layer, m in self.sites if m == "up"}
        # Per site, None: record its input and run the model's weights; else (mask, delta_mask).
        self.masks: dict[tuple, tuple | None] = dict.fromkeys(self.sites)
        self.hidden: dict[tuple, torch.Tensor] = {}
        self.handles = [projection(runner, *site).register_forward_hook(self._hook(site))
                        for site in self.sites]  # fmt: skip

    def _hook(self, site: tuple):
        def hook(_module, args, out):
            state = self.masks[site]
            if state is None:
                self.hidden[site] = args[0].detach().float()
                return None
            return self.parts[site].masked(args[0].float(), *state).to(out.dtype)

        return hook

    def parameters(self, kind: str) -> list[nn.Parameter]:
        """The subcomponents (``"components"``) or the gates (``"gates"``)."""
        if kind == "components":
            return [p for part in self.parts.values() for p in (part.v, part.u)]
        return [p for part in self.parts.values() for p in part.gate_parameters()]

    def logits(self, ids: torch.Tensor, masks: dict[tuple, tuple] | None = None) -> torch.Tensor:
        """The model's logits with ``masks`` on some sites (none: the model's own weights)."""
        self.masks = {site: (masks or {}).get(site) for site in self.sites}
        try:
            return self.runner.model(input_ids=ids).logits.float()
        finally:
            self.masks = dict.fromkeys(self.sites)

    @torch.no_grad()
    def set_means(self, ids: torch.Tensor) -> None:
        """Set each site's mean input from ``ids``, leaving out position 0."""
        self.logits(ids)
        for site, part in self.parts.items():
            part.mean.copy_(self.hidden[site][:, 1:].flatten(0, 1).mean(0))

    def importance(self, ids: torch.Tensor):
        """The logits, and per site the centred input, inner activations and gate logits."""
        with torch.no_grad():
            target = self.logits(ids)
        hidden, inner, logits = {}, {}, {}
        for site, part in self.parts.items():
            hidden[site] = self.hidden[site] - part.mean
            inner[site] = hidden[site] @ part.v
            logits[site] = part.gate_logits(hidden[site], inner[site])
        return target, hidden, inner, logits

    def close(self) -> None:
        """Remove the hooks."""
        for handle in self.handles:
            handle.remove()


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
