"""A neuron pole's top upstream and downstream connections: from its weights (virtual
weights, Elhage et al., 2021) and from activations on prompts where it fires (attribution
patching, Syed et al., 2023). :class:`~openinterp.core.wiring.Wiring` defines each number.

Both readings are direct: they cover the residual-stream path from one unit's write to
another's read, not paths through units in between.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.schemas import NeuronIdentifier
from openinterp.core.wiring import Wiring, WiringEdge
from openinterp.harness.hooks import (
    PRE_MLP_NORM_PATHS,
    NeuronHookRunner,
    _try_path,
    isolate_pole,
    pole_strength,
)

# A neuron's reads of its MLP input, by channel; the first set that resolves wins.
MLP_READ_PATHS = (
    {"gate": "mlp.gate_proj", "up": "mlp.up_proj"},  # Llama, Qwen2, Mistral, Gemma
    {"in": "mlp.c_fc"},  # GPT-2
    {"in": "mlp.dense_h_to_4h"},  # GPT-NeoX, Falcon
    {"in": "mlp.fc_in"},  # GPT-J
    {"in": "mlp.fc1"},  # Phi
    {"in": "fc1"},  # OPT
    {"in": "mlp.up_proj"},  # an ungated MLP named like Llama's
)
# Separate query/key/value projections; GPT-2 fuses them in ``attn.c_attn``.
QKV_PATHS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
FUSED_QKV_PATH = "attn.c_attn"
PRE_ATTN_NORM_PATHS = (
    "input_layernorm",  # Llama, Qwen2, Mistral, Gemma, GPT-NeoX
    "ln_1",  # GPT-2, GPT-J
    "self_attn_layer_norm",  # OPT
)


def _matrix(proj: nn.Module) -> torch.Tensor:
    """A projection's weight as ``[out, in]`` (transformers' Conv1D stores ``[in, out]``)."""
    weight = proj.weight.detach().float()
    return weight if hasattr(proj, "in_features") else weight.T


def _norm(block: nn.Module, paths: Sequence[str]) -> nn.Module | None:
    for path in paths:
        module = _try_path(block, path)
        if isinstance(module, nn.Module):
            return module
    return None


def _fold(norm: nn.Module | None, rows: torch.Tensor) -> torch.Tensor:
    """Read vectors ``[..., d]`` taken through ``norm`` back to the residual stream: the gain,
    then centring for LayerNorm. The per-token scale (1/σ) is dropped."""
    rows = rows.float()
    weight = getattr(norm, "weight", None)
    if weight is not None:
        rows = rows * weight.detach().float().to(rows.device)
    if isinstance(norm, nn.LayerNorm):
        rows = rows - rows.mean(-1, keepdim=True)
    return rows


def mlp_reads(runner: NeuronHookRunner, layer: int) -> dict[str, torch.Tensor]:
    """``{channel: [width, d_model]}``, each neuron's read of the MLP input, norm folded in."""
    block = runner.layers[layer]
    for paths in MLP_READ_PATHS:
        modules = {c: _try_path(block, p) for c, p in paths.items()}
        if all(isinstance(m, nn.Module) for m in modules.values()):
            norm = _norm(block, PRE_MLP_NORM_PATHS)
            return {c: _fold(norm, _matrix(m)) for c, m in modules.items()}
    raise ArchitectureNotSupportedError(
        f"Could not find the MLP input projection under {runner.layer_path}[{layer}]"
    )


def attention_reads(runner: NeuronHookRunner, layer: int) -> dict[str, torch.Tensor] | None:
    """``{"q"|"k"|"v": [n_heads, head_dim, d_model]}`` with the input norm folded in; keys
    and values are repeated across the query heads that share them. ``None`` when the
    architecture's attention isn't supported."""
    block = runner.layers[layer]
    try:
        n, hd = runner.n_heads, runner.head_dim(layer)
    except ArchitectureNotSupportedError:
        return None
    fused = _try_path(block, FUSED_QKV_PATH)
    if isinstance(fused, nn.Module):
        mats = dict(zip("qkv", _matrix(fused).split(n * hd, dim=0), strict=True))
    elif all(isinstance(_try_path(block, p), nn.Module) for p in QKV_PATHS):
        mats = {c: _matrix(_try_path(block, p)) for c, p in zip("qkv", QKV_PATHS, strict=True)}
    else:
        return None
    norm = _norm(block, PRE_ATTN_NORM_PATHS)
    reads = {}
    for c, m in mats.items():
        heads = m.reshape(-1, hd, m.shape[-1])
        reads[c] = _fold(norm, heads.repeat_interleave(n // heads.shape[0], dim=0))
    return reads


def _head_outputs(runner: NeuronHookRunner, layer: int) -> torch.Tensor:
    """W_O per head: ``[n_heads, d_model, head_dim]``."""
    w_o = _matrix(runner.attn_out(layer))
    return w_o.reshape(w_o.shape[0], runner.n_heads, -1).permute(1, 0, 2)


def _top(
    kind: str, scores: Sequence[tuple[int, torch.Tensor]], channels: Sequence[str], k: int
) -> list[WiringEdge]:
    """The ``k`` units of largest |weight| from ``(layer, [units, channels])`` blocks."""
    if not scores:
        return []
    layers = torch.cat([torch.full((s.shape[0],), layer) for layer, s in scores])
    units = torch.cat([torch.arange(s.shape[0]) for _, s in scores])
    values = torch.cat([s.cpu() for _, s in scores])
    pick = values.abs().argmax(-1, keepdim=True)
    weight = values.gather(-1, pick).squeeze(-1)
    order = weight.abs().argsort(descending=True, stable=True)[:k]
    named = len(channels) > 1 or channels[0] != ""
    return [
        WiringEdge(
            kind=kind,
            layer=int(layers[i]),
            index=int(units[i]),
            weight=round(float(weight[i]), 6),
            channels=(
                {c: round(float(v), 6) for c, v in zip(channels, values[i], strict=True)}
                if named
                else {}
            ),
        )
        for i in order.tolist()
    ]


@torch.no_grad()
def weight_wiring(runner: NeuronHookRunner, ident: NeuronIdentifier, *, k: int = 10) -> Wiring:
    """The pole's top connections by virtual weight; see :class:`Wiring`."""
    layer, idx = ident.layer, ident.neuron_idx
    reads = mlp_reads(runner, layer)
    channels = list(reads)
    target = torch.stack([reads[c][idx] for c in channels])  # [channels, d]
    target = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    write = runner.neuron_output_direction(layer, idx).float()
    if ident.pole == "negative":
        write = -write

    up_neurons = [
        (i, _matrix(runner.down_proj(i)).T @ target.T)  # [width, channels]
        for i in range(layer)
    ]
    down_neurons, down_channels = [], [""]
    for i in range(layer + 1, runner.n_layers):
        later = mlp_reads(runner, i)
        down_neurons.append((i, torch.stack([later[c] @ write for c in later], -1)))
        down_channels = list(later)

    up_heads = down_heads = None
    if attention_reads(runner, 0) is not None:
        up_heads, down_heads = [], []
        for i in range(layer + 1):
            v = attention_reads(runner, i)["v"]  # [n, hd, d]
            through_o = torch.einsum("nzh,cz->nch", _head_outputs(runner, i), target)
            ov = torch.einsum("nch,nhd->ncd", through_o, v)
            up_heads.append((i, ov.norm(dim=-1)))  # [n, channels]
        for i in range(layer + 1, runner.n_layers):
            qkv = attention_reads(runner, i)
            down_heads.append((i, torch.stack([(qkv[c] @ write).norm(dim=-1) for c in "qkv"], -1)))
        up_heads = _top("attn_head", up_heads, channels, k)
        down_heads = _top("attn_head", down_heads, list("qkv"), k)
    return Wiring(
        method="weights",
        upstream_neurons=_top("mlp_neuron", up_neurons, channels, k),
        upstream_heads=up_heads,
        downstream_neurons=_top("mlp_neuron", down_neurons, down_channels, k),
        downstream_heads=down_heads,
    )


def _mlp_activations(runner: NeuronHookRunner, layer: int, x: torch.Tensor) -> torch.Tensor:
    """``layer``'s neuron activations for residual-stream inputs ``x`` ``[..., d_model]``."""
    norm = _norm(runner.layers[layer], PRE_MLP_NORM_PATHS)
    captured: list[torch.Tensor] = []
    handle = runner.down_proj(layer).register_forward_pre_hook(
        lambda _m, args: captured.append(args[0])
    )
    try:
        runner.mlp(layer)(norm(x) if norm is not None else x)
    finally:
        handle.remove()
    return captured[-1]


def _read_point(runner: NeuronHookRunner, layer: int) -> nn.Module:
    """The module whose input is the residual stream ``layer``'s MLP reads."""
    return _norm(runner.layers[layer], PRE_MLP_NORM_PATHS) or runner.mlp(layer)


def activation_wiring(
    runner: NeuronHookRunner,
    ident: NeuronIdentifier,
    prompts: Sequence[str],
    threshold: float,
    *,
    k: int = 10,
) -> Wiring | None:
    """The pole's top connections by attribution at its peak on each prompt where it
    reaches ``threshold``; ``None`` if it reaches it on none. See :class:`Wiring`."""
    layer, idx, pole = ident.layer, ident.neuron_idx, ident.pole
    later = range(layer + 1, runner.n_layers)
    try:
        heads = [_head_outputs(runner, i) for i in range(layer + 1)]
    except ArchitectureNotSupportedError:
        heads = None
    write = runner.neuron_output_direction(layer, idx).float()
    up_neurons = {i: 0.0 for i in range(layer)}
    up_heads = {i: 0.0 for i in range(layer + 1)}
    down_neurons = {i: 0.0 for i in later}
    used = 0
    for prompt in prompts:
        resid: dict[int, torch.Tensor] = {}
        acts: dict[int, torch.Tensor] = {}
        z: dict[int, torch.Tensor] = {}

        def keep(store, i):
            return lambda _m, args: store.__setitem__(i, args[0][0].detach().float())

        handles = [
            _read_point(runner, i).register_forward_pre_hook(keep(resid, i))
            for i in (layer, *later)
        ]
        handles += [
            runner.down_proj(i).register_forward_pre_hook(keep(acts, i)) for i in range(layer + 1)
        ]
        if heads is not None:
            handles += [
                runner.attn_out(i).register_forward_pre_hook(keep(z, i)) for i in range(layer + 1)
            ]
        try:
            with torch.no_grad():
                runner.forward(runner.encode(prompt))
        finally:
            for h in handles:
                h.remove()
        strength = pole_strength(acts[layer][:, idx], pole)
        t = int(strength.argmax())
        if float(strength[t]) < threshold:
            continue
        used += 1

        dtype = next(runner.model.parameters()).dtype
        x = resid[layer][t].to(dtype).requires_grad_(True)
        with torch.enable_grad():
            s = pole_strength(_mlp_activations(runner, layer, x[None])[0, idx], pole)
            (grad,) = torch.autograd.grad(s, x)
        grad = grad.float()
        with torch.no_grad():
            for i in range(layer):
                up_neurons[i] = up_neurons[i] + acts[i][t] * (_matrix(runner.down_proj(i)).T @ grad)
            if heads is not None:
                for i in range(layer + 1):
                    per_head = torch.einsum("nzh,z->nh", heads[i], grad)  # W_Oᵀ g per head
                    up_heads[i] = up_heads[i] + (per_head * z[i][t].reshape(per_head.shape)).sum(-1)
            delta = isolate_pole(acts[layer][t, idx], pole) * write
            for i in later:
                both = torch.stack([resid[i][t], resid[i][t] - delta]).to(dtype)
                after = _mlp_activations(runner, i, both).float()
                down_neurons[i] = down_neurons[i] + (after[0] - after[1])
    if not used:
        return None

    def mean(blocks: dict[int, torch.Tensor]) -> list[tuple[int, torch.Tensor]]:
        return [(i, (v / used)[:, None]) for i, v in blocks.items()]

    return Wiring(
        method="activations",
        upstream_neurons=_top("mlp_neuron", mean(up_neurons), [""], k),
        upstream_heads=_top("attn_head", mean(up_heads), [""], k) if heads is not None else None,
        downstream_neurons=_top("mlp_neuron", mean(down_neurons), [""], k),
        downstream_heads=None,
        prompts=used,
    )
