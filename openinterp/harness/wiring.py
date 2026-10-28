"""A unit's top upstream and downstream connections: from its weights (virtual weights,
Elhage et al., 2021) and from activations on prompts where it fires (attribution patching,
Syed et al., 2023). :class:`~openinterp.core.wiring.Wiring` defines each number.

Every unit is described the same way, by where it reads the residual stream and what it
writes back (:class:`Target`):

* a neuron pole reads its MLP's input (``gate``/``up``, or ``in``) and writes ``w_out``;
* a dictionary feature reads its dictionary's input site with its encoder row and writes
  its decoder row into the MLP output;
* an attention head reads its layer's attention input through ``q``, ``k`` and ``v`` and
  writes a subspace, ``W_O``.

Both readings are direct: they cover the residual-stream path from one unit's write to
another's read, not paths through units in between.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.wiring import Wiring, WiringEdge
from openinterp.harness.hooks import (
    PRE_MLP_NORM_PATHS,
    NeuronHookRunner,
    _StopForward,
    _try_path,
    isolate_pole,
    pole_strength,
)
from openinterp.harness.sites import normalize_without_affine
from openinterp.harness.units import FeatureUnit, HeadUnit, NeuronUnit, Unit

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
QKV = ("q", "k", "v")

Site = Literal["attn_in", "mlp_in", "mlp_out"]


def _matrix(proj: nn.Module) -> torch.Tensor:
    """A projection's weight as ``[out, in]`` (transformers' Conv1D stores ``[in, out]``)."""
    weight = proj.weight.detach().float()
    return weight if hasattr(proj, "in_features") else weight.T


def _project(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``proj(x)`` without running its hooks."""
    weight = proj.weight if hasattr(proj, "in_features") else proj.weight.T
    return torch.nn.functional.linear(x, weight, proj.bias)


def _norm(block: nn.Module, paths: Sequence[str]) -> nn.Module | None:
    for path in paths:
        module = _try_path(block, path)
        if isinstance(module, nn.Module):
            return module
    return None


def _fold(norm: nn.Module | None, rows: torch.Tensor, *, gain: bool = True) -> torch.Tensor:
    """Read vectors ``[..., d]`` taken through ``norm`` back to the residual stream: the gain
    (unless ``gain`` is False), then centring for LayerNorm. The per-token scale (1/σ) is
    dropped."""
    rows = rows.float()
    weight = getattr(norm, "weight", None) if gain else None
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


def _qkv_modules(block: nn.Module) -> dict[str, tuple[nn.Module, slice | None]] | None:
    """Each of q/k/v's projection, with its slice of the output when they are fused."""
    fused = _try_path(block, FUSED_QKV_PATH)
    if isinstance(fused, nn.Module):
        width = _matrix(fused).shape[0] // 3
        return {c: (fused, slice(i * width, (i + 1) * width)) for i, c in enumerate(QKV)}
    modules = [_try_path(block, p) for p in QKV_PATHS]
    if all(isinstance(m, nn.Module) for m in modules):
        return {c: (m, None) for c, m in zip(QKV, modules, strict=True)}
    return None


def attention_reads(runner: NeuronHookRunner, layer: int) -> dict[str, torch.Tensor] | None:
    """``{"q"|"k"|"v": [n_heads, head_dim, d_model]}`` with the input norm folded in; keys
    and values are repeated across the query heads that share them. ``None`` when the
    architecture's attention isn't supported."""
    block = runner.layers[layer]
    try:
        n, hd = runner.n_heads, runner.head_dim(layer)
    except ArchitectureNotSupportedError:
        return None
    modules = _qkv_modules(block)
    if modules is None:
        return None
    norm = _norm(block, PRE_ATTN_NORM_PATHS)
    reads = {}
    for c, (module, part) in modules.items():
        m = _matrix(module)
        heads = (m if part is None else m[part]).reshape(-1, hd, m.shape[-1])
        reads[c] = _fold(norm, heads.repeat_interleave(n // heads.shape[0], dim=0))
    return reads


def _head_outputs(runner: NeuronHookRunner, layer: int) -> torch.Tensor:
    """W_O per head: ``[n_heads, d_model, head_dim]``."""
    w_o = _matrix(runner.attn_out(layer))
    return w_o.reshape(w_o.shape[0], runner.n_heads, -1).permute(1, 0, 2)


def _has_attention(runner: NeuronHookRunner) -> bool:
    try:
        runner.attn_out(0)
    except ArchitectureNotSupportedError:
        return False
    return attention_reads(runner, 0) is not None


# -- what a unit reads and writes ---------------------------------------------------------


@dataclass
class Target:
    """Where a unit reads and writes, in the terms both readings need.

    ``reads`` are residual-space read matrices ``[m, d]`` by channel (``m = 1`` for a
    vector); ``write`` is ``[d]`` (a direction) or ``[d, h]`` (a head's subspace). It reads
    at ``site`` of ``layer`` and writes after that layer's attention (heads) or MLP.
    ``strength`` maps the residual stream at its read point ``[b, seq, d]`` to strength
    ``[b, seq]`` (``None`` for a head, which runs the model's attention instead), and
    ``written`` maps the clean pass's captures to its write ``[b, seq, d]``.
    """

    layer: int
    site: Site
    reads: dict[str, torch.Tensor]
    write: torch.Tensor
    strength: Callable[[torch.Tensor], torch.Tensor] | None
    written: Callable[[dict], torch.Tensor]
    head: int | None = None
    mean_output: torch.Tensor | None = None

    @property
    def writes_after_attention(self) -> bool:
        return self.site == "attn_in"


def target_of(runner: NeuronHookRunner, unit: Unit) -> Target:
    layer = unit.ident.layer
    if isinstance(unit, NeuronUnit):
        idx, pole = unit.ident.neuron_idx, unit.ident.pole
        reads = {c: r[idx][None] for c, r in mlp_reads(runner, layer).items()}
        w_out = runner.neuron_output_direction(layer, idx).float()
        return Target(
            layer,
            "mlp_in",
            reads,
            -w_out if pole == "negative" else w_out,
            strength=lambda x: pole_strength(_mlp_activations(runner, layer, x)[..., idx], pole),
            written=lambda cap: (
                isolate_pole(cap["acts"][layer][..., idx], pole)[..., None]
                * w_out.to(cap["acts"][layer].device)
            ),
        )
    if isinstance(unit, FeatureUnit):
        dictionary, f = unit.dictionary(runner), unit.ident.feature_idx
        site = unit.profile.input_site
        norm = runner.mlp_norm(layer) if site == "mlp_in_normalized" else None
        read = dictionary.w_enc[:, f].float()[None]
        if site == "mlp_in_normalized":  # standardised without the gain
            read = _fold(norm, read, gain=False)

        def feature(x: torch.Tensor) -> torch.Tensor:
            if norm is not None:
                x = normalize_without_affine(norm, x)
            return dictionary.feature(x, f)

        w_dec = dictionary.w_dec[f].float()
        return Target(
            layer,
            "mlp_out" if site == "mlp_out" else "mlp_in",
            {"in": read},
            w_dec,
            strength=feature,
            written=lambda cap: feature(cap["read"])[..., None] * w_dec.to(cap["read"].device),
        )
    if isinstance(unit, HeadUnit):
        reads = attention_reads(runner, layer)
        if reads is None:
            raise ArchitectureNotSupportedError("this model's attention reads aren't supported")
        h = unit.ident.head_idx
        w_o = runner.head_output_weights(layer, h).float()  # [d, hd]
        mean = torch.tensor(unit.mean_output, dtype=torch.float32)

        def head_write(cap):
            z = cap["z"][layer]
            z = z.reshape(*z.shape[:-1], runner.n_heads, -1)[..., h, :]
            return (z - mean.to(z.device)) @ w_o.T.to(z.device)

        return Target(
            layer,
            "attn_in",
            {c: r[h] for c, r in reads.items()},
            w_o,
            strength=None,
            written=head_write,
            head=h,
            mean_output=mean,
        )
    raise TypeError(f"no wiring for {type(unit).__name__}")


def _upstream_layers(t: Target) -> tuple[range, range]:
    """Layers of the neurons and heads that write before ``t`` reads."""
    if t.site == "mlp_out":
        return range(t.layer, t.layer + 1), range(0)
    if t.site == "attn_in":
        return range(t.layer), range(t.layer)
    return range(t.layer), range(t.layer + 1)


def _downstream_layers(runner: NeuronHookRunner, t: Target) -> tuple[range, range]:
    """Layers of the neurons and heads that read after ``t`` writes."""
    first = t.layer if t.writes_after_attention else t.layer + 1
    return range(first, runner.n_layers), range(t.layer + 1, runner.n_layers)


# -- ranking --------------------------------------------------------------------------------


def _top(
    kind: str, scores: Sequence[tuple[int, torch.Tensor]], channels: Sequence[str], k: int
) -> list[WiringEdge]:
    """The ``k`` units of largest |weight| from ``(layer, [units, channels])`` blocks."""
    if not scores:
        return []
    layers = torch.cat([torch.full((s.shape[0],), layer) for layer, s in scores])
    units = torch.cat([torch.arange(s.shape[0]) for _, s in scores])
    values = torch.cat([s.float().cpu() for _, s in scores])
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


# -- from the weights -----------------------------------------------------------------------


def _read_vectors(read: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """How much a read ``[m, d]`` picks up of each vector in ``x`` ``[n, d]``: signed for a
    vector read (``m = 1``), else the norm of what it reads."""
    projected = x @ read.T  # [n, m]
    return projected[:, 0] if read.shape[0] == 1 else projected.norm(dim=-1)


@torch.no_grad()
def weight_wiring(runner: NeuronHookRunner, unit: Unit, *, k: int = 10) -> Wiring:
    """The unit's top connections by virtual weight; see :class:`Wiring`."""
    t = target_of(runner, unit)
    channels = list(t.reads)
    # Unit-normalised reads, so channels with different scales compare.
    reads = {c: r / r.norm().clamp(min=1e-12) for c, r in t.reads.items()}
    neuron_layers, head_layers = _upstream_layers(t)
    up_neurons = [
        (
            i,
            torch.stack(
                [_read_vectors(reads[c], _matrix(runner.down_proj(i)).T) for c in channels], -1
            ),
        )
        for i in neuron_layers
    ]
    attention = _has_attention(runner)
    up_heads = None
    if attention:
        up_heads = []
        for i in head_layers:
            ov = torch.einsum(
                "ndz,nzh->ndh", _head_outputs(runner, i), attention_reads(runner, i)["v"]
            )
            # [n, d, d]: each head's OV circuit, read by each channel.
            up_heads.append(
                (i, torch.stack([(reads[c] @ ov).flatten(1).norm(dim=-1) for c in channels], -1))
            )
        up_heads = _top("attn_head", up_heads, channels, k)

    write = t.write
    neuron_layers, head_layers = _downstream_layers(runner, t)
    down_neurons, down_channels = [], [""]
    for i in neuron_layers:
        later = mlp_reads(runner, i)
        if write.dim() == 1:
            scores = [later[c] @ write for c in later]
        else:
            scores = [(later[c] @ write).norm(dim=-1) for c in later]
        down_neurons.append((i, torch.stack(scores, -1)))
        down_channels = list(later)
    down_heads = None
    if attention:
        down_heads = []
        for i in head_layers:
            qkv = attention_reads(runner, i)
            down_heads.append(
                (i, torch.stack([(qkv[c] @ write).flatten(1).norm(dim=-1) for c in QKV], -1))
            )
        down_heads = _top("attn_head", down_heads, list(QKV), k)
    return Wiring(
        method="weights",
        upstream_neurons=_top("mlp_neuron", up_neurons, channels, k),
        upstream_heads=up_heads,
        downstream_neurons=_top("mlp_neuron", down_neurons, down_channels, k),
        downstream_heads=down_heads,
    )


# -- from activations -----------------------------------------------------------------------


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


def _mlp_read_point(runner: NeuronHookRunner, layer: int) -> nn.Module:
    """The module whose input is the residual stream ``layer``'s MLP reads."""
    return _norm(runner.layers[layer], PRE_MLP_NORM_PATHS) or runner.mlp(layer)


def _attn_read_point(runner: NeuronHookRunner, layer: int) -> nn.Module:
    norm = _norm(runner.layers[layer], PRE_ATTN_NORM_PATHS)
    if norm is None:
        raise ArchitectureNotSupportedError("Could not find the norm before attention")
    return norm


def _clean_pass(runner, t: Target, encoded, attention: bool) -> dict:
    """One forward pass keeping what both directions need, all float32 ``[b, seq, ...]``."""
    up_n, up_h = _upstream_layers(t)
    down_n, down_h = _downstream_layers(runner, t)
    cap = {"acts": {}, "z": {}, "mlp_in": {}, "attn_in": {}}

    def keep(store, i, out=False):
        if out:
            return lambda _m, _a, o: store.__setitem__(i, o.detach().float())
        return lambda _m, args: store.__setitem__(i, args[0].detach().float())

    hooks = [(runner.down_proj(i), keep(cap["acts"], i)) for i in {*up_n, t.layer}]
    hooks += [(_mlp_read_point(runner, i), keep(cap["mlp_in"], i)) for i in {*down_n, t.layer}]
    if attention:
        hooks += [(runner.attn_out(i), keep(cap["z"], i)) for i in {*up_h, *down_h, t.layer}]
        hooks += [
            (_attn_read_point(runner, i), keep(cap["attn_in"], i)) for i in {*down_h, t.layer}
        ]
    handles = [m.register_forward_pre_hook(fn) for m, fn in hooks]
    mlp_out: dict[int, torch.Tensor] = {}
    if t.site == "mlp_out":
        handles.append(
            runner.down_proj(t.layer).register_forward_hook(keep(mlp_out, t.layer, out=True))
        )
    try:
        with torch.no_grad():
            runner.forward(encoded)
    finally:
        for h in handles:
            h.remove()
    cap["read"] = {"mlp_in": cap["mlp_in"], "attn_in": cap["attn_in"], "mlp_out": mlp_out}[t.site][
        t.layer
    ]
    return cap


def _head_strength(runner, t: Target, encoded, x: torch.Tensor) -> torch.Tensor:
    """A head's strength ``[b, seq]`` with its layer's attention input replaced by ``x``."""
    z: list[torch.Tensor] = []

    def capture(_m, args):
        z.append(args[0])
        raise _StopForward

    handles = [
        _attn_read_point(runner, t.layer).register_forward_pre_hook(
            lambda _m, args: (x.to(args[0].dtype), *args[1:])
        ),
        runner.attn_out(t.layer).register_forward_pre_hook(capture),
    ]
    try:
        with suppress(_StopForward):
            runner.forward(encoded)
    finally:
        for h in handles:
            h.remove()
    out = z[-1].reshape(*z[-1].shape[:-1], runner.n_heads, -1)[..., t.head, :].float()
    w_o = t.write.to(out.device)
    return ((out - t.mean_output.to(out.device)) @ w_o.T).norm(dim=-1)


def _downstream_head_effects(runner, t: Target, encoded, cap, delta, layer, peaks) -> torch.Tensor:
    """``[b, n_heads, 3]``: how far each head's write at the peak moves when ``delta`` is
    removed from what its ``q``, ``k`` or ``v`` reads, one channel at a time."""
    block = runner.layers[layer]
    norm = _attn_read_point(runner, layer)
    modules = _qkv_modules(block)
    w_o = _head_outputs(runner, layer)  # [n, d, hd]
    rows = torch.arange(len(peaks))
    clean = cap["z"][layer][rows, peaks].reshape(len(peaks), runner.n_heads, -1)
    effects = []
    for c in QKV:
        module, part = modules[c]
        state: dict = {}

        def before(_m, args, state=state):
            if not state.get("busy"):
                state["busy"] = True
                try:
                    state["normed"] = norm(args[0] - delta.to(args[0].dtype))
                finally:
                    state["busy"] = False

        def swap(_m, _args, out, module=module, part=part, state=state):
            alt = _project(module, state["normed"])
            if part is None:
                return alt
            out = out.clone()
            out[..., part] = alt[..., part]
            return out

        z: list[torch.Tensor] = []

        def capture(_m, args, z=z):
            z.append(args[0].float())
            raise _StopForward

        handles = [
            norm.register_forward_pre_hook(before),
            module.register_forward_hook(swap),
            runner.attn_out(layer).register_forward_pre_hook(capture),
        ]
        try:
            with torch.no_grad(), suppress(_StopForward):
                runner.forward(encoded)
        finally:
            for h in handles:
                h.remove()
        changed = z[-1][rows, peaks].reshape(len(peaks), runner.n_heads, -1)
        moved = torch.einsum("ndh,bnh->bnd", w_o.to(changed.device), clean - changed)
        effects.append(moved.norm(dim=-1))
    return torch.stack(effects, -1)


def activation_wiring(
    runner: NeuronHookRunner,
    unit: Unit,
    prompts: Sequence[str],
    threshold: float,
    *,
    k: int = 10,
    batch_size: int = 8,
) -> Wiring | None:
    """The unit's top connections by attribution at its peak on each prompt where it
    reaches ``threshold``; ``None`` if it reaches it on none. See :class:`Wiring`."""
    t = target_of(runner, unit)
    attention = _has_attention(runner)
    if t.site == "attn_in" and not attention:
        raise ArchitectureNotSupportedError("a head's wiring needs supported attention")
    up_n, up_h = _upstream_layers(t)
    down_n, down_h = _downstream_layers(runner, t)
    if not attention:
        up_h = down_h = range(0)
    w_out = {i: _matrix(runner.down_proj(i)) for i in up_n}  # [d, width]
    w_o = {i: _head_outputs(runner, i) for i in up_h} if attention else {}
    dtype = next(runner.model.parameters()).dtype
    sums: dict[tuple[str, int], torch.Tensor] = {}
    used = 0

    def add(key, value):
        sums[key] = sums[key] + value if key in sums else value

    for start in range(0, len(prompts), batch_size):
        encoded = runner.encode(list(prompts[start : start + batch_size]))
        mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
        cap = _clean_pass(runner, t, encoded, attention)
        x = cap["read"].to(dtype).requires_grad_(True)
        with torch.enable_grad():
            s = t.strength(x) if t.strength is not None else _head_strength(runner, t, encoded, x)
            s = s.float().masked_fill(~mask.to(s.device), -1.0)
            peak_values, peaks = s.max(-1)
            fired = peak_values.detach() >= threshold
            if not fired.any():
                continue
            (grad,) = torch.autograd.grad(peak_values[fired].sum(), x)
        grad = grad.float()[fired]
        peaks = peaks[fired]
        used += int(fired.sum())
        with torch.no_grad():
            for i in up_n:
                per_neuron = (grad @ w_out[i].to(grad.device)) * cap["acts"][i][fired]
                add(("up_n", i), per_neuron.sum((0, 1)))
            for i in up_h:
                z = cap["z"][i][fired].reshape(*grad.shape[:2], runner.n_heads, -1)
                per_head = torch.einsum("bsd,ndh,bsnh->n", grad, w_o[i].to(grad.device), z)
                add(("up_h", i), per_head)
            delta = t.written(cap)
            rows = torch.arange(len(peaks))
            at_peak = delta[fired][rows, peaks]
            for i in down_n:
                resid = cap["mlp_in"][i][fired][rows, peaks]
                both = torch.cat([resid, resid - at_peak]).to(dtype)
                after = _mlp_activations(runner, i, both).float()
                add(("down_n", i), (after[: len(peaks)] - after[len(peaks) :]).sum(0))
            if attention and down_h:
                sub = {key: v[fired] for key, v in encoded.items()}
                kept = {**cap, "z": {i: v[fired] for i, v in cap["z"].items()}}
                for i in down_h:
                    effects = _downstream_head_effects(runner, t, sub, kept, delta[fired], i, peaks)
                    add(("down_h", i), effects.sum(0))
    if not used:
        return None

    def mean(kind: str, layers) -> list[tuple[int, torch.Tensor]]:
        blocks = []
        for i in layers:
            v = sums[(kind, i)] / used
            blocks.append((i, v if v.dim() == 2 else v[:, None]))
        return blocks

    return Wiring(
        method="activations",
        upstream_neurons=_top("mlp_neuron", mean("up_n", up_n), [""], k),
        upstream_heads=_top("attn_head", mean("up_h", up_h), [""], k) if attention else None,
        downstream_neurons=_top("mlp_neuron", mean("down_n", down_n), [""], k),
        downstream_heads=_top("attn_head", mean("down_h", down_h), list(QKV), k)
        if attention
        else None,
        prompts=used,
    )
