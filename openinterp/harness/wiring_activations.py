"""A unit's wiring from activations: attribution patching (Syed et al., 2023) upstream,
and its write removed from each later unit's input downstream, at the peaks of prompts
where it fires. :class:`~openinterp.core.wiring.Wiring` defines each number; the unit is
described as in :mod:`openinterp.harness.wiring`.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress

import torch

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.wiring import Wiring
from openinterp.harness.hooks import NeuronHookRunner, _StopForward
from openinterp.harness.units import Unit
from openinterp.harness.wiring import (
    QKV,
    Target,
    _attn_read_point,
    _downstream_layers,
    _has_attention,
    _head_outputs,
    _matrix,
    _mlp_activations,
    _mlp_read_point,
    _project,
    _qkv_modules,
    _top,
    _upstream_layers,
    target_of,
)


def _clean_pass(runner, t: Target, encoded, attention: bool) -> dict:
    """One forward pass keeping what both directions need, all float32 ``[b, seq, ...]``."""
    up_n, up_h = _upstream_layers(t)
    down_n, down_h = _downstream_layers(runner, t)
    cap = {"acts": {}, "z": {}, "mlp_in": {}, "attn_in": {}}

    def keep(store, i, out=False):
        if out:  # a decoder layer may return a tuple whose first item is the residual
            return lambda _m, _a, o: store.__setitem__(
                i, (o[0] if isinstance(o, tuple) else o).detach().float()
            )
        return lambda _m, args: store.__setitem__(i, args[0].detach().float())

    hooks = [(runner.down_proj(i), keep(cap["acts"], i)) for i in {*up_n, t.layer}]
    hooks += [(_mlp_read_point(runner, i), keep(cap["mlp_in"], i)) for i in {*down_n, t.layer}]
    if attention:
        hooks += [(runner.attn_out(i), keep(cap["z"], i)) for i in {*up_h, *down_h, t.layer}]
        hooks += [
            (_attn_read_point(runner, i), keep(cap["attn_in"], i)) for i in {*down_h, t.layer}
        ]
    handles = [m.register_forward_pre_hook(fn) for m, fn in hooks]
    written: dict[int, torch.Tensor] = {}  # the read of a unit reading its layer's output
    if t.site in ("mlp_out", "resid_post"):
        module = runner.mlp_write(t.layer) if t.site == "mlp_out" else runner.layers[t.layer]
        handles.append(module.register_forward_hook(keep(written, t.layer, out=True)))
    try:
        with torch.no_grad():
            runner.forward(encoded)
    finally:
        for h in handles:
            h.remove()
    reads = {"mlp_in": cap["mlp_in"], "attn_in": cap["attn_in"]}
    cap["read"] = reads.get(t.site, written)[t.layer]
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
    modules = _qkv_modules(runner.layers[layer])
    w_o = _head_outputs(runner, layer)  # [n, d, hd]
    rows = torch.arange(len(peaks))
    clean = cap["z"][layer][rows, peaks].reshape(len(peaks), runner.n_heads, -1)
    effects = []
    for c in QKV:
        module, part = modules[c]
        z = _z_without(runner, layer, module, part, delta, encoded)
        changed = z[rows, peaks].reshape(len(peaks), runner.n_heads, -1)
        moved = torch.einsum("ndh,bnh->bnd", w_o.to(changed.device), clean - changed)
        effects.append(moved.norm(dim=-1))
    return torch.stack(effects, -1)


def _z_without(runner, layer: int, module, part, delta, encoded) -> torch.Tensor:
    """The layer's head outputs (float32) when one of ``q``, ``k`` or ``v`` (``module``,
    or its ``part`` of a fused projection) reads the residual without ``delta``."""
    norm = _attn_read_point(runner, layer)
    state: dict = {}

    def before(_m, args):
        if not state.get("busy"):
            state["busy"] = True
            try:
                state["normed"] = norm(args[0] - delta.to(args[0].dtype))
            finally:
                state["busy"] = False

    def swap(_m, _args, out):
        alt = _project(module, state["normed"])
        if part is None:
            return alt
        out = out.clone()
        out[..., part] = alt[..., part]
        return out

    z: list[torch.Tensor] = []

    def capture(_m, args):
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
    return z[-1]


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
    acc = _Sums()
    for start in range(0, len(prompts), batch_size):
        encoded = runner.encode(list(prompts[start : start + batch_size]))
        cap = _clean_pass(runner, t, encoded, attention)
        found = _peak_gradient(runner, t, unit, encoded, cap, dtype, threshold)
        if found is None:
            continue
        grad, peaks, fired = found
        acc.used += int(fired.sum())
        with torch.no_grad():
            _add_upstream(acc, runner, cap, grad, fired, up_n, up_h, w_out, w_o)
            _add_downstream(acc, runner, t, encoded, cap, fired, peaks, down_n, down_h, dtype)
    if not acc.used:
        return None
    return Wiring(
        method="activations",
        upstream_neurons=_top("mlp_neuron", acc.mean("up_n", up_n), [""], k),
        upstream_heads=_top("attn_head", acc.mean("up_h", up_h), [""], k) if attention else None,
        downstream_neurons=_top("mlp_neuron", acc.mean("down_n", down_n), [""], k),
        downstream_heads=_top("attn_head", acc.mean("down_h", down_h), list(QKV), k)
        if attention
        else None,
        prompts=acc.used,
    )


class _Sums:
    """Per-(direction, layer) sums over the prompts where the unit fired, and their count."""

    def __init__(self) -> None:
        self.sums: dict[tuple[str, int], torch.Tensor] = {}
        self.used = 0

    def add(self, key, value) -> None:
        self.sums[key] = self.sums[key] + value if key in self.sums else value

    def mean(self, kind: str, layers) -> list[tuple[int, torch.Tensor]]:
        blocks = []
        for i in layers:
            v = self.sums[(kind, i)] / self.used
            blocks.append((i, v if v.dim() == 2 else v[:, None]))
        return blocks


def _peak_gradient(runner, t: Target, unit: Unit, encoded, cap, dtype, threshold: float):
    """(the gradient of each firing prompt's peak strength with respect to the unit's read,
    the peaks, the mask of prompts that fired); ``None`` if none did."""
    mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).bool()
    x = cap["read"].to(dtype).requires_grad_(True)
    with torch.enable_grad():
        s = t.strength(x) if t.strength is not None else _head_strength(runner, t, encoded, x)
        measured = mask & (mask.cumsum(-1) > int(unit.skip_first))  # protocol 0.8
        s = s.float().masked_fill(~measured.to(s.device), -1.0)
        peak_values, peaks = s.max(-1)
        fired = peak_values.detach() >= threshold
        if not fired.any():
            return None
        (grad,) = torch.autograd.grad(peak_values[fired].sum(), x)
    return grad.float()[fired], peaks[fired], fired


def _add_upstream(acc: _Sums, runner, cap, grad, fired, up_n, up_h, w_out, w_o) -> None:
    for i in up_n:
        per_neuron = (grad @ w_out[i].to(grad.device)) * cap["acts"][i][fired]
        acc.add(("up_n", i), per_neuron.sum((0, 1)))
    for i in up_h:
        z = cap["z"][i][fired].reshape(*grad.shape[:2], runner.n_heads, -1)
        per_head = torch.einsum("bsd,ndh,bsnh->n", grad, w_o[i].to(grad.device), z)
        acc.add(("up_h", i), per_head)


def _add_downstream(acc: _Sums, runner, t, encoded, cap, fired, peaks, down_n, down_h, dtype):
    delta = t.written(cap)
    rows = torch.arange(len(peaks))
    at_peak = delta[fired][rows, peaks]
    for i in down_n:
        resid = cap["mlp_in"][i][fired][rows, peaks]
        both = torch.cat([resid, resid - at_peak]).to(dtype)
        after = _mlp_activations(runner, i, both).float()
        acc.add(("down_n", i), (after[: len(peaks)] - after[len(peaks) :]).sum(0))
    if down_h:  # empty without supported attention
        sub = {key: v[fired] for key, v in encoded.items()}
        kept = {**cap, "z": {i: v[fired] for i, v in cap["z"].items()}}
        for i in down_h:
            effects = _downstream_head_effects(runner, t, sub, kept, delta[fired], i, peaks)
            acc.add(("down_h", i), effects.sum(0))
