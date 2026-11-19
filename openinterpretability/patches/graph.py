"""The canonical model graph that patch operations act on (issue #5229).

A patch names hook points and parameters, never a model's own tensor names. So one patch
format covers every architecture, and a patch stays valid if a model's code renames a layer.

Hook points (``[head=h]`` or ``[index=j]`` may follow, to pick one head or unit):

- ``hook_tokens``: the input token ids (read only).
- ``blocks.<i>.hook_resid_pre`` and ``blocks.<i>.hook_resid_post``: the residual stream
  before and after block ``i``.
- ``blocks.<i>.mlp.hook_in``: the MLP's input. ``blocks.<i>.mlp.hook_post``: its neuron
  activations, the input of ``W_out``. ``blocks.<i>.mlp.hook_out``: its output.
- ``blocks.<i>.attn.hook_z``: the heads' outputs, the input of ``W_O``.
  ``blocks.<i>.attn.hook_pattern``: the attention weights, ``[batch, head, query, key]``
  (read only; the model runs eager attention while a patch reads them).

Parameters, in ``nn.Linear`` orientation (``[out, in]``): ``blocks.<i>.mlp.W_in``,
``W_gate`` (gated MLPs), ``W_out`` and ``blocks.<i>.attn.W_O``, each with a ``b_*`` bias.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from openinterpretability.patches.schema import PatchError

# Where each canonical part sits in a supported architecture, by config.model_type.
ARCHS: dict[str, dict[str, str]] = {
    "gpt_neox": {"layers": "gpt_neox.layers", "embed": "gpt_neox.embed_in", "mlp": "mlp",
                 "attn": "attention", "W_in": "mlp.dense_h_to_4h", "W_out": "mlp.dense_4h_to_h",
                 "W_O": "attention.dense"},
    "gpt2": {"layers": "transformer.h", "embed": "transformer.wte", "mlp": "mlp", "attn": "attn",
             "W_in": "mlp.c_fc", "W_out": "mlp.c_proj", "W_O": "attn.c_proj"},
    "qwen2": {"layers": "model.layers", "embed": "model.embed_tokens", "mlp": "mlp",
              "attn": "self_attn", "W_in": "mlp.up_proj", "W_gate": "mlp.gate_proj",
              "W_out": "mlp.down_proj", "W_O": "self_attn.o_proj"},
}  # fmt: skip
_NAME = re.compile(r"^(?:blocks\.(\d+)\.)?([\w.]+?)(?:\[(head|index)=(\d+)\])?$")
Hook = Callable[[torch.Tensor], torch.Tensor | None]


@dataclass(frozen=True)
class Name:
    """A parsed canonical name: its layer, its part, and an optional head or unit."""

    layer: int | None
    part: str
    select: str | None = None
    index: int | None = None


def parse(name: str) -> Name:
    """Split ``blocks.3.mlp.hook_post[index=5]`` into its layer, part and selector."""
    found = _NAME.match(name)
    if found is None:
        raise PatchError(f"not a canonical name: {name!r}")
    layer, part, select, index = found.groups()
    return Name(None if layer is None else int(layer), part, select,
                None if index is None else int(index))  # fmt: skip


def _get(root: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)  # type: ignore[index]
    return root


class Graph:
    """The canonical graph of one loaded Hugging Face model."""

    def __init__(self, model: nn.Module):
        kind = getattr(model.config, "model_type", "")
        if kind not in ARCHS:
            raise PatchError(f"no canonical graph for model type {kind!r}")
        self.model, self.kind, self.paths = model, kind, ARCHS[kind]
        self.layers = _get(model, self.paths["layers"])
        self.heads = int(model.config.num_attention_heads)

    def block(self, layer: int | None, part: str) -> nn.Module:
        """The module at ``part`` (a key of :data:`ARCHS`) in block ``layer``."""
        if layer is None or not 0 <= layer < len(self.layers):
            raise PatchError(f"no block {layer} in a {len(self.layers)}-layer model")
        if part not in self.paths:
            raise PatchError(f"{self.kind} has no {part}")
        return _get(self.layers[layer], self.paths[part])

    def parameter(self, name: str) -> tuple[nn.Module, str]:
        """The module that holds parameter ``name``, and 'weight' or 'bias'."""
        n = parse(name)
        part = n.part.split(".")[-1]
        key = "W_" + part[2:] if part.startswith("b_") else part
        return self.block(n.layer, key), "bias" if part.startswith("b_") else "weight"

    def weight(self, name: str) -> torch.Tensor:
        """A parameter's current value, in ``nn.Linear`` orientation, as float32."""
        module, which = self.parameter(name)
        value = getattr(module, which).detach().float()
        conv = type(module).__name__ == "Conv1D"  # GPT-2 stores [in, out]
        return value.t() if conv and which == "weight" else value


def on_output(module: nn.Module, fn: Hook, field: int = 0):
    """Hook ``fn`` on the output of ``module``. Its return value replaces the field."""

    def hook(_m, _args, out):
        value = out[field] if isinstance(out, tuple) else out
        new = fn(value)
        if new is None:
            return None
        if not isinstance(out, tuple):
            return new
        return (*out[:field], new, *out[field + 1 :])

    return module.register_forward_hook(hook)


def on_input(module: nn.Module, fn: Hook):
    """Hook ``fn`` on the first input of ``module``, positional or ``hidden_states``."""

    def hook(_m, args, kwargs):
        if args:
            new = fn(args[0])
            return None if new is None else ((new, *args[1:]), kwargs)
        new = fn(kwargs["hidden_states"])
        return None if new is None else (args, {**kwargs, "hidden_states": new})

    return module.register_forward_pre_hook(hook, with_kwargs=True)


def hook_point(graph: Graph, name: str, fn: Hook):
    """Register ``fn`` at the hook point ``name``, and return the handle."""
    n = parse(name)
    if n.part == "hook_tokens":
        return on_input(_get(graph.model, graph.paths["embed"]), fn)
    table = {
        "hook_resid_pre": lambda: on_input(graph.layers[n.layer], fn),
        "hook_resid_post": lambda: on_output(graph.layers[n.layer], fn),
        "mlp.hook_in": lambda: on_input(graph.block(n.layer, "mlp"), fn),
        "mlp.hook_post": lambda: on_input(graph.block(n.layer, "W_out"), fn),
        "mlp.hook_out": lambda: on_output(graph.block(n.layer, "mlp"), fn),
        "attn.hook_z": lambda: on_input(graph.block(n.layer, "W_O"), fn),
        "attn.hook_pattern": lambda: on_output(graph.block(n.layer, "attn"), fn, field=1),
    }
    if n.part not in table:
        raise PatchError(f"unknown hook point {name!r}")
    if n.layer is None or not 0 <= n.layer < len(graph.layers):
        raise PatchError(f"no block {n.layer} in a {len(graph.layers)}-layer model")
    return table[n.part]()


def parameter_hook(graph: Graph, name: str, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Add ``fn(x)`` to the output of the layer that holds parameter ``name``.

    ``x`` is that layer's input. A weight edit ΔW adds ``x @ ΔWᵀ``, and a bias edit adds Δb.
    So the model's own weights are never written, and removal is exact.
    """
    module, _ = graph.parameter(name)
    seen: list[torch.Tensor] = []
    pre = module.register_forward_pre_hook(lambda _m, args: seen.append(args[0]) or None)

    def post(_m, _args, out):
        x = seen.pop()
        return out + fn(x.float()).to(out.dtype)

    return [pre, module.register_forward_hook(post)]
