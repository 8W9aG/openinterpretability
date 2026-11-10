"""The experts forward that openinterp runs mixture-of-experts models with (docs/moe.md).

transformers 5 runs the experts of about 60 MoE architectures through
``ExpertsInterface``: an experts class decorated with ``use_experts_implementation`` looks
its forward up by name. :func:`experts_forward` is registered there as ``openinterp``,
and :func:`use_openinterp_experts` selects it when a model loads. It is the ``eager``
forward written out once for every layout the decorator describes, with two additions:

* Each expert's hidden activation passes through the hook of :func:`expert_hook`
  before the down projection. That is where an expert neuron is read and edited.
* In a streamed model (``--memory stream``) the experts' tensors stay on disk, and the
  forward reads only the routed experts' slices of them (:func:`expert_weight`). One
  expert of Mistral Large 4 is about 113 MB in FP8, and all 128 of a layer 14.5 GB.

``scripts/moe_experts_check.py`` and ``tests/test_moe.py`` compare it with ``eager``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

EXPERTS_IMPLEMENTATION = "openinterp"

# fn(act [tokens, d_expert], expert index, token indices, routing weights) -> act
ExpertHook = Callable[[torch.Tensor, int, torch.Tensor, torch.Tensor], torch.Tensor]

_hooks = threading.local()  # per thread, as the layer-major review needs


def _linear(x: torch.Tensor, weight: torch.Tensor, bias, transposed: bool) -> torch.Tensor:
    y = x @ weight.to(x.dtype) if transposed else x @ weight.to(x.dtype).T
    return y if bias is None else y + bias.to(x.dtype)


def experts_forward(self, hidden, top_k_index, top_k_weights) -> torch.Tensor:
    """The routed experts' summed, weighted outputs ``[tokens, d_model]``.

    ``top_k_index`` and ``top_k_weights`` are each token's chosen experts and their
    routing weights ``[tokens, k]``. An index of ``num_experts`` or more is an
    expert-parallel placeholder, and is skipped.
    """
    out = torch.zeros_like(hidden)
    hook = getattr(_hooks, "by_module", {}).get(id(self))
    up = "gate_up_proj" if self.has_gate else "up_proj"
    for e in torch.unique(top_k_index).tolist():
        if e >= self.num_experts:
            continue
        tokens, slot = torch.where(top_k_index == e)
        pre = _linear(
            hidden[tokens], expert_weight(self, up, e), _bias(self, up, e), self.is_transposed
        )
        act = self._apply_gate(pre) if self.has_gate else self.act_fn(pre)
        weights = top_k_weights[tokens, slot]
        if hook is not None:
            act = hook(act, e, tokens, weights)
        down = expert_weight(self, "down_proj", e)
        y = _linear(act, down, _bias(self, "down_proj", e), self.is_transposed)
        if self.has_post_expert_norm:
            y = self.post_expert_norm(y)
        out.index_add_(0, tokens, (y * weights[:, None]).to(out.dtype))
    return out


def _bias(experts: nn.Module, name: str, e: int) -> torch.Tensor | None:
    return expert_weight(experts, f"{name}_bias", e) if experts.has_bias else None


def expert_weight(experts: nn.Module, name: str, e: int) -> torch.Tensor:
    """Expert ``e``'s slice of the stacked tensor ``name``, read from disk if streamed."""
    param = getattr(experts, name)
    if param.device.type != "meta":
        return param[e]
    weights = experts._hf_hook.weights_map  # streamed: the tensor stays on disk
    return weights.dataset.slice(weights.prefix + name, e).to(param.dtype)


@contextmanager
def expert_hook(experts: nn.Module, fn: ExpertHook) -> Iterator[None]:
    """Pass every expert's hidden activation in ``experts`` through ``fn``, in this thread."""
    by_module = getattr(_hooks, "by_module", None)
    if by_module is None:
        by_module = _hooks.by_module = {}
    previous = by_module.get(id(experts))
    by_module[id(experts)] = fn
    try:
        yield
    finally:
        if previous is None:
            del by_module[id(experts)]
        else:
            by_module[id(experts)] = previous


def experts_modules(model: nn.Module) -> list[nn.Module]:
    """The experts modules that ``use_experts_implementation`` decorated, in model order."""
    return [m for m in model.modules() if hasattr(m, "has_gate") and hasattr(m, "down_proj")]


def use_openinterp_experts(model: nn.Module) -> bool:
    """Run ``model``'s experts with :func:`experts_forward`; False if it has none.

    In a streamed model each experts module's tensors then stay on disk between passes.
    Each pass reads only the routed experts' slices of them.
    """
    experts = experts_modules(model)
    if not experts:
        return False
    _register()
    model.set_experts_implementation(EXPERTS_IMPLEMENTATION)
    for module in experts:
        hook = getattr(module, "_hf_hook", None)
        if hook is not None and hook.offload:
            hook.offload = False  # the forward reads slices: don't load the whole stack
    return True


def experts_implementation(model: nn.Module) -> str | None:
    """The experts forward ``model`` runs with, or ``None`` if it has no experts."""
    if not experts_modules(model):
        return None
    return getattr(model.config, "_experts_implementation", None)


def _register() -> None:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface

    if EXPERTS_IMPLEMENTATION not in ALL_EXPERTS_FUNCTIONS:
        ExpertsInterface.register(EXPERTS_IMPLEMENTATION, experts_forward)
