"""Where a sparse dictionary reads its input and writes its output in the model.

A ``dictionary_interventions`` entry of :meth:`NeuronHookRunner.hooks` is ``(site, fn)``:
the hooks read the dictionary's input x at ``site`` (see ``DictionaryProfile``), then call
``fn(x, out)`` with the down projection's output, which ``fn`` may replace.
"""

from __future__ import annotations

from contextlib import suppress

import torch
from torch import nn

from openinterp.core.schemas import DictionarySite
from openinterp.harness.hooks import (
    DictionaryIntervention,
    Inputs,
    NeuronHookRunner,
    _StopForward,
)


def normalize_without_affine(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """What ``norm`` makes of ``x`` before its gain and bias: standardised for LayerNorm,
    divided by its root mean square for RMSNorm (float32)."""
    eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon", None) or 1e-6
    x = x.float()
    if isinstance(norm, nn.LayerNorm):
        return torch.nn.functional.layer_norm(x, x.shape[-1:], eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def site_hooks(
    runner: NeuronHookRunner, layer: int, site: DictionarySite, fn: DictionaryIntervention
) -> list:
    """Hooks for one ``dictionary_interventions`` entry: read x at ``site``, then call
    ``fn(x, out)`` on the down projection's output."""
    seen: list[torch.Tensor] = []
    handles = []
    if site in ("mlp_in", "mlp_in_normalized"):
        norm = runner.mlp_norm(layer) if site == "mlp_in_normalized" else None

        def record(_module, args):
            seen.append(args[0] if norm is None else normalize_without_affine(norm, args[0]))

        module = runner.mlp(layer) if norm is None else norm
        handles.append(module.register_forward_pre_hook(record))
    elif site != "mlp_out":
        raise ValueError(f"unknown dictionary site {site!r}")

    def output(_module, _args, out):
        return fn(out if site == "mlp_out" else seen.pop(), out)

    handles.append(runner.down_proj(layer).register_forward_hook(output))
    return handles


@torch.no_grad()
def dictionary_io(
    runner: NeuronHookRunner, inputs: Inputs, layer: int, site: DictionarySite
) -> tuple[torch.Tensor, torch.Tensor]:
    """A dictionary's input at ``site`` and the MLP output it reconstructs, both
    ``[batch, seq, d]``. The forward pass stops at ``layer``'s MLP."""
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(x: torch.Tensor, out: torch.Tensor) -> None:
        captured.append((x.detach().clone(), out.detach().clone()))
        raise _StopForward

    with suppress(_StopForward):
        runner.run_with_interventions(inputs, dictionary_interventions={layer: (site, capture)})
    return captured[-1]
