"""Attribution patching: a first-order approximation of activation patching.

For every neuron in the requested layers, the effect of patching the clean
activation into the corrupted run is approximated as

    (a_clean - a_corrupted) * d metric / d a  |_corrupted

which needs one clean forward pass and one corrupted forward + backward pass,
instead of one forward pass per neuron. Use it to triage candidates, then confirm
with exact patching in :mod:`openinterp.harness.patcher`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from openinterp.core.exceptions import ShapeMismatchError
from openinterp.core.schemas import Pole
from openinterp.harness.hooks import Inputs, NeuronHookRunner, isolate_pole
from openinterp.harness.patcher import Metric, Positions, token_logit_metric


def attribution_patching(
    runner: NeuronHookRunner,
    clean: Inputs,
    corrupted: Inputs,
    layers: Sequence[int],
    *,
    metric: Metric | None = None,
    pole: Pole | None = None,
    positions: Positions = "all",
) -> dict[int, torch.Tensor]:
    """Estimated metric change per neuron, ``{layer: tensor[d_mlp]}``.

    With ``pole`` set, only that pole's activation difference is attributed. The
    default metric is the logit of the clean run's top-1 final-position token.
    """
    clean_logits, clean_cache = runner.run_with_cache(clean, layers)
    if metric is None:
        metric = token_logit_metric(int(clean_logits[0, -1].argmax()))

    captured: dict[int, torch.Tensor] = {}

    def capture(layer: int):
        def hook(act: torch.Tensor) -> None:
            captured[layer] = act

        return hook

    def ensure_grad(_module, _args, output: torch.Tensor) -> torch.Tensor:
        # Frozen weights mean nothing upstream requires grad; make the embedding
        # output a fresh leaf so the graph through the MLPs is recorded.
        return output if output.requires_grad else output.detach().requires_grad_(True)

    encoded = runner.encode(corrupted)
    embed_handle = runner.model.get_input_embeddings().register_forward_hook(ensure_grad)
    try:
        with torch.enable_grad(), runner.hooks({layer: capture(layer) for layer in layers}):
            value = metric(runner.forward(encoded))
            ordered = [captured[layer] for layer in layers]
            grads = torch.autograd.grad(value, ordered)
    finally:
        embed_handle.remove()

    scores: dict[int, torch.Tensor] = {}
    for layer, corrupt_act, grad in zip(layers, ordered, grads, strict=True):
        clean_act, corrupt_act = clean_cache[layer], corrupt_act.detach()
        if positions == "last":
            clean_act, corrupt_act, grad = clean_act[:, -1:], corrupt_act[:, -1:], grad[:, -1:]
        elif clean_act.shape != corrupt_act.shape:
            raise ShapeMismatchError(
                f"clean {tuple(clean_act.shape)} vs corrupted {tuple(corrupt_act.shape)}; "
                "use positions='last' or align the prompts."
            )
        if pole is not None:
            clean_act, corrupt_act = isolate_pole(clean_act, pole), isolate_pole(corrupt_act, pole)
        scores[layer] = ((clean_act - corrupt_act) * grad).sum(dim=(0, 1)).detach().cpu()
    return scores


def top_attributed_neurons(
    scores: dict[int, torch.Tensor], k: int = 10
) -> list[tuple[int, int, float]]:
    """Flatten attribution scores to ``(layer, neuron_idx, score)`` sorted by |score|."""
    flat: list[tuple[int, int, float]] = []
    for layer, vec in scores.items():
        for idx in vec.abs().topk(min(k, vec.numel())).indices.tolist():
            flat.append((layer, idx, float(vec[idx])))
    return sorted(flat, key=lambda t: abs(t[2]), reverse=True)[:k]
