"""Input attribution: which context tokens drive a neuron pole at its peak.

For each top example the pole's *signed* activation at the peak token (positive for
the positive pole, negated for the negative pole, so larger is more active and the
gradient is non-zero below the clamp) is attributed to every token up to the peak
with integrated gradients (Sundararajan et al., 2017) over the input token
embeddings:

    attr_i = (x_i − b) · (1/m) Σ_k ∂f/∂x_i |_{b + α_k (x − b)},   α_k = (k + ½)/m

where b is the embedding of one reference token (the pad token) at every position
but the first. The first token stays as it is: most of the model's attention rests on
it (the attention sink), so replacing it moves the whole path off-distribution, and on
Qwen2.5-0.5B the integral then fails to converge and credits the first few tokens of a
window with much of the activation that replacing any one of them doesn't remove. A
peak on the first token is attributed to it alone. Position embeddings, where a model
has them, are left alone. By completeness the attributions sum to f(x) − f(b), and
each is reported as a share of that gap.

The path integral is linear in each token's contribution, so it is checked by
replacing one token at a time with the reference token and re-measuring: the peak
itself, the few tokens before it, and the most attributed tokens. Tokens after the
peak cannot affect it (attention is causal), so each example is cut at its peak, and
every forward pass stops at the audited layer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress

import torch

from openinterp.core.attribution import (
    AttributedToken,
    ContextExample,
    DistanceShare,
    InputAttribution,
)
from openinterp.core.schemas import ActivatingExample, Pole
from openinterp.core.stats import spearman
from openinterp.harness.hooks import NeuronHookRunner, _StopForward

# Tokens back from the peak: the token itself, the one before, then wider bands.
DISTANCE_BANDS: tuple[tuple[int, int | None], ...] = (
    (0, 0),
    (1, 1),
    (2, 4),
    (5, 16),
    (17, 64),
    (65, None),
)

EmbedEdit = Callable[[torch.Tensor], torch.Tensor | None]


def baseline_token_id(tokenizer) -> int:
    """The reference token: the pad token, else end-of-sequence, else id 0."""
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            return value
    return 0


def _pole_values(
    runner: NeuronHookRunner,
    ids: torch.Tensor,
    layer: int,
    neuron_idx: int,
    pole: Pole,
    edit: EmbedEdit,
) -> torch.Tensor:
    """Signed pole activation ``[batch, seq]`` with the token embeddings edited by ``edit``.

    The forward pass stops at ``layer``'s MLP: later layers can't change it.
    """
    captured: list[torch.Tensor] = []

    def capture(act: torch.Tensor) -> None:
        captured.append(act[..., neuron_idx])
        raise _StopForward

    embed = runner.model.get_input_embeddings()
    handle = embed.register_forward_hook(lambda _module, _args, out: edit(out))
    try:
        with suppress(_StopForward), runner.hooks({layer: capture}):
            runner.forward({"input_ids": ids})
    finally:
        handle.remove()
    value = captured[-1].float()
    return value if pole == "positive" else -value


def reference(x: torch.Tensor, baseline: torch.Tensor, keep_first: bool) -> torch.Tensor:
    """The baseline input for embeddings ``x`` ``[batch, seq, d]``: ``baseline`` at every
    position, except the first when ``keep_first``."""
    base = baseline.to(x.dtype).expand_as(x).clone()
    if keep_first:
        base[:, 0] = x[:, 0]
    return base


def integrated_gradients(
    runner: NeuronHookRunner,
    ids: torch.Tensor,
    layer: int,
    neuron_idx: int,
    pole: Pole,
    baseline: torch.Tensor,
    *,
    steps: int = 16,
    batch_size: int = 8,
    keep_first: bool = True,
) -> torch.Tensor:
    """Attribution ``[seq]`` of the last token's signed pole activation, one row ``ids``.

    With ``keep_first`` the first token is not interpolated and gets no attribution."""
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    grad_sum: torch.Tensor | None = None
    diff: torch.Tensor | None = None
    for chunk in alphas.split(batch_size):
        points: list[torch.Tensor] = []

        def interpolate(out: torch.Tensor, chunk=chunk, points=points) -> torch.Tensor:
            nonlocal diff
            x = out.detach()
            base = reference(x, baseline, keep_first)
            alpha = chunk.to(x.device, x.dtype)[:, None, None]
            point = (base + alpha * (x - base)).requires_grad_(True)
            diff = (x[0] - base[0]).float()
            points.append(point)
            return point

        with torch.enable_grad():
            rows = ids.expand(len(chunk), -1)
            values = _pole_values(runner, rows, layer, neuron_idx, pole, interpolate)[:, -1]
            (grad,) = torch.autograd.grad(values.sum(), points[-1])
        part = grad.float().sum(0)
        grad_sum = part if grad_sum is None else grad_sum + part
    assert grad_sum is not None and diff is not None
    return (grad_sum / steps * diff).sum(-1)


@torch.no_grad()
def ablate_positions(
    runner: NeuronHookRunner,
    ids: torch.Tensor,
    layer: int,
    neuron_idx: int,
    pole: Pole,
    baseline: torch.Tensor,
    positions: Sequence[int],
    *,
    batch_size: int = 8,
) -> torch.Tensor:
    """The last token's signed pole activation with each of ``positions`` replaced alone."""
    out: list[torch.Tensor] = []
    for start in range(0, len(positions), batch_size):
        chunk = torch.tensor(positions[start : start + batch_size])

        def replace(emb: torch.Tensor, chunk=chunk) -> torch.Tensor:
            emb = emb.clone()
            emb[torch.arange(len(chunk)), chunk.to(emb.device)] = baseline.to(emb.dtype)
            return emb

        rows = ids.expand(len(chunk), -1)
        out.append(_pole_values(runner, rows, layer, neuron_idx, pole, replace)[:, -1])
    return torch.cat(out)


def _distance_shares(attr: torch.Tensor) -> tuple[list[float], float]:
    """Share of |attr| per band of :data:`DISTANCE_BANDS`, and the weighted mean distance."""
    weight = attr.abs()
    total = float(weight.sum()) or 1.0
    distance = torch.arange(len(attr) - 1, -1, -1, dtype=torch.float32)
    shares = []
    for low, high in DISTANCE_BANDS:
        band = (distance >= low) & (distance <= (high if high is not None else len(attr)))
        shares.append(float(weight[band].sum()) / total)
    return shares, float((weight * distance).sum()) / total


def _converged_attribution(
    runner, ids, args, baseline, gap: float, steps: int, max_steps: int, tolerance: float
) -> tuple[torch.Tensor, int]:
    """Attribution as shares of ``gap``, doubling the steps until Σ is within ``tolerance``
    of 1 or ``max_steps`` is reached. Returns the attribution and the steps it took."""
    keep_first = ids.shape[1] > 1
    while True:
        attr = integrated_gradients(
            runner, ids, *args, baseline, steps=steps, keep_first=keep_first
        ).cpu()
        attr = attr / gap
        if abs(float(attr.sum()) - 1.0) <= tolerance or steps * 2 > max_steps:
            return attr, steps
        steps *= 2


def input_attribution(
    runner: NeuronHookRunner,
    examples: Sequence[ActivatingExample],
    layer: int,
    neuron_idx: int,
    pole: Pole,
    *,
    n_examples: int = 5,
    steps: int = 16,
    max_steps: int = 64,
    tolerance: float = 0.1,
    top_k: int = 5,
    near: int = 4,
    batch_size: int = 8,
) -> InputAttribution:
    """Integrated gradients at the peak of the first ``n_examples`` top examples.

    Each example starts at ``steps`` integration steps and doubles them, up to
    ``max_steps``, until the attribution sums to within ``tolerance`` of the gap. Only
    examples that get there count towards the distance shares and the agreement. Each
    example keeps its ``top_k`` most attributed tokens; the ablation check covers those,
    the peak and the ``near`` tokens before it.
    """
    if runner.tokenizer is None:
        raise ValueError("input_attribution requires a tokenizer")
    tokenizer = runner.tokenizer
    base_id = baseline_token_id(tokenizer)
    baseline = runner.model.get_input_embeddings().weight[base_id].detach()
    args = (layer, neuron_idx, pole)

    results: list[ContextExample] = []
    band_shares: list[list[float]] = []
    distances: list[float] = []
    checked_attr: list[float] = []
    checked_effect: list[float] = []
    for n, example in enumerate(examples[:n_examples]):
        ids = runner.encode(example.text)["input_ids"][:1]
        with torch.no_grad():
            values = _pole_values(runner, ids, *args, lambda _out: None)[0]
            peak = int(values.argmax())
            ids = ids[:, : peak + 1]
            empty = _pole_values(
                runner, ids, *args, lambda out, keep=peak > 0: reference(out, baseline, keep)
            )
        gap = float(values[peak] - empty[0, -1])
        if gap <= 1e-6:
            continue  # the reference input activates the pole as much: nothing to attribute

        attr, used = _converged_attribution(
            runner, ids, args, baseline, gap, steps, max_steps, tolerance
        )
        error = abs(float(attr.sum()) - 1.0)
        top = attr.abs().topk(min(top_k, len(attr))).indices.tolist()
        checked = sorted(set(top) | set(range(max(0, peak - near), peak + 1)))
        ablated = ablate_positions(runner, ids, *args, baseline, checked, batch_size=batch_size)
        effect = dict(
            zip(checked, ((float(values[peak]) - ablated.cpu()) / gap).tolist(), strict=True)
        )
        if error <= tolerance:
            checked_attr += [float(attr[i]) for i in checked]
            checked_effect += [effect[i] for i in checked]
            shares, mean_distance = _distance_shares(attr)
            band_shares.append(shares)
            distances.append(mean_distance)
        token_ids = ids[0].tolist()
        results.append(
            ContextExample(
                example=n,
                peak_token=tokenizer.decode([token_ids[peak]]),
                tokens=peak + 1,
                steps=used,
                converged=error <= tolerance,
                completeness_error=round(error, 4),
                top=[
                    AttributedToken(
                        offset=i - peak,
                        token=tokenizer.decode([token_ids[i]]),
                        attribution=round(float(attr[i]), 4),
                        ablation_effect=round(effect[i], 4),
                    )
                    for i in top
                ],
            )
        )

    agreement = (
        spearman(torch.tensor(checked_attr), torch.tensor(checked_effect))
        if len(checked_attr) >= 3
        else None
    )
    count = len(band_shares)
    return InputAttribution(
        baseline_token=tokenizer.decode([base_id]),
        tolerance=tolerance,
        examples=results,
        distances=[
            DistanceShare(
                min_distance=low,
                max_distance=high,
                share=round(min(1.0, sum(s[b] for s in band_shares) / count), 4),
            )
            for b, (low, high) in enumerate(DISTANCE_BANDS)
        ]
        if count
        else [],
        mean_distance=round(sum(distances) / count, 2) if count else None,
        ablation_agreement=round(agreement, 4) if agreement is not None else None,
    )
