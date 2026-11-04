"""The losses of Stochastic Parameter Decomposition (``train_spd.py``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class _LowerLeaky(torch.autograd.Function):
    """``clamp(x, 0, 1)``; below 0, a gradient of slope ``alpha`` only where it raises x."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        zero = torch.zeros_like(grad)
        below = torch.where(grad < 0, ctx.alpha * grad, zero)
        return torch.where(x <= 0, below, torch.where(x <= 1, grad, zero)), None


def lower_leaky(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """The hard sigmoid used for masks: no value above 1 or below 0."""
    return _LowerLeaky.apply(x, alpha)


def upper_leaky(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """The hard sigmoid used for the importance loss: slope ``alpha`` above 1."""
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))


def first_kept(x: torch.Tensor, value: float) -> torch.Tensor:
    """``x`` with position 0 set to ``value``, so the first token keeps the model's weights."""
    x = x.clone()
    x[:, 0] = value
    return x


def stochastic_masks(ci: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A mask ``g + (1 - g)·U(0, 1)`` per subcomponent and ``U(0, 1)`` for the delta."""
    mask = ci + (1 - ci) * torch.rand_like(ci)
    delta = torch.rand(ci.shape[:-1], device=ci.device)
    return first_kept(mask, 1.0), first_kept(delta, 1.0)


def kl(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(model ‖ masked) per token, averaged over every position but the first."""
    p = F.log_softmax(target[:, 1:], -1)
    q = F.log_softmax(pred[:, 1:], -1)
    return (p.exp() * (p - q)).sum(-1).mean()


def importance_loss(upper: dict, p: float, beta: float, eps: float) -> torch.Tensor:
    """Summed over subcomponents: mean ``(g + eps)^p``, plus ``beta``·mean·log2(1 + total).

    Below ``p = 1`` the slope at ``g = 0`` is ``p·eps^(p-1)``. With the reference's
    ``eps = 1e-12`` it is about 1e7 at ``p = 0.4``; on GPT-2 that froze the gates.
    """
    total = 0.0
    for g in upper.values():
        vals = (g[:, 1:] + eps).pow(p).flatten(0, -2)
        sums = vals.sum(0)
        means = sums / vals.shape[0]
        total = total + (means + beta * means * torch.log2(1 + sums)).sum()
    return total


def faithfulness(parts: dict) -> torch.Tensor:
    """The mean squared error of the subcomponents' sum against each weight."""
    sq = sum(part.delta().pow(2).sum() for part in parts.values())
    return sq / sum(part.w_target.numel() for part in parts.values())


def output_loss(parts: dict, hidden: dict, inner: dict, ci: dict) -> torch.Tensor:
    """Per layer, the CI-masked projection's squared error over its output's variance.

    Averaged over layers, leaving out each piece's first position. The KL losses alone let
    the masked output shrink: a LayerNorm follows it, so the logits barely change.
    """
    total = 0.0
    for layer, part in parts.items():
        y = hidden[layer][:, 1:] @ part.w_target
        y_hat = (inner[layer][:, 1:] * ci[layer][:, 1:]) @ part.u
        total = total + (y - y_hat).pow(2).sum() / (y - y.mean((0, 1))).pow(2).sum()
    return total / len(parts)
