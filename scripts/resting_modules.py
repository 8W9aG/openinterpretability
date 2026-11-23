"""The MLP activation and LayerNorm wrapper of the small models in scripts/train_resting.py."""

from __future__ import annotations

import torch
from figure_svg import LEAKY_SLOPE
from torch import nn


class Activation(nn.Module):
    """The MLP's activation, with an optional fixed resting level for each neuron."""

    def __init__(
        self,
        name: str,
        fixed_rest: float | None,
        learned_width: int | None,
        leaky_slope: float = LEAKY_SLOPE,
        output_shift: float = 0.0,
        centre_output: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.fixed_rest = fixed_rest
        self.leaky_slope = leaky_slope
        self.output_shift = output_shift
        self.centre_output = centre_output
        # When ``record`` is true, each backward pass sets ``rest_grad`` and ``output_grad``.
        # Each is the sum over tokens of dL/dh or dL/da, one value for each neuron.
        # Each forward pass also sets ``mean_slope``, the mean over tokens of the slope a'(h).
        self.record = False
        self.rest_grad: torch.Tensor | None = None
        self.output_grad: torch.Tensor | None = None
        self.mean_slope: torch.Tensor | None = None
        # ``mean_output`` is the mean over tokens of the output that the MLP's c_proj reads.
        self.mean_output: torch.Tensor | None = None
        # ``energy_push`` is −Σ a·a'(h) over tokens: the push if dL/da were a itself, which
        # is −½ d/d(rest) of Σ a². ``output_correlation`` is the correlation over tokens of
        # dL/da with a, one value for each neuron.
        self.energy_push: torch.Tensor | None = None
        self.output_correlation: torch.Tensor | None = None
        # When bins holds edges of h, each backward pass also sets spread_bins, [bins, width].
        # It is the spread term, split by the bin of each token's h. For each bin, it sums
        # (g - mean g)(s - mean s) over the bin's tokens. Here g is -dL/da and s is a'(h).
        # The bins add up to the spread term, whatever the mean of g.
        self.bins: torch.Tensor | None = None
        self.spread_bins: torch.Tensor | None = None
        # ``output_bins`` sums g over each bin's tokens, and ``count_bins`` counts them (#3193).
        self.output_bins: torch.Tensor | None = None
        self.count_bins: torch.Tensor | None = None
        # This control separates the cost of centring from the cost of a shared, fixed rest.
        self.learned_rest = nn.Parameter(torch.zeros(learned_width)) if learned_width else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return the activation of ``h``, ``[batch, seq, width]``."""
        if self.fixed_rest is not None:
            # The batch's mean holds for evaluation too, because a batch has about 4,000
            # tokens. The gradient passes through the mean, as in batch normalisation.
            h = h - h.mean((0, 1), keepdim=True) + self.fixed_rest
        elif self.learned_rest is not None:
            h = h - h.mean((0, 1), keepdim=True) + self.learned_rest
        out = self._shape(h) + self.output_shift
        if self.record and out.requires_grad:
            # dL/dh summed over tokens is the gradient on a shift of the neuron's whole h,
            # after any centring. Without centring, it equals the gradient on the bias.
            h.register_hook(lambda g: setattr(self, "rest_grad", g.sum((0, 1)).detach()))
            out.register_hook(lambda g: setattr(self, "output_grad", g.sum((0, 1)).detach()))
            slope = self._slope(h.detach())
            self.mean_slope = slope.mean((0, 1))
            output = out.detach()
            self.energy_push = -(output * slope).sum((0, 1))
            out.register_hook(lambda g: self._correlate(g.detach(), output))
            if self.bins is not None:
                where = torch.bucketize(h.detach(), self.bins.to(h.device))
                centred = slope - slope.mean((0, 1))
                out.register_hook(lambda g: self._split_spread(g.detach(), centred, where))
        if self.centre_output:
            # dL/da summed over tokens, the output's push, is then 0 on every step.
            out = out - out.mean((0, 1), keepdim=True)
        if self.record:
            self.mean_output = out.detach().mean((0, 1))
        return out

    def _correlate(self, grad: torch.Tensor, output: torch.Tensor) -> None:
        """Keep the correlation over tokens of dL/da with a, for each neuron."""
        grad = grad - grad.mean((0, 1))
        output = output - output.mean((0, 1))
        norm = (grad.square().sum((0, 1)) * output.square().sum((0, 1))).sqrt()
        self.output_correlation = (grad * output).sum((0, 1)) / norm.clamp_min(1e-30)

    def _split_spread(self, grad: torch.Tensor, centred: torch.Tensor, where: torch.Tensor):
        """Keep the spread term split by the bin of h, one row per bin."""
        push = -grad
        product = (push - push.mean((0, 1))) * centred
        index = where.reshape(-1, product.shape[-1])
        self.spread_bins = self._bin_sum(index, product)
        self.output_bins = self._bin_sum(index, push)
        self.count_bins = self._bin_sum(index, torch.ones_like(push))

    def _bin_sum(self, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Sum ``values`` over each bin's tokens, one row per bin and one column per neuron.

        Args:
            index: Each token's bin, ``[tokens, width]``.
            values: The values to sum, ``[batch, seq, width]``.

        Returns:
            ``[bins + 1, width]``.
        """
        out = torch.zeros(len(self.bins) + 1, values.shape[-1], device=values.device)
        return out.scatter_add_(0, index, values.reshape(-1, values.shape[-1]))

    def _slope(self, h: torch.Tensor) -> torch.Tensor:
        """Return the slope a'(h) on each element, in a graph of its own."""
        with torch.enable_grad():
            h = h.clone().requires_grad_(True)
            (slope,) = torch.autograd.grad(self._shape(h).sum(), h)
        return slope

    def _shape(self, h: torch.Tensor) -> torch.Tensor:
        """Return the activation function of ``h``, with no centring."""
        if self.name == "gelu":
            return nn.functional.gelu(h)
        if self.name == "gelu_new":  # the tanh form of GELU that GPT-2 small uses
            return nn.functional.gelu(h, approximate="tanh")
        if self.name == "relu":
            return nn.functional.relu(h)
        if self.name == "tanh":
            return torch.tanh(h)
        if self.name == "mirrored_gelu":
            return -nn.functional.gelu(-h)
        if self.name == "softplus":
            return nn.functional.softplus(h)
        if self.name == "leaky_relu":
            return nn.functional.leaky_relu(h, self.leaky_slope)
        if self.name == "silu":
            return nn.functional.silu(h)
        raise ValueError(f"unknown activation {self.name}")


class CentredNorm(nn.Module):
    """A LayerNorm whose input is first centred over the batch's tokens."""

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the norm of ``x``, ``[batch, seq, d_model]``, less its mean over tokens."""
        return self.norm(x - x.mean((0, 1), keepdim=True))


def parallel_forward(
    self,
    hidden_states,
    past_key_values=None,
    attention_mask=None,
    encoder_hidden_states=None,  # noqa: ARG001  (the forward it replaces takes these)
    encoder_attention_mask=None,  # noqa: ARG001
    use_cache=False,
    **kwargs,
):
    """Run a GPT-2 block with GPT-NeoX's parallel residual: x + attn(ln_1(x)) + mlp(ln_2(x))."""
    attn, _ = self.attn(
        self.ln_1(hidden_states),
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=use_cache,
        **kwargs,
    )
    return hidden_states + attn + self.mlp(self.ln_2(hidden_states))


def warmup(optimizer: torch.optim.Optimizer, steps: int):
    """Return a schedule that grows the learning rate linearly over ``steps``, or None."""
    if not steps:
        return None
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / steps))


def keeper(seen: dict, key: str, layer: int, output: bool = False):
    """A hook that keeps a module's input, or its output, without the first token.

    Args:
        seen: Where the tensors go, by ``(key, layer)``.
        key: The name of what the hook keeps.
        layer: The block's index.
        output: Keep the module's output, not its input.

    Returns:
        The hook, for ``register_forward_pre_hook`` or, with ``output``,
        ``register_forward_hook``.
    """

    def keep(_module, args, result=None):
        """Keep the module's input, or its ``result``, as ``[tokens, features]``.

        Args:
            _module: The hooked module.
            args: Its positional inputs.
            result: Its output, when the hook keeps outputs.
        """
        x = result if output else args[0]
        seen.setdefault((key, layer), []).append(x[:, 1:].reshape(-1, x.shape[-1]).float().cpu())

    if output:
        return lambda module, args, result: keep(module, args, result)
    return lambda module, args: keep(module, args)


def offset_ratio(x: torch.Tensor) -> float:
    """The norm of ``x``'s mean over tokens, over the mean norm of each token's deviation.

    Args:
        x: ``[tokens, d_model]``.

    Returns:
        The ratio. A LayerNorm centres over features, not tokens, so a large ratio leaves
        little of its output for the part that differs between tokens.

    Examples:
        >>> offset_ratio(torch.tensor([[3.0, 1.0], [3.0, -1.0]]))
        3.0
    """
    mean = x.mean(0)
    return (mean.norm() / (x - mean).norm(dim=1).mean()).item()


def offset_stats(seen: dict, layer: int) -> dict:
    """One block's offset measures, from the tensors that :func:`keeper` kept (#3192).

    Args:
        seen: The kept tensors, by ``(key, layer)``.
        layer: The block's index.

    Returns:
        The offset ratio at the input of ``ln_1`` and of ``ln_2``, and the norm of the MLP's
        mean output.
    """
    stream = {key: torch.cat(seen[(key, layer)]) for key in ("ln_1", "ln_2", "mlp_out")}
    return {
        "ln_1_offset_ratio": offset_ratio(stream["ln_1"]),
        "ln_2_offset_ratio": offset_ratio(stream["ln_2"]),
        "mlp_mean_output_norm": stream["mlp_out"].mean(0).norm().item(),
    }
