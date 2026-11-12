"""The recorder of the push on each neuron's rest, for scripts/train_resting.py."""

from __future__ import annotations

import torch
from torch import nn


class PushRecorder:
    """Record, at each step, the push on each neuron's rest and how far the rest moves.

    The push is −dL/dh summed over the batch's tokens: the direction in which gradient
    descent moves the neuron's whole h. It is negative when descent moves the rest down.
    The weights' push is −(dL/dw)·x̄, the gradient on the neuron's input weights along the
    batch's mean input. It is |x̄|² times the push, plus a term from tokens whose input
    lines up with x̄ more or less than the mean does.
    The output's push is −dL/da for the activation's output a. On each token, the rest's
    push is the output's push times the slope of the activation. So the push is the sum of
    two terms. The slope term is the output's push times the mean slope over tokens. The
    spread term is the rest: the output's push on each token times that token's slope minus
    the mean slope. The offset is the MLP's mean output over the batch, ā·W_out + b_out. The
    recorder keeps its cosine with the gradient on b_out, which is Σδ over tokens, and the
    correlation over neurons between ā and the output's push. The movement
    is x̄·ΔW + Δb for one optimizer step, with x̄ the mean of the batch's MLP input.
    """

    def __init__(self, model, mlps: list[nn.Module] | None = None) -> None:
        # ``mlps`` is for a model that is not a GPT-2. Each must have GPT-2's ``c_fc``,
        # ``act`` and ``c_proj``.
        self.mlps = mlps if mlps is not None else [block.mlp for block in model.transformer.h]
        self.inputs: list[torch.Tensor | None] = [None] * len(self.mlps)
        self.before: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, mlp in enumerate(self.mlps):
            mlp.act.record = True
            mlp.register_forward_pre_hook(self._input_hook(i))
        self.steps = 0
        self.run_steps = 0
        self.run_down = [torch.zeros_like(m.c_fc.bias) for m in self.mlps]
        self.run_down_weights = [torch.zeros_like(m.c_fc.bias) for m in self.mlps]
        self._reset()

    def _input_hook(self, i: int):
        def hook(_module, args):
            x = args[0].detach()
            self.inputs[i] = x.reshape(-1, x.shape[-1]).mean(0)

        return hook

    def _reset(self) -> None:
        zeros = [torch.zeros_like(m.c_fc.bias, dtype=torch.float32) for m in self.mlps]
        self.window = {
            key: [z.clone() for z in zeros]
            for key in (
                "push",
                "push_sq",
                "down",
                "output_push",
                "push_weights",
                "push_weights_sq",
                "push_slope",
                "output_correlation",
                "down_weights",
                "moved_weights",
                "moved_bias",
            )
        }
        self.steps = 0
        self.offset = [{"cosine": 0.0, "correlation": 0.0} for _ in self.mlps]
        self.energy = [{"correlation": 0.0} for _ in self.mlps]

    def before_step(self) -> None:
        """Keep the input weights and biases, and add this step's gradients to the window."""
        self.before = [
            (m.c_fc.weight.detach().clone(), m.c_fc.bias.detach().clone()) for m in self.mlps
        ]
        for i, mlp in enumerate(self.mlps):
            push = -mlp.act.rest_grad.float()
            self.window["push"][i] += push
            self.window["push_sq"][i] += push**2
            self.window["down"][i] += (push < 0).float()
            self.run_down[i] += (push < 0).float()
            output_push = -mlp.act.output_grad.float()
            self.window["output_push"][i] += output_push
            self.window["push_slope"][i] += output_push * mlp.act.mean_slope.float()
            push_weights = -(self.inputs[i] @ mlp.c_fc.weight.grad).float()
            self.window["push_weights"][i] += push_weights
            self.window["push_weights_sq"][i] += push_weights**2
            mean_output = mlp.act.mean_output.float()
            weight = mlp.c_proj.weight.detach().float()  # Conv1D: [width, d_model]
            offset = mean_output @ weight + mlp.c_proj.bias.detach().float()
            bias_grad = mlp.c_proj.bias.grad.float()
            cosine = nn.functional.cosine_similarity(offset, bias_grad, dim=0)
            self.offset[i]["cosine"] += cosine.item()
            pair = torch.stack([mean_output, output_push])
            correlation = torch.corrcoef(pair)[0, 1].nan_to_num()  # NaN if ā is constant
            self.offset[i]["correlation"] += correlation.item()
            self.window["output_correlation"][i] += mlp.act.output_correlation.float()
            pair = torch.stack([push, mlp.act.energy_push.float()])
            self.energy[i]["correlation"] += torch.corrcoef(pair)[0, 1].nan_to_num().item()
            self.window["down_weights"][i] += (push_weights < 0).float()
            self.run_down_weights[i] += (push_weights < 0).float()

    def after_step(self) -> None:
        """Add how far this step moved each neuron's rest, at the batch's mean input."""
        for i, (mlp, (weight, bias)) in enumerate(zip(self.mlps, self.before, strict=True)):
            delta = mlp.c_fc.weight.detach() - weight  # Conv1D: [d_model, width]
            self.window["moved_weights"][i] += (self.inputs[i] @ delta).float()
            self.window["moved_bias"][i] += (mlp.c_fc.bias.detach() - bias).float()
        self.steps += 1
        self.run_steps += 1

    def row(self) -> list[dict]:
        """Return each layer's summary of the steps since the last row, and start a new window."""
        layers = []
        for i in range(len(self.mlps)):
            w = {key: values[i].cpu() for key, values in self.window.items()}
            layers.append({**self._push_row(i, w), **self._source_row(i, w)})
        self._reset()
        return layers

    def _push_row(self, i: int, w: dict) -> dict:
        """Return one layer's push, its steadiness and how far it moved the rest."""
        n = self.steps
        mean = w["push"] / n
        rms = (w["push_sq"] / n).sqrt()
        moved = w["moved_weights"] + w["moved_bias"]
        run_down = self.run_down[i].cpu() / self.run_steps
        run_down_weights = self.run_down_weights[i].cpu() / self.run_steps
        weights_rms = (w["push_weights_sq"] / n).sqrt().clamp_min(1e-30)
        return {
            "median_push": mean.median().item(),
            # Adam moves a parameter by about lr × mean / rms of its gradient.
            "median_push_ratio": (mean / rms.clamp_min(1e-30)).median().item(),
            "share_neurons_pushed_down": (mean < 0).float().mean().item(),
            "median_share_steps_down": (w["down"] / n).median().item(),
            "median_run_share_steps_down": run_down.median().item(),
            "median_push_weights": (w["push_weights"] / n).median().item(),
            "median_push_weights_ratio": (w["push_weights"] / n / weights_rms).median().item(),
            "median_share_steps_down_weights": (w["down_weights"] / n).median().item(),
            "median_run_share_steps_down_weights": run_down_weights.median().item(),
            "median_moved": moved.median().item(),
            "median_moved_weights": w["moved_weights"].median().item(),
            "median_moved_bias": w["moved_bias"].median().item(),
            "share_neurons_moved_down": (moved < 0).float().mean().item(),
        }

    def _source_row(self, i: int, w: dict) -> dict:
        """Return one layer's terms of the push: the output's push, slope, spread and offset."""
        n = self.steps
        return {
            "median_output_push": (w["output_push"] / n).median().item(),
            # The slope term of the push; the spread term is the push minus this.
            "median_push_slope": (w["push_slope"] / n).median().item(),
            "median_push_spread": ((w["push"] - w["push_slope"]) / n).median().item(),
            "mean_offset_cosine": self.offset[i]["cosine"] / n,
            "mean_offset_correlation": self.offset[i]["correlation"] / n,
            # Over neurons: the push against −Σ a·a'(h), the push of an L2 pull on a.
            "mean_energy_correlation": self.energy[i]["correlation"] / n,
            # Over tokens: dL/da against a, as a median over neurons.
            "median_output_correlation": (w["output_correlation"] / n).median().item(),
        }
