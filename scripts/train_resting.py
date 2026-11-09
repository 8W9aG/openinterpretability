"""Train small GPT-2s and record where their MLP neurons rest.

Trained GPT-2 small has its neurons' weights against the mean of the MLP's input. So the
pre-activation h rests near −1 (papers/why-neurons-rest/README.md). Each condition changes
one thing about the training of a small model of the same architecture. The evaluation
windows come from the corpus sample, and the training text leaves those documents out.

    uv run python scripts/train_resting.py train --condition gelu --cache tokens.pt --out runs.jsonl
    uv run python scripts/train_resting.py gpt2 --out runs.jsonl
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import click
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import (
    GPT2,
    WINDOW_TOKENS,
    load_split,
    random_init_runner,
    read_mlp_inputs,
    write_row,
)
from figure_svg import LEAKY_SLOPE

from openinterp.cli import _load, resolve_device
from openinterp.cli_evidence import read_corpus
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

PILE_GLOB = "datasets--NeelNanda--pile-10k/snapshots/*/data/*.parquet"
TOKENIZER = "openai-community/gpt2"
N_LAYER = 4
N_EMBD = 256
N_HEAD = 8
BATCH = 32
LEARNING_RATE = 3e-4
GRADIENT_CLIP = 1.0
EVAL_WINDOWS = 128
SGD_MOMENTUM = 0.9
SGD_LEARNING_RATE = 0.1  # the best of 0.01, 0.03, 0.1 and 0.3 over 500 steps that trained stably


@dataclass(frozen=True)
class Condition:
    """One way to train: the baseline, with the fields that the condition changes."""

    activation: str = "gelu"
    dropout: float = 0.1
    weight_decay: float = 0.0
    mlp_ratio: int = 4
    fixed_rest: float | None = None  # h is centred per neuron, then moved to this value
    learned_rest: bool = False  # h is centred per neuron, then moved to a learned value
    rest_lr_factor: float = 1.0  # multiplies the learning rate of the learned value
    optimizer: str = "adamw"  # "adamw", or "sgd" with momentum SGD_MOMENTUM
    learning_rate: float = LEARNING_RATE
    leaky_slope: float = LEAKY_SLOPE  # the slope of leaky ReLU below 0


BASE = Condition()
CONDITIONS = {
    "gelu": BASE,
    "relu": replace(BASE, activation="relu"),
    "tanh": replace(BASE, activation="tanh"),
    "mirrored_gelu": replace(BASE, activation="mirrored_gelu"),
    "dropout_0": replace(BASE, dropout=0.0),
    "dropout_0.3": replace(BASE, dropout=0.3),
    "weight_decay_0.1": replace(BASE, weight_decay=0.1),
    "mlp_1x": replace(BASE, mlp_ratio=1),
    "mlp_16x": replace(BASE, mlp_ratio=16),
    "rest_0": replace(BASE, fixed_rest=0.0),
    "rest_-1": replace(BASE, fixed_rest=-1.0),
    "rest_learned": replace(BASE, learned_rest=True),
    "rest_learned_fast": replace(BASE, learned_rest=True, rest_lr_factor=30.0),
    "softplus": replace(BASE, activation="softplus"),
    "leaky_relu": replace(BASE, activation="leaky_relu"),
    "leaky_relu_0.01": replace(BASE, activation="leaky_relu", leaky_slope=0.01),
    "silu": replace(BASE, activation="silu"),
    "lr_1e-4": replace(BASE, learning_rate=1e-4),
    "lr_1e-3": replace(BASE, learning_rate=1e-3),
    "relu_lr_1e-4": replace(BASE, activation="relu", learning_rate=1e-4),
    "relu_lr_1e-3": replace(BASE, activation="relu", learning_rate=1e-3),
    "sgd": replace(BASE, optimizer="sgd", learning_rate=SGD_LEARNING_RATE),
    "relu_sgd": replace(BASE, activation="relu", optimizer="sgd", learning_rate=SGD_LEARNING_RATE),
}


class Activation(nn.Module):
    """The MLP's activation, with an optional fixed resting level for each neuron."""

    def __init__(
        self,
        name: str,
        fixed_rest: float | None,
        learned_width: int | None,
        leaky_slope: float = LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        self.name = name
        self.fixed_rest = fixed_rest
        self.leaky_slope = leaky_slope
        # When ``record`` is true, each backward pass sets ``rest_grad`` and ``output_grad``.
        # Each is the sum over tokens of dL/dh or dL/da, one value for each neuron.
        self.record = False
        self.rest_grad: torch.Tensor | None = None
        self.output_grad: torch.Tensor | None = None
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
        out = self._shape(h)
        if self.record and out.requires_grad:
            # dL/dh summed over tokens is the gradient on a shift of the neuron's whole h,
            # after any centring. Without centring, it equals the gradient on the bias.
            h.register_hook(lambda g: setattr(self, "rest_grad", g.sum((0, 1)).detach()))
            out.register_hook(lambda g: setattr(self, "output_grad", g.sum((0, 1)).detach()))
        return out

    def _shape(self, h: torch.Tensor) -> torch.Tensor:
        """Return the activation function of ``h``, with no centring."""
        if self.name == "gelu":
            return nn.functional.gelu(h)
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


def build_model(condition: Condition, vocab_size: int, seed: int):
    """Return a small GPT-2 with the condition's dropout, width and activation."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=WINDOW_TOKENS,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_inner=N_EMBD * condition.mlp_ratio,
        resid_pdrop=condition.dropout,
        embd_pdrop=condition.dropout,
        attn_pdrop=condition.dropout,
    )
    model = GPT2LMHeadModel(config)
    learned_width = config.n_inner if condition.learned_rest else None
    for block in model.transformer.h:
        block.mlp.act = Activation(
            condition.activation, condition.fixed_rest, learned_width, condition.leaky_slope
        )
    return model


def training_tokens(tokenizer, corpus: Path, cache: Path) -> torch.Tensor:
    """Return pile-10k's tokens as one vector, without the corpus sample's documents."""
    if cache.exists():
        return torch.load(cache)
    import glob
    import os

    import pyarrow.parquet as pq

    hub = os.path.expanduser("~/.cache/huggingface/hub")
    path = glob.glob(f"{hub}/{PILE_GLOB}")[0]
    held_out, *_ = read_corpus(corpus)
    prefixes = {text[:200] for text in held_out}
    texts = [t for t in pq.read_table(path, columns=["text"])["text"].to_pylist()]
    kept = [t for t in texts if t[:200] not in prefixes]
    click.echo(f"{len(kept)} of {len(texts)} documents kept for training", err=True)
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for row in tokenizer(kept, add_special_tokens=False)["input_ids"]:
        ids.extend(row)
        ids.append(eos)
    tokens = torch.tensor(ids)
    torch.save(tokens, cache)
    return tokens


def sample_batch(tokens: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return ``BATCH`` random windows of the training tokens."""
    starts = torch.randint(len(tokens) - WINDOW_TOKENS, (BATCH,), generator=generator)
    return torch.stack([tokens[s : s + WINDOW_TOKENS] for s in starts.tolist()])


@torch.no_grad()
def evaluate(model, ids: torch.Tensor) -> dict:
    """Return the held-out loss and, per layer, where the neurons rest and why."""
    model.eval()
    inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(N_LAYER)}
    handles = []
    for i, block in enumerate(model.transformer.h):

        def record(_module, args, i=i):
            inputs[i].append(args[0][:, 1:].reshape(-1, args[0].shape[-1]).float().cpu())

        handles.append(block.mlp.register_forward_pre_hook(record))
    device = next(model.parameters()).device
    total = 0.0
    batches = ids.split(BATCH)
    for batch in batches:
        batch = batch.to(device)
        total += model(batch, labels=batch).loss.item()
    for handle in handles:
        handle.remove()
    blocks = enumerate(model.transformer.h)
    layers = [_layer_stats(block.mlp, torch.cat(inputs[i])) for i, block in blocks]
    model.train()
    return {"loss": total / len(batches), "layers": layers}


def _layer_stats(mlp, x: torch.Tensor) -> dict:
    """Return one MLP's resting statistics on its inputs ``x``, ``[tokens, d_model]``."""
    weights = mlp.c_fc.weight.detach().float().cpu()  # Conv1D: [d_model, width]
    bias = mlp.c_fc.bias.detach().float().cpu()
    mean_input = x.mean(0)
    offset = mean_input @ weights
    h = x @ weights + bias
    cosine = nn.functional.cosine_similarity(weights.T, mean_input.unsqueeze(0), dim=1)
    projection = h - h.mean(0)
    learned = getattr(mlp.act, "learned_rest", None)
    return {
        "median_learned_rest": None if learned is None else learned.median().item(),
        "median_h": h.median(0).values.median().item(),
        "mean_h": h.mean(0).median().item(),
        "median_bias": bias.median().item(),
        "median_offset": offset.median().item(),
        "h_sd": h.std(0).median().item(),
        "skew": ((projection**3).mean(0) / projection.std(0) ** 3).median().item(),
        "share_below_zero": (h < 0).float().mean().item(),
        "cosine_to_mean_input": cosine.median().item(),
        "share_against_mean_input": (cosine < 0).float().mean().item(),
        "mean_input_share": (mean_input.norm() / x.norm(dim=1).mean()).item(),
    }


class PushRecorder:
    """Record, at each step, the push on each neuron's rest and how far the rest moves.

    The push is −dL/dh summed over the batch's tokens: the direction in which gradient
    descent moves the neuron's whole h. It is negative when descent moves the rest down.
    The weights' push is −(dL/dw)·x̄, the gradient on the neuron's input weights along the
    batch's mean input. It is |x̄|² times the push, plus a term from tokens whose input
    lines up with x̄ more or less than the mean does.
    The output's push is −dL/da for the activation's output a. On each token, the rest's
    push is the output's push times the slope of the activation. The movement
    is x̄·ΔW + Δb for one optimizer step, with x̄ the mean of the batch's MLP input.
    """

    def __init__(self, model) -> None:
        self.mlps = [block.mlp for block in model.transformer.h]
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
                "down_weights",
                "moved_weights",
                "moved_bias",
            )
        }
        self.steps = 0

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
            self.window["output_push"][i] += -mlp.act.output_grad.float()
            push_weights = -(self.inputs[i] @ mlp.c_fc.weight.grad).float()
            self.window["push_weights"][i] += push_weights
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
        n = self.steps
        for i in range(len(self.mlps)):
            w = {key: values[i].cpu() for key, values in self.window.items()}
            mean = w["push"] / n
            rms = (w["push_sq"] / n).sqrt()
            moved = w["moved_weights"] + w["moved_bias"]
            layers.append(
                {
                    "median_push": mean.median().item(),
                    "median_output_push": (w["output_push"] / n).median().item(),
                    # Adam moves a parameter by about lr × mean / rms of its gradient.
                    "median_push_ratio": (mean / rms.clamp_min(1e-30)).median().item(),
                    "share_neurons_pushed_down": (mean < 0).float().mean().item(),
                    "median_share_steps_down": (w["down"] / n).median().item(),
                    "median_run_share_steps_down": (self.run_down[i].cpu() / self.run_steps)
                    .median()
                    .item(),
                    "median_push_weights": (w["push_weights"] / n).median().item(),
                    "median_share_steps_down_weights": (w["down_weights"] / n).median().item(),
                    "median_run_share_steps_down_weights": (
                        self.run_down_weights[i].cpu() / self.run_steps
                    )
                    .median()
                    .item(),
                    "median_moved": moved.median().item(),
                    "median_moved_weights": w["moved_weights"].median().item(),
                    "median_moved_bias": w["moved_bias"].median().item(),
                    "share_neurons_moved_down": (moved < 0).float().mean().item(),
                }
            )
        self._reset()
        return layers


def make_optimizer(condition: Condition, model) -> torch.optim.Optimizer:
    """Return the condition's optimizer, with the learned rests in a group of their own."""
    # At the base rate, Adam cannot move one parameter by much more than 1 in a short run.
    rests = [p for n, p in model.named_parameters() if n.endswith("learned_rest")]
    others = [p for n, p in model.named_parameters() if not n.endswith("learned_rest")]
    groups = [{"params": others}]
    if rests:
        groups.append({"params": rests, "lr": condition.learning_rate * condition.rest_lr_factor})
    lr, decay = condition.learning_rate, condition.weight_decay
    if condition.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=SGD_MOMENTUM, weight_decay=decay)
    if condition.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=decay)
    raise ValueError(f"unknown optimizer {condition.optimizer}")


def train(
    name: str,
    seed: int,
    steps: int,
    eval_every: int,
    tokens,
    test_ids,
    device,
    out,
    part,
    push_every: int = 0,
) -> None:
    """Train one model and append a row, labelled ``part``, at each evaluation.

    If ``push_every`` is not 0, also append a row labelled ``push`` after each
    ``push_every`` steps. The row summarises the push on the rest over those steps.
    """
    condition = CONDITIONS[name]
    vocab = int(tokens.max()) + 1
    model = build_model(condition, vocab, seed).to(device).train()
    optimizer = make_optimizer(condition, model)
    generator = torch.Generator().manual_seed(seed)
    recorder = PushRecorder(model) if push_every else None
    start = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            row = {"part": part, "condition": name, "seed": seed, "step": step}
            write_row(out, {**row, **evaluate(model, test_ids)})
            click.echo(f"{name} seed {seed} step {step} {time.time() - start:.0f}s", err=True)
        if step == steps:
            break
        batch = sample_batch(tokens, generator).to(device)
        loss = model(batch, labels=batch).loss
        if not loss.isfinite():
            # MPS can fail a command buffer and return NaN, so stop and let a person look.
            raise click.ClickException(f"{name} seed {seed}: loss {loss.item()} at step {step}")
        optimizer.zero_grad()
        loss.backward()
        if recorder:
            recorder.before_step()  # before the clip, which scales every gradient alike
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if recorder:
            recorder.after_step()
            if (step + 1) % push_every == 0:
                row = {"part": "push", "condition": name, "seed": seed, "step": step + 1}
                write_row(out, {**row, "steps": push_every, "layers": recorder.row()})


@click.group()
def cli():
    """Train small models, or measure GPT-2 small, and record where the neurons rest."""


@cli.command("train")
@click.option("--condition", "names", multiple=True, type=click.Choice(sorted(CONDITIONS)))
@click.option("--seed", "seeds", multiple=True, type=int, default=(0,))
@click.option("--steps", type=int, default=20_000, show_default=True)
@click.option("--eval-every", type=int, default=1000, show_default=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--cache", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--part", default="training", show_default=True, help="The label of the rows.")
@click.option(
    "--push-every",
    type=int,
    default=0,
    show_default=True,
    help="Steps in each row of the push on the rest. 0 records no push.",
)
def train_command(names, seeds, steps, eval_every, corpus, cache, device, out, part, push_every):
    """Train each condition with each seed, and append the rows to ``--out``."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokens = training_tokens(tokenizer, corpus, cache)
    split = load_split(tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    click.echo(f"{len(tokens)} training tokens", err=True)
    test_ids = split.test[:EVAL_WINDOWS]
    for seed in seeds:
        for name in names or sorted(CONDITIONS):
            device_ = resolve_device(device)
            train(name, seed, steps, eval_every, tokens, test_ids, device_, out, part, push_every)


@cli.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="auto", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def gpt2(corpus, device, out):
    """Append the resting statistics of GPT-2 small, trained and at random init, to ``--out``."""
    device = resolve_device(device)
    runner = _load(GPT2, device, torch.float32, None, None)
    split = load_split(runner.tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    fresh = random_init_runner(GPT2, runner.tokenizer, 0, device)
    for name, model in (("trained", runner), ("random_init", fresh)):
        for layer in range(model.n_layers):
            x = read_mlp_inputs(model, split.test, layer).float()
            row = {"part": "gpt2_rest", "weights": name, "layer": layer}
            write_row(out, {**row, **_layer_stats(model.mlp(layer), x)})
        click.echo(f"{name} done", err=True)


if __name__ == "__main__":
    cli()
