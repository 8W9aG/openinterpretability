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
import types
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
from resting_modules import Activation, CentredNorm, parallel_forward, warmup
from resting_push import PushRecorder

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
PUSH_BINS = (-1.5, -0.75, -0.25, 0.25, 0.75, 1.5)  # edges of h for --spread-bins
SGD_MOMENTUM = 0.9
ADAM_BETAS = (0.9, 0.999)  # torch's default
# Pythia's training setup (Biderman et al., 2023) uses AdamW at 1e-3, betas (0.9, 0.95)
# and weight decay 0.01. It has no dropout and has parallel attention. It warms up over 1%
# of 143,000 steps.
# Its warm-up is 1,430 of the 4,000 steps that the checkpoint test reads. 715 keeps that
# share in a 2,000-step run.
PYTHIA_LEARNING_RATE = 1e-3
PYTHIA_BETAS = (0.9, 0.95)
PYTHIA_WEIGHT_DECAY = 0.01
PYTHIA_WARMUP = 715
PYTHIA_BATCH_FACTOR = 4  # Pythia's batch is 500 times ours. 4 times fits the time we have.
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
    output_shift: float = 0.0  # a constant added to the activation's output
    centre_output: bool = False  # the activation's output is centred per neuron over each batch
    # "blocks" centres the input of each block's LayerNorms over the batch's tokens, and
    # "all" also the final LayerNorm's. Then no LayerNorm sees a constant offset. "final"
    # centres only the final LayerNorm's input.
    centre_norms: str = ""
    batch: int = BATCH  # windows in each step
    warmup_steps: int = 0  # the learning rate grows linearly over these steps, then holds
    adam_betas: tuple[float, float] = ADAM_BETAS
    parallel: bool = False  # attention and the MLP read the same input, as in GPT-NeoX


# The mean output of ReLU and of GELU at init, where h has an SD of about 0.31. For ReLU it
# is 0.31/√(2π), and for GELU about 0.31² × φ(0). The shifts swap the two means. They test
# whether the mean output sets the sign of the output's push.
INIT_MEAN_RELU = 0.124
INIT_MEAN_GELU = 0.038

BASE = Condition()
CONDITIONS = {
    "gelu": BASE,
    "relu": replace(BASE, activation="relu"),
    "tanh": replace(BASE, activation="tanh"),
    "mirrored_gelu": replace(BASE, activation="mirrored_gelu"),
    "dropout_0": replace(BASE, dropout=0.0),
    "dropout_0.3": replace(BASE, dropout=0.3),
    # With no dropout, a constant offset in the MLP's output stays constant on each token.
    "relu_dropout_0": replace(BASE, activation="relu", dropout=0.0),
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
    # rest_0 with another activation: the push on the rest when every activation rests at 0.
    "relu_rest_0": replace(BASE, activation="relu", fixed_rest=0.0),
    "softplus_rest_0": replace(BASE, activation="softplus", fixed_rest=0.0),
    "leaky_relu_rest_0": replace(BASE, activation="leaky_relu", fixed_rest=0.0),
    "silu_rest_0": replace(BASE, activation="silu", fixed_rest=0.0),
    "tanh_rest_0": replace(BASE, activation="tanh", fixed_rest=0.0),
    # Fixed rests below 0: the push on the rest at each of several depths.
    "relu_rest_-0.5": replace(BASE, activation="relu", fixed_rest=-0.5),
    "relu_rest_-1": replace(BASE, activation="relu", fixed_rest=-1.0),
    "relu_rest_-2": replace(BASE, activation="relu", fixed_rest=-2.0),
    "relu_rest_-4": replace(BASE, activation="relu", fixed_rest=-4.0),
    "softplus_rest_-1": replace(BASE, activation="softplus", fixed_rest=-1.0),
    "softplus_rest_-2": replace(BASE, activation="softplus", fixed_rest=-2.0),
    "softplus_rest_-4": replace(BASE, activation="softplus", fixed_rest=-4.0),
    "rest_-0.5": replace(BASE, fixed_rest=-0.5),
    "leaky_relu_rest_-1": replace(BASE, activation="leaky_relu", fixed_rest=-1.0),
    "silu_rest_-1": replace(BASE, activation="silu", fixed_rest=-1.0),
    # Centring the output removes the output's push, so only the spread term pushes h.
    "gelu_output_centred": replace(BASE, centre_output=True),
    "silu_output_centred": replace(BASE, activation="silu", centre_output=True),
    "relu_output_shift": replace(
        BASE, activation="relu", output_shift=INIT_MEAN_GELU - INIT_MEAN_RELU
    ),
    "gelu_output_shift": replace(BASE, output_shift=INIT_MEAN_RELU - INIT_MEAN_GELU),
    "relu_norms_centred": replace(BASE, activation="relu", centre_norms="blocks"),
    "relu_all_norms_centred": replace(BASE, activation="relu", centre_norms="all"),
    # Every norm centred and no dropout: a constant offset in the residual stream then has no
    # route to the loss.
    "relu_norms_centred_dropout_0": replace(
        BASE, activation="relu", centre_norms="blocks", dropout=0.0
    ),
    "relu_all_norms_centred_dropout_0": replace(
        BASE, activation="relu", centre_norms="all", dropout=0.0
    ),
    # Only the block LayerNorms can see a constant offset.
    "relu_final_norm_centred_dropout_0": replace(
        BASE, activation="relu", centre_norms="final", dropout=0.0
    ),
    # Pythia's setup, one part at a time, on the no-dropout baseline, then all parts.
    "pythia_lr": replace(BASE, dropout=0.0, learning_rate=PYTHIA_LEARNING_RATE),
    "pythia_warmup": replace(
        BASE, dropout=0.0, learning_rate=PYTHIA_LEARNING_RATE, warmup_steps=PYTHIA_WARMUP
    ),
    "pythia_adam": replace(
        BASE, dropout=0.0, adam_betas=PYTHIA_BETAS, weight_decay=PYTHIA_WEIGHT_DECAY
    ),
    "pythia_parallel": replace(BASE, dropout=0.0, parallel=True),
    "pythia_batch": replace(BASE, dropout=0.0, batch=BATCH * PYTHIA_BATCH_FACTOR),
    "pythia_all": replace(
        BASE,
        dropout=0.0,
        learning_rate=PYTHIA_LEARNING_RATE,
        warmup_steps=PYTHIA_WARMUP,
        adam_betas=PYTHIA_BETAS,
        weight_decay=PYTHIA_WEIGHT_DECAY,
        parallel=True,
        batch=BATCH * PYTHIA_BATCH_FACTOR,
    ),
    "sgd": replace(BASE, optimizer="sgd", learning_rate=SGD_LEARNING_RATE),
    "relu_sgd": replace(BASE, activation="relu", optimizer="sgd", learning_rate=SGD_LEARNING_RATE),
}


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
            condition.activation,
            condition.fixed_rest,
            learned_width,
            condition.leaky_slope,
            condition.output_shift,
            condition.centre_output,
        )
        if condition.centre_norms in ("blocks", "all"):
            block.ln_1, block.ln_2 = CentredNorm(block.ln_1), CentredNorm(block.ln_2)
        if condition.parallel:
            block.forward = types.MethodType(parallel_forward, block)
    if condition.centre_norms in ("all", "final"):
        model.transformer.ln_f = CentredNorm(model.transformer.ln_f)
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


def sample_batch(
    tokens: torch.Tensor, generator: torch.Generator, size: int = BATCH
) -> torch.Tensor:
    """Return ``size`` random windows of the training tokens."""
    starts = torch.randint(len(tokens) - WINDOW_TOKENS, (size,), generator=generator)
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
        return torch.optim.AdamW(groups, lr=lr, betas=condition.adam_betas, weight_decay=decay)
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
    push_part: str = "push",
    save_every: int = 0,
    save_dir: Path | None = None,
    spread_bins: bool = False,
) -> None:
    """Train one model and append a row, labelled ``part``, at each evaluation.

    If ``push_every`` is not 0, also append a row labelled ``push_part`` after each
    ``push_every`` steps. The row summarises the push on the rest over those steps.
    If ``spread_bins`` is true, the row also splits the spread term by the bins of h
    that ``PUSH_BINS`` gives.
    If ``save_every`` is not 0, save the weights to ``save_dir`` every ``save_every`` steps,
    as ``checkpoint_path`` names them.
    """
    condition = CONDITIONS[name]
    vocab = int(tokens.max()) + 1
    model = build_model(condition, vocab, seed).to(device).train()
    optimizer = make_optimizer(condition, model)
    schedule = warmup(optimizer, condition.warmup_steps)
    generator = torch.Generator().manual_seed(seed)
    bins = torch.tensor(PUSH_BINS) if spread_bins else None
    recorder = PushRecorder(model, bins=bins) if push_every else None
    start = time.time()
    for step in range(steps + 1):
        if save_every and step % save_every == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path(save_dir, name, seed, step))
        if step % eval_every == 0:
            row = {"part": part, "condition": name, "seed": seed, "step": step}
            write_row(out, {**row, **evaluate(model, test_ids)})
            click.echo(f"{name} seed {seed} step {step} {time.time() - start:.0f}s", err=True)
        if step == steps:
            break
        batch = sample_batch(tokens, generator, condition.batch).to(device)
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
        if schedule:
            schedule.step()
        if recorder:
            recorder.after_step()
            if (step + 1) % push_every == 0:
                row = {"part": push_part, "condition": name, "seed": seed, "step": step + 1}
                write_row(out, {**row, "steps": push_every, "layers": recorder.row()})


def checkpoint_path(save_dir: Path, name: str, seed: int, step: int | str) -> Path:
    """Return the file of one saved checkpoint of a training run. A ``step`` of "*" globs."""
    return save_dir / f"{name}_seed{seed}_step{step}.pt"


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
@click.option("--push-part", default="push", show_default=True, help="The label of the push rows.")
@click.option(
    "--spread-bins",
    is_flag=True,
    help="Also split the push rows' spread term by the bins of h in PUSH_BINS.",
)
@click.option("--save-every", type=int, default=0, show_default=True, help="0 saves no weights.")
@click.option("--save-dir", type=click.Path(path_type=Path), default=None)
def train_command(
    names,
    seeds,
    steps,
    eval_every,
    corpus,
    cache,
    device,
    out,
    part,
    push_every,
    push_part,
    spread_bins,
    save_every,
    save_dir,
):
    """Train each condition with each seed, and append the rows to ``--out``."""
    if save_every and save_dir is None:
        raise click.UsageError("--save-every needs --save-dir")
    if spread_bins and not push_every:
        raise click.UsageError("--spread-bins needs --push-every")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokens = training_tokens(tokenizer, corpus, cache)
    split = load_split(tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    click.echo(f"{len(tokens)} training tokens", err=True)
    test_ids = split.test[:EVAL_WINDOWS]
    for seed in seeds:
        for name in names or sorted(CONDITIONS):
            device_ = resolve_device(device)
            train(
                name,
                seed,
                steps,
                eval_every,
                tokens,
                test_ids,
                device_,
                out,
                part,
                push_every,
                push_part,
                save_every,
                save_dir,
                spread_bins,
            )


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
