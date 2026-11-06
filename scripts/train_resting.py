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
}


class Activation(nn.Module):
    """The MLP's activation, with an optional fixed resting level for each neuron."""

    def __init__(self, name: str, fixed_rest: float | None, learned_width: int | None) -> None:
        super().__init__()
        self.name = name
        self.fixed_rest = fixed_rest
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
        if self.name == "gelu":
            return nn.functional.gelu(h)
        if self.name == "relu":
            return nn.functional.relu(h)
        if self.name == "tanh":
            return torch.tanh(h)
        if self.name == "mirrored_gelu":
            return -nn.functional.gelu(-h)
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
        block.mlp.act = Activation(condition.activation, condition.fixed_rest, learned_width)
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


def train(name: str, seed: int, steps: int, eval_every: int, tokens, test_ids, device, out) -> None:
    """Train one model and append a row at each evaluation."""
    condition = CONDITIONS[name]
    vocab = int(tokens.max()) + 1
    model = build_model(condition, vocab, seed).to(device).train()
    # At the base rate, Adam cannot move one parameter by much more than 1 in a short run.
    rests = [p for n, p in model.named_parameters() if n.endswith("learned_rest")]
    others = [p for n, p in model.named_parameters() if not n.endswith("learned_rest")]
    groups = [{"params": others}]
    if rests:
        groups.append({"params": rests, "lr": LEARNING_RATE * condition.rest_lr_factor})
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE, weight_decay=condition.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    start = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            row = {"part": "training", "condition": name, "seed": seed, "step": step}
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
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()


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
def train_command(names, seeds, steps, eval_every, corpus, cache, device, out):
    """Train each condition with each seed, and append the rows to ``--out``."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokens = training_tokens(tokenizer, corpus, cache)
    split = load_split(tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    click.echo(f"{len(tokens)} training tokens", err=True)
    test_ids = split.test[:EVAL_WINDOWS]
    for seed in seeds:
        for name in names or sorted(CONDITIONS):
            train(name, seed, steps, eval_every, tokens, test_ids, resolve_device(device), out)


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
