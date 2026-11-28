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
from pathlib import Path

import click
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import (
    GPT2,
    load_split,
    random_init_runner,
    read_mlp_inputs,
    write_row,
)
from resting_conditions import (
    BATCH,
    CONDITIONS,
    N_HEAD,
    PYTHIA_ROTARY_BASE,
    PYTHIA_ROTARY_SHARE,
    SGD_MOMENTUM,
    TRAIN_WINDOW,
    Condition,
)
from resting_modules import (
    Activation,
    centre_norms,
    keeper,
    offset_stats,
    parallel_forward,
    rotary_hook,
    warmup,
)
from resting_push import PushRecorder

from openinterpretability.cli import _load, resolve_device
from openinterpretability.cli_evidence import read_corpus
from openinterpretability.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

PILE_GLOB = "datasets--NeelNanda--pile-10k/snapshots/*/data/*.parquet"
TOKENIZER = "openai-community/gpt2"
GRADIENT_CLIP = 1.0
EVAL_WINDOWS = 128
PUSH_BINS = (-1.5, -0.75, -0.25, 0.25, 0.75, 1.5)  # edges of h for --spread-bins


def build_model(condition: Condition, vocab_size: int, seed: int):
    """Return a small GPT-2 with the condition's dropout, width and activation."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=condition.window,
        n_layer=condition.n_layer,
        n_embd=condition.n_embd,
        n_head=N_HEAD,
        n_inner=condition.n_embd * condition.mlp_ratio,
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
        if condition.parallel:
            block.forward = types.MethodType(parallel_forward, block)
        if condition.positions == "rotary":
            dims = int(condition.n_embd // N_HEAD * PYTHIA_ROTARY_SHARE)
            block.attn.c_attn.register_forward_hook(rotary_hook(N_HEAD, dims, PYTHIA_ROTARY_BASE))
    if condition.positions == "rotary":  # no learned position vectors: zero and frozen
        nn.init.zeros_(model.transformer.wpe.weight)
        model.transformer.wpe.weight.requires_grad_(False)
    centre_norms(model, condition.centre_norms)
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
    tokens: torch.Tensor, generator: torch.Generator, size: int = BATCH,
    window: int = TRAIN_WINDOW,
) -> torch.Tensor:  # fmt: skip
    """Return ``size`` random windows of the training tokens, each ``window`` tokens long."""
    starts = torch.randint(len(tokens) - window, (size,), generator=generator)
    return torch.stack([tokens[s : s + window] for s in starts.tolist()])


@torch.no_grad()
def evaluate(model, ids: torch.Tensor) -> dict:
    """Return the held-out loss and, per layer, where the neurons rest and why."""
    model.eval()
    inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(model.transformer.h))}
    # The residual stream at the input of each block's two LayerNorms, and each MLP's output,
    # for the offset measures of #3192.
    seen: dict[tuple[str, int], list[torch.Tensor]] = {}
    handles = []
    for i, block in enumerate(model.transformer.h):

        def record(_module, args, i=i):
            inputs[i].append(args[0][:, 1:].reshape(-1, args[0].shape[-1]).float().cpu())

        handles.append(block.mlp.register_forward_pre_hook(record))
        for key, module in (("ln_1", block.ln_1), ("ln_2", block.ln_2)):
            handles.append(module.register_forward_pre_hook(keeper(seen, key, i)))
        handles.append(block.mlp.register_forward_hook(keeper(seen, "mlp_out", i, output=True)))
    device = next(model.parameters()).device
    total = 0.0
    batches = ids.split(BATCH)
    for batch in batches:
        batch = batch.to(device)
        total += model(batch, labels=batch).loss.item()
    for handle in handles:
        handle.remove()
    blocks = enumerate(model.transformer.h)
    layers = [{**_layer_stats(block.mlp, torch.cat(inputs[i])), **offset_stats(seen, i)}
              for i, block in blocks]  # fmt: skip
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
    micro_batch: int = 0,
) -> None:
    """Train one model and append a row, labelled ``part``, at each evaluation.

    If ``push_every`` is not 0, also append a row labelled ``push_part`` after each
    ``push_every`` steps. The row summarises the push on the rest over those steps.
    If ``spread_bins`` is true, the row also splits the spread term by the bins of h
    that ``PUSH_BINS`` gives.
    If ``save_every`` is not 0, save the weights to ``save_dir`` every ``save_every`` steps,
    as ``checkpoint_path`` names them.
    If ``micro_batch`` is not 0, each step's batch goes through the model in parts of
    ``micro_batch`` windows, as ``accumulate`` does it.
    """
    condition = CONDITIONS[name]
    if spread_bins and 0 < micro_batch < condition.batch:
        # The bins centre each token on the batch's mean, which one part does not hold.
        raise click.UsageError(f"{name}: --spread-bins needs a micro-batch of the whole batch")
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
        batch = sample_batch(tokens, generator, condition.batch, condition.window).to(device)
        optimizer.zero_grad()
        loss = accumulate(model, batch, micro_batch, recorder)
        if not loss.isfinite():
            # MPS can fail a command buffer and return NaN, so stop and let a person look.
            raise click.ClickException(f"{name} seed {seed}: loss {loss.item()} at step {step}")
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


def accumulate(model, batch: torch.Tensor, micro_batch: int = 0, recorder=None) -> torch.Tensor:
    """Add the gradient of the mean loss on ``batch`` to the model's gradients.

    The batch goes through the model in parts of ``micro_batch`` windows, so that a batch
    too large for the device's memory still trains (#6499). Each part's loss is weighted by
    its share of the windows. So the gradient is the same as the whole batch's, except for
    rounding.

    Args:
        model: The model, in training mode. Its gradients are added to, not zeroed.
        batch: The step's windows, ``[windows, tokens]``.
        micro_batch: The windows in each forward pass. 0 sends the whole batch at once.
        recorder: A ``PushRecorder``, which then adds up the push over the parts (#6806).

    Returns:
        The mean loss on the batch, with no graph.
    """
    total = torch.zeros((), device=batch.device)
    if recorder:
        recorder.start_parts()
    for part in batch.split(micro_batch or len(batch)):
        loss = model(part, labels=part).loss * (len(part) / len(batch))
        loss.backward()
        total += loss.detach()
        if recorder:
            recorder.add_part(len(part) / len(batch))
    if recorder:
        recorder.end_parts()
    return total


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
@click.option(
    "--micro-batch",
    type=int,
    default=0,
    show_default=True,
    help="Windows in each forward pass, with the gradient accumulated. 0 is the whole batch.",
)
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
    micro_batch,
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
                micro_batch,
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
