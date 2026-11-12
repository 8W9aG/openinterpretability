"""Record the rest and the push on it in pretrained models, with no optimizer step.

The small models of scripts/train_resting.py show the push while their rest forms
(papers/why-neurons-rest/README.md). The ``gpt2`` command shows trained GPT-2 small, the end
of its training only. The ``pythia`` command shows Pythia-70m at its training checkpoints.
Each MLP gets the recording ``Activation``. Each GPT-NeoX MLP is also swapped for one with
GPT-2's layout and the same weights. Then ``PushRecorder`` and ``_layer_stats`` read every MLP
as they read a small model's.

    uv run python scripts/pretrained_push.py gpt2 --out runs.jsonl
    uv run python scripts/pretrained_push.py pythia --out runs.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import GPT2, load_split, write_row
from resting_modules import Activation
from resting_push import PushRecorder
from train_resting import BATCH, _layer_stats

from openinterp.cli import resolve_device
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

PYTHIA = "EleutherAI/pythia-70m"
# Pythia keeps a checkpoint at step 0, at powers of 2 up to 512, and every 1,000 steps.
CHECKPOINTS = (0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000)
# The rest is read from the first REST_BATCHES batches, which keeps h to a few hundred MB.
REST_BATCHES = 8
corpus_option = click.option(
    "--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl")
)
batches_option = click.option(
    "--batches", default=50, show_default=True, help="Batches of BATCH windows."
)
device_option = click.option("--device", default="auto", show_default=True)
out_option = click.option("--out", type=click.Path(path_type=Path), required=True)


class ConvMLP(nn.Module):
    """A GPT-NeoX MLP with GPT-2's layout: ``c_fc``, ``act`` and ``c_proj`` as Conv1D."""

    def __init__(self, mlp: nn.Module) -> None:
        from transformers.pytorch_utils import Conv1D

        super().__init__()
        d_model, width = mlp.dense_h_to_4h.in_features, mlp.dense_h_to_4h.out_features
        self.c_fc, self.c_proj = Conv1D(width, d_model), Conv1D(d_model, width)
        with torch.no_grad():
            # Conv1D computes x @ weight + bias, so its weight is nn.Linear's transposed.
            self.c_fc.weight.copy_(mlp.dense_h_to_4h.weight.T)
            self.c_fc.bias.copy_(mlp.dense_h_to_4h.bias)
            self.c_proj.weight.copy_(mlp.dense_4h_to_h.weight.T)
            self.c_proj.bias.copy_(mlp.dense_4h_to_h.bias)
        self.act = Activation("gelu", None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the MLP's output for ``x``, ``[batch, seq, d_model]``."""
        return self.c_proj(self.act(self.c_fc(x)))


def corpus_windows(tokenizer, corpus: Path) -> torch.Tensor:
    """Return the fit and test windows of the corpus sample, cut with ``tokenizer``."""
    split = load_split(tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    return torch.cat([split.fit, split.test])


def push_row(model, mlps: list[nn.Module], ids: torch.Tensor, batches: int):
    """Return the loss, and each layer's rest and push, over ``batches`` random batches.

    Each batch's gradient is recorded, and no optimizer step is taken. So the row is the
    push at the model's present weights. Also return, for each layer, each neuron's mean push
    and its rest, the median of h over tokens.
    """
    inputs: list[list[torch.Tensor]] = [[] for _ in mlps]

    def keep(i):
        def hook(_module, args):
            x = args[0][:, 1:].detach()
            inputs[i].append(x.reshape(-1, x.shape[-1]).float().cpu())

        return hook

    handles = [mlp.register_forward_pre_hook(keep(i)) for i, mlp in enumerate(mlps)]
    recorder = PushRecorder(model, mlps)
    generator = torch.Generator().manual_seed(0)
    device = next(model.parameters()).device
    losses = []
    for _ in range(batches):
        batch = ids[torch.randint(len(ids), (BATCH,), generator=generator)].to(device)
        model.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        loss.backward()
        recorder.before_step()
        recorder.after_step()  # no step is taken, so the rest moves by 0
        losses.append(loss.item())
    for handle in handles:
        handle.remove()
    xs = [torch.cat(x[:REST_BATCHES]) for x in inputs]
    rests = [_layer_stats(mlp, x) for mlp, x in zip(mlps, xs, strict=True)]
    neurons = [
        (recorder.window["push"][i].cpu() / recorder.steps, _neuron_rests(mlp, xs[i]))
        for i, mlp in enumerate(mlps)
    ]
    layers = [{**rest, **push} for rest, push in zip(rests, recorder.row(), strict=True)]
    return {"loss": sum(losses) / batches, "batches": batches, "layers": layers}, neurons


def _neuron_rests(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return each neuron's rest, the median over the tokens ``x`` of its h."""
    weights = mlp.c_fc.weight.detach().float().cpu()  # Conv1D: [d_model, width]
    return (x @ weights + mlp.c_fc.bias.detach().float().cpu()).median(0).values


def move_stats(push: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> dict:
    """Return how well each neuron's push predicts how its rest moves to the next checkpoint.

    The push is −dL/dh, so a positive push predicts that the rest moves up.
    """
    moved = after - before
    return {
        "median_moved_to_next": moved.median().item(),
        "push_move_correlation": torch.corrcoef(torch.stack([push, moved]))[0, 1].item(),
        "share_push_sign_moved": (push.sign() == moved.sign()).float().mean().item(),
    }


@click.group()
def cli():
    """Record the rest and the push in pretrained models."""


@cli.command()
@corpus_option
@batches_option
@device_option
@out_option
def gpt2(corpus, batches, device, out):
    """Append one ``gpt2_push`` row for each mode of trained GPT-2 small to ``--out``.

    The ``eval`` mode has no dropout. The ``train`` mode has GPT-2's dropout of 0.1.
    """
    from transformers import AutoTokenizer, GPT2LMHeadModel

    device = resolve_device(device)
    ids = corpus_windows(AutoTokenizer.from_pretrained(GPT2), corpus)
    for mode in ("eval", "train"):
        model = GPT2LMHeadModel.from_pretrained(GPT2).to(device)
        for block in model.transformer.h:
            block.mlp.act = Activation("gelu_new", None, None)
        model.train(mode == "train")
        mlps = [block.mlp for block in model.transformer.h]
        row, _ = push_row(model, mlps, ids, batches)
        write_row(out, {"part": "gpt2_push", "mode": mode, **row})
        click.echo(f"{mode} done", err=True)


@cli.command()
@corpus_option
@click.option("--step", "steps", multiple=True, type=int, help="Default: each of CHECKPOINTS.")
@batches_option
@device_option
@out_option
def pythia(corpus, steps, batches, device, out):
    """Append one ``pythia_push`` row for each checkpoint of Pythia-70m to ``--out``.

    Each row but the last also says how well the push at that checkpoint predicts how each
    neuron's rest moves by the next checkpoint.
    """
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    device = resolve_device(device)
    ids = corpus_windows(AutoTokenizer.from_pretrained(PYTHIA), corpus)
    previous = None
    for step in steps or CHECKPOINTS:
        model = GPTNeoXForCausalLM.from_pretrained(
            PYTHIA, revision=f"step{step}", dtype=torch.float32
        )
        for layer in model.gpt_neox.layers:
            layer.mlp = ConvMLP(layer.mlp)
        model = model.to(device).eval()  # Pythia has no dropout
        mlps = [layer.mlp for layer in model.gpt_neox.layers]
        row, neurons = push_row(model, mlps, ids, batches)
        row = {"part": "pythia_push", "model": PYTHIA, "step": step, **row}
        if previous is not None:
            before_row, before = previous
            for layer, (push, rest), (_, rest_after) in zip(
                before_row["layers"], before, neurons, strict=True
            ):
                layer.update(move_stats(push, rest, rest_after))
            write_row(out, before_row)
        previous = row, neurons
        click.echo(f"step {step} done", err=True)
    if previous is not None:
        write_row(out, previous[0])


if __name__ == "__main__":
    cli()
