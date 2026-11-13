"""Add a constant offset to the input of the final LayerNorm, and measure what it costs.

The loss pulls each MLP's mean output toward 0. The pull goes through the final
LayerNorm (papers/why-neurons-rest/README.md, *Where the push comes from*). One reading
is this. A constant that every token shares takes part of each token's norm. Then the
final LayerNorm divides by a larger norm, so it shrinks the parts that differ between
tokens, and the logits flatten. This script tests it on a trained model.

For each size of offset, the offset ``c`` is added in three ways:

- *full*: ``ln_f(x + c)``, what the model would compute.
- *shrink*: only the LayerNorm's scale changes. The output is ``x``'s, normalised by the
  standard deviation of ``x + c``.
- *shift*: only the direction changes. The output is ``x + c``'s, normalised by the
  standard deviation of ``x``.

Each of these runs with three offsets of the same norm:

- *mean*: along the mean of ``x`` over tokens, which the model already has.
- *mlp*: along the MLPs' mean output, summed over layers. The pull acts on it.
- *random*: one random direction, the same on every token.
- *per_token*: a different random direction on each token, so nothing is shared.

The size of an offset is its norm over the mean norm of ``x - mean(x)``, the part of
``x`` that differs between tokens. Each row has the loss and the mean entropy of the
next-token distribution. One JSON row per model, kind, direction and size goes to
``--out``. Each row also has the mean shrink: the LayerNorm's divisor without the offset
over the divisor with it.

The rows labelled ``ln_offset_slope`` have the derivative of the loss and the entropy as an
offset grows from 0, by autograd. To first order the full slope is the sum of the shrink
slope and the shift slope. So they show which route carries a pull on the mean output.

    uv run python scripts/ln_offset.py gpt2 --out offsets.jsonl
    uv run python scripts/ln_offset.py small --checkpoint run_seed0_step4000.pt --out offsets.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import GPT2, load_split, write_row  # noqa: E402

from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS  # noqa: E402

SIZES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
KINDS = ("full", "shrink", "shift")
DIRECTIONS = ("mean", "mlp", "random", "per_token")
SLOPE_DIRECTIONS = ("mean", "mlp", "random")
WINDOWS = 128  # test windows of 128 tokens, as the paper's evaluation
BATCH = 16


class Offset(nn.Module):
    """``ln_f`` with an offset added to its input, in one of :data:`KINDS`."""

    def __init__(self, norm: nn.LayerNorm, kind: str, offset: torch.Tensor | None):
        super().__init__()
        self.norm, self.kind, self.offset = norm, kind, offset
        self.shrinks: list[float] = []  # mean divisor without the offset over the divisor with it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The LayerNorm of ``x`` with the offset, in the way that ``kind`` says."""
        if self.offset is None:
            return self.norm(x)
        c = self.offset
        if c.dim() == 2:  # one direction for each token
            c = c[: x.shape[0] * x.shape[1]].reshape(x.shape)
        moved = x + c
        with torch.no_grad():
            ratio = (x.var(-1, unbiased=False) + self.norm.eps) / (
                moved.var(-1, unbiased=False) + self.norm.eps
            )
            self.shrinks.append(float(ratio.sqrt().mean()))
        if self.kind == "full":
            return self.norm(moved)
        source = x if self.kind == "shrink" else moved
        scale_from = moved if self.kind == "shrink" else x
        centred = source - source.mean(-1, keepdim=True)
        std = (scale_from.var(-1, unbiased=False, keepdim=True) + self.norm.eps).sqrt()
        return centred / std * self.norm.weight + self.norm.bias


@torch.no_grad()
def residual_stats(model, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The mean input to ``ln_f`` over tokens, the MLPs' summed mean output, and the spread.

    The spread is the mean norm of each token's input to ``ln_f`` minus the mean.
    """
    seen, mlp = [], []
    hooks = [
        model.transformer.ln_f.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].reshape(-1, args[0].shape[-1]).float().cpu())
        )
    ]
    for block in model.transformer.h:
        hooks.append(
            block.mlp.register_forward_hook(
                lambda _m, _a, out: mlp.append(out.reshape(-1, out.shape[-1]).float().mean(0).cpu())
            )
        )
    for batch in ids.split(BATCH):
        model(batch.to(next(model.parameters()).device))
    for hook in hooks:
        hook.remove()
    x = torch.cat(seen)
    mean = x.mean(0)
    batches = len(ids.split(BATCH))
    mlp_mean = torch.stack(mlp).sum(0) / batches  # summed over layers, averaged over batches
    return mean, mlp_mean, float((x - mean).norm(dim=-1).mean())


@torch.no_grad()
def measure(model, ids: torch.Tensor) -> dict:
    """The loss and the mean entropy of the next-token distribution."""
    device = next(model.parameters()).device
    loss, entropy, count = 0.0, 0.0, 0
    for batch in ids.split(BATCH):
        batch = batch.to(device)
        logits = model(batch).logits[:, :-1].float()
        logp = logits.log_softmax(-1)
        target = batch[:, 1:]
        n = target.numel()
        loss += float(-logp.gather(-1, target.unsqueeze(-1)).sum())
        entropy += float(-(logp.exp() * logp).sum())
        count += n
    return {"loss": loss / count, "entropy": entropy / count}


def offsets(direction: str, means: dict, norm: float, tokens: int) -> torch.Tensor:
    """An offset of norm ``norm``: one vector, or one for each of ``tokens`` tokens."""
    width = means["mean"].shape[0]
    generator = torch.Generator().manual_seed(0)
    if direction in ("mean", "mlp"):
        unit = means[direction] / means[direction].norm()
    elif direction == "random":
        unit = torch.randn(width, generator=generator)
        unit = unit / unit.norm()
    else:
        unit = torch.randn(tokens, width, generator=generator)
        unit = unit / unit.norm(dim=-1, keepdim=True)
    return unit * norm


def run(model, name: str, ids: torch.Tensor, out: Path, sizes: bool = True) -> None:
    """Every kind, direction and size, on ``model``; append the rows to ``out``.

    Without ``sizes``, only the slopes.
    """
    model.eval()
    mean, mlp_mean, spread = residual_stats(model, ids)
    means = {"mean": mean, "mlp": mlp_mean}
    cosine = float(torch.nn.functional.cosine_similarity(mean, mlp_mean, dim=0))
    base = measure(model, ids)
    click.echo(f"{name}: loss {base['loss']:.4f}, entropy {base['entropy']:.3f}", err=True)
    click.echo(
        f"  mean of x: norm {float(mean.norm()):.1f}; MLP mean output: norm "
        f"{float(mlp_mean.norm()):.1f}, cosine {cosine:.3f}; spread {spread:.1f}",
        err=True,
    )
    for row in slopes(model, name, ids, means, spread):
        write_row(out, row)
    for row in route_slopes(model, name, ids, spread):
        write_row(out, row)
    if sizes:
        sweep(model, name, ids, out, means, spread, base)


def sweep(model, name: str, ids: torch.Tensor, out: Path, means: dict, spread: float, base: dict):
    """The loss and entropy at each size of offset, for every kind and direction."""
    device = next(model.parameters()).device
    norm = model.transformer.ln_f
    mean, mlp_mean = means["mean"], means["mlp"]
    cosine = float(torch.nn.functional.cosine_similarity(mean, mlp_mean, dim=0))
    tokens = BATCH * ids.shape[1]
    for kind in KINDS:
        for direction in DIRECTIONS:
            for size in SIZES:
                c = offsets(direction, means, size * spread, tokens).to(device) if size else None
                model.transformer.ln_f = wrapped = Offset(norm, kind, c)
                got = measure(model, ids)
                model.transformer.ln_f = norm
                got["shrink"] = (
                    sum(wrapped.shrinks) / len(wrapped.shrinks) if c is not None else 1.0
                )
                row = {
                    "part": "ln_offset",
                    "model": name,
                    "kind": kind,
                    "direction": direction,
                    "size": size,
                    "mean_norm": float(mean.norm()),
                    "mlp_norm": float(mlp_mean.norm()),
                    "mean_mlp_cosine": cosine,
                    "spread": spread,
                    **got,
                    "loss_change": got["loss"] - base["loss"],
                    "entropy_change": got["entropy"] - base["entropy"],
                }
                write_row(out, row)
            click.echo(f"  {kind} {direction} done", err=True)


def slopes(model, name: str, ids: torch.Tensor, means: dict, spread: float) -> list[dict]:
    """d(loss)/dε and d(entropy)/dε at ε = 0, for an offset ε × spread along each direction."""
    device = next(model.parameters()).device
    norm = model.transformer.ln_f
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    for direction in SLOPE_DIRECTIONS:
        unit = offsets(direction, means, spread, 0).to(device)
        for kind in KINDS:
            eps = torch.zeros((), device=device, requires_grad=True)
            d_loss = d_entropy = 0.0
            count = 0
            for batch in ids.split(BATCH):
                batch = batch.to(device)
                model.transformer.ln_f = Offset(norm, kind, unit * eps)  # a new graph each batch
                logp = model(batch).logits[:, :-1].float().log_softmax(-1)
                target = batch[:, 1:]
                loss = -logp.gather(-1, target.unsqueeze(-1)).sum()
                entropy = -(logp.exp() * logp).sum()
                g_loss = torch.autograd.grad(loss, eps, retain_graph=True)[0]
                g_entropy = torch.autograd.grad(entropy, eps)[0]
                d_loss, d_entropy = d_loss + float(g_loss), d_entropy + float(g_entropy)
                count += target.numel()
            model.transformer.ln_f = norm
            rows.append(
                {
                    "part": "ln_offset_slope",
                    "model": name,
                    "kind": kind,
                    "direction": direction,
                    "loss_slope": d_loss / count,
                    "entropy_slope": d_entropy / count,
                    "spread": spread,
                }
            )
        click.echo(f"  slopes along {direction} done", err=True)
    for p in model.parameters():
        p.requires_grad_(True)
    return rows


ROUTES = ("final", "blocks", "all")


@torch.no_grad()
def layer_means(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """Each MLP's mean output over the tokens of ``ids``."""
    sums, count = [0.0] * len(model.transformer.h), 0
    outs: list = []
    hooks = [
        block.mlp.register_forward_hook(lambda _m, _a, out, i=i: outs.append((i, out)))
        for i, block in enumerate(model.transformer.h)
    ]
    for batch in ids.split(BATCH):
        outs.clear()
        model(batch.to(next(model.parameters()).device))
        for i, out in outs:
            sums[i] = sums[i] + out.reshape(-1, out.shape[-1]).float().sum(0).cpu()
        count += batch.numel()
    for hook in hooks:
        hook.remove()
    return [total / count for total in sums]


def readers(model, layer: int, route: str) -> list[tuple[nn.Module, str]]:
    """The LayerNorms, as (parent, name), that read layer ``layer``'s output on ``route``.

    The blocks after ``layer`` read it through ``ln_1`` and ``ln_2``, and the head
    through ``ln_f``.
    """
    found = []
    if route in ("blocks", "all"):
        for block in model.transformer.h[layer + 1 :]:
            found += [(block, "ln_1"), (block, "ln_2")]
    if route in ("final", "all"):
        found.append((model.transformer, "ln_f"))
    return found


def _site_slope(model, ids: torch.Tensor, sites: list, kind: str, unit: torch.Tensor) -> float:
    """d(loss)/dε per token, with ``unit × ε`` added at each LayerNorm in ``sites``."""
    device = next(model.parameters()).device
    originals = [getattr(parent, attr) for parent, attr in sites]
    eps = torch.zeros((), device=device, requires_grad=True)
    d_loss, count = 0.0, 0
    for batch in ids.split(BATCH):
        batch = batch.to(device)
        for (parent, attr), norm in zip(sites, originals, strict=True):
            setattr(parent, attr, Offset(norm, kind, unit * eps))
        logp = model(batch).logits[:, :-1].float().log_softmax(-1)
        target = batch[:, 1:]
        loss = -logp.gather(-1, target.unsqueeze(-1)).sum()
        d_loss += float(torch.autograd.grad(loss, eps)[0])
        count += target.numel()
    for (parent, attr), norm in zip(sites, originals, strict=True):
        setattr(parent, attr, norm)
    return d_loss / count


def route_slopes(model, name: str, ids: torch.Tensor, spread: float) -> list[dict]:
    """d(loss)/dε for each layer's MLP mean output as an offset ε × spread, by route and kind.

    A route is the set of LayerNorms that see the offset. Every other reader sees the
    residual stream as it is. The kinds are those of :class:`Offset`.
    """
    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    for layer, mean in enumerate(layer_means(model, ids)):
        unit = (mean / mean.norm() * spread).to(device)
        for route in ROUTES:
            sites = readers(model, layer, route)
            if not sites:  # the last layer: no later block reads it
                continue
            for kind in KINDS:
                rows.append(
                    {
                        "part": "ln_offset_route",
                        "model": name,
                        "layer": layer,
                        "route": route,
                        "kind": kind,
                        "loss_slope": _site_slope(model, ids, sites, kind, unit),
                        "mean_norm": float(mean.norm()),
                        "spread": spread,
                    }
                )
        click.echo(f"  routes of L{layer} done", err=True)
    for p in model.parameters():
        p.requires_grad_(True)
    return rows


def test_windows(tokenizer, corpus: Path) -> torch.Tensor:
    """The paper's evaluation windows: :data:`WINDOWS` test windows of the corpus sample."""
    split = load_split(tokenizer, corpus, DEFAULT_SAMPLE_TOKENS, DEFAULT_SAMPLE_SEED)
    return split.test[:WINDOWS]


@click.group()
def cli():
    """Measure what a constant offset at the final LayerNorm costs."""


@cli.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="cpu", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def gpt2(corpus, device, out):
    """GPT-2 small."""
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(GPT2)
    model = GPT2LMHeadModel.from_pretrained(GPT2).to(device)
    run(model, "gpt2", test_windows(tokenizer, corpus), out)


@cli.command()
@click.option("--checkpoint", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--slopes-only", is_flag=True, help="Skip the sweep over offset sizes.")
@click.option("--condition", default="gelu", show_default=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="cpu", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def small(checkpoint, slopes_only, condition, corpus, device, out):
    """A small model saved by ``train_resting.py train --save-every``."""
    from train_resting import CONDITIONS, TOKENIZER, build_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    state = torch.load(checkpoint, map_location="cpu")
    vocab = state["transformer.wte.weight"].shape[0]  # training sets it from its tokens
    model = build_model(CONDITIONS[condition], vocab, 0)
    model.load_state_dict(state)
    run(
        model.to(device),
        f"small_{condition}_{checkpoint.stem}",
        test_windows(tokenizer, corpus),
        out,
        sizes=not slopes_only,
    )


@cli.command()
@click.option(
    "--checkpoint", "checkpoints", multiple=True, type=click.Path(path_type=Path, exists=True)
)
@click.option("--condition", default="gelu", show_default=True)
@click.option("--cache", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--out", type=click.Path(path_type=Path), required=True)
def frequency(checkpoints, condition, cache, corpus, out):
    """Which tokens the shift route favours, for each small model.

    The shift adds the MLPs' mean output to every token at ``ln_f``. That changes the
    logits by the unembedding of its normalised direction. Each row correlates that change
    with the log frequency of each token in the training text (``--cache``). It also
    correlates it with the log of the model's mean predicted probability.
    """
    from train_resting import CONDITIONS, TOKENIZER, build_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    ids = test_windows(tokenizer, corpus)
    counts = None
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu")
        model = build_model(CONDITIONS[condition], state["transformer.wte.weight"].shape[0], 0)
        model.load_state_dict(state)
        model.eval()
        if counts is None:
            counts = torch.bincount(torch.load(cache), minlength=model.lm_head.weight.shape[0])
        write_row(
            out,
            {
                "model": f"small_{condition}_{checkpoint.stem}",
                **shift_bias(model, ids, counts, tokenizer),
            },
        )


@torch.no_grad()
def shift_bias(model, ids: torch.Tensor, counts: torch.Tensor, tokenizer) -> dict:
    """The logit change of the shift along the MLPs' mean output, against token frequency."""
    _, mlp_mean, _ = residual_stats(model, ids)
    norm = model.transformer.ln_f
    seen, logits = [], []
    hook = norm.register_forward_pre_hook(
        lambda _m, a: seen.append(a[0].reshape(-1, a[0].shape[-1]))
    )
    for batch in ids.split(BATCH):
        logits.append(model(batch).logits[:, :-1].reshape(-1, counts.shape[0]))
    hook.remove()
    sigma = (torch.cat(seen).var(-1, unbiased=False) + norm.eps).sqrt().mean()
    unit = mlp_mean / mlp_mean.norm()
    bias = model.lm_head.weight @ (norm.weight * (unit - unit.mean()) / sigma)
    predicted = torch.cat(logits).softmax(-1).mean(0)
    corr = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1])  # noqa: E731
    return {
        "part": "ln_offset_frequency",
        "corr_log_frequency": corr(bias, (counts.float() + 1).log()),
        "corr_log_predicted": corr(bias, predicted.log()),
        "favoured": [tokenizer.decode([i]) for i in bias.topk(12).indices.tolist()],
        "disfavoured": [tokenizer.decode([i]) for i in (-bias).topk(12).indices.tolist()],
    }


if __name__ == "__main__":
    cli()
