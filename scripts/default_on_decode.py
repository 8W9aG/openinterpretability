"""What a ridge readout gets from one side of a layer's activations.

The numbers behind papers/default-on-poles/README.md. For each layer of GPT-2 small, the
readout fits on the fit windows and scores on the test windows. It reads the negative
side, the positive side, or both. It predicts the current token, the next token, and the
other side's write to the residual stream. The null reads the negative side of random
features of the same MLP input. Each feature has the median and SD of a real neuron's
pre-activation. So a score the null also reaches needs no learned neuron.

    uv run python scripts/default_on_decode.py --out papers/default-on-poles/results.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import (
    GPT2,
    Split,
    load_split,
    read_mlp_inputs,
    read_pre_activations,
    write_row,
)

from openinterp.cli import _load, resolve_device
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

DECODE_CLASSES = 200
RIDGE_PENALTIES = (1e-3, 1e-2, 1e-1, 1.0)


def ridge(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor) -> torch.Tensor:
    """Return predictions for ``x_test``, with the penalty that does best on held-out fit rows.

    Features are standardised by the fit rows. The penalty is relative to the mean diagonal.
    """
    mean, sd = x.mean(0), x.std(0).clamp(min=1e-6)
    x, x_test = (x - mean) / sd, (x_test - mean) / sd
    y_mean = y.mean(0)
    cut = len(x) * 4 // 5
    scores = {
        p: (x[cut:] @ _solve(x[:cut], y[:cut] - y_mean, p)).sub(y[cut:] - y_mean).square().sum()
        for p in RIDGE_PENALTIES
    }
    return x_test @ _solve(x, y - y_mean, min(scores, key=scores.get)) + y_mean


def _solve(x: torch.Tensor, y: torch.Tensor, penalty: float) -> torch.Tensor:
    """Return ridge coefficients ``[features, targets]``, in double precision."""
    x, y = x.double(), y.double()
    gram = x.T @ x
    gram.diagonal().add_(penalty * gram.diagonal().mean())
    return torch.linalg.solve(gram, x.T @ y).float()


def explained(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return the share of ``target``'s variance over tokens that ``prediction`` explains."""
    residual = (target - prediction).square().sum()
    return 1 - (residual / (target - target.mean(0)).square().sum()).item()


def class_accuracy(x, labels, x_test, labels_test) -> dict:
    """Return a one-vs-rest ridge's accuracy over the fit rows' commonest labels, and chance."""
    classes = labels.bincount().topk(DECODE_CLASSES).indices
    lookup = torch.full((int(labels.max().item()) + 1,), -1)
    lookup[classes] = torch.arange(DECODE_CLASSES)
    y = torch.zeros(len(labels), DECODE_CLASSES)
    keep = lookup[labels] >= 0
    y[keep, lookup[labels[keep]]] = 1
    scores = ridge(x, y, x_test)
    in_range = labels_test <= lookup.shape[0] - 1
    test_labels = torch.where(in_range, labels_test, torch.zeros_like(labels_test))
    test_class = torch.where(in_range, lookup[test_labels], torch.full_like(labels_test, -1))
    seen = test_class >= 0
    right = scores[seen].argmax(1) == test_class[seen]
    return {
        "accuracy": right.float().mean().item(),
        "chance": test_class[seen].bincount().max().item() / int(seen.sum()),
        "tokens": int(seen.sum()),
    }


def random_negative_side(h: torch.Tensor, x: torch.Tensor, h_test, x_test, seed: int):
    """Return the null's negative side on fit and test rows, from random input directions.

    Each random feature takes the median and SD of one real neuron, on the fit rows.
    """
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(x.shape[1], h.shape[1], generator=generator)
    raw, raw_test = x @ directions, x_test @ directions
    median, sd = raw.median(0).values, raw.std(0)
    scale = h.std(0) / sd
    target = h.median(0).values
    gelu = torch.nn.functional.gelu
    return (
        gelu((raw - median) * scale + target).clamp(max=0),
        gelu((raw_test - median) * scale + target).clamp(max=0),
    )


def decode_row(runner, split: Split, layer: int) -> dict:
    """Return how much one side's activations say about the other side, and about tokens."""
    gelu = torch.nn.functional.gelu
    h = read_pre_activations(runner, split.fit, [layer])[layer]
    h_test = read_pre_activations(runner, split.test, [layer])[layer]
    x = read_mlp_inputs(runner, split.fit, layer).float()
    x_test = read_mlp_inputs(runner, split.test, layer).float()
    fit, test = gelu(h), gelu(h_test)
    weights = runner.down_proj(layer).weight.detach().float().cpu()  # Conv1D: [width, d_model]
    sides = {
        "negative": (fit.clamp(max=0), test.clamp(max=0)),
        "positive": (fit.clamp(min=0), test.clamp(min=0)),
        "both": (fit, test),
        "random_negative": random_negative_side(h, x, h_test, x_test, layer),
    }
    writes = {name: (a @ weights, b @ weights) for name, (a, b) in sides.items()}
    row = {"part": "decode", "layer": layer, **_write_overlap(sides, writes)}
    for name, (a, b) in sides.items():
        row[name] = {
            "current_token": class_accuracy(
                a, split.fit[:, 1:].reshape(-1), b, split.test[:, 1:].reshape(-1)
            ),
            "next_token": class_accuracy(*_next_rows(a, split.fit), *_next_rows(b, split.test)),
        }
    return row


def _next_rows(acts: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return activations and the token after each, without each window's last position."""
    seq = ids.shape[1] - 1
    keep = (torch.arange(len(acts)) % seq) != seq - 1
    return acts[keep], ids[:, 2:].reshape(-1)


def _write_overlap(sides: dict, writes: dict) -> dict:
    """Return how far one side's write lines up with the other's, and predicts it."""
    neg = writes["negative"][1] - writes["negative"][1].mean(0)
    pos = writes["positive"][1] - writes["positive"][1].mean(0)
    cosine = torch.nn.functional.cosine_similarity(neg, pos, dim=1)
    null = ridge(sides["random_negative"][0], writes["positive"][0], sides["random_negative"][1])
    return {
        "positive_write_from_random_negative": explained(null, writes["positive"][1]),
        "write_cosine_mean": cosine.mean().item(),
        "write_cosine_abs_mean": cosine.abs().mean().item(),
        "positive_write_from_negative": explained(
            ridge(sides["negative"][0], writes["positive"][0], sides["negative"][1]),
            writes["positive"][1],
        ),
        "negative_write_from_positive": explained(
            ridge(sides["positive"][0], writes["negative"][0], sides["positive"][1]),
            writes["negative"][1],
        ),
    }


@click.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--max-tokens", type=int, default=DEFAULT_SAMPLE_TOKENS, show_default=True)
@click.option("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(corpus, max_tokens, sample_seed, device, out):
    """Append each layer's decode row to ``--out``."""
    runner = _load(GPT2, resolve_device(device), torch.float32, None, None)
    split = load_split(runner.tokenizer, corpus, max_tokens, sample_seed)
    for layer in range(runner.n_layers):
        row = decode_row(runner, split, layer)
        write_row(out, {"model": GPT2, "sample": split.sample, **row})
        click.echo(f"layer {layer} done", err=True)


if __name__ == "__main__":
    main()
