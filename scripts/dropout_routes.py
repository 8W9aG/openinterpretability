"""Split dropout's part in the pull on the MLPs' mean output into its routes (#3192).

The loss pulls each MLP's mean output toward 0 (papers/why-neurons-rest/README.md, *The
pull on the mean output*). Dropout is one route. It drops each element of the MLP's output
on each token, so a mean output ``c`` that every token shares becomes per-token noise:
``c * (m / (1 - p) - 1)``, with ``m`` a mask that keeps each element with probability
``1 - p``. The noise grows with ``c``, so if it costs loss, gradient descent shrinks ``c``.

The noise reaches the loss through the LayerNorms that read the residual stream after the
MLP. This script delivers it to one set of readers at a time:

- *blocks*: only the LayerNorms of the later blocks.
- *final*: only the final LayerNorm.
- *all*: the residual stream itself, so every later reader sees it, as in training.

For each MLP and route, it scales the noise by ``alpha`` and takes the slope of the
expected loss at ``alpha = 1``, by autograd, over ``--draws`` masks. A positive slope is a
pull. The model is otherwise in eval mode, so no other dropout runs. ``extra`` is the loss
with the noise at ``alpha = 1`` minus the loss without it. The noise has mean 0, so its
cost is of second order in its size. So the *all* slope is not the sum of the other two:
it also has a cross term, where the noise reaches the final LayerNorm both directly and
through the later blocks.

    uv run python scripts/dropout_routes.py --condition relu \\
        --checkpoint ckpt/relu_seed0_step2000.pt --out routes.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).parent))

from default_on_poles import write_row  # noqa: E402
from ln_offset import BATCH, test_windows  # noqa: E402

ROUTES = ("blocks", "final", "all")


@torch.no_grad()
def mean_outputs(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """Each MLP's mean output over the windows' tokens, the first tokens left out.

    Args:
        model: The small GPT-2, in eval mode.
        ids: The windows, ``[windows, tokens]``.

    Returns:
        One ``[d_model]`` vector per block.
    """
    device = next(model.parameters()).device
    sums = [torch.zeros(model.config.n_embd) for _ in model.transformer.h]
    count = 0
    hooks = []
    for i, block in enumerate(model.transformer.h):

        def add(_m, _a, out, i=i):
            sums[i] += out[:, 1:].float().sum((0, 1)).cpu()

        hooks.append(block.mlp.register_forward_hook(add))
    for batch in ids.split(BATCH):
        model(batch.to(device))
        count += batch.shape[0] * (batch.shape[1] - 1)
    for hook in hooks:
        hook.remove()
    return [s / count for s in sums]


def readers(model, layer: int, route: str) -> list:
    """The modules whose input gets the noise of MLP ``layer`` on ``route``.

    Args:
        model: The small GPT-2.
        layer: The block whose MLP's output is dropped.
        route: One of :data:`ROUTES`.

    Returns:
        The modules. For *all*, it is the MLP itself, whose output the noise joins.

    Raises:
        ValueError: An unknown route.
    """
    later = model.transformer.h[layer + 1 :]
    if route == "blocks":
        return [m for block in later for m in (block.ln_1, block.ln_2)]
    if route == "final":
        return [model.transformer.ln_f]
    if route == "all":
        return [model.transformer.h[layer].mlp]
    raise ValueError(f"unknown route {route}")


def noise(c: torch.Tensor, shape: tuple, p: float, generator: torch.Generator) -> torch.Tensor:
    """Dropout's noise on a constant ``c``: ``c * (m / (1 - p) - 1)``, on each token.

    Args:
        c: The constant, ``[d_model]``.
        shape: ``(windows, tokens)``.
        p: The dropout probability.
        generator: The mask's random numbers.

    Returns:
        ``[windows, tokens, d_model]``, with mean 0 over masks.

    Examples:
        >>> n = noise(torch.ones(2), (1, 3), 0.0, torch.Generator().manual_seed(0))
        >>> bool((n == 0).all())
        True
    """
    keep = (torch.rand(*shape, c.shape[0], generator=generator) >= p).float()
    return c * (keep / (1 - p) - 1)


def route_slope(model, ids, layer: int, route: str, c: torch.Tensor, p: float, draws: int,
                seed: int = 0) -> dict:  # fmt: skip
    """The slope and the extra loss of one MLP's dropout noise on one route.

    Args:
        model: The small GPT-2, in eval mode.
        ids: The windows.
        layer: The block whose MLP's output is dropped.
        route: One of :data:`ROUTES`.
        c: That MLP's mean output.
        p: The dropout probability.
        draws: The masks for each batch.
        seed: The masks' seed. Each route uses the same masks.

    Returns:
        ``slope``, the mean of ``dL/dalpha`` at ``alpha = 1``, and ``extra``, the mean loss
        with the noise minus the loss without it.
    """
    targets = readers(model, layer, route)
    if not targets:  # the last block's noise has no later block to reach
        return {"slope": 0.0, "extra": 0.0}
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    alpha = torch.ones((), device=device, requires_grad=True)
    current: dict[str, torch.Tensor] = {}
    output = route == "all"

    def add(_m, args, out=None):
        if output:
            return out + alpha * current["noise"]
        return (args[0] + alpha * current["noise"], *args[1:])

    hooks = [m.register_forward_hook(add) if output else m.register_forward_pre_hook(add)
             for m in targets]  # fmt: skip
    slopes, extras = [], []
    try:
        for batch in ids.split(BATCH):
            batch = batch.to(device)
            with torch.no_grad():
                current["noise"] = torch.zeros(1, device=device)
                clean = model(batch, labels=batch).loss.item()
            for _ in range(draws):
                current["noise"] = noise(c, tuple(batch.shape), p, generator).to(device)
                loss = model(batch, labels=batch).loss
                (grad,) = torch.autograd.grad(loss, alpha)
                slopes.append(grad.item())
                extras.append(loss.item() - clean)
    finally:
        for hook in hooks:
            hook.remove()
    return {"slope": sum(slopes) / len(slopes), "extra": sum(extras) / len(extras)}


@click.command()
@click.option(
    "--checkpoint", "checkpoints", multiple=True, type=click.Path(path_type=Path, exists=True)
)
@click.option("--condition", default="gelu", show_default=True)
@click.option("--draws", default=8, show_default=True, help="Masks for each batch.")
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--device", default="cpu", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(checkpoints, condition, draws, corpus, device, out):
    """For each checkpoint, write one row per MLP and route.

    Args:
        checkpoints: Models saved by ``train_resting.py train --save-every``.
        condition: Their training condition, which gives the architecture and ``p``.
        draws: The masks for each batch.
        corpus: The corpus whose sample gives the test windows.
        device: The device of the model.
        out: The JSON Lines file to append to.
    """
    from train_resting import CONDITIONS, TOKENIZER, build_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    ids = test_windows(tokenizer, corpus)
    p = CONDITIONS[condition].dropout
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu")
        model = build_model(CONDITIONS[condition], state["transformer.wte.weight"].shape[0], 0)
        model.load_state_dict(state)
        model.to(device).eval()
        means = mean_outputs(model, ids)
        for layer, c in enumerate(means):
            for route in ROUTES:
                row = route_slope(model, ids, layer, route, c, p, draws)
                write_row(out, {"part": "dropout_routes", "model": checkpoint.stem,
                                "condition": condition, "layer": layer, "route": route,
                                "p": p, "mean_output_norm": c.norm().item(), **row})  # fmt: skip
                click.echo(f"{checkpoint.stem} L{layer} {route}: {row}", err=True)


if __name__ == "__main__":
    main()
