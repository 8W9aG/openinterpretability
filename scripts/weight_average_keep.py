"""Test VERIFIED poles along the line from a base model to its instruct model (#5404).

Δ is the instruct model's weight minus the base's, for each weight. Each condition adds a
change to the base model's weights in place, and copies the weights back afterwards:

* ``interpolate``: α·Δ, for each α in :data:`ALPHAS`. α = 0.5 is the weight average of the
  two models, and α = 1 is the instruct model. α < 0 negates the tuning (task arithmetic),
  and α > 1 goes past it.
* ``random``: Gaussian noise on each weight, with the Frobenius norm of its own |α|·Δ, for
  each α and seed. It is the random change of the same size at that step.

The keep test is ``scripts/random_delta_keep.py``'s: each VERIFIED pole's probes, both gates,
on the base model and on the changed model. Each condition also gets the mean token KL from
the base model's next-token distribution, on :data:`KL_TOKENS` corpus tokens.
``named_parameters`` lists a tied embedding once, so Δ adds it once.

    uv run python scripts/weight_average_keep.py --base Qwen/Qwen2.5-1.5B --device cuda \\
        --out papers/weight-average-keep/results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_delta_keep import (  # noqa: E402
    base_columns,
    base_rows,
    cached_readings,
    changed,
    deltas,
    first_position_activations,
    probe_activations,
    run_condition,
)

from openinterpretability.interference.floors import _noise_like  # noqa: E402
from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    sample_windows,
    verified_records,
)

ALPHAS = (-0.5, 0.25, 0.5, 0.75, 1.0, 1.5)
SEEDS = (0, 1)  # the seeds of the random change at each α
KL_TOKENS = 2048  # the corpus tokens of the token KL: the first batch of the sample


def scaled(delta: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    """α·Δ for each weight.

    Args:
        delta: Δ for each weight.
        alpha: The step along the line from the base model to the instruct model.

    Examples:
        >>> scaled({"w": torch.tensor([2.0, -4.0])}, -0.5)["w"].tolist()
        [-1.0, 2.0]
    """
    return {name: alpha * tensor for name, tensor in delta.items()}


def random_like(delta: dict[str, torch.Tensor], alpha: float, seed: int) -> dict:
    """Gaussian noise for each weight, with the Frobenius norm of its |α|·Δ.

    Args:
        delta: Δ for each weight.
        alpha: The step whose size the noise matches.
        seed: The seed of the noise.

    Examples:
        >>> noise = random_like({"w": torch.tensor([3.0, 4.0])}, -2.0, 0)["w"]
        >>> round(float(noise.norm()), 4)
        10.0
    """
    generator = torch.Generator().manual_seed(seed)
    return {name: _noise_like(abs(alpha) * delta[name], generator) for name in sorted(delta)}


def log_probs(model, batch: torch.Tensor) -> torch.Tensor:
    """The next-token log-probabilities of each row of ``batch``, as float32 on the CPU.

    Args:
        model: A causal language model.
        batch: Token ids, ``[rows, tokens]``.
    """
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for row in batch:  # one row at a time: a row's logits fill a few hundred MB
            logits = model(row[None].to(device)).logits[0].float()
            rows.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.stack(rows)


def token_kl(base: torch.Tensor, changed: torch.Tensor) -> float:
    """The mean KL(base ‖ changed) over the tokens, in nats.

    Args:
        base: The base model's log-probabilities, ``[..., vocabulary]``.
        changed: The changed model's, in the same shape.

    Examples:
        >>> same = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1)
        >>> token_kl(same, same)
        0.0
    """
    kl = (base.exp() * (base - changed)).sum(-1)
    return round(float(kl.mean()), 6)


@click.command()
@click.option("--base", "base_name", required=True, help="The base model, e.g. Qwen/Qwen2.5-1.5B.")
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
def main(base_name: str, wiki_dir: Path, out: Path, device: str) -> None:  # pragma: no cover
    """Test each VERIFIED pole at each α and under the random change of each α's size.

    Args:
        base_name: The base model. Its instruct model is ``<base_name>-Instruct``.
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each condition.
        device: The device of the base model. The instruct model loads on the CPU.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, base_name)
    runner = _load_runner(base_name, device, "float32", None)
    other = _load_runner(f"{base_name}-Instruct", "cpu", "float32", None)
    delta = deltas(runner.model, other.model)
    del other
    batches = sample_windows(runner.tokenizer)
    base = base_columns(runner, batches, records)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {
        r.submission.identifier.slug: cached_readings(runner, r, cache, firsts) for r in records
    }
    kl_batch = batches[0][: KL_TOKENS // batches[0].shape[-1]]  # whole windows of the sample
    base_logp = log_probs(runner.model, kl_batch)
    lines = base_rows(records, on_base, base_name)
    plan = [("interpolate", a, None) for a in ALPHAS]
    plan += [("random", a, s) for a in ALPHAS for s in SEEDS]
    for condition, alpha, seed in plan:
        change = scaled(delta, alpha) if seed is None else random_like(delta, alpha, seed)
        fields = {"model": base_name, "condition": condition, "alpha": alpha, "seed": seed,
                  "sample_tokens": SAMPLE_TOKENS}  # fmt: skip
        rows = run_condition(runner, batches, records, base, on_base, change, fields)
        with changed(runner.model, change):
            rows[0]["token_kl"] = token_kl(base_logp, log_probs(runner.model, kl_batch))
        lines += rows
        click.echo(json.dumps({k: v for k, v in rows[0].items() if k != "flipped"}), err=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
