"""Dormant poles and effective rank through Pythia-70m's training (#5134).

Dohare et al. (2024) find that loss of plasticity comes with more dormant units and a lower
effective rank. This script measures both in Pythia-70m at the 20 checkpoints of
``papers/neuron-development/``, on the same corpus windows. At each step and layer:

* ``dormant``: the share of poles (two for each neuron) active on under 1% of tokens. A
  positive pole is active when ``a > 0`` and a negative pole when ``a < 0``. GELU keeps
  the sign of its input, so this is the rule of ``harness/dead.py``.
* ``never``: the share of poles active on no token, by the same rule.
* ``dormant_line``: the dormant share by the review's line. A pole is active at :data:`LINE`
  times its own maximum at this step. No pole can be never-firing by this rule.
* ``erank`` and ``erank_all``: the effective rank (Roy & Vetterli, 2007) of the token ×
  neuron activation matrix. The first uses a seeded sample of :data:`SAMPLE` tokens, and
  the second all tokens.

Nothing is written in ``wiki/``::

    uv run python scripts/dormant_rank.py --out papers/neuron-development/results.dormant.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_development import MODEL, STEPS  # noqa: E402  (a sibling script)

LINE = 0.25  # the review's activation threshold, a share of the pole's maximum
DORMANT = 0.01  # Dohare et al.'s dormant unit: active on under 1% of inputs
SAMPLE = 2000  # tokens in the effective rank's sample, the size Dohare et al. use
SEED = 0


def active_shares(acts: torch.Tensor) -> torch.Tensor:
    """The share of tokens on which each pole is active, by the sign rule.

    Args:
        acts: The activations, tokens × neurons.

    Returns:
        The shares of the positive poles, then those of the negative poles.

    Examples:
        >>> active_shares(torch.tensor([[1.0, -1.0], [2.0, 0.0]])).tolist()
        [1.0, 0.0, 0.0, 0.5]
    """
    return torch.cat([(acts > 0).double().mean(0), (acts < 0).double().mean(0)])


def line_shares(acts: torch.Tensor) -> torch.Tensor:
    """The share of tokens on which each pole reaches :data:`LINE` times its own maximum.

    A pole that is never on (a maximum of 0 or less) has a share of 0.

    Args:
        acts: The activations, tokens × neurons.

    Returns:
        The shares of the positive poles, then those of the negative poles.

    Examples:
        >>> line_shares(torch.tensor([[1.0, -1.0], [0.1, 1.0], [0.3, 1.0]])).tolist()
        [0.6666666666666666, 0.6666666666666666, 0.0, 0.3333333333333333]
    """
    shares = []
    for strength in (acts, -acts):
        peak = strength.max(0).values
        on = (strength >= LINE * peak) & (peak > 0)
        shares.append(on.double().mean(0))
    return torch.cat(shares)


def effective_rank(matrix: torch.Tensor) -> float:
    """The effective rank of ``matrix``: ``exp(−Σ p log p)``, with ``p = σ / Σσ``.

    Args:
        matrix: Any 2-D matrix. It is not centred.

    Returns:
        A value from 1 to the matrix's rank. A zero matrix gives 0.

    Examples:
        >>> round(effective_rank(torch.eye(4)), 6)
        4.0
        >>> round(effective_rank(torch.ones(3, 5)), 6)
        1.0
    """
    sigma = torch.linalg.svdvals(matrix.double())
    if float(sigma.sum()) == 0:
        return 0.0
    p = sigma / sigma.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def sample_rows(n: int, k: int = SAMPLE, seed: int = SEED) -> torch.Tensor:
    """``k`` row indices of ``n``, drawn without replacement with ``seed``, or all ``n``.

    Args:
        n: The number of rows.
        k: How many to draw.
        seed: The draw's seed.

    Returns:
        The indices, sorted.

    Examples:
        >>> sample_rows(3, 5).tolist()
        [0, 1, 2]
        >>> torch.equal(sample_rows(100, 10), sample_rows(100, 10))
        True
    """
    if n <= k:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:k].sort().values


def layer_row(step: int, layer: int, acts: torch.Tensor) -> dict:
    """One layer's measures at one step.

    Args:
        step: The training step.
        layer: The layer.
        acts: The layer's activations, tokens × neurons.

    Returns:
        The row.
    """
    shares = active_shares(acts)
    return {
        "table": "layer", "step": step, "layer": layer, "tokens": acts.shape[0],
        "poles": shares.numel(),
        "dormant": round(float((shares < DORMANT).double().mean()), 5),
        "never": round(float((shares == 0).double().mean()), 5),
        "dormant_line": round(float((line_shares(acts) < DORMANT).double().mean()), 5),
        "erank": round(effective_rank(acts[sample_rows(acts.shape[0])]), 3),
        "erank_all": round(effective_rank(acts), 3),
    }  # fmt: skip


def step_activations(runner, texts: list[str]) -> dict[int, torch.Tensor]:
    """Every layer's MLP activations on ``texts``, one window at a time, tokens × neurons.

    Args:
        runner: The model.
        texts: The windows.

    Returns:
        Each layer's activations, the windows' tokens stacked.
    """
    layers = list(range(runner.n_layers))
    parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for text in texts:
        for layer, act in runner.layer_activations([text], layers).items():
            parts[layer].append(act.reshape(-1, act.shape[-1]).float().cpu())
    return {layer: torch.cat(chunks) for layer, chunks in parts.items()}


@click.command()
@click.option(
    "--corpus", type=click.Path(dir_okay=False, path_type=Path), default=Path("corpus.jsonl")
)
@click.option("--windows", default=300, show_default=True, help="Fixed corpus windows.")
@click.option("--window-tokens", default=64, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--step", "steps", multiple=True, type=int, help="Default: each of STEPS.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(corpus, windows, window_tokens, device, steps, out):  # pragma: no cover - loads models
    """Measure the dormant shares and effective ranks of Pythia-70m at each step in STEPS.

    Args:
        corpus: The JSONL corpus whose last 200 documents give the windows.
        windows: How many windows to keep.
        window_tokens: The tokens in each window.
        device: Where the model runs.
        steps: The steps to measure. Default: each of STEPS.
        out: The JSONL file for the rows.
    """
    from openinterpretability.cli import _load_runner
    from openinterpretability.harness.explanation_scores import cut_windows

    steps = tuple(sorted(steps)) or STEPS
    docs = [json.loads(line)["text"] for line in corpus.read_text().splitlines()[-200:]]
    final = _load_runner(MODEL, "cpu", "float32", f"step{steps[-1]}")
    texts = cut_windows(final, docs, window_tokens)[:windows]
    del final
    rows = []
    for step in steps:
        runner = _load_runner(MODEL, device, "float32", f"step{step}")
        for layer, acts in step_activations(runner, texts).items():
            rows.append(layer_row(step, layer, acts))
            click.echo(json.dumps(rows[-1]), err=True)
        del runner
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
