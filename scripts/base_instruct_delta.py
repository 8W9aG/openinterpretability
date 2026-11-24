"""Whether the instruct weight change at a pole predicts that its explanation keeps (#5133).

For each Qwen2.5 base and instruct pair of `docs/analyses/base-instruct.md`, Δ is
instruct − base for each MLP matrix. It reads the weights only and trains nothing. It
writes four kinds of row:

* ``layer``: Δ's rank for 90% of its squared singular values, per matrix (Biderman et
  al. 2024, §4.6).
* ``pole``: one row for each pole of the keep test. It gives Δ's norm in the pole's
  ``gate_proj`` and ``up_proj`` rows and ``down_proj`` column, over the base weight's norm
  there. It also gives the share of those weights in Δ's top-r singular directions, with
  r the 90% rank.
* ``test``: each measure as an AUROC for predicting a failure. The null shuffles the keep
  labels within each layer, so depth alone can't win.
* ``rotation``: the share of Δ's energy on the top 5% of neurons, before and after a
  Haar-random rotation of the neuron axis. A spread update would give the rotated share.

    uv run python scripts/base_instruct_delta.py --out docs/analyses/base-instruct-delta.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

MATRICES = ("gate_proj", "up_proj", "down_proj")
PAIRS = {"Qwen/Qwen2.5-0.5B": "docs/analyses/base-instruct-0.5b.jsonl",
         "Qwen/Qwen2.5-1.5B": "docs/analyses/base-instruct.jsonl"}  # fmt: skip
SHUFFLES = 10_000
TOP = 0.05  # the share of neurons whose energy the rotation test reads
SEED = 0


def rank_for(singular: torch.Tensor, share: float = 0.9) -> int:
    """The fewest singular values whose squares hold ``share`` of the total.

    Args:
        singular: Singular values, largest first.
        share: The share of the squared sum to reach.

    Returns:
        The rank.

    Examples:
        >>> rank_for(torch.tensor([3.0, 1.0, 1.0])), rank_for(torch.tensor([1.0, 1.0]))
        (2, 2)
    """
    energy = singular.double() ** 2
    cumulative = energy.cumsum(0) / energy.sum()
    return int((cumulative < share - 1e-12).sum()) + 1


def in_span(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """The share of each row of ``vectors`` that lies in the span of ``basis``'s columns.

    Args:
        vectors: ``[n, d]``.
        basis: ``[d, r]``, orthonormal columns.

    Returns:
        ``[n]``, each from 0 to 1.

    Examples:
        >>> in_span(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), torch.tensor([[1.0], [0.0]])).tolist()
        [1.0, 0.5]
    """
    total = (vectors.double() ** 2).sum(1).clamp_min(1e-30)
    return ((vectors.double() @ basis.double()) ** 2).sum(1) / total


def ranks(scores: list[float]) -> torch.Tensor:
    """Each score's rank from 1, with tied scores sharing their mean rank.

    Args:
        scores: The scores.

    Returns:
        The ranks, as floats.

    Examples:
        >>> ranks([0.5, 0.1, 0.5]).tolist()
        [2.5, 1.0, 2.5]
    """
    values = torch.tensor(scores, dtype=torch.float64)
    order = values.argsort()
    out = torch.empty_like(values)
    out[order] = torch.arange(1, len(values) + 1, dtype=torch.float64)
    for value in values.unique():  # ties share their mean rank
        tied = values == value
        out[tied] = out[tied].mean()
    return out


def auroc(scores: list[float], fails: list[bool]) -> float:
    """The chance that a failing pole scores above a keeping one, with ties counting half.

    Args:
        scores: One per pole.
        fails: Whether each pole fails.

    Returns:
        From 0 to 1; 0.5 is chance.

    Examples:
        >>> auroc([0.1, 0.9, 0.5], [False, True, False]), auroc([1, 1], [True, False])
        (1.0, 0.5)
    """
    return _auroc_from_ranks(ranks(scores), torch.tensor(fails))


def _auroc_from_ranks(rank: torch.Tensor, fails: torch.Tensor) -> float:
    """The Mann–Whitney AUROC of the failing poles, from all poles' ranks.

    Args:
        rank: Every pole's rank, from :func:`ranks`.
        fails: Whether each pole fails.

    Returns:
        From 0 to 1.
    """
    bad = int(fails.sum())
    good = len(fails) - bad
    return float((rank[fails].sum() - bad * (bad + 1) / 2) / (bad * good))


def shuffled_p(scores, fails, layers, shuffles: int = SHUFFLES, seed: int = SEED) -> float:
    """The share of within-layer shuffles of the labels whose AUROC is at least the real one.

    Args:
        scores: One per pole.
        fails: Whether each pole fails.
        layers: Each pole's layer, so a shuffle keeps each layer's count of failures.
        shuffles: How many shuffles.
        seed: The shuffles' seed.

    Returns:
        The one-sided p value, with the real labels counted as one shuffle.

    Examples:
        >>> shuffled_p([0.1, 0.2, 0.9, 0.8], [False, False, True, True], [0, 0, 0, 0], 99)
        0.2
    """
    rank, labels = ranks(scores), torch.tensor(fails)
    real = _auroc_from_ranks(rank, labels)
    generator = torch.Generator().manual_seed(seed)
    groups = [torch.tensor([i for i, x in enumerate(layers) if x == layer])
              for layer in sorted(set(layers))]  # fmt: skip
    beyond = 0
    for _ in range(shuffles):
        shuffled = labels.clone()
        for idx in groups:
            shuffled[idx] = labels[idx[torch.randperm(len(idx), generator=generator)]]
        beyond += _auroc_from_ranks(rank, shuffled) >= real - 1e-12
    return round((beyond + 1) / (shuffles + 1), 4)


def top_share(energy: torch.Tensor, top: float = TOP) -> float:
    """The share of the total that the largest ``top`` of the entries hold.

    Args:
        energy: One non-negative value per neuron.
        top: The share of neurons to keep.

    Returns:
        From ``top`` (spread evenly) to 1 (all on the top ones).

    Examples:
        >>> top_share(torch.tensor([9.0, 1.0, 0.0, 0.0]), 0.25)
        0.9
    """
    k = max(1, round(top * len(energy)))
    return round(float(energy.double().topk(k).values.sum() / energy.double().sum()), 6)


def haar(n: int, seed: int) -> torch.Tensor:
    """A Haar-random ``n × n`` rotation, from the QR of a Gaussian matrix.

    Args:
        n: The size.
        seed: The seed.

    Returns:
        An orthogonal matrix.

    Examples:
        >>> q = haar(4, 0)
        >>> torch.allclose(q @ q.T, torch.eye(4, dtype=q.dtype), atol=1e-10)
        True
    """
    gaussian = torch.randn(n, n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r))  # the sign fix makes the draw uniform


def mlp_weights(model, layer: int) -> dict[str, torch.Tensor]:
    """One Qwen2 layer's MLP matrices, in float64 on the CPU.

    Args:
        model: The causal LM.
        layer: The layer.

    Returns:
        ``{name: weight}``, ``gate_proj`` and ``up_proj`` as ``[width, d]``, ``down_proj`` as
        ``[d, width]``.
    """
    mlp = model.model.layers[layer].mlp
    return {m: getattr(mlp, m).weight.detach().double().cpu() for m in MATRICES}


def neuron_measures(base: dict, delta: dict) -> tuple[dict, dict[str, torch.Tensor]]:
    """Each matrix's 90% rank, and each neuron's measures of Δ.

    A neuron's weights are its ``gate_proj`` and ``up_proj`` rows and its ``down_proj``
    column. Each relative norm is Δ's norm there over the base weight's norm there.

    Args:
        base: :func:`mlp_weights` of the base model.
        delta: The same matrices of instruct − base.

    Returns:
        ``{matrix: rank}``, and ``{measure: [width]}``.
    """
    rows = {"gate_proj": delta["gate_proj"], "up_proj": delta["up_proj"],
            "down_proj": delta["down_proj"].T}  # fmt: skip
    weights = {"gate_proj": base["gate_proj"], "up_proj": base["up_proj"],
               "down_proj": base["down_proj"].T}  # fmt: skip
    rank, measures, shares = {}, {}, []
    for m in MATRICES:
        u, s, vh = torch.linalg.svd(delta[m], full_matrices=False)
        rank[m] = rank_for(s)
        basis = vh[: rank[m]].T if m != "down_proj" else u[:, : rank[m]]
        shares.append(in_span(weights[m], basis))  # the base weights in Δ's top-r directions
        measures[f"relative_{m}"] = rows[m].norm(dim=1) / weights[m].norm(dim=1).clamp_min(1e-30)
    every_delta = sum((rows[m] ** 2).sum(1) for m in MATRICES).sqrt()
    every_weight = sum((weights[m] ** 2).sum(1) for m in MATRICES).sqrt()
    measures["relative_all"] = every_delta / every_weight.clamp_min(1e-30)
    measures["top_r_share"] = torch.stack(shares).mean(0)
    return rank, measures


def neuron_energy(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Each neuron's squared norm over its gate row, up row and down column.

    Args:
        gate: ``[width, d]``.
        up: ``[width, d]``.
        down: ``[d, width]``.

    Returns:
        ``[width]``.

    Examples:
        >>> neuron_energy(torch.ones(2, 1), torch.zeros(2, 1), torch.ones(1, 2)).tolist()
        [2.0, 2.0]
    """
    return (gate**2).sum(1) + (up**2).sum(1) + (down**2).sum(0)


def rotation_row(delta: dict, layer: int) -> dict:
    """Δ's energy on its top neurons, against the same after a Haar rotation of the neuron axis.

    Args:
        delta: The layer's Δ matrices.
        layer: The layer, which seeds the rotation.

    Returns:
        The row.
    """
    q = haar(delta["gate_proj"].shape[0], SEED + layer)

    real = neuron_energy(delta["gate_proj"], delta["up_proj"], delta["down_proj"])
    rotated = neuron_energy(q @ delta["gate_proj"], q @ delta["up_proj"], delta["down_proj"] @ q.T)
    return {"table": "rotation", "layer": layer, "top": TOP, "top_share": top_share(real),
            "top_share_rotated": top_share(rotated)}  # fmt: skip


def pole_rows(path: Path) -> list[dict]:
    """The keep test's poles, from ``base_instruct.py``'s rows.

    Args:
        path: A ``base-instruct*.jsonl`` file.

    Returns:
        Each pole's row.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r["table"] == "pole"]


def neuron_of(slug: str) -> int:
    """The neuron index in a pole's slug.

    Args:
        slug: Such as ``Qwen--Qwen2.5-0.5B/L3/N117-positive``.

    Returns:
        The index.

    Examples:
        >>> neuron_of("Qwen--Qwen2.5-0.5B/L3/N117-positive")
        117
    """
    return int(slug.rsplit("/N", 1)[1].split("-", 1)[0])


def pair_rows(base_name: str, poles: list[dict]) -> list[dict]:
    """Every row for one base and instruct pair.

    Args:
        base_name: The base model; the instruct model is ``<base_name>-Instruct``.
        poles: The pair's keep-test poles.

    Returns:
        The ``layer``, ``rotation``, ``pole`` and ``test`` rows.
    """
    from transformers import AutoModelForCausalLM

    load = {"torch_dtype": torch.float32}
    base = AutoModelForCausalLM.from_pretrained(base_name, **load).eval()
    instruct = AutoModelForCausalLM.from_pretrained(f"{base_name}-Instruct", **load).eval()
    rows, measured = [], []
    for layer in range(base.config.num_hidden_layers):
        b, i = mlp_weights(base, layer), mlp_weights(instruct, layer)
        delta = {m: i[m] - b[m] for m in MATRICES}
        rank, measures = neuron_measures(b, delta)
        rows.append({"table": "layer", "model": base_name, "layer": layer,
                     **{f"rank90_{m}": rank[m] for m in MATRICES}})  # fmt: skip
        rows.append({"model": base_name, **rotation_row(delta, layer)})
        for pole in (p for p in poles if p["layer"] == layer):
            n = neuron_of(pole["slug"])
            values = {k: round(float(v[n]), 6) for k, v in measures.items()}
            measured.append({"table": "pole", "model": base_name, "slug": pole["slug"],
                             "layer": layer, "keeps": pole["keeps"], **values})  # fmt: skip
        print(f"{base_name} layer {layer}: ranks {rank}", file=sys.stderr)
    rows += measured
    fails, layers = [not p["keeps"] for p in measured], [p["layer"] for p in measured]
    for measure in (k for k in measured[0] if k.startswith(("relative_", "top_r"))):
        scores = [p[measure] for p in measured]
        rows.append({"table": "test", "model": base_name, "measure": measure,
                     "poles": len(measured), "fails": sum(fails),
                     "auroc": round(auroc(scores, fails), 4),
                     "p_within_layer_shuffle": shuffled_p(scores, fails, layers)})  # fmt: skip
    return rows


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(out: Path) -> None:
    """Write the rows of every pair to ``out``.

    Args:
        out: Where the rows go.
    """
    rows = []
    for base_name, path in PAIRS.items():
        rows += pair_rows(base_name, pole_rows(Path(path)))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        if r["table"] == "test":
            click.echo(json.dumps(r))


if __name__ == "__main__":
    main()
