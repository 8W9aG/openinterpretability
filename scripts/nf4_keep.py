"""Test VERIFIED poles with the model's linear weights rounded to NF4 (#5158).

QLoRA stores a base model's weights in 4-bit NormalFloat (NF4). It turns them back into
16 bits before it computes with them (Dettmers et al., 2023). This script asks whether the
explanations of the registry's neurons keep through that rounding. It trains nothing.

Each condition adds a change to the base model's weights in place, and copies them back
afterwards (``random_delta_keep.changed``):

* ``nf4``: each linear weight of each block (attention q, k, v, o and MLP gate, up, down)
  rounded to NF4. That means Appendix E's 16 values, blocks of 64 weights in row order, and
  a scale per block set by its largest absolute value. There is no double quantization.
  Biases, norms, the embeddings and the output head stay as they are.
* ``bf16``: the same weights rounded to bfloat16. The model still computes in float32.
* ``random_nf4``: Gaussian noise with the Frobenius norm of NF4's error in each matrix, one
  run for each seed. It is #5154's null for this change.

Each VERIFIED pole of the base model is then tested again as ``scripts/base_instruct.py``
tests it, with the batched readings of ``scripts/random_delta_keep.py``. Each pole row also
gives the drift of each probe's peak, as a share of the pole's scale. That is the measure
of ``docs/analyses/half-precision-reproduce.md``.

    uv run python scripts/nf4_keep.py --base Qwen/Qwen2.5-0.5B --device cuda --out 0.5b.jsonl

More seeds of the null go on the paper's own poles (#7072). ``--poles-from`` keeps the poles of
a results file's ``base`` rows, and ``--first-seed`` starts after the seeds already run. A
pole can get a newer record with the same base readings, which the check of ``--poles-from``
does not see. So ``--wiki-dir`` reads the registry as it was at the paper's commit:

    git archive <the paper's commit> wiki | tar -x -C snapshot
    uv run python scripts/nf4_keep.py --base Qwen/Qwen2.5-0.5B --device cuda \
        --wiki-dir snapshot/wiki --condition random_nf4 --first-seed 3 --seeds 7 \
        --poles-from papers/nf4-keep/results.jsonl --out 0.5b-seeds.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_delta_keep import (  # noqa: E402
    base_columns,
    base_rows,
    cached_readings,
    changed,
    first_position_activations,
    pole_stats,
    probe_activations,
    probe_texts,
    settle_cached,
    summary,
)

from openinterpretability.core.schemas import ReviewPolicy  # noqa: E402
from openinterpretability.harness.hooks import pole_strength  # noqa: E402
from openinterpretability.interference.floors import _noise_like  # noqa: E402
from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    sample_windows,
    verified_records,
)

# The 16 values of NF4, from the QLoRA paper's Appendix E (Dettmers et al., 2023).
NF4 = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)  # fmt: skip
BLOCK = 64  # the weights that share one scale
# The names of a block's linear weights end with these (Qwen2 and Llama).
LINEAR = tuple(f"{part}.weight" for part in (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
))  # fmt: skip
CONDITIONS = ("nf4", "bf16", "random_nf4")


def nf4_round(weight: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """``weight`` rounded to NF4, in blocks of ``block`` values in row order.

    Each block is divided by its largest absolute value, each value goes to the nearest of
    :data:`NF4`, and the block is multiplied back. A block of zeros stays zero.

    Args:
        weight: Any tensor.
        block: The values that share one scale.

    Returns:
        The rounded tensor, in float32, with the shape of ``weight``.
    """
    flat = weight.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    blocks = torch.cat([flat, flat.new_zeros(pad)]).reshape(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    levels = torch.tensor(NF4, device=flat.device)
    # The nearest level: the midpoints between neighbours split the range into 16 bins.
    index = torch.bucketize(blocks / safe, (levels[1:] + levels[:-1]) / 2)
    rounded = levels[index] * scale
    return rounded.reshape(-1)[: flat.numel()].reshape(weight.shape)


def is_linear(name: str) -> bool:
    """True when the parameter ``name`` is one of a block's linear weights.

    Args:
        name: A parameter's name, as ``named_parameters`` gives it.
    """
    return name.endswith(LINEAR)


def change_for(model, condition: str, seed: int = 0) -> dict[str, torch.Tensor]:
    """The change that ``condition`` adds to each linear weight, on the CPU.

    Args:
        model: The base model.
        condition: One of :data:`CONDITIONS`.
        seed: The seed of the noise. Only ``random_nf4`` uses it.

    Returns:
        ``{name: tensor}`` for each linear weight.

    Raises:
        ValueError: ``condition`` is not one of :data:`CONDITIONS`.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name, weight in sorted(model.named_parameters()):
        if not is_linear(name):
            continue
        w = weight.detach().float()
        if condition == "bf16":
            out[name] = (w.bfloat16().float() - w).cpu()
            continue
        error = (nf4_round(w) - w).cpu()
        out[name] = error if condition == "nf4" else _noise_like(error, generator)
    return out


def peak(acts: torch.Tensor, pole: str, skip_first: bool) -> float | None:
    """The pole's peak strength over ``acts``, the way the review reads a probe.

    Args:
        acts: The neuron's raw activations over a text's tokens.
        pole: ``positive`` or ``negative``.
        skip_first: Leave the first token out, for a pole with a first-token floor.

    Returns:
        The peak, or None when no token is measured.
    """
    strengths = pole_strength(acts, pole)[int(skip_first) :]
    return float(strengths.max()) if strengths.numel() else None


def drifts(record, unit, base_cache: dict, cache: dict) -> list[float]:
    """Each probe's |peak after − peak before|, as a share of the pole's scale.

    Args:
        record: One VERIFIED record.
        unit: Its unit, which says whether the first token is read.
        base_cache: The probes' activations on the base model.
        cache: The probes' activations on the changed model.

    Returns:
        One value for each probe that has a measured token, in the probes' order.
    """
    ident, scale = record.submission.identifier, record.submission.max_activation
    out = []
    for text in probe_texts(record):
        key = (text, ident.layer, ident.neuron_idx)
        before = peak(base_cache[key], ident.pole, unit.skip_first)
        after = peak(cache[key], ident.pole, unit.skip_first)
        if before is not None and after is not None:
            out.append(round(abs(after - before) / scale, 6))
    return out


def run_condition(runner, batches, records, base, state, change, fields) -> list[dict]:
    """The pole rows and the summary of one condition, with ``change`` added to the base.

    Args:
        runner: The base model, which the condition changes and then puts back.
        batches: The corpus sample.
        records: The VERIFIED records.
        base: The base model's columns, from ``random_delta_keep.base_columns``.
        state: The base model's readings, probe activations and first positions, as
            ``{"on_base", "cache", "firsts"}``.
        change: The change to add, from :func:`change_for`.
        fields: The condition's fields, copied into each row.

    Returns:
        The summary row, then one row for each pole.
    """
    rows = []
    with changed(runner.model, change):
        stats = pole_stats(runner, batches, records, base)
        cache = probe_activations(runner, records)
        firsts = first_position_activations(runner, records)
        for record in records:
            ident = record.submission.identifier
            s = stats[ident.slug]
            reading = cached_readings(runner, record, cache, firsts, s["ratio"] or 1.0)
            policy = record.verdict.policy or ReviewPolicy()
            unit, _ = settle_cached(record, runner, policy, state["firsts"])
            passes = state["on_base"][ident.slug]["passes"]
            rows.append({
                "table": "pole", **fields, "slug": ident.slug, "layer": ident.layer,
                "pole": ident.pole, **s, "changed": reading, "base_passes": passes,
                "keeps": passes and reading["passes"],
                "drift": drifts(record, unit, state["cache"], cache),
            })  # fmt: skip
    norm = float(torch.sqrt(sum((t.double() ** 2).sum() for t in change.values())))
    return [summary(rows, **fields, change_norm=round(norm, 4)), *rows]


def paper_poles(path: Path, model: str) -> dict[str, dict]:
    """The readings on the base model of each pole in ``path``'s ``base`` rows of ``model``.

    Args:
        path: A results file of this script.
        model: The model's name.

    Returns:
        Each pole's base readings, by slug.

    Examples:
        >>> import tempfile
        >>> rows = [{"table": "base", "model": "m", "slug": "a", "readings": {"passes": True}},
        ...         {"table": "base", "model": "n", "slug": "b", "readings": {}},
        ...         {"table": "summary", "model": "m"}]
        >>> with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        ...     _ = f.write("".join(json.dumps(r) + "\\n" for r in rows))
        >>> paper_poles(Path(f.name), "m")
        {'a': {'passes': True}}
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["slug"]: r["readings"] for r in rows
            if r["table"] == "base" and r["model"] == model}  # fmt: skip


def differing(on_base: dict[str, dict], paper: dict[str, dict], tol: float = 1e-3) -> list[str]:
    """The poles whose readings on the base model differ from the paper's.

    A pole differs when it passes in one and not the other, or when a score moves by more than
    ``tol``. A pole missing from either side differs too.

    Args:
        on_base: This run's readings, by slug.
        paper: The paper's readings, by slug.
        tol: The largest move of a score that still counts as the same.

    Returns:
        The slugs that differ, sorted.

    Examples:
        >>> a = {"x": {"passes": True, "balanced_accuracy": 0.9}, "y": {"passes": True}}
        >>> b = {"x": {"passes": True, "balanced_accuracy": 0.95}, "z": {"passes": True}}
        >>> differing(a, b)
        ['x', 'y', 'z']
    """
    out = []
    for slug in sorted(on_base.keys() | paper.keys()):
        mine, theirs = on_base.get(slug), paper.get(slug)
        if mine is None or theirs is None or mine.get("passes") != theirs.get("passes"):
            out.append(slug)
            continue
        scores = [k for k, v in theirs.items() if isinstance(v, float)]
        if any(abs(mine.get(k, 0.0) - theirs[k]) > tol for k in scores):
            out.append(slug)
    return out


@click.command()
@click.option("--base", "base_name", required=True, help="The model, e.g. Qwen/Qwen2.5-0.5B.")
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--condition", "names", multiple=True, type=click.Choice(CONDITIONS))
@click.option("--seeds", default=3, show_default=True, help="Seeds of random_nf4.")
@click.option("--first-seed", default=0, show_default=True, help="The first seed of random_nf4.")
@click.option("--poles-from", type=click.Path(dir_okay=False, path_type=Path),
              help="Test only the poles of this results file's base rows.")  # fmt: skip
def main(base_name, wiki_dir, out, device, names, seeds, first_seed, poles_from) -> None:
    """Test each VERIFIED pole under each condition, and write the rows to ``--out``.

    Args:
        base_name: The model.
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each condition.
        device: The device of the model.
        names: The conditions to run. With none, every condition runs.
        seeds: The number of seeds of ``random_nf4``.
        first_seed: The first seed of ``random_nf4``.
        poles_from: A results file whose ``base`` rows name the poles to test, or None.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, base_name)
    paper = paper_poles(poles_from, base_name) if poles_from else None
    if paper is not None:  # the paper's poles only, so that new seeds join its old ones
        records = [r for r in records if r.submission.identifier.slug in paper]
    runner = _load_runner(base_name, device, "float32", None)
    batches = sample_windows(runner.tokenizer)
    base = base_columns(runner, batches, records)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {
        r.submission.identifier.slug: cached_readings(runner, r, cache, firsts) for r in records
    }
    if paper is not None and (moved := differing(on_base, paper)):
        click.echo(f"{len(moved)} poles read differently from {poles_from}: {moved}", err=True)
    state = {"on_base": on_base, "cache": cache, "firsts": firsts}
    lines = base_rows(records, on_base, base_name)
    for name in names or CONDITIONS:
        for seed in range(first_seed, first_seed + seeds) if name == "random_nf4" else [None]:
            fields = {"model": base_name, "condition": name, "seed": seed,
                      "sample_tokens": SAMPLE_TOKENS}  # fmt: skip
            change = change_for(runner.model, name, seed or 0)
            rows = run_condition(runner, batches, records, base, state, change, fields)
            lines += rows
            click.echo(json.dumps({k: v for k, v in rows[0].items() if k != "flipped"}), err=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()
