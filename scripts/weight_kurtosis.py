"""Test a weight-only monosemanticity signal as a cheap prior for VERIFIED (#3226).

The signal is from ROTATE (arXiv 2604.06005, §3). It expects monosemantic neurons to have
a high kurtosis of the vocabulary projection of their weights, ``z = wU``. This script
computes that kurtosis from the weights alone, with no forward pass:

* the output kurtosis: ``w_out`` through the final norm and ``U``. This is the
  ``logit_kurtosis`` of :func:`openinterp.harness.jlens._decode`.
* the input kurtosis: the read vector of ``wiring.mlp_reads``, through ``U``. The read
  vector is ``gate`` for a gated MLP and ``in`` for GPT-2, with the pre-MLP norm folded in.

The kurtosis is the mean of z⁴ over the vocabulary, so a Gaussian scores 3. The script
computes it for every neuron of every layer. The rows hold a summary of each layer. They
also hold each random neuron pole of ``scripts/verify_predictor.py``, with its kurtoses
and their percentiles in the layer. Then the script asks whether the kurtoses predict
VERIFIED on the held-out units of that script, alone and added to its features.

    uv run open-interpretability build --output-dir build/site
    uv run python scripts/weight_kurtosis.py --site-dir build/site \\
        --out docs/analyses/weight-kurtosis.jsonl
"""

from __future__ import annotations

import bisect
import json
import random
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_predictor as vp  # noqa: E402

CHUNK = 256  # neurons per matmul, so that [chunk, vocab] scores fit in memory
SIDES = ("out", "in")
FEATURES = ("kurtosis_out", "kurtosis_in", "kurtosis_out_pct", "kurtosis_in_pct")
QUANTILES = (0.1, 0.5, 0.9, 0.99)
TOP = 0.9  # the top decile of a layer's neurons, by kurtosis
LATE = 0.5  # relative depth from which a layer counts as mid-to-late, as ROTATE uses
BOOTSTRAP = vp.BOOTSTRAP


def moments(rows: torch.Tensor, unembed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(kurtosis, skewness) of each row's vocabulary projection ``unembed @ row``.

    ``rows`` is ``[n, d]`` and ``unembed`` is ``[vocab, d]``. Both statistics use the
    population standard deviation of the ``vocab`` scores of one row.
    """
    kurt, skew = [], []
    for start in range(0, len(rows), CHUNK):
        scores = rows[start : start + CHUNK].to(unembed) @ unembed.T
        z = (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True).clamp(min=1e-12)
        kurt.append((z**4).mean(1).cpu())
        skew.append((z**3).mean(1).cpu())
    return torch.cat(kurt).double(), torch.cat(skew).double()


def write_rows(runner, layer: int) -> torch.Tensor:
    """``[width, d_model]``: each neuron's ``w_out`` through the final norm (see ``jlens``)."""
    from openinterp.harness import wiring

    w = wiring._matrix(runner.down_proj(layer)).T  # [width, d_model]
    norm = runner.final_norm()
    if isinstance(norm, torch.nn.LayerNorm):
        w = w - w.mean(-1, keepdim=True)
    gain = getattr(norm, "weight", None)
    return w * gain.detach().float() if gain is not None else w


def read_rows(runner, layer: int) -> torch.Tensor:
    """``[width, d_model]``: each neuron's read vector, ``gate`` when the MLP has one."""
    from openinterp.harness import wiring

    reads = wiring.mlp_reads(runner, layer)
    return reads.get("gate", reads.get("in", next(iter(reads.values()))))


def model_kurtoses(model: str, device: str) -> dict[int, dict[str, torch.Tensor]]:
    """``{layer: {"kurtosis_out", "kurtosis_in", "skew_out": [width]}}`` for ``model``."""
    from openinterp.cli import _load_runner

    runner = _load_runner(model, device, "float32", None)
    unembed = runner.model.get_output_embeddings().weight.detach().float().to(device)
    out = {}
    for layer in range(len(runner.layers)):
        k_out, s_out = moments(write_rows(runner, layer), unembed)
        k_in, _ = moments(read_rows(runner, layer), unembed)
        out[layer] = {"kurtosis_out": k_out, "kurtosis_in": k_in, "skew_out": s_out}
        click.echo(f"{model} L{layer}: median kurtosis out {k_out.median():.2f}, "
                   f"in {k_in.median():.2f}", err=True)  # fmt: skip
    del runner
    return out


def percentile(sorted_values: list[float], value: float) -> float:
    """The share of ``sorted_values`` below ``value``, counting ties as half."""
    lo, hi = bisect.bisect_left(sorted_values, value), bisect.bisect_right(sorted_values, value)
    return (lo + hi) / 2 / len(sorted_values)


def layer_summary(model: str, layer: int, stats: dict[str, torch.Tensor]) -> dict:
    """One row: the quantiles of each kurtosis over the layer's neurons."""
    row = {"table": "layer", "model": model, "layer": layer, "neurons": len(stats["kurtosis_out"])}
    for side in SIDES:
        values = stats[f"kurtosis_{side}"]
        q = torch.quantile(values, torch.tensor(QUANTILES, dtype=values.dtype))
        row[f"kurtosis_{side}"] = {f"q{round(p * 100)}": round(float(v), 3)
                                   for p, v in zip(QUANTILES, q, strict=True)}  # fmt: skip
    return row


def add_kurtoses(units: list[dict], kurtoses: dict) -> None:
    """Give each unit its neuron's kurtoses, their percentiles in the layer, and its skew.

    The skew is signed by the pole, so a positive value means the pole promotes a heavy
    tail of tokens. Kurtosis is the same for both poles of a neuron.
    """
    ordered = {}
    for unit in units:
        stats = kurtoses[unit["model_name"]][unit["layer"]]
        for side in SIDES:
            key = (unit["model_name"], unit["layer"], side)
            if key not in ordered:
                ordered[key] = sorted(stats[f"kurtosis_{side}"].tolist())
            value = float(stats[f"kurtosis_{side}"][unit["neuron_idx"]])
            unit[f"kurtosis_{side}"] = value
            unit[f"kurtosis_{side}_pct"] = percentile(ordered[key], value)
        sign = -1.0 if unit["pole"] == "negative" else 1.0
        unit["skew_out_signed"] = sign * float(stats["skew_out"][unit["neuron_idx"]])


def ci(values: list[float]) -> list[float]:
    """The 2.5% and 97.5% points of bootstrap ``values``."""
    values = sorted(values)
    return [round(values[int(0.025 * len(values))], 4), round(values[int(0.975 * len(values))], 4)]


def alone(units: list[dict], name: str, seed: int = 0) -> dict:
    """The AUC of ``name`` alone for VERIFIED over ``units``, with a bootstrap interval."""
    p = torch.tensor([u[name] for u in units], dtype=torch.float64)
    y = torch.tensor([float(u["verified"]) for u in units], dtype=torch.float64)
    rng, n = random.Random(seed), len(units)
    boot = []
    for _ in range(BOOTSTRAP):
        pick = [rng.randrange(n) for _ in range(n)]
        value = vp.auc(p[pick], y[pick])
        if value is not None:
            boot.append(value)
    return {"table": "alone", "feature": name, "units": n, "auc": vp.auc(p, y), "auc_ci": ci(boot)}


def added(train: list[dict], test: list[dict], seed: int = 0) -> dict:
    """The predictor's features with and without the kurtoses, scored on ``test``.

    Each gain is the score of the model with the kurtoses minus the score of the model
    without them, per test unit. A positive gain means that the kurtoses help. The
    intervals come from a paired bootstrap.
    """
    y = torch.tensor([float(u["verified"]) for u in test], dtype=torch.float64)
    _, pen_a, p_a = vp.fit_and_predict(train, test)
    w_b, pen_b, p_b = vp.fit_and_predict(train, test, numeric=vp.NUMERIC + FEATURES)
    p_a, p_b = p_a.clamp(1e-6, 1 - 1e-6), p_b.clamp(1e-6, 1 - 1e-6)
    brier = (p_a - y) ** 2 - (p_b - y) ** 2
    log = (y * p_b.log() + (1 - y) * (1 - p_b).log()) - (y * p_a.log() + (1 - y) * (1 - p_a).log())
    rng, n = random.Random(seed), len(test)
    boot = {"brier_gain": [], "log_gain": [], "auc_gain": []}
    for _ in range(BOOTSTRAP):
        pick = [rng.randrange(n) for _ in range(n)]
        boot["brier_gain"].append(float(brier[pick].mean()))
        boot["log_gain"].append(float(log[pick].mean()))
        a, b = vp.auc(p_a[pick], y[pick]), vp.auc(p_b[pick], y[pick])
        if a is not None and b is not None:
            boot["auc_gain"].append(b - a)
    names = [f"{f}{s}" for f in vp.NUMERIC + FEATURES for s in ("", " missing")]
    weights = {n: round(float(v), 4) for n, v in zip(names, w_b[1:], strict=False)
               if n in FEATURES}  # fmt: skip
    return {
        "table": "added", "train_units": len(train), "test_units": n,
        "penalty_without": pen_a, "penalty_with": pen_b,
        "without": vp.scores(p_a, y), "with": vp.scores(p_b, y),
        "brier_gain": [round(float(brier.mean()), 5), *ci(boot["brier_gain"])],
        "log_gain": [round(float(log.mean()), 5), *ci(boot["log_gain"])],
        "auc_gain": ci(boot["auc_gain"]), "kurtosis_weights": weights,
    }  # fmt: skip


def tail(units: list[dict]) -> list[dict]:
    """The share VERIFIED among poles in the top 10% of their layer, and among the rest.

    ROTATE's own test (§3) asks where known concept vectors fall among the neurons of a
    layer. So this compares the top decile with the rest, for each side and model.
    """
    out = []
    for side in SIDES:
        for model in ["all", *sorted({u["model_name"] for u in units})]:
            sel = [u for u in units if model in ("all", u["model_name"])]
            top = [u["verified"] for u in sel if u[f"kurtosis_{side}_pct"] >= TOP]
            rest = [u["verified"] for u in sel if u[f"kurtosis_{side}_pct"] < TOP]
            out.append({
                "table": "tail", "side": side, "model": model,
                "top_units": len(top), "top_verified": round(sum(top) / len(top), 4),
                "rest_units": len(rest), "rest_verified": round(sum(rest) / len(rest), 4),
            })  # fmt: skip
    return out


def pearson(a: list[float], b: list[float]) -> float:
    """Pearson's r of two equal-length lists."""
    x, y = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    return round(float(torch.corrcoef(torch.stack([x, y]))[0, 1]), 4)


def checks(site_dir: Path, units: list[dict]) -> dict:
    """How the output kurtosis compares with the recorded lens kurtoses of the same units."""
    import duckdb

    con = duckdb.connect()
    data = site_dir / "data" / "registry.parquet"
    rows = con.execute(
        f"SELECT slug, json_extract(record_json, '$.submission.vocabulary_lens.logit_kurtosis')"  # noqa: S608
        f" FROM read_parquet('{data}')"
    ).fetchall()
    recorded = {slug: float(v) for slug, v in rows if v is not None}
    pairs = [(u["kurtosis_out"], recorded[u["slug"]]) for u in units if u["slug"] in recorded]
    jac = [(u["kurtosis_out"], u["jacobian_kurtosis"]) for u in units
           if u["jacobian_kurtosis"] is not None]  # fmt: skip
    return {
        "table": "checks", "units_with_logit_kurtosis": len(pairs),
        "max_abs_diff_logit_kurtosis": round(max(abs(a - b) for a, b in pairs), 4),
        "pearson_out_vs_jacobian": pearson(*zip(*jac, strict=True)),
        "pearson_out_vs_in": pearson([u["kurtosis_out"] for u in units],
                                     [u["kurtosis_in"] for u in units]),
    }  # fmt: skip


@click.command()
@click.option("--site-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True, help="Where to read the weights.")
def main(site_dir: Path, out: Path, device: str) -> None:
    """Compute the kurtoses, test them against VERIFIED, and write the rows."""
    units = vp.load_units(site_dir)
    vp.add_output_norms(units, device)
    kurtoses = {m: model_kurtoses(m, device) for m in sorted({u["model_name"] for u in units})}
    add_kurtoses(units, kurtoses)
    results = [layer_summary(m, layer, s) for m, by in kurtoses.items() for layer, s in by.items()]
    cut = int(len(units) * vp.TRAIN_SHARE)
    train, test = units[:cut], units[cut:]
    for name in (*FEATURES, "skew_out_signed", "jacobian_kurtosis"):
        known = [u for u in test if u[name] is not None]
        results.append(alone(known, name) | {"units_from": "test"})
    late = [u for u in units if u["depth"] >= LATE]
    for name in FEATURES[:2]:
        results.append(alone(late, name) | {"units_from": f"all, depth >= {LATE}"})
    results.extend(tail(units))
    results.append(added(train, test))
    results.append(checks(site_dir, units))
    for i, u in enumerate(units):
        results.append({
            "table": "unit", "slug": u["slug"], "verified": u["verified"],
            "split": "train" if i < cut else "test",
            **{k: round(u[k], 4) for k in (*FEATURES, "skew_out_signed")},
        })  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in results))
    for r in results:
        if r["table"] in ("alone", "added", "checks", "tail"):
            click.echo(json.dumps({k: v for k, v in r.items() if k != "table"}))


if __name__ == "__main__":
    main()
