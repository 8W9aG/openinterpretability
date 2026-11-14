"""Predict whether a random neuron pole will verify, from what is known before review (#3372).

One row per random MLP neuron pole, from its first reviewed experiment, as in
``scripts/registry_stats.py``. The outcome is VERIFIED on that first reading. The features
come from the scan and the weights, before any agent writes a hypothesis. The units are
split by time: the earliest 70% train, the latest 30% test, so no unit is on both sides.

The model is an L2-regularised logistic regression. Its rival is the base rate of VERIFIED
in the training units, the forecast that needs no features. Both are scored on the test
units with the Brier and log scores. A bootstrap over test units gives the interval of
the difference.

    uv run open-interpretability build --output-dir build/site
    uv run python scripts/verify_predictor.py --site-dir build/site \\
        --out docs/analyses/verify-predictor.jsonl
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import click
import duckdb
import torch

TRAIN_SHARE = 0.7
ACTIVE_LINE = 0.25  # a token is active above this share of the pole's maximum (PROTOCOL §4)
PENALTIES = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)  # L2 penalties to choose from, by time-ordered validation
BOOTSTRAP = 2000
FOCUS_PROPOSER = "deepseek-flash"  # the proposer with the most units in both halves
NUMERIC = (
    "token_r2", "position_r2", "active_share", "floor_share", "log_max_activation",
    "depth", "jacobian_kurtosis", "lens_overlap", "logit_promoted_z", "current_token_share",
    "top_partner_lift", "output_norm",
)  # fmt: skip

UNITS = """
SELECT slug, model_name, layer, neuron_idx, pole, model_num_layers, status, max_activation,
       token_r2, position_r2, jacobian_kurtosis, lens_overlap,
       logit_promoted_z[1] AS logit_promoted_z,  -- the strongest promoted token's z
       current_token_share, top_partner_lift, submitting_agent_model,
       epoch(reviewed_at) AS reviewed_at,
       json_extract(record_json, '$.submission.activation_histogram') AS histogram
FROM r
WHERE kind = 'mlp_neuron_pole' AND sister_of IS NULL
QUALIFY row_number() OVER (PARTITION BY slug ORDER BY reviewed_at, experiment_id) = 1
"""


def load_units(site_dir: Path) -> list[dict]:
    """The first reading of every random neuron pole, oldest first, with its features."""
    from openinterp.core.agent_names import canonical_agent

    con = duckdb.connect()
    data = site_dir / "data" / "registry.parquet"
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{data}')")  # noqa: S608
    cur = con.execute(UNITS.replace("WHERE", "WHERE selection = 'random' AND", 1))
    names = [d[0] for d in cur.description]
    rows = [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
    for row in rows:
        row["proposer"] = canonical_agent(row.pop("submitting_agent_model")).removeprefix(
            "external/"
        )
        row |= histogram_features(json.loads(row.pop("histogram") or "null"))
        row["log_max_activation"] = math.log(max(row["max_activation"] or 0.0, 1e-6))
        row["depth"] = row["layer"] / max(row["model_num_layers"] - 1, 1)
        row["verified"] = row["status"] == "VERIFIED"
    return sorted(rows, key=lambda r: r["reviewed_at"])


def histogram_features(histogram: dict | None) -> dict:
    """The active share and the lowest bin's share of the scan's activation histogram.

    A bin counts as active when its lower edge is at least :data:`ACTIVE_LINE` of the top
    edge. So the share is about the share of tokens at or above that line.
    """
    if not histogram or not sum(histogram["counts"]):
        return {"active_share": None, "floor_share": None}
    edges, counts = histogram["bin_edges"], histogram["counts"]
    total, top = sum(counts), edges[-1]
    active = sum(c for c, lo in zip(counts, edges, strict=False) if lo >= ACTIVE_LINE * top)
    return {"active_share": active / total, "floor_share": counts[0] / total}


def add_output_norms(rows: list[dict], device: str) -> None:
    """Each row's ``output_norm``: the norm of its neuron's row of the down projection."""
    from openinterp.cli import _load_runner

    for model in sorted({r["model_name"] for r in rows}):
        runner = _load_runner(model, device, "float32", None)
        for row in rows:
            if row["model_name"] == model:
                vector = runner.neuron_output_direction(row["layer"], row["neuron_idx"])
                row["output_norm"] = float(vector.norm())
        del runner


def design(
    rows: list[dict],
    reference: list[dict],
    extra: tuple[str, ...] = (),
    numeric: tuple[str, ...] = NUMERIC,
):
    """The feature matrix of ``rows``, standardised and filled from ``reference``.

    Each numeric feature gets an indicator for a missing value. Models, poles and the
    ``extra`` categorical columns are one-hot, from the values in ``reference``.
    """
    columns = []
    for name in numeric:
        known = [r[name] for r in reference if r[name] is not None]
        mean = sum(known) / len(known)
        sd = (sum((v - mean) ** 2 for v in known) / len(known)) ** 0.5 or 1.0
        columns.append([((r[name] if r[name] is not None else mean) - mean) / sd for r in rows])
        columns.append([float(r[name] is None) for r in rows])
    for name in ("model_name", "pole", *extra):
        for value in sorted({r[name] for r in reference})[1:]:  # the first is the baseline
            columns.append([float(r[name] == value) for r in rows])
    return torch.tensor(columns, dtype=torch.float64).T


def fit(x: torch.Tensor, y: torch.Tensor, penalty: float) -> torch.Tensor:
    """Weights of an L2-penalised logistic regression (the intercept is not penalised)."""
    x1 = torch.cat([torch.ones(len(x), 1, dtype=x.dtype), x], dim=1)
    w = torch.zeros(x1.shape[1], dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        value = torch.nn.functional.binary_cross_entropy_with_logits(x1 @ w, y)
        value = value + penalty * (w[1:] ** 2).sum() / len(x)
        value.backward()
        return value

    opt.step(loss)
    return w.detach()


def predict(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The fitted probability of VERIFIED for each row of ``x``."""
    return torch.sigmoid(torch.cat([torch.ones(len(x), 1, dtype=x.dtype), x], dim=1) @ w)


def scores(p: torch.Tensor, y: torch.Tensor) -> dict:
    """Mean Brier score, mean log score (natural log, lower is better) and AUC."""
    p = p.clamp(1e-6, 1 - 1e-6)
    brier = float(((p - y) ** 2).mean())
    log = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())
    return {"brier": round(brier, 5), "log": round(log, 5), "auc": auc(p, y)}


def auc(p: torch.Tensor, y: torch.Tensor) -> float | None:
    """The chance that a VERIFIED unit gets a higher forecast than one that isn't."""
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()
    return round(float(wins.mean()), 4)


def gain_interval(p: torch.Tensor, base: float, y: torch.Tensor, seed: int = 0) -> dict:
    """Brier and log-score gains of ``p`` over the constant ``base``, with 95% intervals."""
    p = p.clamp(1e-6, 1 - 1e-6)
    b = torch.full_like(p, base)
    brier = (b - y) ** 2 - (p - y) ** 2  # per unit: positive when the model is better
    log = -(y * b.log() + (1 - y) * (1 - b).log()) + (y * p.log() + (1 - y) * (1 - p).log())
    rng = random.Random(seed)
    n = len(y)
    out = {}
    for name, gain in (("brier_gain", brier), ("log_gain", log)):
        means = sorted(
            float(gain[[rng.randrange(n) for _ in range(n)]].mean()) for _ in range(BOOTSTRAP)
        )
        lo, hi = means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP)]
        out[name] = [round(float(gain.mean()), 5), round(lo, 5), round(hi, 5)]
    aucs = []
    for _ in range(BOOTSTRAP):
        pick = [rng.randrange(n) for _ in range(n)]
        value = auc(p[pick], y[pick])
        if value is not None:
            aucs.append(value)
    aucs.sort()
    out["auc_ci"] = [aucs[int(0.025 * len(aucs))], aucs[int(0.975 * len(aucs))]]
    return out


def choose_penalty(train: list[dict], extra: tuple[str, ...], numeric=NUMERIC) -> float:
    """The penalty with the best log score on the latest 30% of the training units."""
    cut = int(len(train) * TRAIN_SHARE)
    early, late = train[:cut], train[cut:]
    y_early = torch.tensor([float(r["verified"]) for r in early], dtype=torch.float64)
    y_late = torch.tensor([float(r["verified"]) for r in late], dtype=torch.float64)
    x_early = design(early, early, extra, numeric)
    x_late = design(late, early, extra, numeric)
    return min(
        PENALTIES, key=lambda c: scores(predict(fit(x_early, y_early, c), x_late), y_late)["log"]
    )


def fit_and_predict(train, test, extra: tuple[str, ...] = (), numeric=NUMERIC):
    """(weights, penalty, test forecasts) of the model chosen and fitted on ``train``."""
    penalty = choose_penalty(train, extra, numeric)
    y_train = torch.tensor([float(r["verified"]) for r in train], dtype=torch.float64)
    w = fit(design(train, train, extra, numeric), y_train, penalty)
    return w, penalty, predict(w, design(test, train, extra, numeric))


def evaluate(train, test, extra: tuple[str, ...] = (), label: str = "neuron features") -> dict:
    """Fit on ``train``, and score the model and the base rate on ``test``."""
    w, penalty, p = fit_and_predict(train, test, extra)
    y_train = torch.tensor([float(r["verified"]) for r in train], dtype=torch.float64)
    y_test = torch.tensor([float(r["verified"]) for r in test], dtype=torch.float64)
    base = float(y_train.mean())
    names = [f"{n}{s}" for n in NUMERIC for s in ("", " missing")]
    names += [f"{c}={v}" for c in ("model_name", "pole", *extra)
              for v in sorted({r[c] for r in train})[1:]]  # fmt: skip
    weights = {n: round(float(v), 4) for n, v in zip(names, w[1:], strict=True)}
    return {
        "weights": weights,
        "table": "evaluation", "model": label, "penalty": penalty,
        "train_units": len(train), "test_units": len(test), "train_base_rate": round(base, 4),
        "test_rate": round(float(y_test.mean()), 4),
        "fitted": scores(p, y_test), "base_rate": scores(torch.full_like(p, base), y_test),
        **gain_interval(p, base, y_test),
    }  # fmt: skip


@click.command()
@click.option("--site-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True, help="Where to read the weights.")
def main(site_dir: Path, out: Path, device: str) -> None:
    """Fit, score against the base rate, and write the rows."""
    rows = load_units(site_dir)
    add_output_norms(rows, device)
    cut = int(len(rows) * TRAIN_SHARE)
    train, test = rows[:cut], rows[cut:]
    results = [evaluate(train, test)]
    results.append(evaluate(train, test, ("proposer",), "neuron features and proposer"))
    focus = [r for r in rows if r["proposer"] == FOCUS_PROPOSER]
    fcut = int(len(focus) * TRAIN_SHARE)
    results.append(
        evaluate(focus[:fcut], focus[fcut:], (), f"neuron features, {FOCUS_PROPOSER} only")
    )
    split_at = rows[cut]["reviewed_at"]
    results.append({"table": "split", "units": len(rows), "test_from": split_at})
    out.write_text("".join(json.dumps(r) + "\n" for r in results))
    for r in results[:-1]:
        fitted, base = r["fitted"], r["base_rate"]
        click.echo(
            f"{r['model']}: Brier {fitted['brier']} vs {base['brier']}, "
            f"log {fitted['log']} vs {base['log']}, AUC {fitted['auc']} {r['auc_ci']}"
        )


if __name__ == "__main__":
    main()
