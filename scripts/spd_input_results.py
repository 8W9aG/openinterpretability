"""Write the results of papers/spd-input-projections/ from four SPD runs.

``--logs`` holds each run's ``train.log`` as ``A.log``, ``B.log``, ``C.log`` and ``D.log``. A
log read back from Google Drive has Markdown escapes, and this script removes them. It needs
no model and no checkpoint.

    uv run python scripts/spd_input_results.py --logs exchange/spd-logs \\
        --out papers/spd-input-projections/results.jsonl
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import click

ANALYSES = Path(__file__).parents[1] / "docs" / "analyses"
COMPARES = {"C": "spd-baseline-colab-both-random.json", "D": "spd-baseline-colab-both-neurons.json"}
CURVES = "spd-baseline-up-neurons.json"
LAYERS = range(12)
RUNS = "ABCD"
# The weight of each logged loss, from the notebook's settings and train_spd.py's defaults.
WEIGHTS = {"faith": 1e7, "imp": 6e-5, "stoch": 0.5, "layerwise": 0.5, "ci_masked": 1.0,
           "ci_output": 10.0}  # fmt: skip
SETTLED_FROM = 40_000  # the evaluations that give each run's mean objective
ALIVE = re.compile(r"^L(\d+(?:\.up)?): (\d+) of \d+ subcomponents alive")


def read_log(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Return a log's evaluation rows, and each site's count of alive subcomponents."""
    rows, alive = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.replace("\\", "")
        if clean.startswith("step "):
            rows.append(json.loads(clean[clean.index("{") : clean.rindex("}") + 1]))
        elif found := ALIVE.match(clean):
            alive[found[1]] = int(found[2])
    return rows, alive


def neurons_for_fit(curve: dict[str, float], fit: float) -> float:
    """Return the neurons per token that reach ``fit``, interpolated on a log scale of k."""
    points = sorted((int(k), value) for k, value in curve.items())
    if fit <= points[0][1]:
        return float(points[0][0])
    for (k1, fit1), (k2, fit2) in zip(points, points[1:], strict=False):
        if fit1 < fit <= fit2:
            return k1 * (k2 / k1) ** ((fit - fit1) / (fit2 - fit1))
    return float(points[-1][0])


def site_row(run: str, site: str, row: dict, curve: dict, alive: dict) -> dict:
    """Return one site's results in one run."""
    fit, active = row["spd_explained_variance"], row["spd_l0"]
    same = curve.get(str(round(active)), row.get("neurons_at_spd_l0"))
    need = neurons_for_fit(curve, fit)
    return {
        "kind": "site", "run": run, "site": site, "layer": int(site.split(".")[0]),
        "matrix": "up" if site.endswith(".up") else "down",
        "explained_variance": fit, "active": active,
        "best_neuron_cosine": row["best_neuron_cosine_median"],
        "share_over_0_7": row["share_best_neuron_cosine_over_0.7"],
        "drift_cosine": row["drift_cosine_median"],
        "neurons_same_count": same, "neurons_for_fit": round(need, 1),
        "neurons_per_active": round(need / active, 3), "alive": alive[site],
    }  # fmt: skip


def summary_row(run: str, rows: list[dict]) -> dict:
    """Return one run's whole-model results, with its objective over the settled evaluations."""
    settled = [row for row in rows if row["step"] >= SETTLED_FROM]
    objective = [sum(w * row[name] for name, w in WEIGHTS.items()) for row in settled]
    mean = {name: statistics.mean(row[name] for row in settled) for name in WEIGHTS}
    return {
        "kind": "summary", "run": run, "evaluations": len(settled),
        "active_total": round(sum(rows[-1]["l0"].values()), 1),
        "objective": round(statistics.mean(objective), 3),
        "objective_sd": round(statistics.stdev(objective), 3),
        "importance_term": round(WEIGHTS["imp"] * mean["imp"], 3),
        "output_term": round(WEIGHTS["ci_output"] * mean["ci_output"], 3),
        "kl": rows[-1]["kl_ci"], "unrecovered": rows[-1]["ce_unrecovered"],
        "kl_settled": round(statistics.mean(row["kl_ci"] for row in settled), 4),
        "unrecovered_settled": round(statistics.mean(r["ce_unrecovered"] for r in settled), 4),
    }  # fmt: skip


def trace_row(run: str, row: dict) -> dict:
    """Return one evaluation of one run, with its per-site values when the log has them."""
    trace = {"kind": "trace", "run": run, "step": row["step"], "kl": row["kl_ci"],
             "unrecovered": row["ce_unrecovered"]}  # fmt: skip
    if "explained_variance" in row:
        trace |= {"explained_variance": row["explained_variance"], "active": row["l0"]}
    return trace


@click.command()
@click.option("--logs", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(logs, out):
    """Write the paper's results to ``--out``."""
    curves = json.loads((ANALYSES / CURVES).read_text())["curves"]
    results = []
    for run in RUNS:
        rows, alive = read_log(logs / f"{run}.log")
        results.append(summary_row(run, rows))
        results += [trace_row(run, row) for row in rows]
        if run in COMPARES:
            layers = json.loads((ANALYSES / COMPARES[run]).read_text())["layers"]
            for site, row in layers.items():
                curve = curves[site] if site.endswith(".up") else row["neuron_curve"]
                results.append(site_row(run, site, row, curve, alive))
    results += [{"kind": "up_curve", "site": s, "curve": c} for s, c in curves.items()]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as lines:
        lines.writelines(json.dumps(row) + "\n" for row in results)
    click.echo(f"Wrote {out}: {len(results)} rows")


if __name__ == "__main__":
    main()
