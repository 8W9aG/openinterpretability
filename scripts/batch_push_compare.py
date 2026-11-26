"""The push in training with 32 times the batch, against the no-dropout baseline (#6806).

Both runs record the push on each neuron's rest every 100 steps (``train --push-every``).
The 32× run adds the push over its micro-batches (``PushRecorder.add_part``). For each run
and range of steps, this prints three values, each the median of L1–L3 and then the mean
over the windows:

* The median push.
* Its ratio of mean to root mean square over the window's steps. That is how steady it
  is, since Adam moves a parameter by about the rate times this ratio.
* How far the rest moved, at the batch's mean input.

It also gives the number of windows in which the push points down. The ranges compare the
runs at the same steps, at the same number of tokens and at the same loss.

    uv run python scripts/batch_push_compare.py papers/why-neurons-rest/results.jsonl
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import click

LAYERS = (1, 2, 3)  # the layers that the checkpoint test counts in the small models
RUNS = {"push_batch_32x": "32x", "push_batch_base": "base"}
# (run, first step, last step, what the range matches)
RANGES = (
    ("push_batch_32x", 1000, 2000, "same steps"),
    ("push_batch_base", 1000, 2000, "same steps"),
    ("push_batch_32x", 0, 500, "same tokens"),
    ("push_batch_base", 0, 16000, "same tokens"),
    ("push_batch_32x", 1500, 2000, "same loss"),
    ("push_batch_base", 8000, 12000, "same loss"),
)


def summarise(rows: list[dict], part: str, first: int, last: int) -> dict:
    """The push, its steadiness and the rest's move over the windows from ``first`` to ``last``.

    Args:
        rows: The paper's results.
        part: The push rows' label.
        first: The step before the first window.
        last: The step that ends the last window.

    Returns:
        The means over the windows, and the windows in which the push points down.
    """
    windows = [r for r in rows if r.get("part") == part and first < r["step"] <= last]
    # A window's row is labelled with the step that ends it, so a range starts after first.

    def mean_of(key: str) -> float:
        """The mean over the windows of the median of ``key`` over L1–L3.

        Args:
            key: A field of each layer's push row.
        """
        return statistics.mean(
            statistics.median(w["layers"][i][key] for i in LAYERS) for w in windows
        )

    pushes = [statistics.median(w["layers"][i]["median_push"] for i in LAYERS) for w in windows]
    return {
        "windows": len(windows),
        "median_push": float(f"{mean_of('median_push'):.3g}"),
        "push_ratio": round(mean_of("median_push_ratio"), 4),
        "moved": float(f"{mean_of('median_moved'):.3g}"),
        "windows_down": sum(p < 0 for p in pushes),
    }


def losses(rows: list[dict]) -> dict[str, dict[int, float]]:
    """Each run's loss at each evaluation, by step.

    Args:
        rows: The paper's results.
    """
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        if r.get("part") == "training_batch_push" and "loss" in r:
            out.setdefault(r["condition"], {})[r["step"]] = round(r["loss"], 3)
    return out


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(results: Path) -> None:
    """Print each range's summary, and each run's losses, one JSON line each.

    Args:
        results: The paper's results.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    for part, first, last, matches in RANGES:
        row = {"run": RUNS[part], "steps": [first, last], "matches": matches}
        click.echo(json.dumps({**row, **summarise(rows, part, first, last)}))
    click.echo(json.dumps({"losses": losses(rows)}))


if __name__ == "__main__":
    main()
