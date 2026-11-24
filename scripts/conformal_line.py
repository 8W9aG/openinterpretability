"""A conformal activation line for each VERIFIED pole, and the verdicts it changes (#6417).

A probe fires a pole when its peak strength reaches 0.25 of the pole's maximum. The share
is the same for every pole, and the line states no error rate. Split conformal prediction
(Angelopoulos & Bates) gives one. Take calibration windows of corpus text, and put the line
at the ⌈(n + 1)(1 − α)⌉-th smallest peak. Then a new window from the same text crosses it
with probability at most α.

For each model with VERIFIED poles, this script cuts ``corpus.jsonl`` into windows about
as long as a probe. A fixed seed splits them into calibration and held-out windows. One
pass per window reads each pole's peak strength. The script then writes, for each pole
and each α in :data:`ALPHAS`:

* the conformal line, and the record's own line (0.25 × ``max_activation``);
* the held-out coverage: the share of held-out windows under the line;
* how many of the record's probes predicted quiet reach the line;
* the specificity and sensitivity of the record's probes with that line, and whether both
  still pass the record's gates.

The probes' peaks are in the records, so step two runs no model. Nothing is written in
``wiki/``::

    uv run python scripts/conformal_line.py --out papers/conformal-line/results.jsonl
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALPHAS = (0.05, 0.01)
WINDOW_TOKENS = 20  # about a probe's length: the probes' median is 14 words
WINDOWS = 2000  # half calibrate the line, half check its coverage
SEED = 0
PROBES = ("hard_negative", "positive_holdout")


def newest_verified(wiki_dir: Path) -> dict[str, list[dict]]:
    """The newest record of each MLP neuron pole, where it is VERIFIED, by model slug.

    Args:
        wiki_dir: The wiki folder.

    Returns:
        Each model's records, each with its ``_slug``, sorted by slug.
    """
    newest: dict[Path, Path] = {}
    for path in sorted((wiki_dir / "experiments").glob("*/L*/N*/*.json")):
        newest[path.parent] = path  # names sort by time, so the last one is the newest
    out: dict[str, list[dict]] = {}
    for folder, path in sorted(newest.items()):
        record = json.loads(path.read_text())
        ident = record["submission"]["identifier"]
        if record["verdict"]["status"] != "VERIFIED" or ident.get("kind") != "mlp_neuron_pole":
            continue
        slug = str(folder.relative_to(wiki_dir / "experiments"))
        out.setdefault(slug.split("/")[0], []).append(record | {"_slug": slug})
    return out


def conformal_line(peaks: list[float], alpha: float) -> float:
    """The split-conformal line: just above the ⌈(n + 1)(1 − α)⌉-th smallest of ``peaks``.

    Conformal prediction bounds the chance that a new peak is *above* that quantile. A probe
    fires when its peak *reaches* the line, so the line is the next float up. Then the bound
    holds for the review's rule even where peaks tie, as a layer-0 pole's often do.

    Args:
        peaks: The calibration windows' peak strengths.
        alpha: The share of new windows that may cross the line.

    Returns:
        The line, or infinity when too few windows calibrate it.

    Examples:
        >>> line = conformal_line([float(i) for i in range(1, 20)], 0.1)
        >>> 18.0 < line < 18.000001
        True
        >>> conformal_line([1.0, 2.0], 0.1)
        inf
    """
    rank = math.ceil((len(peaks) + 1) * (1 - alpha))
    if rank > len(peaks):
        return math.inf
    return math.nextafter(sorted(peaks)[rank - 1], math.inf)


def probe_rates(record: dict, line: float) -> dict:
    """The record's probes read again with ``line``: their pass rates, and the gates.

    A probe fires when its peak reaches the line, as in the review. A probe that wasn't
    measured counts in no rate. The rates weight near-duplicate probes as the review does
    (``core.stats.effective_counts``), so the record's own line gives its stored rates.

    Args:
        record: The record.
        line: The activation line.

    Returns:
        ``specificity``, ``sensitivity`` and ``holds`` (both rates reach the record's gates).
        The rates group probes by kind, as the review does, so an inverse claim's hard
        negatives are its probes predicted active. Then ``quiet``: the raw count of probes
        predicted quiet, whatever their kind, and ``quiet_fired``: how many reach the line.
        Their share is what the conformal guarantee bounds on corpus text.
    """
    from openinterpretability.core.stats import effective_counts

    policy = record["verdict"]["policy"]
    probes = [t for t in record["verdict"]["adversarial_test_results"]
              if t["kind"] in PROBES and t["activation"] is not None]  # fmt: skip
    rates = {}
    for name, kind in (("specificity", "hard_negative"), ("sensitivity", "positive_holdout")):
        own = [t for t in probes if t["kind"] == kind]  # by kind, as the review counts them
        passed = [(t["activation"] >= line) == t["expected_active"] for t in own]
        found = effective_counts([t["prompt"] for t in own], passed,
                                 policy["probe_cluster_threshold"])  # fmt: skip
        rates[name] = math.nan if found.rate is None else found.rate
    holds = (rates["specificity"] >= policy["min_specificity"]
             and rates["sensitivity"] >= policy["min_sensitivity"])  # fmt: skip
    quiet = [t["activation"] >= line for t in probes if not t["expected_active"]]
    return rates | {"holds": holds, "quiet": len(quiet), "quiet_fired": sum(quiet)}


def record_line(record: dict) -> float:
    """The record's own line: its policy's share of its ``max_activation``.

    Args:
        record: The record.

    Returns:
        The line.
    """
    share = record["verdict"]["policy"]["activation_threshold"]
    return share * record["submission"]["max_activation"]


def split(windows: list[str], seed: int = SEED) -> tuple[list[str], list[str]]:
    """The windows shuffled with ``seed``, cut into calibration and held-out halves.

    Args:
        windows: The corpus windows.
        seed: The shuffle's seed.

    Returns:
        The calibration windows and the held-out windows.

    Examples:
        >>> cal, held = split(list("abcdef"))
        >>> sorted(cal + held) == list("abcdef") and len(cal) == 3
        True
    """
    order = list(windows)
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    return order[:half], order[half:]


def pole_row(record: dict, cal: list[float], held: list[float]) -> dict:
    """One pole's row: its own line and each conformal line, with what each changes.

    Args:
        record: The pole's newest record.
        cal: The pole's peaks on the calibration windows.
        held: Its peaks on the held-out windows.

    Returns:
        The row.
    """
    ident = record["submission"]["identifier"]
    own = record_line(record)
    row = {"table": "pole", "slug": record["_slug"], "layer": ident["layer"],
           "pole": ident["pole"], "own_line": round(own, 6),
           "own": probe_rates(record, own),
           "own_cover": sum(p < own for p in held) / len(held)}  # fmt: skip
    for alpha in ALPHAS:
        line = conformal_line(cal, alpha)
        row[f"a{alpha}"] = probe_rates(record, line) | {
            "line": line,
            "cover": sum(p < line for p in held) / len(held),
        }
    return row


def window_peaks(runner, windows: list[str], records: list[dict]) -> list[list[float]]:
    """Each pole's peak strength on each window, from one forward pass per window.

    The peak skips the first token for a pole with a first-token floor, as the review does
    (``harness.units.peak_strength``).

    Args:
        runner: The model.
        windows: The texts.
        records: The poles' records.

    Returns:
        For each pole, its peak on each window.
    """
    from openinterpretability.harness.hooks import pole_strength

    idents = [r["submission"]["identifier"] for r in records]
    skips = [int(bool(r["submission"].get("first_token_floor"))) for r in records]
    layers = sorted({i["layer"] for i in idents})
    peaks: list[list[float]] = [[] for _ in records]
    for text in windows:
        acts = runner.layer_activations([text], layers)
        for k, (ident, skip) in enumerate(zip(idents, skips, strict=True)):
            act = acts[ident["layer"]][0, skip:, ident["neuron_idx"]]
            peaks[k].append(float(pole_strength(act.float(), ident["pole"]).max()))
    return peaks


def model_rows(model: str, records: list[dict], docs: list[str], device: str) -> list[dict]:
    """The rows of one model: a summary row, then a row for each of its poles.

    Args:
        model: The model's name.
        records: Its poles' records.
        docs: The corpus's documents.
        device: Where the model runs.

    Returns:
        The rows.
    """
    from openinterpretability.cli import _load_runner
    from openinterpretability.harness.explanation_scores import cut_windows

    runner = _load_runner(model, device, "float32", None)
    windows = cut_windows(runner, docs, WINDOW_TOKENS)[: WINDOWS * 4]
    cal, held = split(windows)
    cal, held = cal[: WINDOWS // 2], held[: WINDOWS // 2]
    peaks_cal, peaks_held = window_peaks(runner, cal, records), window_peaks(runner, held, records)
    rows = [pole_row(r, c, h) | {"model": model}
            for r, c, h in zip(records, peaks_cal, peaks_held, strict=True)]  # fmt: skip
    summary = {"table": "model", "model": model, "poles": len(records),
               "calibration": len(cal), "held_out": len(held)}  # fmt: skip
    return [summary, *rows]


@click.command()
@click.option("--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("wiki"))
@click.option(
    "--corpus", type=click.Path(dir_okay=False, path_type=Path), default=Path("corpus.jsonl")
)
@click.option("--device", default="auto", show_default=True)
@click.option("--model", "models", multiple=True, help="Only these model slugs.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(wiki_dir, corpus, device, models, out):  # pragma: no cover - loads models
    """Measure a conformal line for each VERIFIED pole, and re-read its probes.

    Args:
        wiki_dir: The wiki folder.
        corpus: The JSONL corpus that the windows come from.
        device: Where each model runs.
        models: Only these model slugs, for example ``EleutherAI--pythia-70m``.
        out: The JSONL file for the rows.
    """
    docs = [json.loads(line)["text"] for line in corpus.read_text().splitlines()]
    rows = []
    for slug, records in newest_verified(wiki_dir).items():
        if models and slug not in models:
            continue
        model = records[0]["submission"]["identifier"]["model_name"]
        click.echo(f"{model}: {len(records)} VERIFIED poles", err=True)
        rows += model_rows(model, records, docs, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))  # each model as it ends
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
