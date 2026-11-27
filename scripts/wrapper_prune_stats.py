"""The tables of papers/wrapper-prune/README.md, from the rows of scripts/wrapper_prune.py (#5163).

For each pruned condition, it counts the poles that fail on the instruct model and pass again
(*recovered*). It also counts the poles that keep on the instruct model and fail once pruned
(*lost*). Top pruning and each random prune of the same K are compared on the same poles. The exact
McNemar test runs once for the recovered poles and once for the lost ones. It prints one
JSON line for each condition, and one for each layer's share of the top neurons.

``--unit`` and ``--loss`` pick the variant of ``scripts/wrapper_prune.py`` (#7071). Rows from
before the variants have neither field, and count as ``write`` and ``base``. An input prune
switches a pruned neuron's own poles off, so those poles count as neither recovered nor lost.
With the failed loss, the recovered poles are also counted in each half of the split.

    uv run python scripts/wrapper_prune_stats.py papers/wrapper-prune/results.jsonl
    uv run python scripts/wrapper_prune_stats.py papers/wrapper-prune/results.jsonl --unit input
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_change_stats import mcnemar  # noqa: E402

SLUG_NEURON = re.compile(r"/L(\d+)/N(\d+)-")  # the layer and neuron in a pole's slug


def variant_rows(rows: list[dict], unit: str = "write", loss: str = "base") -> list[dict]:
    """The rows of one variant. Rows with no ``unit`` or ``loss`` are ``write`` and ``base``.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
        unit: What the prune zeroed.
        loss: The old task of the score.
    """
    return [r for r in rows if r.get("unit", "write") == unit and r.get("loss", "base") == loss]


def neuron_of(slug: str) -> tuple[int, int]:
    """The ``(layer, neuron)`` of a pole's slug.

    Args:
        slug: A slug such as ``Qwen--Qwen2.5-0.5B/L3/N12-positive``.
    """
    layer, neuron = SLUG_NEURON.search(slug).groups()
    return int(layer), int(neuron)


def switched_off(none: dict[str, bool], summary: dict, unit: str) -> set[str]:
    """The poles that a condition switches off: an input prune zeroed their own neuron.

    Args:
        none: ``{slug: keeps}`` on the instruct model.
        summary: The condition's summary row, which names its pruned neurons.
        unit: The variant's prune. A write prune switches no pole off.
    """
    if unit != "input":
        return set()
    pruned = {tuple(n) for n in summary["neurons"]}
    return {s for s in none if neuron_of(s) in pruned}


def pole_counts(none: dict[str, bool], got: dict[str, bool], gone: set[str]) -> tuple[set, ...]:
    """The failed, kept, recovered and lost poles of one condition, without ``gone``.

    Args:
        none: ``{slug: keeps}`` on the instruct model.
        got: ``{slug: keeps}`` after the prune.
        gone: The poles to leave out.

    Returns:
        ``(failed, kept, recovered, lost)``.
    """
    failed = {s for s, keeps in none.items() if not keeps} - gone
    kept = {s for s, keeps in none.items() if keeps} - gone
    return failed, kept, {s for s in failed if got[s]}, {s for s in kept if not got[s]}


def pole_sets(rows: list[dict]) -> dict[tuple, dict[str, bool]]:
    """``{(condition, K, seed): {slug: keeps}}`` over the poles that reproduce on the base.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
    """
    out: dict[tuple, dict[str, bool]] = {}
    for r in rows:
        if r["table"] == "pole" and r["base_passes"]:
            out.setdefault((r["condition"], r["k"], r["seed"]), {})[r["slug"]] = r["keeps"]
    return out


def changes(rows: list[dict], unit: str = "write", loss: str = "base") -> list[dict]:
    """For each pruned condition, its recovered and lost poles, against the instruct model.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
        unit: The variant's prune. With ``input``, a pruned neuron's own poles are left out.
        loss: The variant's old task. With ``failed``, each half of the split is counted too.

    Returns:
        One row for each condition. A random row also gives its McNemar tests against the top
        prune of the same K. ``recovered_p`` is on the failed poles, and ``lost_p`` on the kept.
        With the failed loss, ``held_recovered_p`` is on the held-out half.
    """
    rows = variant_rows(rows, unit, loss)
    sets = pole_sets(rows)
    none = sets["none", 0, None]
    summaries = {(r["condition"], r["k"], r["seed"]): r for r in rows if r["table"] == "summary"}
    split = next((r for r in rows if r["table"] == "split"), None)
    held = set(split["held"]) if split else set()
    off = lambda key: switched_off(none, summaries[key], unit)  # noqa: E731
    counts = lambda key: pole_counts(none, sets[key], off(key))  # noqa: E731
    out = []
    for key in sorted(sets, key=lambda k: (k[1], k[0] != "top", k[2] or 0)):
        if key[0] == "none":
            continue
        failed, kept, recovered, lost = counts(key)
        row = {"table": "change", "unit": unit, "loss": loss, "condition": key[0], "k": key[1],
               "seed": key[2], "failed_on_instruct": len(failed), "recovered": len(recovered),
               "kept_on_instruct": len(kept), "lost": len(lost),
               "held_kl": summaries[key]["held_kl"]}  # fmt: skip
        row |= {"switched_off": len(off(key))} if unit == "input" else {}
        if split:
            row["held_failed"] = len(failed & held)
            row["held_recovered"] = len(recovered & held)
            row["scored_recovered"] = len(recovered - held)
        if key[0] == "random":
            top = sets["top", key[1], None]
            # Both conditions are compared on the poles that neither switches off.
            both = off(("top", key[1], None)) | off(key)
            top_recovered = {s for s in failed if top[s]} - both
            top_lost = {s for s in kept if not top[s]} - both
            recovered, lost = recovered - both, lost - both
            p_rec = mcnemar(len(top_recovered - recovered), len(recovered - top_recovered))
            p_lost = mcnemar(len(top_lost - lost), len(lost - top_lost))
            row["recovered_p"] = float(f"{p_rec:.3g}")  # three significant figures
            row["lost_p"] = float(f"{p_lost:.3g}")
            if split:
                p_held = mcnemar(len((top_recovered - recovered) & held),
                                 len((recovered - top_recovered) & held))  # fmt: skip
                row["held_recovered_p"] = float(f"{p_held:.3g}")
        out.append(row)
    return out


def layers_of_top(rows: list[dict], unit: str = "write", loss: str = "base") -> list[dict]:
    """For each K, how many of the top neurons sit in each layer, most first.

    Args:
        rows: The rows of ``scripts/wrapper_prune.py``.
        unit: The variant's prune.
        loss: The variant's old task.
    """
    rows = variant_rows(rows, unit, loss)
    return [
        {"table": "top_layers", "k": r["k"],
         "layers": Counter(layer for layer, _ in r["neurons"]).most_common()}
        for r in rows if r["table"] == "summary" and r["condition"] == "top"
    ]  # fmt: skip


@click.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--unit", type=click.Choice(("write", "input")), default="write", show_default=True)
@click.option("--loss", type=click.Choice(("base", "failed")), default="base", show_default=True)
def main(results: Path, unit: str, loss: str) -> None:
    """Print the recovered and lost poles of each condition, and the layers of the top neurons.

    Args:
        results: The rows of ``scripts/wrapper_prune.py``.
        unit: The variant's prune.
        loss: The variant's old task.
    """
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    for row in changes(rows, unit, loss) + layers_of_top(rows, unit, loss):
        click.echo(json.dumps(row))


if __name__ == "__main__":
    main()
