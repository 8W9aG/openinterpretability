"""Grow GPT-2's wiring circuit by attribution, and measure it with its heads (#4284, part B).

The pilot (#4283, ``docs/analyses/circuit-pilot.md``) measured C1's five neurons alone,
and they carried no measurable share of two-token word completion. This uses
``openinterp/harness/circuit_measures.py`` to measure the whole cluster, heads included.
Then it grows the cluster by attribution patching on the task, to each size in
:data:`SIZES`. It measures each size against a null of random sets like it::

    uv run python scripts/circuit_grown.py --out docs/analyses/circuit-grown.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from circuit_study import MODEL, predictable, two_token_words, word_task

from openinterp.harness import circuit_measures as cm

CLUSTER = ["L4/N2999", "L5/N1208", "L5/N2567", "L5/N616", "L6/N37", "L7/H10", "L7/H2"]
SIZES = (7, 25, 50, 100, 200)
NULL_SETS = 32


@click.command()
@click.option("--device", default="auto", show_default=True)
@click.option("--pairs", default=120, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(device: str, pairs: int, seed: int, out: Path) -> None:
    """Measure C1 with its heads, then grown by attribution, against a random null."""
    from openinterp.cli import _load_runner

    runner = _load_runner(MODEL, device, "float32", None)
    words = predictable(runner, two_token_words(runner.tokenizer, seed))
    task = word_task(runner.tokenizer, pairs, seed, words)
    task = cm.PairTask(task.clean.tolist(), task.corrupted.tolist(), task.good.tolist(),
                       task.bad.tolist())  # fmt: skip
    means = cm.task_means(runner, task)
    scores = cm.attribution(runner, task)
    seed_members = [cm.Member.parse(m) for m in CLUSTER]
    rows = []
    for size in SIZES:
        members = cm.grow(seed_members, scores, size)
        found = cm.measures(runner, task, means, members)
        null = cm.null_necessity(runner, task, means, members, NULL_SETS, seed)
        p95 = null[int(0.95 * len(null)) - 1]
        rows.append({"table": "size", "size": size, "heads": sum(m.kind == "H" for m in members),
                     "faithfulness": round(found["faithfulness"], 4),
                     "necessity": round(found["necessity"], 4),
                     "null_necessity_p95": round(p95, 4),
                     "beats_null": found["necessity"] > p95})  # fmt: skip
        click.echo(rows[-1], err=True)
    rows.append({"table": "circuit", "model": MODEL, "cluster": CLUSTER, "words": words,
                 "top": [str(m) for m in cm.grow(seed_members, scores, 25)]})  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {len(rows)} rows → {out}", err=True)


if __name__ == "__main__":
    main()
