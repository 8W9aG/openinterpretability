"""Do VERIFIED explanations keep through a sequence of narrow fine-tunes (#5125)?

The script fine-tunes ``--base`` and its ``-Instruct`` model on one sequence of narrow tasks
(``interference/tuning.py``): SST-2 sentiment, then XSum summaries, then GSM8K maths. Each
stage trains every weight. Before the first stage and after each stage, it measures three
things. It runs the keep test of ``docs/analyses/base-instruct.md`` on each VERIFIED neuron
pole of the base model. It scores ARC-Easy, zero-shot, with a Wilson interval. And it gives
the held-out loss of each task. Each stage also gives the KL on its own held-out data
between the model before and after it. The scale ratio of the keep test is the pole's
maximum on the stage model over its maximum on the base model, on one sample of
``corpus.jsonl``. The script writes no record and saves no weights.

    uv run python scripts/sequential_tuning.py --device cuda \\
        --out papers/sequential-tuning/results.jsonl
"""

from __future__ import annotations

import copy
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.interference.keep import (  # noqa: E402
    layer_columns,
    pearson,
    pole_max,
    readings,
    sample_windows,
    verified_records,
    wilson,
)
from openinterpretability.interference.tuning import (  # noqa: E402
    ARC,
    TASKS,
    arc_accuracy,
    heldout_loss,
    load_rows,
    pair,
    sample,
    stage_kl,
    train_stage,
)

BASE = "Qwen/Qwen2.5-0.5B"
ARMS = ("base", "instruct")
SEED = 0
TRAIN = 2_000  # training pairs for each stage
HELDOUT = 200  # held-out pairs for each task's loss
KL_PAIRS = 50  # held-out pairs for each stage's KL
EPOCHS = 2
LR = 2e-5  # Luo et al. (2023)
BATCH, ACCUMULATE = 1, 8


def progress(message: str, started: float) -> None:
    """Print ``message`` to stderr after the minutes since ``started``."""
    print(f"[{(time.monotonic() - started) / 60:6.1f} min] {message}", file=sys.stderr, flush=True)


def _free(device: str) -> None:
    """Return freed memory to the device, so the next model or copy fits."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def by_layer(records: list) -> dict[int, list]:
    """``{layer: [identifier, ...]}`` of the records' poles, in layer order."""
    out: dict[int, list] = defaultdict(list)
    for record in records:
        out[record.submission.identifier.layer].append(record.submission.identifier)
    return dict(sorted(out.items()))


def columns(runner, batches, layers: dict[int, list]) -> dict[int, tuple[list, torch.Tensor]]:
    """``{layer: (neurons, activations)}`` over the sample, for each layer's poles."""
    out = {}
    for layer, idents in layers.items():
        neurons = sorted({i.neuron_idx for i in idents})
        out[layer] = (neurons, layer_columns(runner, batches, layer, neurons))
    return out


def pole_rows(runner, records: list, base: dict, now: dict, arm: str, stage: int,
              started: float) -> list[dict]:  # fmt: skip
    """One ``pole`` row for each record, read on ``runner`` at ``stage`` of ``arm``.

    Args:
        runner: The model at this stage.
        records: The VERIFIED records.
        base: :func:`columns` of the base model before any stage.
        now: :func:`columns` of ``runner``.
        arm: ``base`` or ``instruct``.
        stage: 0 before any fine-tune, else the number of stages done.
        started: When the run started, for progress lines.

    Returns:
        The rows. ``ratio`` scales the record's own scale, as in ``base_instruct.py``.
    """
    rows = []
    for i, record in enumerate(records, 1):
        ident = record.submission.identifier
        neurons, a = base[ident.layer]
        b = now[ident.layer][1]
        col = neurons.index(ident.neuron_idx)
        top = pole_max(a[:, col], ident.pole)
        ratio = pole_max(b[:, col], ident.pole) / top if top > 0 else None
        read = readings(runner, record, ratio or 1.0)
        rows.append({
            "table": "pole", "arm": arm, "stage": stage, "slug": ident.slug,
            "layer": ident.layer, "pole": ident.pole, "ratio": ratio,
            "pearson": pearson(a[:, col], b[:, col]), "specificity": read["specificity"],
            "sensitivity": read["sensitivity"], "keeps": read["passes"],
        })  # fmt: skip
        if i % 50 == 0:
            progress(f"{arm} stage {stage}: pole {i}/{len(records)}", started)
    return rows


def stage_summary(rows: list[dict], reproduced: set[str]) -> dict:
    """The keep share and the median Pearson r of one stage's ``pole`` rows.

    Args:
        rows: The stage's ``pole`` rows.
        reproduced: The slugs that keep on the base model before any stage.

    Returns:
        ``keeps``, ``poles``, ``keep_share``, ``keep_ci95`` and ``median_pearson``, over the
        reproduced poles only.
    """
    got = [r for r in rows if r["slug"] in reproduced]
    kept = sum(r["keeps"] for r in got)
    rs = sorted(r["pearson"] for r in got if r["pearson"] is not None)
    return {"poles": len(got), "keeps": kept,
            "keep_share": round(kept / len(got), 4) if got else None,
            "keep_ci95": wilson(kept, len(got)),
            "median_pearson": rs[len(rs) // 2] if rs else None}  # fmt: skip


def first_failures(rows: list[dict], reproduced: set[str], stages: int) -> list[dict]:
    """For each arm, how many reproduced poles first fail at each stage.

    Args:
        rows: Every ``pole`` row.
        reproduced: The slugs that keep on the base model before any stage.
        stages: The number of fine-tune stages.

    Returns:
        One ``first_fail`` row for each arm. ``counts[s]`` is the number of poles whose
        first failure is at stage ``s``. ``never`` counts the poles that keep at every
        stage, and ``back`` the poles that fail at one stage and keep at a later one.
    """
    out = []
    for arm in ARMS:
        seen: dict[str, list[bool]] = defaultdict(lambda: [False] * (stages + 1))
        for r in rows:
            if r["arm"] == arm and r["slug"] in reproduced:
                seen[r["slug"]][r["stage"]] = r["keeps"]
        counts = [0] * (stages + 1)
        never = back = 0
        for keeps in seen.values():
            if all(keeps):
                never += 1
                continue
            first = keeps.index(False)
            counts[first] += 1
            back += any(keeps[first + 1 :])
        out.append({"table": "first_fail", "arm": arm, "counts": counts, "never": never,
                    "back": back})  # fmt: skip
    return out


def task_data(count: int) -> dict[str, dict]:
    """``{task: {"train", "heldout"}}``: the sampled pairs of each task, drawn with :data:`SEED`."""
    out = {}
    for task in TASKS:
        train = sample(load_rows(task, "train"), count, SEED)
        heldout = sample(load_rows(task, "heldout"), HELDOUT, SEED)
        out[task.name] = {"train": [pair(task, r) for r in train],
                          "heldout": [pair(task, r) for r in heldout]}  # fmt: skip
    return out


def tune(runner, pairs: dict, name: str, stage: int, device: str, started: float) -> dict:
    """Train ``runner``'s model on one task, and give the stage's KL and training losses.

    Args:
        runner: The model, changed in place.
        pairs: The task's ``train`` and ``heldout`` pairs.
        name: The stage's name in progress lines.
        stage: The stage's number, which also offsets the seed of the training order.
        device: The device of the model.
        started: When the run started, for progress lines.

    Returns:
        ``kl`` on the task's held-out pairs, the first and last step's loss, and the steps.
    """
    before = copy.deepcopy(runner.model).to("cpu", torch.bfloat16)
    losses = train_stage(runner.model, runner.tokenizer, pairs["train"], device=device,
                         epochs=EPOCHS, lr=LR, batch=BATCH, accumulate=ACCUMULATE,
                         seed=SEED + stage,
                         progress=lambda m: progress(f"{name}: {m}", started))  # fmt: skip
    _free(device)
    before = before.to(device, torch.float32)
    kl = stage_kl(before, runner.model, runner.tokenizer, pairs["heldout"][:KL_PAIRS], device)
    del before
    _free(device)
    return {"kl": round(kl, 5), "train_loss_first": round(losses[0], 4),
            "train_loss_last": round(losses[-1], 4), "steps": len(losses)}  # fmt: skip


def measure_stage(runner, data: dict, arc_rows: list, device: str) -> dict:
    """ARC-Easy accuracy with its interval, and each task's held-out loss, at one stage."""
    correct, items = arc_accuracy(runner.model, runner.tokenizer, arc_rows, device)
    losses = {name: round(heldout_loss(runner.model, runner.tokenizer, d["heldout"], device), 4)
              for name, d in data.items()}  # fmt: skip
    return {"arc_correct": correct, "arc_items": items,
            "arc_accuracy": round(correct / items, 4) if items else None,
            "arc_ci95": wilson(correct, items), "task_loss": losses}  # fmt: skip


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--base", "base_name", default=BASE, show_default=True)
@click.option("--corpus", default="corpus.jsonl", show_default=True)
@click.option("--train", "count", default=TRAIN, show_default=True, help="Pairs per stage.")
@click.option("--limit", type=int, default=None, help="Read only the first N poles: a timing test.")
@click.option("--arc-limit", type=int, default=None, help="Score only the first N ARC items.")
def main(wiki_dir: Path, out: Path, device: str, base_name: str, corpus: str, count: int,
         limit: int | None, arc_limit: int | None) -> None:  # fmt: skip
    """Fine-tune both models through the sequence, and write the rows after each stage.

    Args:
        wiki_dir: The registry's wiki.
        out: The JSON Lines file to write.
        device: The device of the models.
        base_name: The base model. Its instruct model is ``<base>-Instruct``.
        corpus: The corpus of the keep test's sample.
        count: The training pairs of each stage.
        limit: Read only this many poles, or every pole when None.
        arc_limit: Score only this many ARC items, or all of them when None.
    """
    from openinterpretability.cli import _load_runner

    started = time.monotonic()
    torch.manual_seed(SEED)
    records = verified_records(wiki_dir, base_name)[:limit]
    layers = by_layer(records)
    data = task_data(count)
    arc_rows = load_rows(ARC, "heldout")[:arc_limit]
    progress(f"{len(records)} poles, {len(arc_rows)} ARC items, {count} pairs a stage", started)
    base = _load_runner(base_name, device, "float32", None)
    batches = sample_windows(base.tokenizer, corpus=corpus)
    base_cols = columns(base, batches, layers)
    progress("base sample read", started)
    poles: list[dict] = []
    stages: list[dict] = []
    reproduced: set[str] = set()
    for arm in ARMS:
        runner = base if arm == "base" else _load_runner(f"{base_name}-Instruct", device,
                                                         "float32", None)  # fmt: skip
        for stage in range(len(TASKS) + 1):
            row: dict = {"table": "stage", "arm": arm, "stage": stage,
                         "task": TASKS[stage - 1].name if stage else None}  # fmt: skip
            if stage:
                row.update(tune(runner, data[TASKS[stage - 1].name], f"{arm} stage {stage}",
                                stage, device, started))  # fmt: skip
            row.update(measure_stage(runner, data, arc_rows, device))
            progress(f"{arm} stage {stage}: ARC {row['arc_accuracy']}, "
                     f"losses {row['task_loss']}, KL {row.get('kl')}", started)  # fmt: skip
            now = base_cols if (arm, stage) == ("base", 0) else columns(runner, batches, layers)
            got = pole_rows(runner, records, base_cols, now, arm, stage, started)
            if (arm, stage) == ("base", 0):
                reproduced = {r["slug"] for r in got if r["keeps"]}
            poles += got
            row.update(stage_summary(got, reproduced))
            stages.append(row)
            progress(f"{arm} stage {stage}: {row['keeps']}/{row['poles']} keep", started)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in [*stages, *poles]))
        runner = base = None  # the next arm needs only the base model's cached columns
        _free(device)
    head = {"table": "summary", "base": base_name, "instruct": f"{base_name}-Instruct",
            "tasks": [t.name for t in TASKS], "train_pairs": count, "epochs": EPOCHS, "lr": LR,
            "batch": BATCH * ACCUMULATE, "seed": SEED, "poles": len(records),
            "reproduced": len(reproduced),
            "revisions": {t.name: t.revision for t in (*TASKS, ARC)}}  # fmt: skip
    fails = first_failures(poles, reproduced, len(TASKS))
    out.write_text("".join(json.dumps(r) + "\n" for r in [head, *stages, *fails, *poles]))
    click.echo(json.dumps(head))


if __name__ == "__main__":
    main()
