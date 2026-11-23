"""Do VERIFIED explanations keep through a sequence of narrow fine-tunes (#5125)?

The script fine-tunes ``--base`` and its ``-Instruct`` model on one sequence of narrow tasks
(``interference/tuning.py``). The tasks are SST-2 sentiment, then XSum summaries, then GSM8K
maths. Each stage trains every weight. Before the first stage and after each stage, it
measures three things. It runs the keep test of ``docs/analyses/base-instruct.md`` on each
VERIFIED neuron pole of the base model. It scores ARC-Easy, zero-shot, with a Wilson
interval. And it gives the held-out loss of each task. Each stage also gives the KL on its
own held-out data between the model before and after it. The keep test scales each pole by a
ratio. It is the pole's maximum on the stage model over its maximum on the base model, on
one sample of ``corpus.jsonl``. The script writes no record and saves no weights.

    uv run python scripts/sequential_tuning.py --device cuda \\
        --out papers/sequential-tuning/results.jsonl
"""

from __future__ import annotations

import copy
import gc
import json
import math
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
    """Print ``message`` to stderr after the minutes since ``started``.

    Args:
        message: The line to print.
        started: When the run started.
    """
    print(f"[{(time.monotonic() - started) / 60:6.1f} min] {message}", file=sys.stderr, flush=True)


def _free(device: str) -> None:
    """Return freed memory to the device, so the next model or copy fits.

    Args:
        device: The device of the models.
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def by_layer(records: list) -> dict[int, list]:
    """``{layer: [identifier, ...]}`` of the records' poles, in layer order.

    Args:
        records: The VERIFIED records.
    """
    out: dict[int, list] = defaultdict(list)
    for record in records:
        out[record.submission.identifier.layer].append(record.submission.identifier)
    return dict(sorted(out.items()))


def columns(runner, batches, layers: dict[int, list]) -> dict[int, tuple[list, torch.Tensor]]:
    """``{layer: (neurons, activations)}`` over the sample, for each layer's poles.

    Args:
        runner: The model's ``NeuronHookRunner``.
        batches: The corpus sample, from ``sample_windows``.
        layers: The poles of each layer, from :func:`by_layer`.
    """
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


def binomial_p(k: int, n: int) -> float:
    """The exact two-sided binomial p of ``k`` successes in ``n`` at one half.

    Args:
        k: The successes.
        n: The trials.

    Returns:
        The p value, at most 1.
    """
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _p(value: float) -> float:
    """A p value to three significant figures, so that a very small one stays above zero.

    Args:
        value: The p value.
    """
    return float(f"{value:.3g}")


def compare(rows: list[dict], reproduced: set[str], stages: int) -> list[dict]:
    """The paired tests of the paper, from the ``pole`` rows.

    Args:
        rows: Every ``pole`` row.
        reproduced: The slugs that keep on the base model before any stage.
        stages: The number of fine-tune stages.

    Returns:
        - One ``change`` row for each stage of each arm: the poles that stop and start
          keeping at that stage, with an exact McNemar p, and the mean change of each gate.
        - One ``between`` row for each stage: the poles that keep on one model only.
        - One ``depth`` row for each arm and third of the layers, after the last stage.
        - One ``overlap`` row: the poles that fail after the base model's first stage,
          against those that fail on the instruct model before any stage.
    """
    at = {(r["arm"], r["stage"], r["slug"]): r for r in rows if r["slug"] in reproduced}
    keeps = lambda arm, stage, slug: at[(arm, stage, slug)]["keeps"]  # noqa: E731
    out = []
    for arm in ARMS:
        for stage in range(1, stages + 1):
            lost = sum(keeps(arm, stage - 1, x) and not keeps(arm, stage, x) for x in reproduced)
            gained = sum(keeps(arm, stage, x) and not keeps(arm, stage - 1, x)
                         for x in reproduced)  # fmt: skip
            row = {"table": "change", "arm": arm, "stage": stage, "lost": lost,
                   "gained": gained, "p_keep": _p(binomial_p(lost, lost + gained))}  # fmt: skip
            for gate in ("specificity", "sensitivity"):
                deltas = [at[(arm, stage, x)][gate] - at[(arm, stage - 1, x)][gate]
                          for x in reproduced]  # fmt: skip
                row[f"mean_{gate}_change"] = round(sum(deltas) / len(deltas), 4)
            out.append(row)
    for stage in range(stages + 1):
        base = sum(keeps("base", stage, x) and not keeps("instruct", stage, x) for x in reproduced)
        tuned = sum(keeps("instruct", stage, x) and not keeps("base", stage, x)
                    for x in reproduced)  # fmt: skip
        out.append({"table": "between", "stage": stage, "base_only": base,
                    "instruct_only": tuned, "p": _p(binomial_p(base, base + tuned))})  # fmt: skip
    for arm in ARMS:
        for low in (0, 8, 16):
            got = [x for x in reproduced if low <= at[(arm, stages, x)]["layer"] < low + 8]
            kept = sum(keeps(arm, stages, x) for x in got)
            out.append({"table": "depth", "arm": arm, "layers": [low, low + 7],
                        "poles": len(got), "keeps": kept})  # fmt: skip
    first = {x for x in reproduced if not keeps("base", 1, x)}
    tuned = {x for x in reproduced if not keeps("instruct", 0, x)}
    both, n = len(first & tuned), len(reproduced)
    ways = sum(math.comb(len(tuned), k) * math.comb(n - len(tuned), len(first) - k)
               for k in range(both, min(len(first), len(tuned)) + 1))  # fmt: skip
    tail = ways / math.comb(n, len(first))  # the hypergeometric upper tail
    out.append({"table": "overlap", "base_stage_1_fail": len(first),
                "instruct_stage_0_fail": len(tuned), "both": both,
                "expected": round(len(first) * len(tuned) / n, 2),
                "p_hypergeometric": _p(tail)})  # fmt: skip
    return out


def analysed(rows: list[dict], stages: int) -> list[dict]:
    """``rows`` with their first-failure and comparison rows made again from the pole rows.

    Args:
        rows: The results: the summary, stage and pole rows, with any analysis rows.
        stages: The number of fine-tune stages.

    Returns:
        The summary and stage rows, then the analysis rows, then the pole rows.
    """
    poles = [r for r in rows if r["table"] == "pole"]
    reproduced = {r["slug"] for r in poles if (r["arm"], r["stage"]) == ("base", 0) and r["keeps"]}
    kept = [r for r in rows if r["table"] in ("summary", "stage")]
    return [
        *kept,
        *first_failures(poles, reproduced, stages),
        *compare(poles, reproduced, stages),
        *poles,
    ]


def task_data(count: int) -> dict[str, dict]:
    """``{task: {"train", "heldout"}}``: the sampled pairs of each task, drawn with :data:`SEED`.

    Args:
        count: The training pairs of each task.
    """
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
    """ARC-Easy accuracy with its interval, and each task's held-out loss, at one stage.

    Args:
        runner: The model at this stage.
        data: Each task's pairs, from :func:`task_data`.
        arc_rows: The ARC-Easy rows.
        device: The device of the model.
    """
    correct, items = arc_accuracy(runner.model, runner.tokenizer, arc_rows, device)
    losses = {name: round(heldout_loss(runner.model, runner.tokenizer, d["heldout"], device), 4)
              for name, d in data.items()}  # fmt: skip
    return {"arc_correct": correct, "arc_items": items,
            "arc_accuracy": round(correct / items, 4) if items else None,
            "arc_ci95": wilson(correct, items), "task_loss": losses}  # fmt: skip


def stage_rows(runner, arm: str, stage: int, run: dict) -> tuple[dict, list[dict]]:
    """Tune ``runner`` for ``stage`` when it is above 0, then measure it.

    Args:
        runner: The arm's model, changed in place.
        arm: ``base`` or ``instruct``.
        stage: 0 before any fine-tune, else the stage to train.
        run: The run's fixed inputs: ``data``, ``arc``, ``device``, ``started``,
            ``records``, ``batches``, ``layers`` and the base model's columns, ``base``.

    Returns:
        The stage row, without its keep summary, and the stage's ``pole`` rows.
    """
    device, started = run["device"], run["started"]
    row: dict = {"table": "stage", "arm": arm, "stage": stage,
                 "task": TASKS[stage - 1].name if stage else None}  # fmt: skip
    if stage:
        row.update(tune(runner, run["data"][TASKS[stage - 1].name], f"{arm} stage {stage}",
                        stage, device, started))  # fmt: skip
    row.update(measure_stage(runner, run["data"], run["arc"], device))
    progress(f"{arm} stage {stage}: ARC {row['arc_accuracy']}, "
             f"losses {row['task_loss']}, KL {row.get('kl')}", started)  # fmt: skip
    first = (arm, stage) == ("base", 0)  # the base model unchanged: its columns are cached
    now = run["base"] if first else columns(runner, run["batches"], run["layers"])
    return row, pole_rows(runner, run["records"], run["base"], now, arm, stage, started)


def reanalyse(out: Path) -> None:
    """Make the analysis rows of the results file ``out`` again, from its pole rows.

    Args:
        out: The results file, rewritten in place.
    """
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    out.write_text("".join(json.dumps(r) + "\n" for r in analysed(rows, len(TASKS))))


def summary_row(base_name: str, count: int, poles: int, reproduced: int) -> dict:
    """The ``summary`` row: the run's settings, its counts and each dataset's commit.

    Args:
        base_name: The base model.
        count: The training pairs of each stage.
        poles: The VERIFIED poles read.
        reproduced: The poles that keep on the base model before any stage.

    Returns:
        The row.
    """
    return {"table": "summary", "base": base_name, "instruct": f"{base_name}-Instruct",
            "tasks": [t.name for t in TASKS], "train_pairs": count, "epochs": EPOCHS, "lr": LR,
            "batch": BATCH * ACCUMULATE, "seed": SEED, "poles": poles, "reproduced": reproduced,
            "revisions": {t.name: t.revision for t in (*TASKS, ARC)}}  # fmt: skip


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--base", "base_name", default=BASE, show_default=True)
@click.option("--corpus", default="corpus.jsonl", show_default=True)
@click.option("--train", "count", default=TRAIN, show_default=True, help="Pairs per stage.")
@click.option("--limit", type=int, default=None, help="Read only the first N poles: a timing test.")
@click.option("--arc-limit", type=int, default=None, help="Score only the first N ARC items.")
@click.option("--analyse", is_flag=True, help="Remake the analysis rows of --out; train nothing.")
def main(wiki_dir: Path, out: Path, device: str, base_name: str, corpus: str, count: int,
         limit: int | None, arc_limit: int | None, analyse: bool) -> None:  # fmt: skip
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
        analyse: Only make the analysis rows of ``out`` again.
    """
    from openinterpretability.cli import _load_runner

    if analyse:
        reanalyse(out)
        return
    started = time.monotonic()
    torch.manual_seed(SEED)
    records = verified_records(wiki_dir, base_name)[:limit]
    layers, arc_rows = by_layer(records), load_rows(ARC, "heldout")[:arc_limit]
    progress(f"{len(records)} poles, {len(arc_rows)} ARC items, {count} pairs a stage", started)
    base = _load_runner(base_name, device, "float32", None)
    batches = sample_windows(base.tokenizer, corpus=corpus)
    run = {"data": task_data(count), "arc": arc_rows, "device": device, "started": started,
           "records": records, "batches": batches, "layers": layers,
           "base": columns(base, batches, layers)}  # fmt: skip
    progress("base sample read", started)
    poles: list[dict] = []
    stages: list[dict] = []
    reproduced: set[str] = set()
    for arm in ARMS:
        runner = base if arm == "base" else _load_runner(f"{base_name}-Instruct", device,
                                                         "float32", None)  # fmt: skip
        for stage in range(len(TASKS) + 1):
            row, got = stage_rows(runner, arm, stage, run)
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
    head = summary_row(base_name, count, len(records), len(reproduced))
    done = analysed([head, *stages, *poles], len(TASKS))
    out.write_text("".join(json.dumps(r) + "\n" for r in done))
    click.echo(json.dumps(head))


if __name__ == "__main__":
    main()
