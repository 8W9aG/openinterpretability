"""A pilot of the single-judge rescore: 200 records, more windows, two seeds (issue #4102).

The registry's explanation scores mix judges and use one seed of about 50 windows.
Window noise and judge differences can blur how well a score separates VERIFIED records
(``docs/analyses/verified-vs-detection.md``). This rescores a sample with one judge and
twice the windows, and measures how much of the spread is window noise.

* ``sample`` draws the latest record of 100 VERIFIED and 100 other neuron-pole units.
  Each arm is split over the models in proportion to the registry.
* ``emit`` writes two judge tasks per record (``score --emit-task``, two seeds) under
  ``exchange/rescore-pilot/``, with twice the default windows and held-out text.
* One ``explanation-judge`` subagent answers each prompt (a workflow, outside this script).
* ``collect`` scores the answered tasks (``score --task``) and writes one row per record
  and seed to ``docs/analyses/rescore-pilot.jsonl``.
* ``report`` writes ``docs/analyses/rescore-pilot.json``. It reads only files in git: the
  sample and the rows.

    uv run python scripts/rescore_pilot.py sample
    uv run python scripts/rescore_pilot.py emit
    uv run python scripts/rescore_pilot.py collect --judge claude-haiku-5-5
    uv run python scripts/rescore_pilot.py report
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))
from score_backfill import CLI, CORPUS, corpus_for, seed_of, sha256  # noqa: E402
from verified_vs_detection import BACKFILL, backfilled, normalise  # noqa: E402

WORK = Path("exchange/rescore-pilot")
SAMPLE = Path("docs/analyses/rescore-pilot-sample.jsonl")  # in git, so `report` reruns from it
ROWS = Path("docs/analyses/rescore-pilot.jsonl")
REPORT = Path("docs/analyses/rescore-pilot.json")
PER_ARM = 100
SEEDS = 2
PER_QUANTILE = 10  # twice the default of 5 active windows per bin
MAX_WINDOWS = 8000  # twice the default held-out text
KINDS = ("detection", "fuzzing")
BOOTSTRAP = 2000


def latest_records(wiki_dir: Path = Path("wiki")) -> list[dict]:
    """The newest neuron-pole record of each unit, with its status, model and old scores."""
    from openinterpretability.core.schemas import AuditRecord

    extra = backfilled(BACKFILL)
    best: dict[str, dict] = {}
    for path in sorted((wiki_dir / "experiments").rglob("*.json")):
        record = AuditRecord.model_validate_json(path.read_text())
        verdict, sub = record.verdict, record.submission
        if verdict is None or sub.identifier.kind != "mlp_neuron_pole":
            continue
        scores = verdict.explanation_scores
        if scores is None or scores.detection is None:
            scores = extra.get(str(path), scores)
        row = {
            "record": str(path), "slug": sub.identifier.slug, "status": verdict.status,
            "model": sub.identifier.model_name, "time": verdict.timestamp.isoformat(),
            "corpus_sha256": sub.provenance.corpus_sha256 if sub.provenance else None,
            "old_judge": normalise(scores.judge_model) if scores else None,
            "old": {k: getattr(getattr(scores, k, None), "balanced_accuracy", None)
                    for k in KINDS},
        }  # fmt: skip
        if row["slug"] not in best or row["time"] > best[row["slug"]]["time"]:
            best[row["slug"]] = row
    return sorted(best.values(), key=lambda r: r["slug"])


def quotas(rows: list[dict], total: int) -> dict[str, int]:
    """``total`` split over the rows' models in proportion, by largest remainder."""
    counts = Counter(r["model"] for r in rows)
    exact = {m: total * n / len(rows) for m, n in counts.items()}
    out = {m: math.floor(x) for m, x in exact.items()}
    for m in sorted(exact, key=lambda m: out[m] - exact[m])[: total - sum(out.values())]:
        out[m] += 1
    return out


def draw(rows: list[dict], per_arm: int = PER_ARM, seed: int = 0) -> list[dict]:
    """``per_arm`` VERIFIED and ``per_arm`` other rows, each arm split over models."""
    rng, out = random.Random(seed), []
    for verified in (True, False):
        arm = [r for r in rows if (r["status"] == "VERIFIED") == verified]
        for model, n in sorted(quotas(arm, per_arm).items()):
            pool = [r for r in arm if r["model"] == model]
            out += rng.sample(pool, min(n, len(pool)))
    return out


def replace(chosen: list[dict], rows: list[dict], bad: set[str], seed: int = 1) -> list[dict]:
    """``chosen`` with each row whose corpus is in ``bad`` swapped for an unchosen row.

    The new row has the same model and arm, and a corpus not in ``bad``. The draw is
    seeded, so the same inputs give the same sample.
    """
    rng, taken = random.Random(seed), {r["record"] for r in chosen}
    out = []
    for row in chosen:
        if row["corpus_sha256"] in bad:
            verified = row["status"] == "VERIFIED"
            pool = [r for r in rows if r["record"] not in taken and r["model"] == row["model"]
                    and (r["status"] == "VERIFIED") == verified
                    and r["corpus_sha256"] not in bad]  # fmt: skip
            if not pool:
                continue
            row = rng.choice(pool)
            taken.add(row["record"])
        out.append(row)
    return out


def read_sample() -> list[dict]:
    """The sampled records, one per line of :data:`SAMPLE`."""
    return [json.loads(line) for line in SAMPLE.read_text().splitlines() if line.strip()]


def task_dir(record: str, seed: int) -> Path:
    """``exchange/rescore-pilot/<pole slug>/<experiment id>/seed<k>/``."""
    path = Path(record)
    return WORK / path.relative_to("wiki/experiments").parent / path.stem / f"seed{seed}"


@click.group()
def main() -> None:
    """The single-judge rescore pilot."""


@main.command()
def sample() -> None:
    """Draw the 200 records, among those whose scan corpus is known, and save the list.

    A record scanned on the built-in corpus is replaced: that corpus has 40 short texts,
    too few to draw held-out windows from.
    """
    from openinterpretability.data import sample_corpus

    rows = [r for r in latest_records() if r["corpus_sha256"]]
    builtin = hashlib.sha256("\n".join(sample_corpus()).encode()).hexdigest()
    chosen = replace(draw(rows), rows, {builtin})
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text("".join(json.dumps(r) + "\n" for r in chosen))
    click.echo(f"{len(chosen)} records: {Counter((r['status'], r['model']) for r in chosen)}")


@main.command()
@click.option("--shard", default="0/1", show_default=True, help="i/n: every n-th record from i.")
@click.option("--reverse", is_flag=True, help="Last record first, to help a slower worker.")
def emit(shard: str, reverse: bool) -> None:
    """Write each sampled record's tasks, one per seed, skipping those already written."""
    i, n_shards = map(int, shard.split("/"))
    chosen = read_sample()[i::n_shards]
    chosen = chosen[::-1] if reverse else chosen
    known: dict[str, Path | None] = {sha256(CORPUS): CORPUS} if CORPUS.exists() else {}
    for n, row in enumerate(chosen, 1):
        corpus = corpus_for(row, known)
        for k in range(SEEDS):
            out = task_dir(row["record"], k)
            if corpus is None or (out / "score.task.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            args = ["score", "--audit-file", row["record"], "--corpus", str(corpus),
                    "--seed", str(seed_of(row["record"]) + k), "--device", "auto",
                    "--per-quantile", str(PER_QUANTILE), "--max-windows", str(MAX_WINDOWS),
                    "--embedding-model", "",
                    "--emit-task", str(out / "score.task.json")]  # fmt: skip
            done = subprocess.run([*CLI, *args], capture_output=True, text=True)
            status = "ok" if done.returncode == 0 else f"failed: {done.stderr.strip()[-200:]}"
            click.echo(f"{n}/{len(chosen)} seed {k} {row['record']}: {status}", err=True)
        if corpus is None:
            click.echo(f"skipped (corpus can't be rebuilt): {row['record']}", err=True)


@main.command()
@click.option("--judge", required=True, help="The model that answered, e.g. claude-haiku-5-5.")
def collect(judge: str) -> None:
    """Score every answered task, and write one row per record and seed."""
    rows = []
    for row in read_sample():
        for k in range(SEEDS):
            out = task_dir(row["record"], k)
            task = out / "score.task.json"
            if not task.exists():
                continue
            # A pole with too few active windows has no fuzzing prompt, so no answer is due.
            answers = [out / f"score.{kind}.json" for kind in KINDS
                       if (out / f"score.{kind}.prompt.md").exists()]  # fmt: skip
            scores = out / "scores.json"
            if all(a.exists() for a in answers) and not scores.exists():
                args = ["score", "--task", str(task), "--judge-model", f"external/{judge}",
                        "--embedding-model", "", "--out", str(scores)]  # fmt: skip
                done = subprocess.run([*CLI, *args], capture_output=True, text=True)
                if done.returncode != 0:
                    click.echo(f"failed {out}: {done.stderr.strip()[-200:]}", err=True)
            found = json.loads(scores.read_text()) if scores.exists() else None
            rows.append({**{key: row[key] for key in ("record", "slug", "status", "model")},
                         "seed": k, **summarise(found, task)})  # fmt: skip
    ROWS.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    click.echo(f"Wrote {len(rows)} rows to {ROWS}", err=True)


def summarise(scores: dict | None, task: Path) -> dict:
    """The few numbers ``report`` needs, from a scores file or, unanswered, the task."""
    if scores is None:
        drawn = json.loads(task.read_text())
        return {"active_windows": drawn.get("active_windows"), "detection": None,
                "fuzzing": None, "judge": None}  # fmt: skip
    return {
        "active_windows": scores["active_windows"], "judge": scores["judge_model"],
        **{k: (scores[k] or {}).get("balanced_accuracy") for k in KINDS},
    }  # fmt: skip


def pearson(a: list[float], b: list[float]) -> float | None:
    """The Pearson correlation of two equal-length lists; ``None`` when one is constant."""
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None


def median_of(xs: list[int]) -> int | None:
    """The middle value (the upper one of an even count), or ``None`` for no values."""
    return sorted(xs)[len(xs) // 2] if xs else None


def spearman_brown(r: float, k: float) -> float:
    """The reliability of the mean of ``k`` seeds, from one seed's reliability ``r``."""
    return k * r / (1 + (k - 1) * r)


def separation(pairs: list[tuple[float, bool]], seed: int = 0) -> dict:
    """AUROC and VERIFIED-minus-other difference, each with a 95% bootstrap interval."""
    from openinterpretability.harness.explanation_scores import auroc

    def stats(sample: list[tuple[float, bool]]) -> tuple[float | None, float | None]:
        ok = [s for s, y in sample if y]
        other = [s for s, y in sample if not y]
        if not ok or not other:
            return None, None
        return auroc(*zip(*sample, strict=True)), sum(ok) / len(ok) - sum(other) / len(other)

    rng, draws = random.Random(seed), []
    for _ in range(BOOTSTRAP):
        draws.append(stats([rng.choice(pairs) for _ in pairs]))
    out = {}
    for i, name in enumerate(("auroc", "difference")):
        values = sorted(d[i] for d in draws if d[i] is not None)
        out[name] = stats(pairs)[i]
        out[f"{name}_ci"] = [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]
    out["n_verified"] = sum(y for _, y in pairs)
    out["n_other"] = len(pairs) - out["n_verified"]
    return out


def by_record(rows: list[dict], kind: str) -> dict[str, list[float]]:
    """Each record's scores of one kind, one per answered seed."""
    out: dict[str, list[float]] = {}
    for row in rows:
        if row[kind] is not None:
            out.setdefault(row["record"], []).append(row[kind])
    return out


def kind_report(rows: list[dict], sample: dict[str, dict], kind: str) -> dict:
    """Every number for one kind: new and old separation, and seed reliability."""
    scores = by_record(rows, kind)
    verified = {r: sample[r]["status"] == "VERIFIED" for r in scores}
    new = [(sum(s) / len(s), verified[r]) for r, s in scores.items()]
    first = [(s[0], verified[r]) for r, s in scores.items()]
    old = [(sample[r]["old"][kind], verified[r]) for r in scores
           if sample[r]["old"][kind] is not None]  # fmt: skip
    pairs = [s for s in scores.values() if len(s) == SEEDS]
    r1 = pearson([s[0] for s in pairs], [s[1] for s in pairs])
    return {
        "records": len(scores),
        "new_mean_of_seeds": separation(new),
        "new_one_seed": separation(first),
        "old_same_records": separation(old),
        "seed_reliability": r1,
        "projected_reliability": {str(k): spearman_brown(r1, k) for k in (2, 4, 8)} if r1 else None,
        "old_vs_new": pearson(*zip(*[(sample[r]["old"][kind], sum(s) / len(s))
                                    for r, s in scores.items()
                                    if sample[r]["old"][kind] is not None], strict=True)),
    }  # fmt: skip


@main.command()
def report() -> None:
    """Compare the new scores with the old ones on the same records, and write the JSON."""
    sample = {r["record"]: r for r in read_sample()}
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    low = [r for r in sample.values() if all(v is None for v in r["old"].values())]
    active = {r["record"]: r["active_windows"] for r in rows if r["seed"] == 0}
    result = {
        "sample": len(sample), "rows": len(rows),
        "old_judges": dict(Counter(r["old_judge"] for r in sample.values())),
        "unscored_before": len(low),
        "unscored_before_now_scored": sum(r["record"] in by_record(rows, "detection") for r in low),
        "median_active_windows": median_of([v for v in active.values() if v is not None]),
        **{kind: kind_report(rows, sample, kind) for kind in KINDS},
    }  # fmt: skip
    REPORT.write_text(json.dumps(result, indent=1) + "\n")
    for kind in KINDS:
        k = result[kind]
        click.echo(f"{kind}: AUROC new {k['new_mean_of_seeds']['auroc']:.3f} vs old "
                   f"{k['old_same_records']['auroc']:.3f}; "
                   f"seed r {k['seed_reliability']}")  # fmt: skip


if __name__ == "__main__":
    main()
