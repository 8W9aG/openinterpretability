"""Score the neuron-pole records that have no explanation scores (issue #3499).

Records are write-once, so the scores go beside them, in
``docs/analyses/explanation-scores-backfill.jsonl``. ``scripts/verified_vs_detection.py``
reads them for a record with no scores of its own.

* ``emit`` writes each record's judge task (``score --emit-task``) under
  ``exchange/score-backfill/``, VERIFIED records first. Windows come from the record's
  own scan corpus: ``corpus.jsonl`` when its SHA-256 matches, else the corpus rebuilt
  with ``corpus --same-as``. A record whose corpus can't be rebuilt is listed and left
  out. Each record's seed comes from its experiment id, so a rerun draws the same windows.
* One ``explanation-judge`` subagent answers each prompt (a workflow, outside this script).
* ``collect`` turns the answered tasks into scores (``score --task``) and writes the rows.

    uv run python scripts/score_backfill.py emit
    uv run python scripts/score_backfill.py collect --judge claude-haiku-5-5
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import click

WORK = Path("exchange/score-backfill")
OUT = Path("docs/analyses/explanation-scores-backfill.jsonl")
CORPUS = Path("corpus.jsonl")
ORDER = {"VERIFIED": 0, "NEEDS_REVISION": 1, "REJECTED": 2}
CLI = [str(Path(sys.executable).parent / "open-interpretability")]
HEX64 = re.compile(r"[0-9a-f]{64}")


def unscored(wiki_dir: Path = Path("wiki")) -> list[dict]:
    """The neuron-pole records with no detection and no fuzzing score, VERIFIED first."""
    from openinterpretability.core.schemas import AuditRecord

    found = []
    for path in sorted((wiki_dir / "experiments").rglob("*.json")):
        record = AuditRecord.model_validate_json(path.read_text())
        verdict, sub = record.verdict, record.submission
        if verdict is None or sub.identifier.kind != "mlp_neuron_pole":
            continue
        scores = verdict.explanation_scores
        if scores is None or (scores.detection is None and scores.fuzzing is None):
            sha = sub.provenance.corpus_sha256 if sub.provenance else None
            found.append({"record": str(path), "status": verdict.status, "corpus_sha256": sha})
    return sorted(found, key=lambda r: (ORDER[r["status"]], r["record"]))


def task_dir(record: str) -> Path:
    """``exchange/score-backfill/<pole slug>/<experiment id>/``."""
    path = Path(record)
    return WORK / path.relative_to("wiki/experiments").parent / path.stem


def seed_of(record: str) -> int:
    """A seed from the experiment id, so the same record always draws the same windows."""
    return int.from_bytes(hashlib.sha256(Path(record).stem.encode()).digest()[:4], "big")


def sha256(path: Path) -> str:
    """The file's SHA-256, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_for(row: dict, known: dict[str, Path | None]) -> Path | None:
    """The record's scan corpus on disk, rebuilt once per SHA-256; None if it can't be.

    A hash that is not 64 lowercase hex digits gives None, because the hash names the file.
    A file on disk whose SHA-256 is not the hash is deleted and rebuilt.
    """
    sha = row["corpus_sha256"]
    if sha is None or not HEX64.fullmatch(sha):
        return None
    if sha not in known:
        out = WORK / "corpora" / f"{sha}.jsonl"
        if out.exists() and sha256(out) != sha:
            out.unlink()
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            done = subprocess.run([*CLI, "corpus", "--same-as", row["record"], "--out", str(out)])
            if done.returncode != 0 or not out.exists() or sha256(out) != sha:
                out.unlink(missing_ok=True)
                out = None
        known[sha] = out
    return known[sha]


@click.group()
def main() -> None:
    """Backfill explanation scores for unscored records."""


@main.command()
@click.option("--limit", type=int, default=None, help="Only the first N records.")
def emit(limit: int | None) -> None:
    """Write every unscored record's judge task, skipping those already written."""
    rows = unscored()[:limit]
    known: dict[str, Path | None] = {sha256(CORPUS): CORPUS} if CORPUS.exists() else {}
    skipped = []
    for n, row in enumerate(rows, 1):
        out = task_dir(row["record"])
        if (out / "score.task.json").exists():
            continue
        corpus = corpus_for(row, known)
        if corpus is None:
            skipped.append(row["record"])
            continue
        out.mkdir(parents=True, exist_ok=True)
        args = ["score", "--audit-file", row["record"], "--corpus", str(corpus),
                "--seed", str(seed_of(row["record"])), "--device", "cpu",
                "--emit-task", str(out / "score.task.json")]  # fmt: skip
        done = subprocess.run([*CLI, *args], capture_output=True, text=True)
        status = "ok" if done.returncode == 0 else f"failed: {done.stderr.strip()[-200:]}"
        click.echo(f"{n}/{len(rows)} {row['status']} {row['record']}: {status}", err=True)
    for record in skipped:
        click.echo(f"skipped (corpus can't be rebuilt): {record}", err=True)


@main.command()
@click.option("--judge", required=True, help="The model that answered, e.g. claude-haiku-5-5.")
def collect(judge: str) -> None:
    """Score every answered task, and write one row per record to the backfill file."""
    rows = []
    for row in unscored():
        out = task_dir(row["record"])
        answers = [out / "score.detection.json", out / "score.fuzzing.json"]
        if not (out / "score.task.json").exists() or not all(a.exists() for a in answers):
            continue
        scores = out / "scores.json"
        args = ["score", "--task", str(out / "score.task.json"),
                "--judge-model", f"external/{judge}", "--out", str(scores)]  # fmt: skip
        done = subprocess.run([*CLI, *args], capture_output=True, text=True)
        if done.returncode != 0:
            click.echo(f"failed {row['record']}: {done.stderr.strip()[-200:]}", err=True)
            continue
        found = json.loads(scores.read_text())
        rows.append({"record": row["record"], "experiment_id": Path(row["record"]).stem,
                     "status": row["status"], "scores": found})  # fmt: skip
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    click.echo(f"Wrote {len(rows)} rows to {OUT}", err=True)


if __name__ == "__main__":
    main()
