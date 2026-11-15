"""Whether another scan corpus changes a verdict (#3187, docs/analyses/corpus-stability.md).

The corpus enters a verdict through the scan's maximum activation, which is the scale. A
probe is active at ``activation_threshold`` times it, and counterfactual Δs are divided by
it. The confound baselines use the same threshold. The scan also gives the active share
that the default-on gate reads. Each probe's raw activation is stored. So the same
hypothesis and probes can be judged again with another corpus's scan, with no LLM and no
new probe measurement.

``select`` writes a stratified sample of records. ``rescan`` scans each one's pole on
the other corpus, with the record's token budget and windows. ``judge`` judges each record
again with the new scan, and writes the rows. As a control, ``judge`` also judges each
record with its own scan, which must give the stored verdict.

    uv run python scripts/corpus_stability.py select --out sample.json
    uv run python scripts/corpus_stability.py rescan --sample sample.json \\
        --corpus fineweb.jsonl --scans scans/
    uv run python scripts/corpus_stability.py judge --sample sample.json --scans scans/ \\
        --corpus fineweb.jsonl --out docs/analyses/corpus-stability.jsonl
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import click

DEFAULT_CORPUS = "ded48cfc4898c00bd663c4e26dd02baddc48f54f53469988097030686dd2a2ad"  # pile-10k
MODELS = ("openai-community/gpt2", "Qwen/Qwen2.5-0.5B")
PER_CELL = 10  # records per model and per VERIFIED-or-not


@click.group()
def cli() -> None:
    """Judge records again with a scan on another corpus."""


def _first_readings(wiki: Path) -> list[Path]:
    """First experiments of random native neuron poles, protocol 0.8, on the default corpus."""
    from openinterp.core.schemas import AuditRecord

    first: dict[str, tuple[str, Path]] = {}
    for path in sorted((wiki / "experiments").rglob("*.json")):
        record = AuditRecord.model_validate_json(path.read_bytes())
        sub, verdict = record.submission, record.verdict
        prov = sub.provenance
        if (
            verdict is None
            or verdict.policy is None
            or verdict.metrics is None
            or sub.identifier.kind != "mlp_neuron_pole"
            or sub.identifier.model_name not in MODELS
            or sub.sister_of is not None
            or prov is None
            or prov.selection != "random"
            or prov.corpus_sha256 != DEFAULT_CORPUS
            or verdict.policy.protocol_version != "0.8"
        ):
            continue
        slug, stamp = sub.identifier.slug, verdict.timestamp.isoformat()
        if slug not in first or stamp < first[slug][0]:
            first[slug] = (stamp, path)
    return [path for _, path in first.values()]


@cli.command()
@click.option("--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default="wiki")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--seed", default=0, show_default=True)
def select(wiki_dir: Path, out: Path, seed: int) -> None:
    """Write the sample: PER_CELL records per model, VERIFIED and not, at random."""
    from openinterp.core.schemas import AuditRecord

    cells: dict[tuple[str, bool], list[str]] = {}
    for path in _first_readings(wiki_dir):
        record = AuditRecord.model_validate_json(path.read_bytes())
        key = (record.submission.identifier.model_name, record.verdict.status == "VERIFIED")
        cells.setdefault(key, []).append(str(path))
    rng = random.Random(seed)
    sample = [p for key in sorted(cells) for p in rng.sample(sorted(cells[key]), PER_CELL)]
    out.write_text(json.dumps(sample, indent=1) + "\n")
    click.echo(
        f"{len(sample)} records from cells {[(k, len(v)) for k, v in sorted(cells.items())]}"
    )


def _scan_path(scans: Path, record_path: str) -> Path:
    return scans / (record_path.replace("/", "--") + ".ev.json")


@cli.command()
@click.option("--sample", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--scans", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--device", default="auto", show_default=True)
def rescan(sample: Path, corpus: Path, scans: Path, device: str) -> None:
    """Scan each record's pole on ``corpus``, as the record's scan was set up."""
    from openinterp.core.schemas import AuditRecord

    scans.mkdir(parents=True, exist_ok=True)
    for record_path in json.loads(sample.read_text()):
        out = _scan_path(scans, record_path)
        if out.exists():
            continue
        record = AuditRecord.model_validate_json(Path(record_path).read_bytes())
        ident, prov = record.submission.identifier, record.submission.provenance
        args = ["audit", "--model", ident.model_name, "--layer", str(ident.layer)]
        args += ["--neuron", str(ident.neuron_idx), "--pole", ident.pole, "--corpus", str(corpus)]
        budget = prov.corpus_sample.max_tokens if prov.corpus_sample else 0  # 0: every document
        args += ["--max-tokens", str(budget), "--device", device]
        if prov.snippet_tokens:
            args += ["--window-tokens", str(prov.snippet_tokens)]
            args += ["--stride", str(prov.snippet_stride or prov.snippet_tokens // 2)]
        command = [sys.executable, "-c", "from openinterp.cli import main; main()", *args]
        done = subprocess.run(
            [*command, "--evidence-out", str(out)], capture_output=True, text=True
        )
        status = "ok" if done.returncode == 0 else f"failed: {done.stderr[-400:]}"
        click.echo(f"{ident.slug}: {status}", err=True)


def judge_with(record, scale: float, histogram, confounds, tokenizer) -> dict:
    """The verdict that ``record``'s stored results get with a scan's scale and statistics."""
    from openinterp.agents.review_measurements import _delta_agrees
    from openinterp.agents.reviewer import AdversarialProbe, best_baseline, decide, review_metrics
    from openinterp.harness.confounds import baseline_predictions

    verdict, sub = record.verdict, record.submission
    policy, stored = verdict.policy, verdict.metrics
    old = sub.max_activation
    threshold = policy.activation_threshold * scale
    results, probes, actual = [], [], []
    for r in verdict.adversarial_test_results:
        if r.kind in ("hard_negative", "positive_holdout") and r.measured:
            active = scale > 0 and r.activation >= threshold
            r = r.model_copy(update={"passed": active == r.expected_active})
            probes.append(
                AdversarialProbe(text=r.prompt, expected_active=r.expected_active, rationale="")
            )
            actual.append(active)
        elif r.kind == "counterfactual" and r.measured and r.observed_delta is not None:
            observed = r.observed_delta * old / scale
            r = r.model_copy(update={"passed": _delta_agrees(r.expected_delta, observed, policy)})
        results.append(r)
    baseline = None
    if confounds is not None and probes:
        skip = bool(sub.first_token_floor)
        texts = [p.text for p in probes]
        candidates = baseline_predictions(confounds, tokenizer, texts, threshold, skip)
        candidates["never active"] = [False] * len(probes)
        baseline = best_baseline(candidates, probes, actual, policy)
    share = histogram.share_at_least(threshold) if histogram is not None else None
    metrics = review_metrics(
        results, policy, independent=stored.independent_review, baseline=baseline,
        active_share=share, first_token_floor=stored.first_token_floor,
    )  # fmt: skip
    failed = sorted(g for g, ok in metrics.gates.items() if not ok)
    return {"status": decide(metrics, policy), "failed_gates": failed,
            "balanced_accuracy": round(metrics.balanced_accuracy, 4)}  # fmt: skip


@cli.command()
@click.option("--sample", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--scans", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def judge(sample: Path, scans: Path, corpus: Path, out: Path) -> None:
    """Judge each record with its own scan (the control) and with the other corpus's scan."""
    from transformers import AutoTokenizer

    from openinterp.core.examples import Histogram
    from openinterp.core.schemas import AuditRecord, ConfoundProfile
    from openinterp.evidence import read_corpus

    other_sha = read_corpus(corpus)[2]
    tokenizers, rows = {}, []
    for record_path in json.loads(sample.read_text()):
        scan_file = _scan_path(scans, record_path)
        if not scan_file.exists():
            continue
        record = AuditRecord.model_validate_json(Path(record_path).read_bytes())
        sub = record.submission
        model = sub.identifier.model_name
        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(
                model, revision=sub.provenance.model_revision
            )
        tok = tokenizers[model]
        own = judge_with(record, sub.max_activation, sub.activation_histogram, sub.confounds, tok)
        ev = json.loads(scan_file.read_text())
        scale = max(e["max_activation"] for e in ev["top_examples"])
        hist = Histogram.model_validate(ev["activation_histogram"])
        confounds = ConfoundProfile.model_validate(ev["confounds"]) if ev.get("confounds") else None
        other = judge_with(record, scale, hist, confounds, tok)
        rows.append({
            "record": record_path, "model": model, "slug": sub.identifier.slug,
            "stored": record.verdict.status, "own_scan": own, "other_corpus": other,
            "scale": round(sub.max_activation, 6), "other_scale": round(scale, 6),
            "control_ok": own["status"] == record.verdict.status, "other_corpus_sha256": other_sha,
        })  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    flips = sum(r["other_corpus"]["status"] != r["stored"] for r in rows)
    controls = sum(r["control_ok"] for r in rows)
    click.echo(f"{len(rows)} records: control {controls}/{len(rows)}, status changed in {flips}")


if __name__ == "__main__":
    cli()
