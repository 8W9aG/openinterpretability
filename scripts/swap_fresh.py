"""The swapped-unit null with fresh probes (``docs/analyses/null-calibration.md``).

``null_calibration.py swap`` reuses neuron A's probes on neuron B. Here a sample of those
swaps is reviewed again from scratch: the reviewer gets A's hypothesis with B's scan (its
top examples, confounds and scale) and writes new probes, which are measured on B.

    uv run python scripts/swap_fresh.py prepare --swaps docs/analyses/null-calibration-swap.jsonl
    # each exchange/swap-fresh/sNN/review.prompt.md is answered in review.probes.json
    uv run python scripts/swap_fresh.py measure --agent-model <reviewer>
    uv run python scripts/swap_fresh.py report --out docs/analyses/null-calibration-fresh.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click

WORK = Path("exchange/swap-fresh")


@click.group()
def cli() -> None:
    """The swapped-unit null with fresh probes."""


@cli.command()
@click.option("--swaps", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--n", default=40, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--corpus", default="corpus.jsonl", type=click.Path(exists=True, path_type=Path))
def prepare(swaps, n, seed, corpus):
    """Sample reviewed swaps, rescan each null neuron and write the reviewer's prompt."""
    from null_calibration import _NoPartners, null_submission, scan_request, scan_texts

    import openinterp.harness.scan
    from openinterp.cli import _load_runner
    from openinterp.cli_evidence import open_scan_cache
    from openinterp.cli_review import write_review_prompt
    from openinterp.core.schemas import AuditRecord
    from openinterp.evidence import read_corpus, run_scan

    openinterp.harness.scan.PartnerSearch = _NoPartners
    rows = [json.loads(line) for line in swaps.open()]
    rows = random.Random(seed).sample([r for r in rows if "gates" in r], n)
    builtin = read_corpus(None)[2]
    cache, samples = open_scan_cache(True), {}
    run = _load_runner(rows[0]["model"], "auto", "float32")
    for i, row in enumerate(sorted(rows, key=lambda r: (r["model"], r["layer"]))):
        if (WORK / f"s{i:02d}" / "review.prompt.md").exists():
            continue
        record = AuditRecord.model_validate_json(Path(row["record"]).read_text())
        if record.submission.identifier.model_name != run.model_name:
            run = _load_runner(record.submission.identifier.model_name, "auto", "float32")
        null = record.submission.identifier.model_copy(update={"neuron_idx": row["null_neuron"]})
        source = None if record.submission.provenance.corpus_sha256 == builtin else corpus
        request = scan_request(record.submission, null, source)
        scan = run_scan(run, request, null, None, scan_texts(run, request, samples), cache)
        sub = null_submission(record.submission, null, scan).model_copy(
            update={"input_attribution": None, "vocabulary_lens": None, "opposite_pole": None}
        )
        out = WORK / f"s{i:02d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "submission.json").write_text(sub.model_dump_json(indent=2))
        (out / "swap.json").write_text(json.dumps(row))
        write_review_prompt(sub, out / "review.prompt.md", 20, False)


@cli.command()
@click.option("--agent-model", required=True, help="The reviewer's model, as recorded.")
def measure(agent_model):
    """Measure every answered probe set on its null neuron (result.json)."""
    from null_calibration import review_probes

    from openinterp.agents.reviewer import ProbeSet
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import NeuronAuditSubmission

    runners: dict[str, object] = {}
    for out in sorted(WORK.glob("s*/review.probes.json")):
        out = out.parent
        if (out / "result.json").exists():
            continue
        sub = NeuronAuditSubmission.model_validate_json((out / "submission.json").read_text())
        model = sub.identifier.model_name
        if model not in runners:
            runners[model] = _load_runner(model, "auto", "float32")
        try:
            probes = ProbeSet.model_validate_json((out / "review.probes.json").read_text()).probes
        except ValueError as exc:  # left for the reviewer to fix; the others go ahead
            click.echo(f"{out.name}: invalid probes, skipped: {exc}", err=True)
            continue
        result = review_probes(sub, runners[model], probes, agent_model)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        click.echo(f"{out.name} {sub.identifier}: {result['outcome']}")


@cli.command()
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def report(out):
    """One JSON line per measured swap: the reused-probe row and the fresh review."""
    with out.open("w") as fh:
        for result in sorted(WORK.glob("s*/result.json")):
            swap = json.loads((result.parent / "swap.json").read_text())
            fresh = json.loads(result.read_text())
            reused = {k: swap.get(k) for k in ("outcome", "probe_verified", "balanced_accuracy")}
            fh.write(json.dumps(swap | {"reused": reused, "fresh": fresh}) + "\n")
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    cli()
