"""Command-line interface: ``openinterp audit | verify | build | deploy``."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

import click

from openinterp.core.exceptions import OpenInterpError


def _load_runner(model_name: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from openinterp.harness.hooks import NeuronHookRunner

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    click.echo(f"Loading {model_name} on {device}…", err=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    model.eval()
    return NeuronHookRunner(model, tokenizer)


def _read_corpus(path: Path | None) -> tuple[list[str], str, str]:
    """Return (snippets, description, sha256 of the corpus source)."""
    if path is None:
        from openinterp.data import sample_corpus

        texts = sample_corpus()
        return texts, "builtin:sample_corpus", _sha256("\n".join(texts).encode())
    raw = path.read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        lines = [json.loads(line)["text"] for line in lines]
    return lines, path.name, _sha256(raw)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


model_options = [
    click.option("--device", default="auto", show_default=True, help="cpu, cuda, mps or auto."),
    click.option(
        "--dtype",
        type=click.Choice(["auto", "float32", "bfloat16", "float16"]),
        default="float32",
        show_default=True,
    ),
]


wiki_option = click.option(
    "--wiki-dir",
    default=Path("wiki"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Registry root: the experiment's record and page go in experiments/<slug>/. "
    "Pass --wiki-dir '' to skip (then --out is required).",
)


def _skip_wiki(wiki_dir) -> bool:
    return not str(wiki_dir) or str(wiki_dir) == "."


def _write_wiki(record, wiki_dir, **kwargs) -> None:
    if _skip_wiki(wiki_dir):
        return
    from openinterp.builder.wiki import archive_path, write_experiment_page

    page = write_experiment_page(record, wiki_dir, **kwargs)
    if page is not None:
        click.echo(f"Experiment: {archive_path(wiki_dir, record)}")
        click.echo(f"Wiki: {page}")


def _write_copy(record, out) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(record.model_dump_json(indent=2))
        click.echo(f"Wrote {out}")


def with_model_options(fn):
    for option in reversed(model_options):
        fn = option(fn)
    return fn


@click.group()
@click.version_option(package_name="openinterp")
def cli() -> None:
    """Open Interpretability: agent-audited registry of native neurons."""


@cli.command()
@click.option(
    "--model", "model_name", required=True, help="Hugging Face repo, e.g. Qwen/Qwen2.5-0.5B"
)
@click.option("--layer", type=int, help="Required unless --random.")
@click.option("--neuron", type=int, help="Required unless --random.")
@click.option("--pole", type=click.Choice(["positive", "negative"]), help="Default: positive.")
@click.option(
    "--random",
    "random_pick",
    is_flag=True,
    help="Sample unspecified layer/neuron/pole uniformly (the unbiased selection method).",
)
@click.option("--seed", type=int, help="Seed for --random (recorded in provenance).")
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    help="Text file (one snippet per line) or .jsonl with a 'text' field.",
)
@click.option("--top-k", default=20, show_default=True, help="Examples shown to the proposer.")
@click.option(
    "--agent-model",
    default=None,
    help="Proposer. LiteLLM model string, or external/<name> to answer a prompt file.",
)
@click.option(
    "--out", type=click.Path(path_type=Path), help="Output JSON (default: submissions/<slug>.json)"
)
@with_model_options
def audit(
    model_name,
    layer,
    neuron,
    pole,
    random_pick,
    seed,
    corpus,
    top_k,
    agent_model,
    out,
    device,
    dtype,
):
    """Scan a neuron pole, ask the proposer agent for a hypothesis, and write a submission."""
    import random

    from openinterp import __version__
    from openinterp.agents.base import DEFAULT_MODEL
    from openinterp.agents.proposer import ProposerAgent
    from openinterp.core.schemas import ExperimentProvenance, NeuronIdentifier
    from openinterp.harness.hooks import activation_histogram
    from openinterp.harness.jlens import vocabulary_lens
    from openinterp.harness.scan import scan_neuron

    if not random_pick and (layer is None or neuron is None):
        raise click.UsageError("--layer and --neuron are required unless --random is given.")
    runner = _load_runner(model_name, device, dtype)
    if random_pick:
        seed = secrets.randbits(32) if seed is None else seed
        rng = random.Random(seed)
        layer = rng.randrange(runner.n_layers) if layer is None else layer
        neuron = rng.randrange(runner.mlp_width(layer)) if neuron is None else neuron
        pole = rng.choice(["positive", "negative"]) if pole is None else pole
    ident = NeuronIdentifier(
        model_name=model_name, layer=layer, neuron_idx=neuron, pole=pole or "positive"
    )
    pole = ident.pole
    texts, corpus_name, corpus_sha = _read_corpus(corpus)
    click.echo(f"Scanning {len(texts)} snippets for {ident}…", err=True)
    scan = scan_neuron(runner, texts, layer, neuron, pole, k=top_k)
    examples, values = scan.examples, scan.values
    if not examples or examples[0].max_activation <= 0:
        raise click.ClickException(
            f"{ident} never activates on this corpus; try a larger --corpus."
        )
    click.echo("Reading the output direction through the logit and Jacobian lenses…", err=True)
    lens = vocabulary_lens(runner, texts[:100], layer, neuron, pole)

    agent = ProposerAgent(agent_model or DEFAULT_MODEL)
    click.echo(f"Asking {agent.model} for a hypothesis…", err=True)
    draft = agent.propose(ident, examples, lens, scan.confounds, scan.correlated)
    provenance = ExperimentProvenance(
        model_revision=getattr(runner.model.config, "_commit_hash", None),
        dtype=dtype,
        corpus=corpus_name,
        corpus_sha256=corpus_sha,
        n_snippets=len(texts),
        tokens_scanned=len(values),
        selection="random" if random_pick else "manual",
        seed=seed if random_pick else None,
        harness_version=__version__,
    )
    submission = agent.build_submission(
        ident,
        draft,
        runner=runner,
        examples=examples,
        histogram=activation_histogram(values),
        provenance=provenance,
        confounds=scan.confounds,
        correlated=scan.correlated,
        vocabulary_lens=lens,
    )
    out = out or Path("submissions") / f"{ident.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(submission.model_dump_json(indent=2))
    click.echo(f"Hypothesis: {submission.natural_language_explanation}")
    click.echo(f"Wrote {out}")


@cli.command()
@click.option("--audit-file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--agent-model",
    default=None,
    help="Reviewer. LiteLLM model string, or external/<name> to answer a prompt file.",
)
@click.option("--n-probes", default=16, show_default=True, type=click.IntRange(10, 20))
@click.option("--out", type=click.Path(path_type=Path), help="Also write the record here.")
@wiki_option
@with_model_options
def verify(audit_file, agent_model, n_probes, out, wiki_dir, device, dtype):
    """Adversarially review a submission and post it as a new experiment."""
    from openinterp.agents.base import DEFAULT_MODEL
    from openinterp.agents.reviewer import ReviewerAgent
    from openinterp.core.schemas import AuditRecord, NeuronAuditSubmission

    if _skip_wiki(wiki_dir) and out is None:
        raise click.UsageError("--out is required when --wiki-dir '' skips the registry.")
    raw = audit_file.read_text()
    if "submission" in json.loads(raw):  # an existing audit record: re-review its submission
        submission = AuditRecord.model_validate_json(raw).submission
    else:
        submission = NeuronAuditSubmission.model_validate_json(raw)
    runner = _load_runner(submission.identifier.model_name, device, dtype)
    reviewer = ReviewerAgent(agent_model or DEFAULT_MODEL)
    if reviewer.model == submission.submitting_agent_model:
        click.echo("warning: reviewer and submitter use the same model", err=True)
    click.echo(f"Generating {n_probes} adversarial probes with {reviewer.model}…", err=True)
    probes = reviewer.generate_probes(submission, n=n_probes).probes
    verdict = reviewer.review(submission, runner, probes=probes)

    record = AuditRecord(submission=submission, verdict=verdict)
    click.echo(f"{verdict.status} (confidence {verdict.confidence_score:.2f}) — {verdict.notes}")
    _write_copy(record, out)
    _write_wiki(record, wiki_dir, kind="review")


cli.add_command(verify, name="review")


@cli.command()
@click.option("--record", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), help="Also write the new record here.")
@wiki_option
@with_model_options
def reevaluate(record, out, wiki_dir, device, dtype):
    """Re-measure a record's stored probes under the current ReviewPolicy (no LLM calls).

    The result is a new experiment; the original record is never modified.
    """
    from openinterp.agents.reviewer import evaluate_submission, probes_from_verdict
    from openinterp.core.schemas import AuditRecord

    if _skip_wiki(wiki_dir) and out is None:
        raise click.UsageError("--out is required when --wiki-dir '' skips the registry.")
    old = AuditRecord.model_validate_json(record.read_text())
    if old.verdict is None:
        raise click.ClickException(f"{record} has no verdict to re-evaluate.")
    runner = _load_runner(old.submission.identifier.model_name, device, dtype)
    verdict = evaluate_submission(
        old.submission, runner, probes_from_verdict(old.verdict), old.verdict.reviewer_agent_model
    )
    new = AuditRecord(submission=old.submission, verdict=verdict)
    click.echo(f"{old.verdict.status} -> {verdict.status} — {verdict.notes}")
    _write_wiki(old, wiki_dir, kind="backfill")  # no-op when the original is already archived
    _write_copy(new, out)
    _write_wiki(new, wiki_dir, kind="reevaluation", previous_status=old.verdict.status)


@cli.command()
@click.option(
    "--wiki-dir",
    default=Path("wiki"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Experiment archive (wiki/experiments/**/<id>.json).",
)
@click.option(
    "--output-dir", default=Path("dist"), show_default=True, type=click.Path(path_type=Path)
)
def build(wiki_dir, output_dir):
    """Compile every archived experiment into static pages and data/registry.parquet."""
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.wiki import load_archive

    report = compile_site(load_archive(wiki_dir), output_dir)
    click.echo(
        f"Published {report.experiments_published} experiments: {len(report.pages)} pages, "
        f"{report.parquet} → {output_dir}"
    )


@cli.command()
@click.option(
    "--dist-dir",
    default=Path("dist"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--bucket", required=True, help="R2 bucket name.")
@click.option("--prefix", default="", help="Key prefix inside the bucket.")
@click.option("--delete", is_flag=True, help="Delete remote objects missing locally.")
@click.option("--dry-run", is_flag=True, help="Show what would change without uploading.")
@click.option(
    "--site-url",
    envvar="OPENINTERP_SITE_URL",
    help="Public URL of the bucket root. With CF_ZONE_ID and CF_API_TOKEN set, the "
    "changed URLs are purged from Cloudflare's cache.",
)
def deploy(dist_dir, bucket, prefix, delete, dry_run, site_url):
    """Delta-sync the built site to Cloudflare R2 (a no-op when nothing changed)."""
    import os

    from openinterp.sync.r2 import (
        R2Config,
        make_client,
        purge_cloudflare_cache,
        purge_targets,
        sync_directory,
    )

    client = make_client(R2Config.from_env())
    plan = sync_directory(
        dist_dir, bucket, client, prefix=prefix, delete=delete, dry_run=dry_run, log=click.echo
    )
    verb = "Would upload" if dry_run else "Uploaded"
    click.echo(
        f"{verb} {len(plan.uploads)}, unchanged {len(plan.unchanged)}, "
        f"deleted {len(plan.deletes)}, derived skipped {len(plan.skipped_derived)}"
    )

    zone, token = os.environ.get("CF_ZONE_ID"), os.environ.get("CF_API_TOKEN")
    if not (site_url and zone and token):
        return
    urls = purge_targets(plan, site_url, prefix)
    if not urls:
        click.echo("Nothing changed; no cache purge.")
    elif dry_run:
        click.echo(f"Would purge {len(urls)} URLs")
    else:
        purge_cloudflare_cache(zone, token, urls)
        click.echo(f"Purged {len(urls)} URLs from the Cloudflare cache")


# Registers the registry-maintenance commands on `cli`.
from openinterp import cli_registry  # noqa: E402, F401


def main() -> None:
    try:
        cli(standalone_mode=True)
    except OpenInterpError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
