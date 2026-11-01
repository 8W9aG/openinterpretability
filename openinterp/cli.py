"""Command-line interface: the `openinterp` group, shared options and core commands."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.core.exceptions import OpenInterpError


def _load_runner(
    model_name: str,
    device: str,
    dtype: str,
    revision: str | None = None,
    attn_implementation: str | None = None,
):
    """Load ``model_name`` at ``revision``. Pass ``attn_implementation="eager"`` when
    attention patterns are needed (SDPA and flash attention don't return them)."""
    import torch

    from openinterp.core import costs

    device = resolve_device(device)
    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    click.echo(f"Loading {model_name} on {device}…", err=True)
    with costs.stage("model_load"):
        return _load(model_name, device, torch_dtype, revision, attn_implementation)


def resolve_device(device: str) -> str:
    """``auto``: CUDA, else Apple's GPU (MPS), else the CPU. float32 activations on MPS
    match the CPU's to about 1e-4 of a neuron's peak, inside reproduce's tolerance."""
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_unit_runner(ident, device: str, dtype: str, revision: str | None = None):
    """Load the model an identifier names, with eager attention for a head."""
    from openinterp.harness.units import attn_implementation

    attn = attn_implementation(ident)
    return _load_runner(ident.model_name, device, dtype, revision, attn)


def _load(model_name, device, torch_dtype, revision, attn_implementation):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from openinterp.harness.hooks import NeuronHookRunner

    commit = _resolve_revision(model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=commit)
    extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, revision=commit, **extra
    ).to(device)
    model.eval()
    _align_parameters(model)
    runner = NeuronHookRunner(model, tokenizer)
    runner.model_name = model_name
    runner.model_revision = commit
    return runner


def _align_parameters(model, alignment: int = 64) -> None:
    """Copy parameters that sit at a misaligned address into fresh memory.

    Weights memory-mapped from a safetensors file start wherever the file puts them;
    GPT-2's are not even 4-byte aligned, and a matrix-vector product on them (the
    vocabulary lens's) dies with SIGBUS in Apple's BLAS. Shared (tied) parameters are
    one object, so they stay tied.
    """
    for param in model.parameters():
        if param.data_ptr() % alignment:
            param.data = param.data.clone()


def _resolve_revision(model_name: str, revision: str | None) -> str | None:
    """The commit hash of ``revision`` (default: main), so records pin exact weights.

    ``config._commit_hash`` is not set by every transformers version; the snapshot
    folder Hugging Face caches the files in is always named after the commit.
    """
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(model_name, revision=revision, allow_patterns=["config.json"])
        ).name
    except Exception:  # offline with nothing cached, or a local path: load as given
        return revision


model_options = [
    click.option(
        "--device",
        default="auto",
        show_default=True,
        help="cpu, cuda, mps or auto (CUDA, else MPS, else CPU).",
    ),
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


explanation_scores_option = click.option(
    "--explanation-scores",
    "scores_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="ExplanationScores JSON from `openinterp score` for this explanation, recorded in "
    "the verdict (reported, not gated).",
)


def recorded_model_option(_ctx, _param, value):
    """Click callback for model options: the name to record, after
    ``$OPENINTERP_MODEL_ALIASES`` (``openinterp.agents.base.recorded_model``)."""
    from openinterp.agents.base import recorded_model

    try:
        return recorded_model(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _skip_wiki(wiki_dir) -> bool:
    return not str(wiki_dir) or str(wiki_dir) == "."


def _write_wiki(record, wiki_dir, **kwargs) -> None:
    if _skip_wiki(wiki_dir):
        return
    from openinterp.builder.wiki import archive_path, write_experiment_page
    from openinterp.signing import SignatureError, auto_sign

    page = write_experiment_page(record, wiki_dir, **kwargs)
    if page is not None:
        archived = archive_path(wiki_dir, record)
        click.echo(f"Experiment: {archived}")
        click.echo(f"Wiki: {page}")
        if kwargs.get("kind") == "backfill":
            return  # an imported record this run didn't produce: never vouch for it
        try:
            signer = auto_sign(archived)
        except SignatureError as exc:
            click.echo(f"warning: could not sign {archived}: {exc}", err=True)
        else:
            if signer:
                click.echo(f"Signed as {signer}: {archived}.sig (CI skips reproducing it)")


def _write_copy(record, out) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(record.model_dump_json(indent=2))
        click.echo(f"Wrote {out}")


def _clear_costs(submission):
    """``submission`` without the costs of the experiment that first audited it."""
    prov = submission.provenance
    if prov is None or prov.costs is None:
        return submission
    return submission.model_copy(update={"provenance": prov.model_copy(update={"costs": None})})


def _add_costs(submission, *parts):
    """``submission`` with ``parts`` and everything this process measured in its costs."""
    from openinterp.cli_evidence import with_costs
    from openinterp.core import costs

    provenance = with_costs(submission.provenance, *parts, costs.measured())
    return submission.model_copy(update={"provenance": provenance})


def quiet_libraries(quiet: bool) -> None:
    """Keep Hugging Face and torch noise out of the output agents read.

    Always drop torch's macOS note about redirects; drop progress bars when stderr
    isn't a terminal or ``quiet``; with ``quiet``, also library warnings. Set before
    ``transformers`` and ``huggingface_hub`` are imported, which read the environment
    then; variables the user set win."""
    import logging
    import os
    import sys
    import warnings

    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    if quiet or not sys.stderr.isatty():
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if quiet:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        warnings.filterwarnings("ignore")


def with_model_options(fn):
    for option in reversed(model_options):
        fn = option(fn)
    return fn


@click.group()
@click.version_option(package_name="openinterp")
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="No progress bars or library warnings. Progress bars are already off when "
    "stderr isn't a terminal (an agent reading the output).",
)
@click.pass_context
def cli(ctx, quiet) -> None:
    """Open Interpretability: agent-audited registry of native neurons."""
    import time

    from openinterp.core import costs

    quiet_libraries(quiet)
    costs.reset()
    start = time.perf_counter()
    command = ctx.invoked_subcommand or "openinterp"
    if command not in ("costs", "usage"):  # bookkeeping, not an experiment step
        ctx.call_on_close(lambda: costs.append_ledger(command, time.perf_counter() - start))


@cli.command()
@click.option("--record", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), help="Also write the new record here.")
@explanation_scores_option
@wiki_option
@with_model_options
def reevaluate(record, out, scores_file, wiki_dir, device, dtype):
    """Re-measure a record's stored probes under the current ReviewPolicy (no LLM calls).

    The result is a new experiment; the original record is never modified.
    """
    from openinterp.agents.reviewer import evaluate_submission, probes_from_verdict
    from openinterp.cli_score import load_scores
    from openinterp.core.schemas import AuditRecord

    if _skip_wiki(wiki_dir) and out is None:
        raise click.UsageError("--out is required when --wiki-dir '' skips the registry.")
    old = AuditRecord.model_validate_json(record.read_text())
    if old.verdict is None:
        raise click.ClickException(f"{record} has no verdict to re-evaluate.")
    scores = load_scores(scores_file, old.submission)
    runner = _load_unit_runner(old.submission.identifier, device, dtype)
    submission = _clear_costs(old.submission)
    verdict = evaluate_submission(
        submission, runner, probes_from_verdict(old.verdict), old.verdict.reviewer_agent_model
    )
    scored = scores and scores.costs
    scores = scores and scores.model_copy(update={"costs": None})
    verdict = verdict.model_copy(update={"explanation_scores": scores})
    new = AuditRecord(submission=_add_costs(submission, scored), verdict=verdict)
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
@click.option(
    "--wiring",
    "wiring_path",
    default=Path("docs/analyses/wiring.jsonl"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Backfilled wiring (`openinterp wiring-backfill`) for records without their own; "
    "skipped if missing.",
)
@click.option(
    "--corpus-sources",
    "source_paths",
    multiple=True,
    default=[Path("corpus.source.json")],
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Corpus sidecars that give the source of records which scanned their corpus "
    "(matched by SHA-256) without recording one, shown as inferred; skipped if missing.",
)
@click.option(
    "--alignment",
    "alignment_path",
    default=Path("docs/analyses/dictionary-alignment.jsonl"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Dictionary latents aligned with each neuron pole (`openinterp crossref`); "
    "skipped if missing.",
)
def build(wiki_dir, output_dir, wiring_path, source_paths, alignment_path):
    """Compile every archived experiment into static pages and data/registry.parquet."""
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.wiki import load_archive
    from openinterp.core.draws import load_draws
    from openinterp.corpus import load_sources
    from openinterp.crossref import load_rows as load_alignment
    from openinterp.wiring_backfill import load_rows

    report = compile_site(
        load_archive(wiki_dir),
        output_dir,
        wiring_rows=load_rows(wiring_path),
        draws=load_draws(wiki_dir),
        corpus_sources=load_sources(source_paths),
        alignment_rows=load_alignment(alignment_path),
    )
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
from openinterp import (  # noqa: E402, F401
    cli_baseline,
    cli_concepts,
    cli_corpus,
    cli_costs,
    cli_crossref,
    cli_evidence,
    cli_probing,
    cli_registry,
    cli_review,
    cli_scan,
    cli_score,
    cli_wiring,
)


def main() -> None:
    try:
        cli(standalone_mode=True)
    except OpenInterpError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
