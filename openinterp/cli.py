"""Command-line interface: ``openinterp audit | verify | build | deploy``."""

from __future__ import annotations

import json
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

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    click.echo(f"Loading {model_name} on {device}…", err=True)
    with costs.stage("model_load"):
        return _load(model_name, device, torch_dtype, revision, attn_implementation)


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
        model_name, torch_dtype=torch_dtype, revision=commit, **extra
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


explanation_scores_option = click.option(
    "--explanation-scores",
    "scores_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="ExplanationScores JSON from `openinterp score` for this explanation, recorded in "
    "the verdict (reported, not gated).",
)


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


def with_model_options(fn):
    for option in reversed(model_options):
        fn = option(fn)
    return fn


@click.group()
@click.version_option(package_name="openinterp")
@click.pass_context
def cli(ctx) -> None:
    """Open Interpretability: agent-audited registry of native neurons."""
    import time

    from openinterp.core import costs

    costs.reset()
    start = time.perf_counter()
    command = ctx.invoked_subcommand or "openinterp"
    if command != "costs":  # summarising the ledger isn't part of any experiment
        ctx.call_on_close(lambda: costs.append_ledger(command, time.perf_counter() - start))


@cli.command()
@click.option("--audit-file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--agent-model",
    default=None,
    help="Reviewer. LiteLLM model string, or external/<name> to answer a prompt file. "
    "With --probes, the name recorded for whoever wrote them.",
)
@click.option("--n-probes", default=16, show_default=True, type=click.IntRange(10, 20))
@click.option(
    "--emit-prompt",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the reviewer's prompt here, e.g. review.prompt.md (answer in "
    "review.probes.json), and exit. "
    "Loads no model, so the reviewer sees no probe activations.",
)
@click.option(
    "--probes",
    "probes_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="ProbeSet JSON answering --emit-prompt. No LLM is called.",
)
@click.option("--out", type=click.Path(path_type=Path), help="Also write the record here.")
@explanation_scores_option
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    help="Without --explanation-scores: the scan's corpus, to score the explanation on.",
)
@click.option(
    "--judge-model", help="Without --explanation-scores: the judge (default: the reviewer)."
)
@wiki_option
@with_model_options
def verify(
    audit_file,
    agent_model,
    n_probes,
    emit_prompt,
    probes_file,
    out,
    scores_file,
    corpus,
    judge_model,
    wiki_dir,
    device,
    dtype,
):
    """Adversarially review a submission and post it as a new experiment.

    Every review records explanation scores on held-out text (`openinterp score`). Pass
    them with --explanation-scores; in one step (no --probes), they are measured here.
    """
    from openinterp.agents.base import DEFAULT_MODEL
    from openinterp.agents.reviewer import ProbeSet, ReviewerAgent
    from openinterp.cli_evidence import answer_path
    from openinterp.cli_score import load_scores, score_submission
    from openinterp.core import costs
    from openinterp.core.schemas import AuditRecord, NeuronAuditSubmission

    raw = audit_file.read_text()
    if "submission" in json.loads(raw):  # an existing audit record: re-review its submission
        submission = AuditRecord.model_validate_json(raw).submission
        submission = _clear_costs(submission)  # its audit belonged to another experiment
    else:
        submission = NeuronAuditSubmission.model_validate_json(raw)
    if emit_prompt is not None:
        answer = answer_path(emit_prompt, ".probes.json")
        task = ReviewerAgent.probe_prompt(submission, n=n_probes)
        emit_prompt.parent.mkdir(parents=True, exist_ok=True)
        emit_prompt.write_text(ReviewerAgent.render_prompt(ProbeSet, task, answer))
        click.echo(f"Reviewer prompt: {emit_prompt} (answer in {answer})")
        return
    if _skip_wiki(wiki_dir) and out is None:
        raise click.UsageError("--out is required when --wiki-dir '' skips the registry.")
    if probes_file is not None and not agent_model:
        raise click.UsageError(
            "--agent-model is required with --probes: name the model that wrote them."
        )
    if scores_file is None and (probes_file is not None or corpus is None):
        raise click.UsageError(
            "--explanation-scores is required (from `openinterp score`, docs/PROTOCOL.md §4); "
            "without --probes, --corpus <the scan's corpus> also works."
        )
    scores = load_scores(scores_file, submission)
    reviewer = ReviewerAgent(agent_model or DEFAULT_MODEL)
    if reviewer.model == submission.submitting_agent_model:
        click.echo(
            f"warning: {reviewer.model} both proposed and reviewed this, so the review is not "
            "independent. It will be recorded and badged as self-reviewed; a different "
            "reviewer model is preferred (docs/PROTOCOL.md §2).",
            err=True,
        )
    if probes_file is not None:
        probes = ProbeSet.model_validate_json(probes_file.read_text()).probes
        # --emit-prompt review.prompt.md is answered in review.probes.json.
        stem = probes_file.name.removesuffix(".probes.json")
        prompt_file = probes_file.with_name(stem + ".prompt.md")
        costs.add_external("reviewer", reviewer.model, prompt_file, probes_file)
    else:
        click.echo(f"Generating {n_probes} adversarial probes with {reviewer.model}…", err=True)
        probes = reviewer.generate_probes(submission, n=n_probes).probes
    runner = _load_unit_runner(submission.identifier, device, dtype)
    if scores is None:  # one step: score here, on the scan's corpus (checked above)
        scores = score_submission(runner, submission, corpus, judge_model or reviewer.model)
    verdict = reviewer.review(submission, runner, probes=probes)
    # The scores' costs move into the record's provenance with everything else.
    scored = scores and scores.costs
    scores = scores and scores.model_copy(update={"costs": None})
    verdict = verdict.model_copy(update={"explanation_scores": scores})

    record = AuditRecord(submission=_add_costs(submission, scored), verdict=verdict)
    click.echo(f"{verdict.status} (confidence {verdict.confidence_score:.2f}) — {verdict.notes}")
    _write_copy(record, out)
    _write_wiki(record, wiki_dir, kind="review")


cli.add_command(verify, name="review")


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
from openinterp import (  # noqa: E402, F401
    cli_baseline,
    cli_corpus,
    cli_costs,
    cli_evidence,
    cli_registry,
    cli_scan,
    cli_score,
)


def main() -> None:
    try:
        cli(standalone_mode=True)
    except OpenInterpError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
