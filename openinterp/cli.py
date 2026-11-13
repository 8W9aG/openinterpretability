"""Command-line interface: the `openinterp` group, shared options and core commands."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli_memory import memory_option
from openinterp.core.contributors import CONTRIBUTOR_ENV, contributor_problem
from openinterp.core.exceptions import OpenInterpError
from openinterp.core.paths import DEFAULT_WIKI_DIR
from openinterp.core.project import site_url as project_site_url


def _load_runner(
    model_name: str,
    device: str,
    dtype: str,
    revision: str | None = None,
    attn_implementation: str | None = None,
):
    """Load ``model_name`` at ``revision``.

    Pass ``attn_implementation="eager"`` when you need attention patterns. SDPA and flash
    attention don't return them.
    """
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
        runner = _load(model_name, device, torch_dtype, revision, attn_implementation)
    from openinterp.harness import devices

    if devices.scans_on_ane():
        from openinterp.harness.memory import is_streamed

        if is_streamed(runner.model) and (why := devices.streamed_scans_skip_ane()):
            click.echo(f"Scans run on the CPU, not the ANE, because {why}.", err=True)
            devices.note_choice(f"{devices.choice()}; scans on the cpu, because {why}")
        else:
            _attach_ane(runner)
    return runner


def _attach_ane(runner) -> None:
    """Run ``runner``'s scans on Apple's Neural Engine, if it can run the model."""
    from openinterp.harness import devices
    from openinterp.harness.ane.backend import AneScan

    backend = AneScan.for_model(runner.model, runner.model_name, runner.model_revision)
    if backend is None:
        click.echo(
            f"Scans run on {runner.device.type}: the ANE needs coremltools (Apple-silicon "
            "Macs), and a model that passed scripts/ane_qualify.py "
            "(openinterp/harness/ane/qualified.py).",
            err=True,
        )
        return
    runner.scan_backend = backend
    devices.take_lease("ane")


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to a free CUDA GPU, else Apple's GPU (MPS) if it is free, else the CPU.

    The device planner (``openinterp/harness/devices.py``) decides if the GPU is free.
    A process that uses a GPU keeps a lease on it, so other processes see that it is busy.
    float32 activations on MPS match the CPU's to about 1e-4 of a neuron's peak. That is
    inside the tolerance of ``openinterp reproduce``.
    """
    from openinterp.harness import devices

    if devices.forced():  # a retry after a GPU failure: scans on the ANE if they can be
        if devices.ane_possible():
            devices.force_ane()
        devices.note_choice(f"{devices.forced()}: a retry after the GPU failed")
        return devices.forced()
    asked = device
    if device == "ane":  # scans on the ANE; the rest where the planner puts it
        devices.force_ane()
        device = "auto"
    if device == "auto":
        plan = devices.planned()
        if note := devices.notice():
            click.echo(note, err=True)
        device = plan.device
        rest = "scans on the ane, the rest on " if asked == "ane" else ""
        devices.note_choice(f"{asked}: {rest}{device}, because {plan.reason}")
    else:
        devices.note_choice(f"--device {device}")
    if device == "mps" and (stuck := devices.mps_stuck()):
        click.echo(f"Using the CPU, because {stuck}.", err=True)
        if devices.ane_possible():
            devices.force_ane()
        devices.note_choice(f"{asked}: cpu, because {stuck}")
        return "cpu"
    devices.take_lease(device)  # a GPU only
    return device


def _load_unit_runner(ident, device: str, dtype: str, revision: str | None = None):
    """Load the model an identifier names, with eager attention for a head."""
    from openinterp.harness.units import attn_implementation

    attn = attn_implementation(ident)
    return _load_runner(ident.model_name, device, dtype, revision, attn)


def _load(model_name, device, torch_dtype, revision, attn_implementation):
    from transformers import AutoTokenizer

    from openinterp.harness.hooks import NeuronHookRunner
    from openinterp.harness.memory import load_causal_lm

    commit = _resolve_revision(model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=commit)
    extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = load_causal_lm(model_name, device, torch_dtype, commit, **extra)
    # Nothing here trains the model. Gradients are only taken for inputs and
    # activations. A weight that requires a gradient makes autograd keep more. On a
    # streamed Qwen2.5-1.5B, integrated gradients then peaked at 3.7 GB, not 1.8 GB.
    model.eval().requires_grad_(False)
    _align_parameters(model)
    runner = NeuronHookRunner(model, tokenizer)
    runner.model_name = model_name
    runner.model_revision = commit
    return runner


def _align_parameters(model, alignment: int = 64) -> None:
    """Copy parameters that sit at a misaligned address into fresh memory.

    Weights memory-mapped from a safetensors file start where the file puts them.
    GPT-2's weights are not even 4-byte aligned. A matrix-vector product on them (as the
    vocabulary lens does) dies with SIGBUS in Apple's BLAS. Shared (tied) parameters are
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
    memory_option,
    click.option(
        "--device",
        default="auto",
        show_default=True,
        help="cpu, cuda, cuda:N, mps, ane or auto: a free CUDA GPU, else MPS if it is free, "
        "else scans on the ANE (Apple's Neural Engine) and the rest on the CPU. See "
        "docs/experiments.md, Devices.",
    ),
    click.option(
        "--dtype",
        type=click.Choice(["auto", "float32", "bfloat16", "float16"]),
        default="float32",
        show_default=True,
    ),
]


def wiki_dir_option(help: str | None = None, *, exists: bool = False):
    """The ``--wiki-dir`` option that every command shares.

    It is the registry root, by default :data:`~openinterp.core.paths.DEFAULT_WIKI_DIR`.
    """
    return click.option(
        "--wiki-dir",
        default=DEFAULT_WIKI_DIR,
        show_default=True,
        type=click.Path(exists=exists, file_okay=False, path_type=Path),
        help=help,
    )


wiki_option = wiki_dir_option(
    "Registry root: the experiment's record and page go in experiments/<slug>/. "
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
    """Click callback for model options: return the name to record.

    The name is after ``$OPENINTERP_MODEL_ALIASES``
    (``openinterp.agents.base.recorded_model``).
    """
    from openinterp.agents.base import recorded_model

    try:
        return recorded_model(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _contributor(_ctx, _param, value):
    if value and (problem := contributor_problem(value)):
        raise click.BadParameter(f"{value!r}: {problem}")
    return value or None


def contributor_option(role: str):
    """``--contributor``: the human behind this step, recorded beside the agent model."""
    return click.option(
        "--contributor",
        envvar=CONTRIBUTOR_ENV,
        callback=_contributor,
        help=f"Your public handle (e.g. GitHub username, never an email), recorded as the "
        f"{role}'s contributor (or ${CONTRIBUTOR_ENV}). Default: none recorded.",
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
        click.echo(f"Experiment: {archived}\nWiki: {page}")
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
    then; variables the user set win.
    """
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
    """Add the shared model options (:data:`model_options`) to a command."""
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
    # The stored probes are still the old reviewer's, so are its model and contributor.
    update = {"explanation_scores": scores, "contributor": old.verdict.contributor}
    verdict = verdict.model_copy(update=update)
    new = AuditRecord(submission=_add_costs(submission, scored), verdict=verdict)
    click.echo(f"{old.verdict.status} -> {verdict.status} — {verdict.notes}")
    _write_wiki(old, wiki_dir, kind="backfill")  # no-op when the original is already archived
    _write_copy(new, out)
    _write_wiki(new, wiki_dir, kind="reevaluation", previous_status=old.verdict.status)


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
    default=project_site_url,
    help="Public URL of the bucket root (default: [project.urls] Homepage). With CF_ZONE_ID "
    "and CF_API_TOKEN set, the changed URLs are purged from Cloudflare's cache.",
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
    cli_benchmark,
    cli_build,
    cli_check,
    cli_concepts,
    cli_corpus,
    cli_costs,
    cli_crossref,
    cli_evidence,
    cli_neuronpedia,
    cli_pause,
    cli_probing,
    cli_registry,
    cli_reproduce,
    cli_review,
    cli_scan,
    cli_score,
    cli_sisters,
    cli_wiring,
)


def main() -> None:
    """The console entry point: run the CLI, and turn an ``OpenInterpError`` into a message.

    If the GPU (MPS or CUDA) fails, it runs the command again on the CPU (``harness/devices.py``).
    """
    from openinterp.harness import devices

    try:
        cli(standalone_mode=True)
    except OpenInterpError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except Exception as exc:
        if gpu := devices.gpu_failed(exc):
            devices.rerun_on_cpu(exc, gpu)
        raise


if __name__ == "__main__":
    main()
