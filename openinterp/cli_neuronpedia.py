"""Neuronpedia's explanations through the adversarial review.

The command is ``open-interpretability neuronpedia-review``.
See :mod:`openinterp.neuronpedia_review` for the rule and the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterp.cli import cli, recorded_model_option, wiki_dir_option, with_model_options

N_PROBES = 20


@cli.command("neuronpedia-review")
@wiki_dir_option()
@click.option(
    "--work-dir",
    default="exchange/neuronpedia-review",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Submissions, reviewer prompts, probe sets and measured verdicts.",
)
@click.option(
    "--reviewer-model",
    default="external/claude-sonnet-5-5",
    show_default=True,
    callback=recorded_model_option,
    help="The model that wrote the probe sets under --work-dir.",
)
@click.option(
    "--out",
    default="docs/analyses/neuronpedia-review",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Writes <out>.jsonl (one row per latent) and <out>.md (the comparison).",
)
@with_model_options
def neuronpedia_review(wiki_dir, work_dir, reviewer_model, out, device, dtype):
    """Review Neuronpedia's explanation of each audited latent, against the agent's.

    Nothing is written to the registry, and nothing is sent to Neuronpedia.

    \b
    Run it once to write a reviewer prompt per latent under --work-dir.
    Answer each prompt with its own neuron-reviewer.
    Then run it again to measure the probes and write the comparison.
    """
    from openinterp.neuronpedia_review import report

    rows, pending, runners = [], 0, {}
    for slug, record in sorted(_newest_latent_records(wiki_dir).items()):
        row = _row(record, work_dir / slug, reviewer_model, runners, device)
        if row is None:
            continue
        pending += row.neuronpedia is None
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
    out.with_suffix(".md").write_text(report(rows))
    click.echo(report(rows))
    if pending:
        click.echo(f"{pending} latents await probe sets under {work_dir}.")
    click.echo(f"Wrote {out.with_suffix('.jsonl')} and {out.with_suffix('.md')}")


def _newest_latent_records(wiki_dir) -> dict:
    """The newest reviewed record of each dictionary latent that Neuronpedia hosts, by slug."""
    from openinterp.builder.wiki import load_archive_paths
    from openinterp.crossref import latent_url

    newest = {}
    for _, r in load_archive_paths(wiki_dir):
        ident = r.submission.identifier
        if ident.kind != "dictionary_feature" or r.verdict is None or r.verdict.metrics is None:
            continue
        if latent_url(ident.dictionary, ident.layer, ident.feature_idx) is None:
            continue
        old = newest.get(ident.slug)
        if old is None or r.verdict.timestamp > old.verdict.timestamp:
            newest[ident.slug] = r
    return newest


def _row(record, unit_dir: Path, reviewer_model: str, runners: dict, device: str):
    """The latent's ReviewRow; ``None`` if Neuronpedia has no machine-written explanation."""
    from openinterp.builder.wiki import experiment_id
    from openinterp.crossref import latent_explanations, latent_url, machine_explanation
    from openinterp.neuronpedia_review import ReviewRow, side

    ident = record.submission.identifier
    found = latent_explanations(ident.dictionary, ident.layer, ident.feature_idx)
    if found is None:
        raise click.ClickException(f"could not read Neuronpedia for {ident}; try again later")
    theirs = machine_explanation(found)
    if theirs is None:
        return None
    sub = baseline_submission(record, theirs)
    verdict = _verdict(record, sub, unit_dir, reviewer_model, runners, device)
    return ReviewRow(
        slug=ident.slug,
        experiment_id=experiment_id(record),
        url=latent_url(ident.dictionary, ident.layer, ident.feature_idx),
        agent_status=record.verdict.status,
        agent_explanation=record.submission.natural_language_explanation,
        neuronpedia_explanation=theirs.description,
        explainer=theirs.explainer,
        reviewer_model=verdict and reviewer_model,
        agent=side(record.verdict.metrics),
        neuronpedia=verdict and side(verdict.metrics),
    )


def baseline_submission(record, theirs):
    """A copy of the record's submission that claims only Neuronpedia's explanation.

    It claims no activating tokens and a retrospective mechanism, so no claim of the
    agent's reaches the reviewer. It is credited to Neuronpedia's explainer, so the review
    counts as independent. It keeps the scan evidence the reviewer sees for any hypothesis.
    """
    from openinterp.cli import _clear_costs

    sub = _clear_costs(record.submission)
    return sub.model_copy(
        update={
            "natural_language_explanation": theirs.description,
            "activating_tokens": [],
            "lookahead": sub.lookahead
            and sub.lookahead.model_copy(
                update={"is_lookahead": False, "target_upcoming_tokens": []}
            ),
            "submitting_agent_model": f"neuronpedia/{theirs.explainer}",
            "contributor": None,
            "supersedes": None,
            "sister_of": None,
            "prior_art": None,  # else the reviewer reads Neuronpedia's text twice
        }
    )


def _verdict(record, sub, unit_dir: Path, reviewer_model: str, runners: dict, device: str):
    """The measured verdict of ``sub``: cached, or measured now, or ``None`` without probes.

    The first time, it writes the reviewer's prompt.
    """
    from openinterp.agents.reviewer import ProbeSet, ReviewerAgent
    from openinterp.cli import _load_unit_runner
    from openinterp.cli_review import refuse_unisolated, write_review_prompt
    from openinterp.core.schemas import VerificationVerdict
    from openinterp.reproduce import pinned_environment

    prompt, probes = unit_dir / "review.prompt.md", unit_dir / "review.probes.json"
    measured = unit_dir / "verdict.json"
    (unit_dir / "submission.json").parent.mkdir(parents=True, exist_ok=True)
    (unit_dir / "submission.json").write_text(sub.model_dump_json(indent=2))
    if measured.exists():
        return VerificationVerdict.model_validate_json(measured.read_text())
    if not probes.exists():
        if not prompt.exists():
            write_review_prompt(sub, prompt, N_PROBES, full_snippets=False)
        return None
    refuse_unisolated(probes)
    revision, pinned = pinned_environment(record)
    key = (sub.identifier.model_name, revision, pinned)
    if key not in runners:
        runners[key] = _load_unit_runner(sub.identifier, device, pinned, revision)
    click.echo(f"Measuring the probes for {sub.identifier}…", err=True)
    probe_set = ProbeSet.model_validate_json(probes.read_text()).probes
    verdict = ReviewerAgent(reviewer_model).review(sub, runners[key], probes=probe_set)
    measured.write_text(json.dumps(json.loads(verdict.model_dump_json()), indent=2))
    return verdict
