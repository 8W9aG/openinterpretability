"""``openinterp review``: adversarially review a submission and post the experiment."""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterp.cli import (
    cli,
    contributor_option,
    explanation_scores_option,
    recorded_model_option,
    wiki_option,
    with_model_options,
)


@cli.command()
@click.option("--audit-file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--agent-model",
    default=None,
    callback=recorded_model_option,
    help="Reviewer. LiteLLM model string, or external/<name> to answer a prompt file. "
    "With --probes, the name recorded for whoever wrote them.",
)
@contributor_option("reviewer")
@click.option("--n-probes", default=20, show_default=True, type=click.IntRange(10, 20))
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
@click.option(
    "--check-probes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="ProbeSet JSON to show its reviewer as the model's tokens (and, for a head, the "
    "token runs each probe repeats): writes <name>.check.prompt.md, answered by "
    "rewriting the same file, and exits. Loads only the tokenizer.",
)
@click.option(
    "--full-snippets",
    is_flag=True,
    help="Show the reviewer whole scanned documents, without windowing or dropping "
    "near-duplicates: the prompt before trimming, for A/B comparison.",
)
@click.option(
    "--dictionary-latents",
    is_flag=True,
    help="Show the reviewer the published dictionary latents aligned with the pole "
    "(`openinterp crossref`) and their descriptions, fetched from Neuronpedia.",
)
@click.option(
    "--sisters/--no-sisters",
    default=True,
    show_default=True,
    help="Show the reviewer the pole's sister neurons in other models that this registry "
    "has studied (`openinterp sisters`), with their explanations.",
)
@click.option("--out", type=click.Path(path_type=Path), help="Also write the record here.")
@explanation_scores_option
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    help="Without --explanation-scores: the scan's corpus, to score the explanation on.",
)
@click.option(
    "--judge-model",
    callback=recorded_model_option,
    help="Without --explanation-scores: the judge (default: the reviewer).",
)
@wiki_option
@with_model_options
def verify(
    audit_file,
    agent_model,
    contributor,
    n_probes,
    emit_prompt,
    probes_file,
    check_probes,
    full_snippets,
    dictionary_latents,
    sisters,
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
    from openinterp.agents.reviewer import ReviewerAgent
    from openinterp.cli import _add_costs, _load_unit_runner, _write_copy, _write_wiki
    from openinterp.cli_score import load_scores, score_submission
    from openinterp.core.agent_names import same_agent
    from openinterp.core.schemas import AuditRecord

    submission = load_submission(audit_file)
    latents = aligned_latents_text(submission, wiki_dir) if dictionary_latents else ""
    tested = [submission.sister_of.rsplit("/", 1)[0]] if submission.sister_of else []
    latents += sisters_text(submission.identifier, wiki_dir, "reviewer", tested) if sisters else ""
    if emit_prompt is not None:
        return write_review_prompt(submission, emit_prompt, n_probes, full_snippets, latents)
    if check_probes is not None:
        return write_token_check(submission, check_probes)
    check_review_options(wiki_dir, out, probes_file, agent_model, scores_file, corpus)
    scores = load_scores(scores_file, submission)
    reviewer = ReviewerAgent(agent_model or DEFAULT_MODEL)
    if same_agent(reviewer.model, submission.submitting_agent_model):
        click.echo(
            f"warning: {reviewer.model} both proposed and reviewed this, so the review is not "
            "independent. It will be recorded and badged as self-reviewed; a different "
            "reviewer model is preferred (docs/PROTOCOL.md §2).",
            err=True,
        )
    probes = review_probes(reviewer, submission, probes_file, n_probes, full_snippets, latents)
    runner = _load_unit_runner(submission.identifier, device, dtype)
    if scores is None:  # one step: score here, on the scan's corpus (checked above)
        scores = score_submission(runner, submission, corpus, judge_model or reviewer.model)
    verdict = reviewer.review(submission, runner, probes=probes)
    # The scores' costs move into the record's provenance with everything else.
    scored = scores and scores.costs
    scores = scores and scores.model_copy(update={"costs": None})
    verdict = verdict.model_copy(update={"explanation_scores": scores, "contributor": contributor})

    record = AuditRecord(submission=_add_costs(submission, scored), verdict=verdict)
    click.echo(f"{verdict.status} (confidence {verdict.confidence_score:.2f}) — {verdict.notes}")
    _write_copy(record, out)
    _write_wiki(record, wiki_dir, kind="review")


def load_submission(audit_file: Path):
    """The submission in ``audit_file``, or the one an existing audit record re-reviews."""
    from openinterp.cli import _clear_costs
    from openinterp.core.schemas import AuditRecord, NeuronAuditSubmission

    raw = audit_file.read_text()
    if "submission" in json.loads(raw):  # an existing audit record: re-review its submission
        submission = AuditRecord.model_validate_json(raw).submission
        return _clear_costs(submission)  # its audit belonged to another experiment
    return NeuronAuditSubmission.model_validate_json(raw)


def sisters_text(identifier, wiki_dir, role: str, exclude=()) -> str:
    """The agent's section on the unit's studied sister neurons, or "" if there are none;
    ``exclude`` leaves out the pole a sister study tests."""
    from openinterp.cli import _skip_wiki
    from openinterp.sisters import sister_context

    if _skip_wiki(wiki_dir):
        return ""
    text = sister_context(identifier, wiki_dir, role, exclude=exclude)
    if text:
        click.echo(f"Showing the {role} {text.count(chr(10)) - 1} studied sister neurons", err=True)
    return text


def aligned_latents_text(submission, wiki_dir) -> str:
    """The reviewer's section on aligned dictionary latents, or "" if there are none."""
    from openinterp.builder.wiki import load_archive
    from openinterp.cli import _skip_wiki
    from openinterp.crossref import describe, format_for_reviewer, load_rows
    from openinterp.crossref import registry_explanations as explanations

    rows = load_rows().get(submission.identifier.slug, [])
    if not rows:
        click.echo(f"warning: no aligned latents for {submission.identifier.slug}", err=True)
        return ""
    registry = {} if _skip_wiki(wiki_dir) else explanations(load_archive(wiki_dir))
    return format_for_reviewer(describe(rows, registry))


def write_review_prompt(
    submission, emit_prompt: Path, n_probes: int, full_snippets: bool, latents: str = ""
):
    from openinterp.agents.reviewer import ProbeSet, ReviewerAgent
    from openinterp.cli_evidence import answer_path

    answer = answer_path(emit_prompt, ".probes.json")
    task = ReviewerAgent.probe_prompt(submission, n_probes, full_snippets, latents)
    emit_prompt.parent.mkdir(parents=True, exist_ok=True)
    emit_prompt.write_text(ReviewerAgent.render_prompt(ProbeSet, task, answer))
    click.echo(f"Reviewer prompt: {emit_prompt} (answer in {answer})")


def write_token_check(submission, check_probes: Path):
    from transformers import AutoTokenizer

    from openinterp.agents.reviewer import ProbeSet, ReviewerAgent

    probes = ProbeSet.model_validate_json(check_probes.read_text()).probes
    revision = submission.provenance and submission.provenance.model_revision
    tokenizer = AutoTokenizer.from_pretrained(submission.identifier.model_name, revision=revision)
    task = ReviewerAgent.token_check_prompt(submission, probes, tokenizer)
    stem = check_probes.name.removesuffix(".json").removesuffix(".probes")
    prompt = check_probes.with_name(stem + ".check.prompt.md")
    prompt.write_text(ReviewerAgent.render_prompt(ProbeSet, task, check_probes))
    click.echo(f"Token check: {prompt} (answer by rewriting {check_probes})")


def check_review_options(wiki_dir, out, probes_file, agent_model, scores_file, corpus) -> None:
    from openinterp.cli import _skip_wiki

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


def review_probes(
    reviewer, submission, probes_file, n_probes: int, full_snippets: bool, latents: str = ""
):
    """The probes in ``probes_file`` (an external reviewer's), or the reviewer LLM's."""
    from openinterp.agents.reviewer import ProbeSet
    from openinterp.core import costs

    if probes_file is not None:
        probes = ProbeSet.model_validate_json(probes_file.read_text()).probes
        # --emit-prompt review.prompt.md is answered in review.probes.json.
        stem = probes_file.name.removesuffix(".probes.json")
        prompt_file = probes_file.with_name(stem + ".prompt.md")
        costs.add_external("reviewer", reviewer.model, prompt_file, probes_file)
        return probes
    click.echo(f"Generating {n_probes} adversarial probes with {reviewer.model}…", err=True)
    return reviewer.generate_probes(
        submission, n=n_probes, full_snippets=full_snippets, related=latents
    ).probes


cli.add_command(verify, name="review")
