"""``openinterp review``: adversarially review a submission and post the experiment."""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterp.cli import (
    cli,
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
    n_probes,
    emit_prompt,
    probes_file,
    check_probes,
    full_snippets,
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
    from openinterp.cli import (
        _add_costs,
        _clear_costs,
        _load_unit_runner,
        _skip_wiki,
        _write_copy,
        _write_wiki,
    )
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
        task = ReviewerAgent.probe_prompt(submission, n=n_probes, full_snippets=full_snippets)
        emit_prompt.parent.mkdir(parents=True, exist_ok=True)
        emit_prompt.write_text(ReviewerAgent.render_prompt(ProbeSet, task, answer))
        click.echo(f"Reviewer prompt: {emit_prompt} (answer in {answer})")
        return
    if check_probes is not None:
        from transformers import AutoTokenizer

        probes = ProbeSet.model_validate_json(check_probes.read_text()).probes
        revision = submission.provenance and submission.provenance.model_revision
        tokenizer = AutoTokenizer.from_pretrained(
            submission.identifier.model_name, revision=revision
        )
        task = ReviewerAgent.token_check_prompt(submission, probes, tokenizer)
        stem = check_probes.name.removesuffix(".json").removesuffix(".probes")
        prompt = check_probes.with_name(stem + ".check.prompt.md")
        prompt.write_text(ReviewerAgent.render_prompt(ProbeSet, task, check_probes))
        click.echo(f"Token check: {prompt} (answer by rewriting {check_probes})")
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
        probe_set = reviewer.generate_probes(submission, n=n_probes, full_snippets=full_snippets)
        probes = probe_set.probes
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
