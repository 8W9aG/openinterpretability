"""The steps of ``open-interpretability audit`` after the proposer answers.

The steps are counterfactuals, the Patchscope, steering, WordNet specificity and wiring
readouts, and writing the submission.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli_evidence import with_costs
from openinterp.core.paths import DEFAULT_SUBMISSIONS_DIR


def read_patchscope(runner, submission):
    """Return the model's own reading of the unit's output direction.

    It runs after the proposer answers, so it can't steer the hypothesis. ``None`` for a
    head, which writes a subspace.
    """
    from openinterp.core import costs
    from openinterp.harness.patchscope import patchscope, write_direction
    from openinterp.harness.units import submission_unit

    direction = write_direction(runner, submission_unit(submission))
    if direction is None or runner.tokenizer is None:
        return None
    click.echo("Reading the output direction back through a Patchscope…", err=True)
    with costs.stage("patchscope"):
        return patchscope(runner, submission.identifier.layer, direction)


def read_prior_art(runner, submission):
    """Return published explanations of the unit, or of latents aligned with it.

    It runs after the proposer answers, so it can't steer the hypothesis. The reviewer
    sees it (:mod:`openinterp.prior_art`).
    """
    from openinterp.prior_art import enabled, find_prior_art

    if enabled():
        click.echo("Looking up prior art on Neuronpedia…", err=True)
    return find_prior_art(submission, runner)


def read_steering(runner, submission):
    """Return neutral prompts continued at baseline, with the unit ablated and clamped high.

    It runs after the proposer answers. ``None`` for a head, which has no "high" to clamp
    to.
    """
    from openinterp.core import costs
    from openinterp.harness.steering import steering_samples
    from openinterp.harness.units import HeadUnit, submission_unit

    unit = submission_unit(submission)
    if isinstance(unit, HeadUnit) or not submission.max_activation or runner.tokenizer is None:
        return None
    click.echo("Continuing neutral prompts with the unit clamped and ablated…", err=True)
    with costs.stage("steering"):
        return steering_samples(runner, unit, submission.max_activation)


def read_specificity(runner, submission, draft):
    """Return the unit on a WordNet category's held-out words against its siblings' words.

    The category is the draft's ``wordnet_synset``, or else the largest category of the
    peak words. It runs after the proposer answers. ``None`` without WordNet, without a
    category, or when either side has too few words.
    """
    from openinterp import wordnet
    from openinterp.core import costs
    from openinterp.harness.specificity import specificity
    from openinterp.harness.units import submission_unit

    lex = wordnet.lexicon()
    if lex is None or not submission.max_activation or runner.tokenizer is None:
        return None
    peaks = wordnet.peak_words(submission.top_examples)
    synset, source = draft.wordnet_synset, "proposer"
    if synset is not None and not lex.has(synset):
        click.echo(f"Skipping the WordNet check: no noun synset {synset!r}.", err=True)
        return None
    if synset is None:
        groups = wordnet.categorise(peaks, lex).groups
        if not groups:
            return None
        synset, source = groups[0].synset, "peak_words"
    click.echo(f"Measuring {synset} words against its sibling categories…", err=True)
    with costs.stage("specificity"):
        unit = submission_unit(submission)
        return specificity(runner, unit, lex, synset, source, submission.max_activation, peaks)


def read_wiring(runner, submission):
    """Return the unit's top connections by virtual weight.

    It runs after the proposer answers. ``None`` on an unsupported architecture.
    """
    from openinterp.core import costs
    from openinterp.core.exceptions import ArchitectureNotSupportedError
    from openinterp.harness.units import submission_unit
    from openinterp.harness.wiring import weight_wiring

    click.echo("Reading the unit's wiring from the weights…", err=True)
    with costs.stage("wiring"):
        try:
            return weight_wiring(runner, submission_unit(submission))
        except ArchitectureNotSupportedError as exc:
            click.echo(f"Skipping wiring: {exc}", err=True)
            return None


def readouts(runner, submission, draft) -> dict:
    """The readouts taken after the proposer answers, by submission field."""
    return {
        "patchscope": read_patchscope(runner, submission),
        "steering": read_steering(runner, submission),
        "wordnet_specificity": read_specificity(runner, submission, draft),
        "wiring": read_wiring(runner, submission),
        "prior_art": read_prior_art(runner, submission),
    }


def write_submission(
    agent,
    draft,
    runner,
    evidence,
    superseded,
    out: Path | None,
    contributor: str | None = None,
    sister_file: Path | None = None,
) -> None:
    """Build the submission (counterfactuals, Patchscope, steering, specificity, wiring), write it.

    In a sister study (``sister_file``), the explanation is the sister's, word for word.
    """
    from openinterp.builder.wiki import experiment_id
    from openinterp.core import costs
    from openinterp.sisters import sister_ref

    ident = evidence.identifier
    if superseded is not None and superseded.submission.identifier.slug != ident.slug:
        raise click.UsageError(
            f"--supersedes names an experiment on {superseded.submission.identifier}, "
            f"not on {ident}."
        )
    sister, draft = sister_draft(sister_file, ident, draft)
    submission = build_submission(agent, draft, runner, evidence)
    after = readouts(runner, submission, draft)
    # An evidence file carries the scan's costs; a one-step audit measured them here.
    provenance = with_costs(submission.provenance, costs.measured())
    revises = superseded and experiment_id(superseded)
    submission = submission.model_copy(
        update={
            "contributor": contributor,
            "opposite_pole": evidence.opposite_pole,
            "provenance": provenance,
            "supersedes": revises,
            "sister_of": sister and sister_ref(sister),
            **after,
        }
    )
    out = out or DEFAULT_SUBMISSIONS_DIR / f"{ident.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(submission.model_dump_json(indent=2))
    click.echo(f"Hypothesis: {submission.natural_language_explanation}")
    if revises:
        click.echo(f"Supersedes: {revises} ({superseded.verdict.status})")
    if sister:
        click.echo(f"Sister study of: {submission.sister_of}")
    click.echo(f"Wrote {out}")


def sister_draft(sister_file, ident, draft):
    """Return ``(sister record, draft)``.

    In a sister study, the draft has the sister's explanation, word for word. Otherwise
    the result is ``(None, draft)``.
    """
    from openinterp.cli_evidence import load_sister

    if sister_file is None:
        return None, draft
    sister = load_sister(sister_file, ident)
    text = sister.submission.natural_language_explanation
    return sister, draft.model_copy(update={"explanation": text})


def build_submission(agent, draft, runner, evidence):
    """Build the submission from the proposer's draft, timed as the counterfactuals stage."""
    from openinterp.core import costs

    with costs.stage("counterfactuals"):
        try:
            return agent.build_submission(
                evidence.identifier,
                draft,
                runner=runner,
                examples=evidence.top_examples,
                histogram=evidence.activation_histogram,
                provenance=evidence.provenance,
                confounds=evidence.confounds,
                correlated=evidence.correlated_neurons,
                vocabulary_lens=evidence.vocabulary_lens,
                input_attribution=evidence.input_attribution,
                head_profile=evidence.head_profile,
                dictionary_profile=evidence.dictionary_profile,
                first_token_floor=evidence.first_token_floor,
                superposition_profile=evidence.superposition_profile,
            )
        except ValueError as exc:  # e.g. a one-token counterfactual prompt
            raise click.ClickException(f"Counterfactual: {exc}") from exc
