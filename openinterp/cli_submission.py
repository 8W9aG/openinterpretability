"""The steps of ``openinterp audit`` after the proposer answers: counterfactuals, the
Patchscope, steering and wiring readouts, and writing the submission."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli_evidence import with_costs


def read_patchscope(runner, submission):
    """The model's own reading of the unit's output direction, once the proposer has
    answered so it can't steer the hypothesis; ``None`` for a head, which writes a subspace."""
    from openinterp.core import costs
    from openinterp.harness.patchscope import patchscope, write_direction
    from openinterp.harness.units import submission_unit

    direction = write_direction(runner, submission_unit(submission))
    if direction is None or runner.tokenizer is None:
        return None
    click.echo("Reading the output direction back through a Patchscope…", err=True)
    with costs.stage("patchscope"):
        return patchscope(runner, submission.identifier.layer, direction)


def read_steering(runner, submission):
    """Neutral prompts continued with the unit clamped high, at baseline and ablated, once
    the proposer has answered; ``None`` for a head, which has no "high" to clamp to."""
    from openinterp.core import costs
    from openinterp.harness.steering import steering_samples
    from openinterp.harness.units import HeadUnit, submission_unit

    unit = submission_unit(submission)
    if isinstance(unit, HeadUnit) or not submission.max_activation or runner.tokenizer is None:
        return None
    click.echo("Continuing neutral prompts with the unit clamped and ablated…", err=True)
    with costs.stage("steering"):
        return steering_samples(runner, unit, submission.max_activation)


def read_wiring(runner, submission):
    """The unit's top connections by virtual weight, once the proposer has answered;
    ``None`` on an unsupported architecture."""
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


def write_submission(
    agent, draft, runner, evidence, superseded, out: Path | None, contributor: str | None = None
) -> None:
    """Build the submission (counterfactuals, Patchscope, steering, wiring) and write it."""
    from openinterp.builder.wiki import experiment_id
    from openinterp.core import costs

    ident = evidence.identifier
    if superseded is not None and superseded.submission.identifier.slug != ident.slug:
        raise click.UsageError(
            f"--supersedes names an experiment on {superseded.submission.identifier}, "
            f"not on {ident}."
        )
    submission = build_submission(agent, draft, runner, evidence)
    readout = read_patchscope(runner, submission)
    steering = read_steering(runner, submission)
    wiring = read_wiring(runner, submission)
    # An evidence file carries the scan's costs; a one-step audit measured them here.
    provenance = with_costs(submission.provenance, costs.measured())
    revises = superseded and experiment_id(superseded)
    submission = submission.model_copy(
        update={
            "contributor": contributor,
            "opposite_pole": evidence.opposite_pole,
            "provenance": provenance,
            "supersedes": revises,
            "patchscope": readout,
            "steering": steering,
            "wiring": wiring,
        }
    )
    out = out or Path("submissions") / f"{ident.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(submission.model_dump_json(indent=2))
    click.echo(f"Hypothesis: {submission.natural_language_explanation}")
    if revises:
        click.echo(f"Supersedes: {revises} ({superseded.verdict.status})")
    click.echo(f"Wrote {out}")


def build_submission(agent, draft, runner, evidence):
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
            )
        except ValueError as exc:  # e.g. a one-token counterfactual prompt
            raise click.ClickException(f"Counterfactual: {exc}") from exc
