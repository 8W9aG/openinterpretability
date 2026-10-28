"""``openinterp audit``, and its helpers: scan caching and prompt/answer files. Corpus
loading and the scan itself are in :mod:`openinterp.evidence`."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli, recorded_model_option, with_model_options
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS
from openinterp.evidence import (  # noqa: F401  (re-exported)
    Unauditable,
    read_corpus,
    scan_evidence,
    sha256,
)

scan_cache_option = click.option(
    "--scan-cache/--no-scan-cache",
    "use_scan_cache",
    default=True,
    show_default=True,
    help="Replay a cached scan of the layer, or store this one (see `openinterp scan-cache`).",
)


def open_scan_cache(enabled: bool):
    from openinterp.harness.layer_cache import LayerCache

    return LayerCache.default() if enabled else None


def load_superseded(path: Path):
    """The archived, reviewed record that ``audit --supersedes`` names."""
    from openinterp.core.schemas import AuditRecord

    try:
        record = AuditRecord.model_validate_json(path.read_text())
    except ValueError as exc:
        raise click.BadParameter(f"not an audit record: {exc}", param_hint="--supersedes") from exc
    if record.verdict is None:
        raise click.BadParameter("that record has no verdict yet", param_hint="--supersedes")
    return record


def answer_path(prompt_path: Path, suffix: str) -> Path:
    """``review.prompt.md`` or ``review.md`` -> ``review<suffix>``."""
    name = prompt_path.name.removesuffix(".md").removesuffix(".prompt")
    return prompt_path.with_name(name + suffix)


def with_costs(provenance, *parts):
    """``provenance`` with ``parts`` merged into its costs; unchanged if it is ``None``."""
    from openinterp.core.costs import merge

    if provenance is None:
        return None
    return provenance.model_copy(update={"costs": merge(provenance.costs, *parts)})


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


def record_draw(exc: Unauditable, random_pick: bool, wiki_dir: Path) -> None:
    """Keep a random draw the scan stopped on, so random-selection statistics count it."""
    from openinterp.cli import _skip_wiki
    from openinterp.core.draws import write_draw

    if random_pick and exc.draw is not None and not _skip_wiki(wiki_dir):
        path = write_draw(exc.draw, wiki_dir)
        click.echo(f"Recorded the {exc.draw.outcome.replace('_', '-')} draw: {path}", err=True)


def write_evidence(evidence, path: Path) -> None:
    from openinterp.agents.proposer import HypothesisDraft, ProposerAgent
    from openinterp.core import costs

    evidence = evidence.model_copy(
        update={"provenance": with_costs(evidence.provenance, costs.measured())}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(evidence.model_dump_json(indent=2))
    prompt_path = path.with_suffix(".prompt.md")
    answer_path = path.with_suffix(".hypothesis.json")
    task = ProposerAgent.proposal_prompt(
        evidence.identifier,
        evidence.top_examples,
        evidence.vocabulary_lens,
        evidence.confounds,
        evidence.correlated_neurons,
        evidence.head_profile,
        evidence.input_attribution,
        evidence.dictionary_profile,
        bool(evidence.first_token_floor),
    )
    prompt_path.write_text(ProposerAgent.render_prompt(HypothesisDraft, task, answer_path))
    click.echo(f"Unit: {evidence.identifier}")
    click.echo(f"Wrote {path}")
    click.echo(f"Proposer prompt: {prompt_path} (answer in {answer_path})")


@cli.command()
@click.option(
    "--model",
    "model_name",
    help="Hugging Face repo, e.g. Qwen/Qwen2.5-0.5B. Required unless --evidence.",
)
@click.option("--layer", type=int, help="Required unless --random.")
@click.option("--neuron", type=int, help="Required unless --random or --head.")
@click.option("--pole", type=click.Choice(["positive", "negative"]), help="Default: positive.")
@click.option(
    "--head",
    type=int,
    help="Audit this attention head of --layer instead of a neuron (implies --unit head).",
)
@click.option(
    "--feature",
    type=int,
    help="Audit this latent of --dictionary on --layer instead of a neuron (implies --unit "
    "feature).",
)
@click.option(
    "--dictionary",
    help="Sparse dictionary (transcoder or SAE) whose latent to audit, e.g. "
    "gpt2-small-transcoders; it implies --model. See openinterp/harness/dictionaries.py.",
)
@click.option(
    "--unit",
    type=click.Choice(["neuron", "head", "feature"]),
    help="What --random samples: an MLP neuron pole (default), an attention head, or a "
    "latent of --dictionary (the learned-basis baseline).",
)
@click.option(
    "--random",
    "random_pick",
    is_flag=True,
    help="Sample unspecified layer/neuron/pole (or head) uniformly: the unbiased selection.",
)
@click.option("--seed", type=int, help="Seed for --random (recorded in provenance).")
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    help="Text file (one snippet per line) or .jsonl with a 'text' field.",
)
@click.option(
    "--max-tokens",
    default=DEFAULT_SAMPLE_TOKENS,
    show_default=True,
    type=click.IntRange(0),
    help="Scan a seeded sample of whole documents with at most this many tokens, so the "
    "scan's cost and scale don't grow with the corpus (recorded in provenance). "
    "0 scans every document.",
)
@click.option(
    "--sample-seed",
    default=DEFAULT_SAMPLE_SEED,
    show_default=True,
    type=int,
    help="Seed for the --max-tokens sample. Keep the default so audits of one corpus read "
    "the same documents.",
)
@click.option("--top-k", default=20, show_default=True, help="Examples shown to the proposer.")
@click.option(
    "--window-tokens",
    type=click.IntRange(2),
    help="Scan each document as overlapping windows of this many tokens (≥ 512 recommended) "
    "instead of whole. Top examples are then windows. Default: whole documents.",
)
@click.option(
    "--stride",
    type=click.IntRange(1),
    help="Tokens between window starts (default: half of --window-tokens). Each token counts "
    "once, with at least --window-tokens minus --stride tokens of context.",
)
@scan_cache_option
@click.option(
    "--agent-model",
    default=None,
    callback=recorded_model_option,
    help="Proposer. LiteLLM model string, or external/<name> to answer a prompt file. "
    "With --hypothesis, the name recorded for whoever wrote it (e.g. external/claude-opus-5-5).",
)
@click.option(
    "--evidence-out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Scan only: write the evidence to this JSON file and the proposer's prompt next to "
    "it (<name>.prompt.md), then exit. Answer the prompt, then run with --evidence.",
)
@click.option(
    "--evidence",
    "evidence_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Evidence from --evidence-out; skips the scan. Requires --hypothesis.",
)
@click.option(
    "--hypothesis",
    "hypothesis_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="HypothesisDraft JSON answering the --evidence-out prompt. No LLM is called.",
)
@click.option(
    "--supersedes",
    "superseded_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Archived record (wiki/experiments/<slug>/<id>.json) this submission revises, e.g. a "
    "NEEDS_REVISION one. Recorded in the submission; it must be on the same unit.",
)
@click.option(
    "--out", type=click.Path(path_type=Path), help="Output JSON (default: submissions/<slug>.json)"
)
@click.option(
    "--wiki-dir",
    default=Path("wiki"),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Registry root. A --random draw the scan stops on (dead or default-on) is recorded "
    "in draws/<slug>/, as an outcome of random selection. Pass --wiki-dir '' to skip.",
)
@with_model_options
def audit(
    model_name,
    layer,
    neuron,
    pole,
    head,
    feature,
    dictionary,
    unit,
    random_pick,
    seed,
    corpus,
    max_tokens,
    sample_seed,
    top_k,
    window_tokens,
    stride,
    use_scan_cache,
    agent_model,
    evidence_out,
    evidence_file,
    hypothesis_file,
    superseded_file,
    out,
    wiki_dir,
    device,
    dtype,
):
    """Scan a neuron pole, attention head or dictionary feature, get a hypothesis, and write
    a submission.

    \b
    In one step, the proposer is an LLM (or external/<name>, answered via exchange/).
    In two steps, any agent can propose without an API key or a waiting process:
      openinterp audit --model M --random --evidence-out ev.json
      (answer ev.prompt.md by writing ev.hypothesis.json)
      openinterp audit --evidence ev.json --hypothesis ev.hypothesis.json \\
        --agent-model external/<model that wrote it>
    """
    from openinterp.agents.base import DEFAULT_MODEL
    from openinterp.agents.proposer import HypothesisDraft, ProposerAgent
    from openinterp.builder.wiki import experiment_id
    from openinterp.cli import _load_runner  # looked up per call, so tests can swap it
    from openinterp.core import costs
    from openinterp.core.schemas import ScanEvidence
    from openinterp.harness.dictionaries import dictionary_spec
    from openinterp.harness.units import attn_implementation

    if unit is None:
        unit = "head" if head is not None else "neuron"
        unit = "feature" if feature is not None or dictionary is not None else unit
    if unit != "neuron" and (neuron is not None or pole is not None):
        raise click.UsageError(f"--neuron and --pole are for neurons; a {unit} has neither.")
    if head is not None and unit != "head":
        raise click.UsageError("--head needs --unit head (or no --unit).")
    if unit != "feature" and (feature is not None or dictionary is not None):
        raise click.UsageError("--feature and --dictionary need --unit feature (or no --unit).")
    if unit == "feature" and evidence_file is None:
        if dictionary is None:
            raise click.UsageError("--unit feature needs --dictionary.")
        try:
            spec = dictionary_spec(dictionary)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--dictionary") from exc
        if model_name not in (None, spec.model_name):
            raise click.UsageError(f"{dictionary} is a dictionary for {spec.model_name}.")
        model_name = spec.model_name
        limit = spec.context_tokens
        if limit and (window_tokens is None or window_tokens > limit):
            raise click.UsageError(
                f"{dictionary} only fits contexts of up to {limit} tokens; "
                f"scan with --window-tokens {limit}."
            )
    if (evidence_file is None) != (hypothesis_file is None):
        raise click.UsageError("--evidence and --hypothesis go together.")
    if stride is not None and (window_tokens is None or stride > window_tokens):
        raise click.UsageError("--stride needs --window-tokens, and can't exceed it.")
    if superseded_file is not None and evidence_out is not None:
        raise click.UsageError(
            "--supersedes goes on the step that writes the submission (with --hypothesis)."
        )
    superseded = superseded_file and load_superseded(superseded_file)
    if evidence_file is not None:
        if evidence_out is not None:
            raise click.UsageError("--evidence-out scans; --evidence reuses a scan. Pick one.")
        if not agent_model:
            raise click.UsageError(
                "--agent-model is required with --hypothesis: name the model that wrote it."
            )
        evidence = ScanEvidence.model_validate_json(evidence_file.read_text())
        draft = HypothesisDraft.model_validate_json(hypothesis_file.read_text())
        prompt_file = evidence_file.with_suffix(".prompt.md")
        costs.add_external("proposer", agent_model, prompt_file, hypothesis_file)
        attn = attn_implementation(evidence.identifier)
        runner = _load_runner(evidence.identifier.model_name, device, dtype, None, attn)
        agent = ProposerAgent(agent_model)
    else:
        if model_name is None:
            raise click.UsageError("--model is required unless --evidence is given.")
        attn = "eager" if unit == "head" else None  # heads need attention patterns
        runner = _load_runner(model_name, device, dtype, None, attn)
        try:
            evidence = scan_evidence(
                runner,
                model_name,
                layer,
                neuron,
                pole,
                random_pick,
                seed,
                corpus,
                top_k,
                dtype,
                window_tokens=window_tokens,
                stride=stride,
                head=head,
                unit=unit,
                max_tokens=max_tokens,
                sample_seed=sample_seed,
                scan_cache=open_scan_cache(use_scan_cache),
                feature=feature,
                dictionary=dictionary,
            )
        except Unauditable as exc:
            record_draw(exc, random_pick, wiki_dir)
            raise
        if evidence_out is not None:
            write_evidence(evidence, evidence_out)
            return
        agent = ProposerAgent(agent_model or DEFAULT_MODEL)
        click.echo(f"Asking {agent.model} for a hypothesis…", err=True)
        draft = agent.propose(
            evidence.identifier,
            evidence.top_examples,
            evidence.vocabulary_lens,
            evidence.confounds,
            evidence.correlated_neurons,
            evidence.head_profile,
            evidence.input_attribution,
            evidence.dictionary_profile,
            bool(evidence.first_token_floor),
        )

    ident = evidence.identifier
    if superseded is not None and superseded.submission.identifier.slug != ident.slug:
        raise click.UsageError(
            f"--supersedes names an experiment on {superseded.submission.identifier}, "
            f"not on {ident}."
        )
    with costs.stage("counterfactuals"):
        try:
            submission = agent.build_submission(
                ident,
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
    readout = read_patchscope(runner, submission)
    steering = read_steering(runner, submission)
    wiring = read_wiring(runner, submission)
    # An evidence file carries the scan's costs; a one-step audit measured them here.
    provenance = with_costs(submission.provenance, costs.measured())
    revises = superseded and experiment_id(superseded)
    submission = submission.model_copy(
        update={
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
