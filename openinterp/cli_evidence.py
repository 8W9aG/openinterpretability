"""``openinterp audit``, and its helpers: scan caching and prompt/answer files. Corpus
loading and the scan itself are in :mod:`openinterp.evidence`."""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import (
    cli,
    contributor_option,
    recorded_model_option,
    wiki_dir_option,
    with_model_options,
)
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS
from openinterp.evidence import (  # noqa: F401  (re-exported)
    ScanRequest,
    Unauditable,
    read_corpus,
    scan_evidence,
    sha256,
)

REGISTRY_CORPUS = Path("corpus.jsonl")

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


def record_draw(exc: Unauditable, random_pick: bool, wiki_dir: Path) -> None:
    """Keep a random draw the scan stopped on, so random-selection statistics count it."""
    from openinterp.cli import _skip_wiki
    from openinterp.core.draws import write_draw

    if random_pick and exc.draw is not None and not _skip_wiki(wiki_dir):
        path = write_draw(exc.draw, wiki_dir)
        click.echo(f"Recorded the {exc.draw.outcome.replace('_', '-')} draw: {path}", err=True)


def write_evidence(evidence, path: Path, full_snippets: bool = False) -> None:
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
        full_snippets,
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
    "gpt2-small-transcoders or gemma-scope-2b-mlp-saes; it implies --model. See "
    "openinterp/harness/dictionaries.py.",
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
    help="Text file (one snippet per line) or .jsonl with a 'text' field. "
    "Default: ./corpus.jsonl, the registry's scan corpus (see the README).",
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
    "--full-snippets",
    is_flag=True,
    help="Show the proposer whole scanned documents, without windowing or dropping "
    "near-duplicates: the prompt before trimming, for A/B comparison.",
)
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
@contributor_option("proposer")
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
@wiki_dir_option(
    "Registry root. A --random draw the scan stops on (dead or default-on) is recorded "
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
    full_snippets,
    window_tokens,
    stride,
    use_scan_cache,
    agent_model,
    contributor,
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
    unit = resolve_unit(unit, neuron, pole, head, feature, dictionary)
    if unit == "feature" and evidence_file is None:
        model_name = check_dictionary(dictionary, model_name, window_tokens)
    check_steps(evidence_file, hypothesis_file, evidence_out, superseded_file, agent_model)
    check_scan(evidence_file, model_name, stride, window_tokens)
    superseded = superseded_file and load_superseded(superseded_file)
    if evidence_file is not None:
        evidence, draft, runner, agent = load_answer(
            evidence_file, hypothesis_file, agent_model, device, dtype
        )
    else:
        request = ScanRequest(
            model_name=model_name,
            unit=unit,
            layer=layer,
            neuron=neuron,
            pole=pole,
            head=head,
            feature=feature,
            dictionary=dictionary,
            random_pick=random_pick,
            seed=seed,
            corpus=scan_corpus(corpus),
            top_k=top_k,
            dtype=dtype,
            window_tokens=window_tokens,
            stride=stride,
            max_tokens=max_tokens,
            sample_seed=sample_seed,
        )
        runner, evidence = scan_or_record(request, use_scan_cache, wiki_dir, device)
        if evidence_out is not None:
            write_evidence(evidence, evidence_out, full_snippets)
            return
        agent, draft = ask_proposer(agent_model, evidence, full_snippets)
    from openinterp.cli_submission import write_submission

    write_submission(agent, draft, runner, evidence, superseded, out, contributor)


def resolve_unit(unit, neuron, pole, head, feature, dictionary) -> str:
    """The unit ``--unit`` names, or the one the other options imply."""
    if unit is None:
        unit = "head" if head is not None else "neuron"
        unit = "feature" if feature is not None or dictionary is not None else unit
    if unit != "neuron" and (neuron is not None or pole is not None):
        raise click.UsageError(f"--neuron and --pole are for neurons; a {unit} has neither.")
    if head is not None and unit != "head":
        raise click.UsageError("--head needs --unit head (or no --unit).")
    if unit != "feature" and (feature is not None or dictionary is not None):
        raise click.UsageError("--feature and --dictionary need --unit feature (or no --unit).")
    return unit


def check_dictionary(dictionary, model_name, window_tokens) -> str:
    """The model ``--dictionary`` is for, after checking it fits the scan's windows."""
    from openinterp.harness.dictionaries import dictionary_spec

    if dictionary is None:
        raise click.UsageError("--unit feature needs --dictionary.")
    try:
        spec = dictionary_spec(dictionary)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--dictionary") from exc
    if model_name not in (None, spec.model_name):
        raise click.UsageError(f"{dictionary} is a dictionary for {spec.model_name}.")
    limit = spec.context_tokens
    if limit and (window_tokens is None or window_tokens > limit):
        raise click.UsageError(
            f"{dictionary} only fits contexts of up to {limit} tokens; "
            f"scan with --window-tokens {limit}."
        )
    return spec.model_name


def check_steps(evidence_file, hypothesis_file, evidence_out, superseded_file, agent_model):
    """Options that go together, or apart, across the one- and two-step audits."""
    if (evidence_file is None) != (hypothesis_file is None):
        raise click.UsageError("--evidence and --hypothesis go together.")
    if superseded_file is not None and evidence_out is not None:
        raise click.UsageError(
            "--supersedes goes on the step that writes the submission (with --hypothesis)."
        )
    if evidence_file is not None and evidence_out is not None:
        raise click.UsageError("--evidence-out scans; --evidence reuses a scan. Pick one.")
    if evidence_file is not None and not agent_model:
        raise click.UsageError(
            "--agent-model is required with --hypothesis: name the model that wrote it."
        )


def check_scan(evidence_file, model_name, stride, window_tokens) -> None:
    if stride is not None and (window_tokens is None or stride > window_tokens):
        raise click.UsageError("--stride needs --window-tokens, and can't exceed it.")
    if evidence_file is None and model_name is None:
        raise click.UsageError("--model is required unless --evidence is given.")


def scan_corpus(corpus: Path | None) -> Path:
    """The corpus to scan: ``--corpus``, else the registry's ``corpus.jsonl`` if it's here.
    Never the built-in sample, whose ~552 tokens make a scan that says little."""
    if corpus is not None:
        return corpus
    if REGISTRY_CORPUS.exists():
        click.echo(f"Scanning {REGISTRY_CORPUS} (no --corpus given).", err=True)
        return REGISTRY_CORPUS
    raise click.UsageError(
        f"No --corpus given and no {REGISTRY_CORPUS} here. Build the registry's scan corpus "
        "with `openinterp corpus` (see the README), copy it in, or pass --corpus."
    )


def load_answer(evidence_file, hypothesis_file, agent_model, device, dtype):
    """Step two: the evidence, the external proposer's draft, the model and the agent."""
    from openinterp.agents.proposer import HypothesisDraft, ProposerAgent
    from openinterp.cli import _load_runner  # looked up per call, so tests can swap it
    from openinterp.core import costs
    from openinterp.core.schemas import ScanEvidence
    from openinterp.harness.units import attn_implementation

    evidence = ScanEvidence.model_validate_json(evidence_file.read_text())
    draft = HypothesisDraft.model_validate_json(hypothesis_file.read_text())
    prompt_file = evidence_file.with_suffix(".prompt.md")
    costs.add_external("proposer", agent_model, prompt_file, hypothesis_file)
    attn = attn_implementation(evidence.identifier)
    runner = _load_runner(evidence.identifier.model_name, device, dtype, None, attn)
    return evidence, draft, runner, ProposerAgent(agent_model)


def scan_or_record(request, use_scan_cache: bool, wiki_dir: Path, device):
    """Load the model and scan; keep a random draw the scan stops on (see record_draw)."""
    from openinterp.cli import _load_runner  # looked up per call, so tests can swap it

    attn = "eager" if request.unit == "head" else None  # heads need attention patterns
    runner = _load_runner(request.model_name, device, request.dtype, None, attn)
    try:
        return runner, scan_evidence(runner, request, open_scan_cache(use_scan_cache))
    except Unauditable as exc:
        record_draw(exc, request.random_pick, wiki_dir)
        raise


def ask_proposer(agent_model, evidence, full_snippets: bool):
    from openinterp.agents.base import DEFAULT_MODEL
    from openinterp.agents.proposer import ProposerAgent

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
        full_snippets,
    )
    return agent, draft
