"""``openinterp audit``, and its helpers: corpus loading, the scan, and prompt/answer files."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

import click

from openinterp.cli import cli, with_model_options
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS


def read_corpus(path: Path | None):
    """Return (snippets, description, sha256 of the corpus file, its CorpusSource or None)."""
    from openinterp.corpus import load_source

    if path is None:
        from openinterp.data import sample_corpus

        texts = sample_corpus()
        return texts, "builtin:sample_corpus", sha256("\n".join(texts).encode()), None
    raw = path.read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        lines = [json.loads(line)["text"] for line in lines]
    digest = sha256(raw)
    source = load_source(path, digest)
    return lines, str(source) if source else path.name, digest, source


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def scan_evidence(
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
    window_tokens=None,
    stride=None,
    head=None,
    unit="neuron",
    max_tokens=None,
    sample_seed=0,
    scan_cache=None,
    feature=None,
    dictionary=None,
):
    """Pick the unit, scan the corpus (or a ``max_tokens`` sample of it), then read the
    vocabulary lens (neurons and dictionary features) and attribute the top examples'
    peaks to their context tokens (neurons only). A neuron scan replays ``scan_cache``
    when it holds the layer."""
    import random

    from openinterp import __version__
    from openinterp.core import costs
    from openinterp.core.schemas import (
        ExperimentProvenance,
        FeatureIdentifier,
        HeadIdentifier,
        NeuronIdentifier,
        ScanEvidence,
    )
    from openinterp.harness.dictionaries import dictionary_spec, load_spec
    from openinterp.harness.features import scan_feature
    from openinterp.harness.heads import scan_head
    from openinterp.harness.hooks import activation_histogram, run_environment
    from openinterp.harness.input_attribution import input_attribution
    from openinterp.harness.jlens import direction_lens, vocabulary_lens
    from openinterp.harness.scan import cut_pieces, default_stride, scan_neuron

    index = {"head": head, "feature": feature}.get(unit, neuron)
    spec = dictionary_spec(dictionary) if unit == "feature" else None
    if not random_pick and (layer is None or index is None):
        raise click.UsageError(f"--layer and --{unit} are required unless --random is given.")
    if random_pick:
        seed = secrets.randbits(32) if seed is None else seed
        rng = random.Random(seed)
        if layer is None:  # a dictionary's layers are the ones its features can come from
            layer = rng.choice(spec.layers) if spec else rng.randrange(runner.n_layers)
        if unit == "feature":
            width = load_spec(spec, layer, runner.device).n_features
        else:
            width = runner.n_heads if unit == "head" else runner.mlp_width(layer)
        index = rng.randrange(width) if index is None else index
        if unit == "neuron":
            pole = rng.choice(["positive", "negative"]) if pole is None else pole
    if unit == "head":
        ident = HeadIdentifier(model_name=model_name, layer=layer, head_idx=index)
    elif unit == "feature":
        ident = FeatureIdentifier(
            model_name=model_name, layer=layer, dictionary=dictionary, feature_idx=index
        )
    else:
        ident = NeuronIdentifier(
            model_name=model_name, layer=layer, neuron_idx=index, pole=pole or "positive"
        )
    texts, corpus_name, corpus_sha, corpus_source = read_corpus(corpus)
    sample = None
    if max_tokens:
        from openinterp.corpus import sample_tokens

        texts, sample = sample_tokens(runner.tokenizer, texts, max_tokens, sample_seed)
        click.echo(f"Sampled {sample}", err=True)
    if window_tokens is not None:
        stride = stride or default_stride(window_tokens)
        how = f" as {window_tokens}-token windows every {stride} tokens"
    else:
        how = ""
        check_context(runner, texts)
    click.echo(f"Scanning {len(texts)} snippets{how} for {ident}…", err=True)
    windows = {"window_tokens": window_tokens, "stride": stride}
    with costs.stage("scan"):
        if unit == "head":
            scan = scan_head(runner, texts, ident, k=top_k, **windows)
        elif unit == "feature":
            scan = scan_feature(runner, texts, ident, spec, k=top_k, **windows)
        else:
            scan = scan_neuron(
                runner, texts, layer, index, ident.pole, k=top_k, cache=scan_cache, **windows
            )
    if unit == "neuron" and scan_cache is not None and scan_cache.last is not None:
        verb = "Replayed" if scan_cache.replayed else "Cached"
        click.echo(f"{verb} layer {layer}'s scan: {scan_cache.last}", err=True)
    if not scan.examples or scan.examples[0].max_activation <= 0:
        raise click.ClickException(
            f"{ident} never activates on this corpus; try a larger --corpus."
        )
    lens = attribution = None
    if unit != "head":  # a head writes a subspace, not one direction a lens can read
        click.echo("Reading the output direction through the logit and Jacobian lenses…", err=True)
        # The first 100 documents, each cut to its first window when scanning windows.
        pieces = cut_pieces(runner.tokenizer, texts[:100], window_tokens, stride)
        lens_texts = [p.text for p in pieces if p.start == 0]
        with costs.stage("lenses"):
            if unit == "feature":
                direction = load_spec(spec, layer, runner.device).w_dec[index]
                lens = direction_lens(runner, lens_texts, layer, direction)
            else:
                lens = vocabulary_lens(runner, lens_texts, layer, index, ident.pole)
    if unit == "neuron":
        click.echo("Attributing the top examples' peaks to their context tokens…", err=True)
        with costs.stage("input_attribution"):
            attribution = input_attribution(runner, scan.examples, layer, index, ident.pole)
    return ScanEvidence(
        identifier=ident,
        top_examples=scan.examples,
        activation_histogram=activation_histogram(scan.values),
        provenance=ExperimentProvenance(
            model_revision=run_environment(runner).model_revision,
            dtype=dtype,
            corpus=corpus_name,
            corpus_sha256=corpus_sha,
            n_snippets=len(texts),
            tokens_scanned=len(scan.values),
            snippet_tokens=window_tokens,
            snippet_stride=stride if window_tokens is not None else None,
            selection="random" if random_pick else "manual",
            seed=seed if random_pick else None,
            harness_version=__version__,
            environment=run_environment(runner),
            corpus_source=corpus_source,
            corpus_sample=sample,
        ),
        confounds=scan.confounds,
        correlated_neurons=scan.correlated,
        vocabulary_lens=lens,
        input_attribution=attribution,
        head_profile=scan.profile if unit == "head" else None,
        model_n_heads=runner.n_heads if unit == "head" else None,
        dictionary_profile=scan.profile if unit == "feature" else None,
    )


def check_context(runner, texts) -> None:
    """Whole documents must fit the model's context (GPT-2 reads 1,024 positions)."""
    config = getattr(runner.model, "config", None)
    limit = getattr(config, "n_positions", None) or getattr(config, "max_position_embeddings", 0)
    if not limit or runner.tokenizer is None:
        return
    longest = max((len(runner.tokenizer.encode(t)) for t in texts), default=0)
    if longest > limit:
        raise click.UsageError(
            f"a document has {longest:,} tokens but the model reads at most {limit:,}; "
            f"scan windows instead, e.g. --window-tokens {limit // 2}."
        )


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
        )

    ident = evidence.identifier
    if superseded is not None and superseded.submission.identifier.slug != ident.slug:
        raise click.UsageError(
            f"--supersedes names an experiment on {superseded.submission.identifier}, "
            f"not on {ident}."
        )
    with costs.stage("counterfactuals"):
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
        )
    # An evidence file carries the scan's costs; a one-step audit measured them here.
    provenance = with_costs(submission.provenance, costs.measured())
    revises = superseded and experiment_id(superseded)
    submission = submission.model_copy(update={"provenance": provenance, "supersedes": revises})
    out = out or Path("submissions") / f"{ident.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(submission.model_dump_json(indent=2))
    click.echo(f"Hypothesis: {submission.natural_language_explanation}")
    if revises:
        click.echo(f"Supersedes: {revises} ({superseded.verdict.status})")
    click.echo(f"Wrote {out}")
