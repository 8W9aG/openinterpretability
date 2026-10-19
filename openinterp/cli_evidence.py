"""Helpers for ``openinterp audit``: corpus loading, the scan, and prompt/answer files."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

import click


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


def scan_evidence(runner, model_name, layer, neuron, pole, random_pick, seed, corpus, top_k, dtype):
    """Pick the neuron pole, scan the corpus and read the vocabulary lens."""
    import random

    from openinterp import __version__
    from openinterp.core.schemas import ExperimentProvenance, NeuronIdentifier, ScanEvidence
    from openinterp.harness.hooks import activation_histogram, run_environment
    from openinterp.harness.jlens import vocabulary_lens
    from openinterp.harness.scan import scan_neuron

    if not random_pick and (layer is None or neuron is None):
        raise click.UsageError("--layer and --neuron are required unless --random is given.")
    if random_pick:
        seed = secrets.randbits(32) if seed is None else seed
        rng = random.Random(seed)
        layer = rng.randrange(runner.n_layers) if layer is None else layer
        neuron = rng.randrange(runner.mlp_width(layer)) if neuron is None else neuron
        pole = rng.choice(["positive", "negative"]) if pole is None else pole
    ident = NeuronIdentifier(
        model_name=model_name, layer=layer, neuron_idx=neuron, pole=pole or "positive"
    )
    texts, corpus_name, corpus_sha, corpus_source = read_corpus(corpus)
    click.echo(f"Scanning {len(texts)} snippets for {ident}…", err=True)
    scan = scan_neuron(runner, texts, layer, neuron, ident.pole, k=top_k)
    if not scan.examples or scan.examples[0].max_activation <= 0:
        raise click.ClickException(
            f"{ident} never activates on this corpus; try a larger --corpus."
        )
    click.echo("Reading the output direction through the logit and Jacobian lenses…", err=True)
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
            selection="random" if random_pick else "manual",
            seed=seed if random_pick else None,
            harness_version=__version__,
            environment=run_environment(runner),
            corpus_source=corpus_source,
        ),
        confounds=scan.confounds,
        correlated_neurons=scan.correlated,
        vocabulary_lens=vocabulary_lens(runner, texts[:100], layer, neuron, ident.pole),
    )


def answer_path(prompt_path: Path, suffix: str) -> Path:
    """``review.prompt.md`` or ``review.md`` -> ``review<suffix>``."""
    name = prompt_path.name.removesuffix(".md").removesuffix(".prompt")
    return prompt_path.with_name(name + suffix)


def write_evidence(evidence, path: Path) -> None:
    from openinterp.agents.proposer import HypothesisDraft, ProposerAgent

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
    )
    prompt_path.write_text(ProposerAgent.render_prompt(HypothesisDraft, task, answer_path))
    click.echo(f"Neuron: {evidence.identifier}")
    click.echo(f"Wrote {path}")
    click.echo(f"Proposer prompt: {prompt_path} (answer in {answer_path})")
