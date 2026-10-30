"""Scanning one unit into :class:`~openinterp.core.schemas.ScanEvidence`: reading the
corpus, picking the unit, the scan, and the lens and attribution that follow it."""

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
        OppositePole,
        ScanEvidence,
    )
    from openinterp.harness.dictionaries import dictionary_spec, load_spec
    from openinterp.harness.environment import run_environment
    from openinterp.harness.features import scan_feature
    from openinterp.harness.heads import scan_head
    from openinterp.harness.hooks import activation_histogram
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
    if corpus is not None and corpus_source is None:
        from openinterp.corpus import source_path

        click.echo(
            f"warning: {corpus} has no {source_path(corpus).name} describing these bytes, "
            "so nobody can rebuild this scan's corpus. Build corpora with `openinterp corpus`; "
            "for the registry's corpus, see the README.",
            err=True,
        )
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
    provenance = ExperimentProvenance(
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
    )
    if not scan.examples or scan.examples[0].max_activation <= 0:
        raise Unauditable(
            f"{ident} never activates on this corpus; try a larger --corpus.",
            draw(ident, "dead", None, provenance),
        )
    check_default_on(ident, scan.values, scan.examples[0].max_activation, provenance)
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
            attribution = input_attribution(
                runner, scan.examples, layer, index, ident.pole, skip_first=scan.first_token_floor
            )
    other = scan.opposite  # a neuron's other pole, recorded if it ever fired
    opposite = None
    if other is not None and other.examples and other.examples[0].max_activation > 0:
        opposite = OppositePole(
            pole=other.pole,
            top_examples=other.examples,
            activation_histogram=activation_histogram(other.values),
        )
    return ScanEvidence(
        identifier=ident,
        top_examples=scan.examples,
        activation_histogram=activation_histogram(scan.values),
        first_token_floor=scan.first_token_floor,
        provenance=provenance,
        confounds=scan.confounds,
        correlated_neurons=scan.correlated,
        vocabulary_lens=lens,
        input_attribution=attribution,
        head_profile=scan.profile if unit == "head" else None,
        model_n_heads=runner.n_heads if unit == "head" else None,
        dictionary_profile=scan.profile if unit == "feature" else None,
        opposite_pole=opposite,
    )


class Unauditable(click.ClickException):
    """The scan stopped on a unit that can't be audited; ``draw`` records it."""

    def __init__(self, message: str, draw):
        super().__init__(message)
        self.draw = draw


def draw(ident, outcome, share, provenance):
    from datetime import UTC, datetime

    from openinterp.core.draws import UnauditableDraw

    share = None if share is None else round(share, 6)
    return UnauditableDraw(
        identifier=ident,
        outcome=outcome,
        active_share=share,
        provenance=provenance,
        timestamp=datetime.now(UTC),
    )


def check_default_on(ident, values, scale: float, provenance=None) -> None:
    """Stop if the unit is default-on (protocol 0.8): active on so much of the scan that a
    hard negative of a few words would still fire, so it can't be audited as a detector.
    Most of GPT-2's negative poles are (GELU's dip; docs/analyses/position-zero.md). The
    share is read from the histogram, as the review's `not_default_on` gate reads it."""
    from openinterp.core.schemas import ReviewPolicy
    from openinterp.harness.hooks import activation_histogram

    policy = ReviewPolicy()
    share = activation_histogram(values).share_at_least(policy.activation_threshold * scale)
    if share >= policy.max_active_share:
        raise Unauditable(
            f"{ident} is default-on: it is active on {share:.0%} of the scan's tokens, so no "
            "hard negative can stay inactive and it can't be audited as a detector "
            "(docs/PROTOCOL.md §1).",
            draw(ident, "default_on", share, provenance) if provenance else None,
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
