"""Scanning one unit into :class:`~openinterp.core.schemas.ScanEvidence`: reading the
corpus, picking the unit, the scan, and the lens and attribution that follow it."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
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


def read_labels(path: Path | None) -> dict[str, list[str | None]]:
    """The corpus's document labels (:data:`~openinterp.corpus.DOCUMENT_LABELS`), by label
    set, one per document in :func:`read_corpus` order; only the sets some document has."""
    from openinterp.corpus import DOCUMENT_LABELS

    if path is None or path.suffix != ".jsonl":
        return {}
    lines = path.read_bytes().decode("utf-8").splitlines()  # as read_corpus splits them
    rows = [json.loads(line) for line in lines if line.strip()]
    return {
        name: [row.get(field) for row in rows]
        for field, name in DOCUMENT_LABELS.items()
        if any(row.get(field) is not None for row in rows)
    }


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ScanRequest:
    """What ``openinterp audit`` scans: the unit (or how to draw it) and the corpus.

    ``None`` for ``layer`` or the unit's index means "draw it" (with ``random_pick``)."""

    model_name: str
    unit: str = "neuron"
    layer: int | None = None
    neuron: int | None = None
    pole: str | None = None
    head: int | None = None
    feature: int | None = None
    dictionary: str | None = None
    random_pick: bool = False
    seed: int | None = None
    corpus: Path | None = None
    top_k: int = 20
    dtype: str | None = None
    window_tokens: int | None = None
    stride: int | None = None
    max_tokens: int | None = None
    sample_seed: int = 0


@dataclass
class _Texts:
    """The scanned snippets and where they came from."""

    texts: list[str]
    name: str
    sha256: str
    source: object
    sample: object
    stride: int | None
    labels: dict[str, list[str | None]] = field(default_factory=dict)  # by label set

    def taggers(self):
        """One document tagger per label set the corpus carries."""
        from openinterp.harness.annotations import DocumentLabels

        return [DocumentLabels(name, labels) for name, labels in self.labels.items()]


def scan_evidence(runner, request: ScanRequest, scan_cache=None):
    """Pick the unit, scan the corpus (or a ``max_tokens`` sample of it), then read the
    vocabulary lens (neurons and dictionary features) and attribute the top examples'
    peaks to their context tokens (neurons only). A neuron scan replays ``scan_cache``
    when it holds the layer."""
    from openinterp.core.schemas import ScanEvidence
    from openinterp.harness.dictionaries import dictionary_spec
    from openinterp.harness.hooks import activation_histogram

    unit = request.unit
    spec = dictionary_spec(request.dictionary) if unit == "feature" else None
    ident, seed = pick_unit(runner, request, spec)
    corpus = load_texts(runner, request)
    scan = run_scan(runner, request, ident, spec, corpus, scan_cache)
    provenance = scan_provenance(runner, request, seed, corpus, len(scan.values))
    if not scan.examples or scan.examples[0].max_activation <= 0:
        raise Unauditable(
            f"{ident} never activates on this corpus; try a larger --corpus.",
            draw(ident, "dead", None, provenance),
        )
    check_default_on(ident, scan.values, scan.examples[0].max_activation, provenance)
    lens, attribution = read_lenses(runner, request, ident, spec, corpus, scan)
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
        opposite_pole=opposite_pole(scan),
    )


def pick_unit(runner, request: ScanRequest, spec):
    """The unit's identifier and the ``--random`` seed, drawing whatever ``request`` leaves
    unset. The draws come in a fixed order (layer, index, pole), so a seed replays."""
    import random

    from openinterp.harness.dictionaries import load_spec

    unit, layer, pole, seed = request.unit, request.layer, request.pole, request.seed
    index = {"head": request.head, "feature": request.feature}.get(unit, request.neuron)
    if not request.random_pick and (layer is None or index is None):
        raise click.UsageError(f"--layer and --{unit} are required unless --random is given.")
    if request.random_pick:
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
    return identifier(request, layer, index, pole), seed


def identifier(request: ScanRequest, layer: int, index: int, pole: str | None):
    from openinterp.core.schemas import FeatureIdentifier, HeadIdentifier, NeuronIdentifier

    model_name = request.model_name
    if request.unit == "head":
        return HeadIdentifier(model_name=model_name, layer=layer, head_idx=index)
    if request.unit == "feature":
        return FeatureIdentifier(
            model_name=model_name, layer=layer, dictionary=request.dictionary, feature_idx=index
        )
    return NeuronIdentifier(
        model_name=model_name, layer=layer, neuron_idx=index, pole=pole or "positive"
    )


def load_texts(runner, request: ScanRequest) -> _Texts:
    """Read the corpus, take the ``max_tokens`` sample, and settle the window stride."""
    from openinterp.harness.scan import default_stride

    corpus = request.corpus
    texts, corpus_name, corpus_sha, corpus_source = read_corpus(corpus)
    if corpus is not None and corpus_source is None:
        from openinterp.corpus import source_path

        click.echo(
            f"warning: {corpus} has no {source_path(corpus).name} describing these bytes, "
            "so nobody can rebuild this scan's corpus. Build corpora with `openinterp corpus`; "
            "for the registry's corpus, see the README.",
            err=True,
        )
    sample, labels = None, read_labels(corpus)
    if request.max_tokens:
        from openinterp.corpus import sample_indices

        picked, sample = sample_indices(
            runner.tokenizer, texts, request.max_tokens, request.sample_seed
        )
        texts = [texts[i] for i in picked]
        labels = {name: [values[i] for i in picked] for name, values in labels.items()}
        click.echo(f"Sampled {sample}", err=True)
    stride = request.stride
    if request.window_tokens is not None:
        stride = stride or default_stride(request.window_tokens)
    else:
        check_context(runner, texts)
    return _Texts(texts, corpus_name, corpus_sha, corpus_source, sample, stride, labels)


def run_scan(runner, request: ScanRequest, ident, spec, corpus: _Texts, scan_cache):
    from openinterp.core import costs
    from openinterp.harness.features import scan_feature
    from openinterp.harness.heads import scan_head
    from openinterp.harness.scan import scan_neuron

    texts, unit, top_k = corpus.texts, request.unit, request.top_k
    windows = {"window_tokens": request.window_tokens, "stride": corpus.stride}
    how = ""
    if request.window_tokens is not None:
        how = f" as {request.window_tokens}-token windows every {corpus.stride} tokens"
    click.echo(f"Scanning {len(texts)} snippets{how} for {ident}…", err=True)
    with costs.stage("scan"):
        if unit == "head":
            scan = scan_head(runner, texts, ident, k=top_k, **windows)
        elif unit == "feature":
            scan = scan_feature(runner, texts, ident, spec, k=top_k, **windows)
        else:
            scan = scan_neuron(
                runner,
                texts,
                ident.layer,
                ident.neuron_idx,
                ident.pole,
                k=top_k,
                cache=scan_cache,
                taggers=corpus.taggers(),
                **windows,
            )
    if unit == "neuron" and scan_cache is not None and scan_cache.last is not None:
        verb = "Replayed" if scan_cache.replayed else "Cached"
        click.echo(f"{verb} layer {ident.layer}'s scan: {scan_cache.last}", err=True)
    return scan


def scan_provenance(runner, request: ScanRequest, seed, corpus: _Texts, tokens_scanned: int):
    from openinterp import __version__
    from openinterp.core.schemas import ExperimentProvenance
    from openinterp.harness.environment import run_environment

    windowed = request.window_tokens is not None
    return ExperimentProvenance(
        model_revision=run_environment(runner).model_revision,
        dtype=request.dtype,
        corpus=corpus.name,
        corpus_sha256=corpus.sha256,
        n_snippets=len(corpus.texts),
        tokens_scanned=tokens_scanned,
        snippet_tokens=request.window_tokens,
        snippet_stride=corpus.stride if windowed else None,
        selection="random" if request.random_pick else "manual",
        seed=seed if request.random_pick else None,
        harness_version=__version__,
        environment=run_environment(runner),
        corpus_source=corpus.source,
        corpus_sample=corpus.sample,
    )


def read_lenses(runner, request: ScanRequest, ident, spec, corpus: _Texts, scan):
    """The vocabulary lens (not for a head, which writes a subspace, not one direction a
    lens can read) and, for a neuron, the input attribution of the top examples."""
    from openinterp.core import costs
    from openinterp.harness.dictionaries import load_spec
    from openinterp.harness.input_attribution import input_attribution
    from openinterp.harness.jlens import direction_lens, vocabulary_lens
    from openinterp.harness.scan import cut_pieces

    lens = attribution = None
    kind = request.unit
    if kind != "head":
        click.echo("Reading the output direction through the logit and Jacobian lenses…", err=True)
        # The first 100 documents, each cut to its first window when scanning windows.
        texts, window = corpus.texts[:100], request.window_tokens
        pieces = cut_pieces(runner.tokenizer, texts, window, corpus.stride)
        lens_texts = [p.text for p in pieces if p.start == 0]
        with costs.stage("lenses"):
            if kind == "feature":
                direction = load_spec(spec, ident.layer, runner.device).w_dec[ident.feature_idx]
                lens = direction_lens(runner, lens_texts, ident.layer, direction)
            else:
                lens = vocabulary_lens(
                    runner, lens_texts, ident.layer, ident.neuron_idx, ident.pole
                )
    if kind == "neuron":
        click.echo("Attributing the top examples' peaks to their context tokens…", err=True)
        with costs.stage("input_attribution"):
            attribution = input_attribution(
                runner,
                scan.examples,
                ident.layer,
                ident.neuron_idx,
                ident.pole,
                skip_first=scan.first_token_floor,
            )
    return lens, attribution


def opposite_pole(scan):
    """A neuron's other pole, recorded if it ever fired."""
    from openinterp.core.schemas import OppositePole
    from openinterp.harness.hooks import activation_histogram

    other = scan.opposite
    if other is None or not other.examples or other.examples[0].max_activation <= 0:
        return None
    return OppositePole(
        pole=other.pole,
        top_examples=other.examples,
        activation_histogram=activation_histogram(other.values),
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
