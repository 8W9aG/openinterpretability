"""The learned-basis baseline on the home page: random latents against random neuron poles."""

import pyarrow.parquet as pq
import pytest

from openinterp.builder.compiler import compile_site, registry_stats
from openinterp.builder.learned_baseline import fisher_exact, wilson
from openinterp.core.schemas import (
    AuditRecord,
    DictionaryProfile,
    ExperimentProvenance,
    FeatureIdentifier,
)
from tests.test_compiler import experiment


def _provenance(selection, window):
    return ExperimentProvenance(
        dtype="float32",
        corpus="corpus.jsonl",
        corpus_sha256="ab" * 32,
        n_snippets=1,
        tokens_scanned=100,
        snippet_tokens=window,
        selection=selection,
        harness_version="test",
    )


def _selected(record, selection="random", window=128):
    sub = record.submission.model_copy(update={"provenance": _provenance(selection, window)})
    return AuditRecord(submission=sub, verdict=record.verdict)


def _feature(record, idx, status, hours=0):
    ident = FeatureIdentifier(
        model_name=record.submission.identifier.model_name,
        layer=3,
        dictionary="tiny-tc",
        feature_idx=idx,
    )
    profile = DictionaryProfile(
        kind="transcoder",
        repo="local",
        path="tc3/sae.safetensors",
        input_site="mlp_in_normalized",
        activation="relu",
        n_features=48,
        density=0.01,
        explained_variance=0.6,
        mean_l0=40.0,
        dead_share=0.25,
        tokens=100,
    )
    native = experiment(record, idx=0, status=status, counts=(5, 0, 5, 0), hours=hours)
    sub = native.submission.model_copy(update={"identifier": ident, "dictionary_profile": profile})
    return _selected(AuditRecord(submission=sub, verdict=native.verdict))


def _neuron(record, idx, status, layer=3, selection="random", window=128, **kw):
    native = experiment(record, idx=idx, status=status, counts=(5, 0, 5, 0), **kw)
    ident = native.submission.identifier.model_copy(update={"layer": layer})
    sub = native.submission.model_copy(update={"identifier": ident})
    return _selected(AuditRecord(submission=sub, verdict=native.verdict), selection, window)


@pytest.fixture
def baseline_records(record):
    return [
        _feature(record, 1, "VERIFIED"),
        _feature(record, 2, "REJECTED"),
        _feature(record, 3, "NEEDS_REVISION"),
        # A later revision of latent 3 doesn't count: each unit counts by its first audit.
        _feature(record, 3, "VERIFIED", hours=5),
        _neuron(record, 10, "VERIFIED"),
        _neuron(record, 11, "VERIFIED", pole="negative"),
        _neuron(record, 12, "REJECTED"),
        # Not compared: chosen by hand, on a layer the dictionary doesn't cover, or
        # scanned in longer windows than the latents.
        _neuron(record, 13, "VERIFIED", selection="manual"),
        _neuron(record, 14, "VERIFIED", layer=7),
        _neuron(record, 15, "VERIFIED", window=512),
    ]


def test_random_latents_against_random_poles_of_the_same_layers(tmp_path, baseline_records):
    report = compile_site(baseline_records, tmp_path)
    rows = pq.read_table(report.parquet).to_pylist()
    assert {r["selection"] for r in rows} == {"random", "manual"}

    (b,) = registry_stats(report.parquet)["baseline"]
    assert (b.dictionary, b.layers, b.explained_variance, b.mean_l0) == ("tiny-tc", (3,), 0.6, 40)
    assert (b.dead_share, b.window_tokens, b.corpus_sha256) == (0.25, 128, "ab" * 32)
    assert (b.features.units, b.features.verified) == (3, 1)
    assert (b.neurons.units, b.neurons.verified) == (3, 2)
    assert (b.positive_poles.units, b.positive_poles.verified) == (2, 1)
    assert (b.negative_poles.units, b.negative_poles.verified) == (1, 1)
    assert b.p_value == pytest.approx(1.0)

    home = (tmp_path / "index.html").read_text()
    assert 'id="learned-baseline"' in home and "dead latents 25%" in home
    assert "33% (6%–79%)" in home  # 1 of 3, Wilson interval
    assert "128-token windows of corpus" in home


def test_no_baseline_without_random_latents(tmp_path, baseline_records):
    natives = [r for r in baseline_records if r.submission.identifier.kind != "dictionary_feature"]
    report = compile_site(natives, tmp_path)
    assert registry_stats(report.parquet)["baseline"] == []
    assert 'id="learned-baseline"' not in (tmp_path / "index.html").read_text()


def test_wilson_interval_and_fisher_exact_test():
    assert wilson(1, 3) == pytest.approx((0.0615, 0.7923), abs=1e-4)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    # scipy.stats.fisher_exact on [[1, 9], [11, 3]] and [[8, 2], [1, 5]]
    assert fisher_exact(1, 10, 11, 14) == pytest.approx(0.0027594562)
    assert fisher_exact(8, 10, 1, 6) == pytest.approx(0.0349650350)
    assert fisher_exact(3, 6, 3, 6) == pytest.approx(1.0)
