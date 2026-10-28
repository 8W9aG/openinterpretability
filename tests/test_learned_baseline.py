"""The learned-basis baseline on the home page: random latents against random neuron poles."""

import pyarrow.parquet as pq
import pytest

from openinterp.builder.compiler import _environment, compile_site, registry_stats
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
    assert b.p_positive == pytest.approx(fisher_exact(1, 3, 1, 2))

    home = (tmp_path / "index.html").read_text()
    assert 'id="learned-baseline"' in home and "dead latents 25%" in home
    assert "33% (6%–79%)" in home  # 1 of 3, Wilson interval
    assert "128-token windows of corpus" in home
    assert "p = 1.000 vs positive poles (1.000 vs all)" in home


def test_small_p_values_are_not_rounded_to_zero():
    pval = _environment().filters["pval"]
    assert (pval(None), pval(0.18), pval(4.3e-5)) == ("—", "0.180", "4.3e-05")


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


def _draw(record, ident, outcome, window=128):
    from datetime import UTC, datetime

    from openinterp.core.draws import UnauditableDraw

    return UnauditableDraw(
        identifier=ident,
        outcome=outcome,
        active_share=0.9 if outcome == "default_on" else None,
        provenance=_provenance("random", window),
        timestamp=datetime(2026, 10, 1, tzinfo=UTC),
    )


def test_unauditable_draws_are_counted_with_their_arm(tmp_path, baseline_records, record):
    pole = record.submission.identifier.model_copy(update={"layer": 3, "neuron_idx": 20})
    latent = baseline_records[0].submission.identifier.model_copy(update={"feature_idx": 9})
    draws = [
        _draw(record, latent, "dead"),
        _draw(record, latent.model_copy(update={"feature_idx": 8}), "dead"),
        _draw(record, pole.model_copy(update={"pole": "negative"}), "default_on"),
        _draw(record, pole.model_copy(update={"layer": 7}), "dead"),  # not a compared layer
        _draw(record, pole, "dead", window=512),  # not the latents' scan
    ]
    report = compile_site(baseline_records, tmp_path, draws=draws)
    (b,) = registry_stats(report.parquet, draws)["baseline"]
    assert (b.features.dead, b.features.default_on) == (2, 0)
    assert (b.neurons.dead, b.neurons.default_on) == (0, 1)
    assert (b.negative_poles.default_on, b.positive_poles.default_on) == (1, 0)
    assert "2 dead" in (tmp_path / "index.html").read_text()
    rows = pq.read_table(tmp_path / "data" / "draws.parquet").to_pylist()
    assert [r["outcome"] for r in rows] == ["dead", "dead", "default_on", "dead", "dead"]
    assert rows[2]["pole"] == "negative" and rows[0]["dictionary"] == "tiny-tc"
