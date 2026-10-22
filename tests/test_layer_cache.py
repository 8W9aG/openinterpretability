"""The per-layer scan cache: replaying it gives the model's scan exactly."""

from dataclasses import astuple
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.layer_cache import LayerCache, default_root
from openinterp.harness.scan import cut_pieces, scan_neuron
from tests.conftest import FakeTokenizer, ToyModel
from tests.test_agents import TEXTS

LONG = [t * 4 for t in TEXTS]


@pytest.fixture
def named(toy_runner):
    toy_runner.model_name = "toy/model"
    return toy_runner


@pytest.fixture
def cache(tmp_path):
    return LayerCache(tmp_path / "scans")


def no_model(runner, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("ran the model")

    monkeypatch.setattr(runner, "run_with_cache", fail)


@pytest.mark.parametrize("windows", [{}, {"window_tokens": 8, "stride": 3}])
@pytest.mark.parametrize("pole", ["positive", "negative"])
def test_replaying_a_cached_layer_matches_the_model(named, cache, monkeypatch, windows, pole):
    fresh = scan_neuron(named, LONG, 1, 2, pole, batch_size=3, **windows)
    stored = scan_neuron(named, LONG, 1, 2, pole, batch_size=3, cache=cache, **windows)
    assert not cache.replayed and cache.last.exists()

    with monkeypatch.context() as patch:
        no_model(named, patch)
        replayed = scan_neuron(named, LONG, 1, 2, pole, batch_size=3, cache=cache, **windows)
        assert cache.replayed
        # Any other neuron of the layer replays the same entry.
        other = scan_neuron(named, LONG, 1, 7, pole, batch_size=3, cache=cache, **windows)
    assert astuple(fresh) == astuple(stored) == astuple(replayed)
    assert astuple(other) == astuple(scan_neuron(named, LONG, 1, 7, pole, batch_size=3, **windows))
    assert len(cache.entries()) == 1


def test_bfloat16_activations_round_trip(cache, monkeypatch):
    torch.manual_seed(0)
    runner = NeuronHookRunner(ToyModel().to(torch.bfloat16).eval(), FakeTokenizer())
    runner.model_name = "toy/bf16"
    fresh = scan_neuron(runner, TEXTS, 1, 2, "positive")
    scan_neuron(runner, TEXTS, 1, 2, "positive", cache=cache)
    entry = cache.entries()[0]
    assert entry.meta["storage"] == "int16"
    replayed = torch.cat([c.acts for c in entry.chunks(8)])
    assert replayed.dtype == torch.bfloat16
    assert torch.equal(entry.column(2), replayed[:, 2].float())
    no_model(runner, monkeypatch)
    assert astuple(scan_neuron(runner, TEXTS, 1, 2, "positive", cache=cache)) == astuple(fresh)


def test_the_key_covers_what_sets_the_numbers(named, cache):
    pieces = list(cut_pieces(named.tokenizer, TEXTS, None, None))
    key = LayerCache.fields(named, 1, pieces, 8)
    assert key["model"] == "toy/model" and key["dtype"] == "float32" and key["device"] == "cpu"
    windowed = list(cut_pieces(named.tokenizer, TEXTS, 4, 2))
    others = [
        LayerCache.fields(named, 2, pieces, 8),
        LayerCache.fields(named, 1, pieces[:-1], 8),
        LayerCache.fields(named, 1, windowed, 8),
        LayerCache.fields(named, 1, pieces, 4),
    ]
    assert len({cache.path(f) for f in [key, *others]}) == 5

    scan_neuron(named, TEXTS, 1, 2, "positive", cache=cache)
    scan_neuron(named, TEXTS, 2, 2, "positive", cache=cache)
    scan_neuron(named, TEXTS[:-1], 1, 2, "positive", cache=cache)
    assert not cache.replayed and len(cache.entries()) == 3
    assert cache.cached_layers("toy/model") == {1, 2} and not cache.cached_layers("other")


def test_an_unnamed_model_is_not_cached(toy_runner, cache):
    scan_neuron(toy_runner, TEXTS, 1, 2, "positive", cache=cache)
    assert cache.last is None and not cache.entries()


def test_an_unfinished_scan_leaves_nothing(named, cache):
    from openinterp.harness.scan import layer_chunks

    pieces = list(cut_pieces(named.tokenizer, TEXTS, None, None))
    chunks = cache.through(named, 1, pieces, 2, layer_chunks(named, pieces, 1, 2))
    next(chunks)
    chunks.close()
    assert not cache.entries() and not any(cache.root.rglob("acts.bin"))


def test_least_recently_used_scans_are_evicted(named, cache):
    scan_neuron(named, TEXTS, 0, 2, "positive", cache=cache)
    size = cache.entries()[0].size
    cache.max_bytes = 2 * size
    scan_neuron(named, TEXTS, 1, 2, "positive", cache=cache)
    scan_neuron(named, TEXTS, 0, 2, "positive", cache=cache)  # replayed: now most recent
    scan_neuron(named, TEXTS, 2, 2, "positive", cache=cache)
    assert cache.cached_layers("toy/model") == {0, 2}


def test_the_default_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENINTERP_SCAN_CACHE", "off")
    assert default_root() is None and LayerCache.default() is None
    monkeypatch.delenv("OPENINTERP_SCAN_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_root() == tmp_path / "openinterp" / "scans"


@pytest.fixture
def cli_runner(tmp_path, monkeypatch, named):
    named.model.config = SimpleNamespace()
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *_a, **_k: named)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    return named


def run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_audit_caches_the_layer_and_activations_replays_it(cli_runner, monkeypatch):
    audit = run("audit", "--model", "toy/model", "--layer", 1, "--neuron", 2,
                "--corpus", "corpus.txt", "--evidence-out", "ev.json")  # fmt: skip
    assert "Cached layer 1's scan" in audit.stderr

    with monkeypatch.context() as patch:
        no_model(cli_runner, patch)
        out = run("activations", "--model", "toy/model", "--layer", 1, "--neuron", 5,
                  "--corpus", "corpus.txt", "--top", 5, "--context", 3)  # fmt: skip
    assert "Replayed layer 1's scan" in out.stderr
    header, *rows = out.stdout.splitlines()
    assert header.split("\t") == ["doc", "window_start", "index", "token_id", "token",
                                  "strength", "context"]  # fmt: skip
    strengths = [float(r.split("\t")[5]) for r in rows]
    assert len(rows) == 5 and strengths == sorted(strengths, reverse=True)
    doc, _, index = (int(x) for x in rows[0].split("\t")[:3])
    direct = cli_runner.neuron_activations(TEXTS[doc], 1, 5, "positive")[0, index]
    assert strengths[0] == pytest.approx(float(direct), abs=1e-5)

    every = run("activations", "--model", "toy/model", "--layer", 1, "--neuron", 5,
                "--corpus", "corpus.txt", "--top", 0, "--format", "jsonl")  # fmt: skip
    assert len(every.stdout.splitlines()) == sum(len(t) for t in TEXTS)

    tok = FakeTokenizer()
    token = tok.decode([tok._id("a")])
    only = run("activations", "--model", "toy/model", "--layer", 1, "--neuron", 5,
               "--corpus", "corpus.txt", "--top", 0, "--token", token)  # fmt: skip
    rows = only.stdout.splitlines()[1:]
    assert rows and all(r.split("\t")[4] == token for r in rows)
    assert len(rows) == sum(tok._id(c) == tok._id("a") for t in TEXTS for c in t)

    listing = run("scan-cache").stdout
    assert "toy/model L1" in listing and "1 scans" in listing
    assert "Deleted 1 cached scans" in run("scan-cache", "--clear").stdout


def test_no_scan_cache_skips_it(cli_runner):
    run("audit", "--model", "toy/model", "--layer", 1, "--neuron", 2, "--corpus", "corpus.txt",
        "--no-scan-cache", "--evidence-out", "ev.json")  # fmt: skip
    assert "0 scans" in run("scan-cache").stdout
