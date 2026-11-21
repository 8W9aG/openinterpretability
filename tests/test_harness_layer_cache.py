"""The scan cache's cap, its key without transformers, and entries it does not count."""

import sys

import pytest

from openinterpretability.harness.layer_cache import LayerCache
from openinterpretability.harness.scan import cut_pieces, scan_neuron
from tests.test_agents import TEXTS


@pytest.fixture
def named(toy_runner):
    toy_runner.model_name = "toy/model"
    return toy_runner


def test_a_given_cap_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_SCAN_CACHE_GB", "1")
    assert LayerCache(tmp_path, max_bytes=5).max_bytes == 5
    assert LayerCache(tmp_path).max_bytes == 10**9


def test_the_key_has_no_transformers_version_without_transformers(named, tmp_path, monkeypatch):
    pieces = list(cut_pieces(named.tokenizer, TEXTS, None, None))
    monkeypatch.setitem(sys.modules, "transformers", None)  # its import fails
    fields = LayerCache.fields(named, 1, pieces, 16)
    assert fields["transformers"] is None and fields["model"] == "toy/model"


def test_the_scan_just_stored_stays_although_it_is_over_the_cap(named, tmp_path):
    cache = LayerCache(tmp_path / "scans", max_bytes=0)
    scan_neuron(named, TEXTS, 1, 2, "positive", cache=cache)
    (entry,) = cache.entries()
    assert entry.path == cache.last and entry.size > cache.max_bytes
    assert cache.evict() == [entry] and cache.entries() == []


def test_unfinished_and_unreadable_entries_are_not_counted(tmp_path):
    cache = LayerCache(tmp_path)
    for name, text in ((".writing", "{}"), ("broken", "not json")):
        folder = tmp_path / "toy--model" / "L1" / name
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(text)
    assert cache.entries() == [] and cache.cached_layers("toy/model") == set()
