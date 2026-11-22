"""The ANE scan backend's limits, failures and uncached builds, with Core ML replaced."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openinterpretability.harness.ane import backend, cache
from openinterpretability.harness.scan_backend import AneError
from tests.test_ane import tiny_qwen2

__all__ = ["tiny_qwen2"]  # the fixture, used by name below


@pytest.fixture
def small_shapes(monkeypatch):
    monkeypatch.setattr(backend, "SEQS", (16, 32, 64))
    monkeypatch.setattr(backend, "SHAPE_TOKENS", 64)


def test_a_piece_longer_than_the_largest_shape_keeps_the_scan_on_torch(tiny_qwen2, small_shapes):
    ane = backend.AneScan(tiny_qwen2, "tiny", None, None)
    assert ane.prepare([10, 100], 1, 256) == "a piece has 100 tokens, more than the ANE's 64"


def test_a_loaded_pass_counts_as_cached_and_no_cache_holds_nothing(tiny_qwen2, small_shapes):
    ane = backend.AneScan(tiny_qwen2, "tiny", None, None)
    assert ane._cached(1, 4, 16) is False  # the cache is turned off
    ane._loaded[(1, 4, 16)] = []
    assert ane._cached(1, 4, 16) is True


def test_a_batch_wider_than_every_shape_is_refused(tiny_qwen2, small_shapes):
    ane = backend.AneScan(tiny_qwen2, "tiny", None, None)
    batch = SimpleNamespace(ids=torch.zeros((1, 100), dtype=torch.long), pieces=[0])
    with pytest.raises(ValueError, match="no ANE shape holds 100 tokens"):
        ane.batch_acts(batch, 1)


def test_a_core_ml_failure_in_a_batch_is_an_ane_error(tiny_qwen2, small_shapes, monkeypatch):
    ane = backend.AneScan(tiny_qwen2, "tiny", None, None)

    def compiled(*args):
        raise RuntimeError("the stream was reset")

    monkeypatch.setattr(ane, "compiled", compiled)
    batch = SimpleNamespace(ids=torch.zeros((2, 10), dtype=torch.long), pieces=[0, 1])
    with pytest.raises(AneError, match="RuntimeError: the stream was reset"):
        ane.batch_acts(batch, 1)


@pytest.fixture
def fake_core_ml(monkeypatch):
    """Core ML replaced: a build writes a folder, and a prediction returns ``out[0]``."""
    out = [None]

    def compile_to(net, sample, dest):
        dest.mkdir(parents=True)
        (dest / "weights.bin").write_bytes(b"w")

    monkeypatch.setattr(backend.convert, "compile_to", compile_to)
    monkeypatch.setattr(backend.convert, "load", lambda path: path)
    monkeypatch.setattr(backend.convert, "feed", lambda extra: extra)
    monkeypatch.setattr(backend.convert, "predict", lambda model, x, extra: out[0])
    return out


def test_without_the_cache_a_chunk_is_built_in_a_scratch_folder_and_close_deletes_it(
    tiny_qwen2, fake_core_ml
):
    ane = backend.AneScan(tiny_qwen2, "tiny", None, None)
    net, x, ref = SimpleNamespace(extra={}), torch.zeros(2, 4), torch.ones(2, 4)
    fake_core_ml[0] = ref * (1 + cache.MAX_ERROR / 2)  # close enough to fp32
    path = ane._get({}, net, x, ref)
    assert path.name == "model.mlmodelc" and (path / "weights.bin").exists()
    assert ane._scratch == [path.parent]
    ane.close()
    assert not path.parent.exists() and ane._scratch == []


def test_without_the_cache_a_chunk_far_from_fp32_is_rejected(tiny_qwen2, fake_core_ml):
    ane = backend.AneScan(tiny_qwen2, "tiny", None, None)
    net, x, ref = SimpleNamespace(extra={}), torch.zeros(2, 4), torch.ones(2, 4)
    fake_core_ml[0] = ref * 2  # relative error 1
    with pytest.raises(cache.ConversionRejected, match="relative error 1.0000 is more than"):
        ane._get({}, net, x, ref)
    ane.close()
    assert ane._scratch == []
