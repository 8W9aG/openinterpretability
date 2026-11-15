"""The sister search's random rotations are cached on disk (harness/sisters.py, issue #3646)."""

import torch

from openinterp.harness import sisters


def test_a_cached_rotation_is_the_fresh_one_bit_for_bit(tmp_path, monkeypatch):
    monkeypatch.setenv(sisters.ROTATION_CACHE_ENV, "0")
    fresh = sisters.rotation("m/x", 3, 48)
    monkeypatch.setenv(sisters.ROTATION_CACHE_ENV, str(tmp_path))
    first = sisters.rotation("m/x", 3, 48)  # computed, then saved
    path = sisters.rotation_cache("m/x", 3, 48)
    assert path.exists() and torch.equal(first, fresh)
    assert torch.equal(sisters.rotation("m/x", 3, 48), fresh)  # read back from disk
    assert torch.allclose(fresh @ fresh.T, torch.eye(48), atol=1e-5)


def test_the_cache_reads_its_own_file_not_another_layers(tmp_path, monkeypatch):
    monkeypatch.setenv(sisters.ROTATION_CACHE_ENV, str(tmp_path))
    torch.save(torch.zeros(48, 48), sisters.rotation_cache("m/x", 4, 48))  # a planted file
    assert torch.equal(sisters.rotation("m/x", 4, 48), torch.zeros(48, 48))
    assert not torch.equal(sisters.rotation("m/x", 5, 48), torch.zeros(48, 48))


def test_the_name_holds_the_torch_version_and_0_turns_it_off(tmp_path, monkeypatch):
    monkeypatch.setenv(sisters.ROTATION_CACHE_ENV, str(tmp_path))
    path = sisters.rotation_cache("org/model", 2, 64)
    assert path.parent == tmp_path and torch.__version__ in path.name
    assert path.name.startswith("org--model--L2--w64--seed")
    monkeypatch.setenv(sisters.ROTATION_CACHE_ENV, "0")
    assert sisters.rotation_cache("org/model", 2, 64) is None
