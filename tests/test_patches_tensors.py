"""The patch tensor store: repeated puts, hash checks and finding every reference."""

import pytest
import torch

from openinterpretability.patches.tensors import KEY, _path, load, put_tensor, refs_in


def test_putting_a_stored_tensor_again_keeps_the_one_blob(tmp_path):
    ref = put_tensor(torch.arange(3.0), tmp_path)
    path = _path(ref[KEY], tmp_path)
    before = path.stat().st_mtime_ns
    assert put_tensor(torch.arange(3.0), tmp_path) == ref
    assert path.stat().st_mtime_ns == before
    torch.testing.assert_close(load(ref, tmp_path), torch.arange(3.0))


def test_a_blob_that_does_not_match_its_hash_is_refused(tmp_path):
    ref = put_tensor(torch.ones(2), tmp_path)
    _path(ref[KEY], tmp_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match=f"tensor blob {ref[KEY][:12]} does not match its hash"):
        load(ref, tmp_path)


def test_refs_in_finds_every_reference_in_a_parameter_tree():
    a, b = {KEY: "a" * 64}, {KEY: "b" * 64}
    tree = {"vector": a, "list": [1, {"inner": b}, [a]], "scale": 0.5, "name": "x"}
    assert refs_in(tree) == ["a" * 64, "b" * 64, "a" * 64]
    assert refs_in(a) == ["a" * 64] and refs_in(2.0) == []
