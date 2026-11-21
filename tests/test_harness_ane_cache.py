"""The on-disk cache of compiled Core ML models (harness/ane/cache.py)."""

import hashlib
import os

from openinterpretability.harness.ane import cache
from tests.test_ane import _fake, _key


class _BuiltWhileWaiting(cache.ConversionCache):
    """A cache where another process builds the entry while this one waits for the lock."""

    def __init__(self, root, other):
        super().__init__(root)
        self.other = other

    def _lock(self, entry):
        self.other()  # takes the lock, builds the entry and lets the lock go
        return super()._lock(entry)


def test_an_entry_built_while_waiting_for_the_lock_is_not_built_again(tmp_path):
    other_calls, other_build, check = _fake()
    calls, build, _ = _fake()
    first = cache.ConversionCache(tmp_path)
    store = _BuiltWhileWaiting(tmp_path, lambda: first.get(_key(), other_build, check))
    path = store.get(_key(), build, check)
    assert (path.parent / cache.META).exists()
    assert other_calls["build"] == 1 and calls["build"] == 0


def test_weights_already_linked_to_their_blob_stay_as_they_are(tmp_path):
    weights = b"w" * 64
    blobs = tmp_path / cache.BLOBS
    blobs.mkdir()
    blob = blobs / f"{hashlib.sha256(weights).hexdigest()}.bin"
    blob.write_bytes(weights)

    def build(dest):  # a build that links the stored blob in place of a new file
        (dest / "weights").mkdir(parents=True)
        os.link(blob, dest / "weights" / "weight.bin")

    store = cache.ConversionCache(tmp_path)
    path = store.get(_key(), build, lambda _path: 0.001)
    linked = path / "weights" / "weight.bin"
    assert linked.samefile(blob) and linked.read_bytes() == weights
    assert list(blobs.glob("*.bin")) == [blob]


def test_eviction_keeps_the_entry_in_use_and_drops_a_newer_one(tmp_path):
    store = cache.ConversionCache(tmp_path)
    _, build, check = _fake()
    keep = store.get(_key(layer=1), build, check).parent  # the oldest entry
    newer = store.get(_key(layer=2), build, check).parent
    store.max_bytes = store.size() - 1  # one entry must go
    assert store.evict(keep=keep) == [newer]
    assert keep.exists() and not newer.exists()
