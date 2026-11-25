"""The site build's on-disk memo: turned off, and saved before any lookup."""

from __future__ import annotations

from labinacan.site.disk_memo import DiskMemo
from openinterpretability.core.cache_dirs import build_cache

ENV = "OPENINTERPRETABILITY_BUILD_CACHE"


def test_a_memo_that_is_turned_off_keeps_results_only_in_memory(monkeypatch):
    monkeypatch.setenv(ENV, "off")
    memo = DiskMemo("off-test", b"v", build_cache)
    assert memo.path() is None
    memo.put("k", [1, 2])
    memo.save()  # nothing to write to
    assert memo.get("k") == [1, 2]
    assert DiskMemo("off-test", b"v", build_cache).get("k") is None


def test_a_save_before_any_lookup_writes_the_file_and_a_later_lookup_reads_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(ENV, str(tmp_path / "build"))
    memo = DiskMemo("early-save", b"v", build_cache)
    memo.put("k", {"a": 1})
    memo.save()
    assert memo._stored is None and memo._new == {}  # nothing loaded yet, nothing left to save
    path = memo.path()
    assert path.parent == tmp_path / "build" and path.exists()
    assert memo.get("k") == {"a": 1}  # read back from the file
    assert DiskMemo("early-save", b"v", build_cache).get("k") == {"a": 1}
    assert DiskMemo("early-save", b"other version", build_cache).get("k") is None


def test_the_memo_file_is_named_by_the_memo_and_its_versions_hash(tmp_path):
    import hashlib

    path = DiskMemo("m", b"v1", lambda: tmp_path).path()
    assert path == tmp_path / f"m-{hashlib.sha256(b'v1').hexdigest()[:16]}.sqlite"


def test_the_build_cache_is_the_cache_folders_build_folder(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert build_cache() == tmp_path / "openinterpretability" / "build"
