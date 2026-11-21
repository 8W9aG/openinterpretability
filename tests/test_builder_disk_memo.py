"""The site build's on-disk memo: turned off, and saved before any lookup."""

from __future__ import annotations

from openinterpretability.builder.disk_memo import DiskMemo

ENV = "OPENINTERPRETABILITY_BUILD_CACHE"


def test_a_memo_that_is_turned_off_keeps_results_only_in_memory(monkeypatch):
    monkeypatch.setenv(ENV, "off")
    memo = DiskMemo("off-test", b"v")
    assert memo.path() is None
    memo.put("k", [1, 2])
    memo.save()  # nothing to write to
    assert memo.get("k") == [1, 2]
    assert DiskMemo("off-test", b"v").get("k") is None


def test_a_save_before_any_lookup_writes_the_file_and_a_later_lookup_reads_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(ENV, str(tmp_path / "build"))
    memo = DiskMemo("early-save", b"v")
    memo.put("k", {"a": 1})
    memo.save()
    assert memo._stored is None and memo._new == {}  # nothing loaded yet, nothing left to save
    path = memo.path()
    assert path.parent == tmp_path / "build" and path.exists()
    assert memo.get("k") == {"a": 1}  # read back from the file
    assert DiskMemo("early-save", b"v").get("k") == {"a": 1}
    assert DiskMemo("early-save", b"other version").get("k") is None
