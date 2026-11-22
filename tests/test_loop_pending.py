"""The records of a loop's open PRs, counted before they merge (#5826)."""

import subprocess
from pathlib import Path

import pytest

from openinterpretability.loop_pending import PendingRecords


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def branch(tmp_path, monkeypatch):
    """A clone whose branch adds two records and a file outside wiki/, against origin/main."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    _git("init", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    _git("clone", str(origin), str(work), cwd=tmp_path)
    for key, value in (("user.name", "t"), ("user.email", "t@example.com")):
        _git("config", key, value, cwd=work)
    (work / "wiki").mkdir()
    (work / "wiki" / "old.json").write_text("old")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "origin", "HEAD:main", cwd=work)
    _git("switch", "-c", "experiment/x", cwd=work)
    (work / "wiki" / "a").mkdir()
    (work / "wiki" / "a" / "rec.json").write_text("rec")
    (work / "wiki" / "old.json").write_text("edited")  # changed, not added: not kept
    (work / "notes.txt").write_text("outside wiki")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "experiment", cwd=work)
    monkeypatch.chdir(work)
    return work


def test_keep_takes_only_the_files_the_branch_adds_under_wiki(branch):
    pending = PendingRecords()
    assert pending.keep() == ["wiki/a/rec.json"]
    assert pending.files == {"wiki/a/rec.json": b"rec"}


def test_applied_writes_missing_files_and_removes_them_after(branch):
    pending = PendingRecords()
    pending.keep()
    _git("checkout", "--quiet", "--detach", "origin/main", cwd=branch)
    record = Path("wiki/a/rec.json")
    assert not record.exists()
    with pending.applied() as written:
        assert written == [record] and record.read_text() == "rec"
    assert not record.exists()


def test_a_file_already_there_is_left_as_it_is_and_kept_after(tmp_path, monkeypatch):
    # Its PR merged, so main holds it: the block must neither overwrite nor delete it.
    monkeypatch.chdir(tmp_path)
    merged = tmp_path / "wiki" / "rec.json"
    merged.parent.mkdir()
    merged.write_text("merged")
    pending = PendingRecords({"wiki/rec.json": b"pending"})
    with pending.applied() as written:
        assert written == []
    assert merged.read_text() == "merged"


def test_the_files_are_removed_when_the_block_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pending = PendingRecords({"wiki/a/rec.json": b"rec"})
    with pytest.raises(RuntimeError), pending.applied():
        raise RuntimeError("the draw failed")
    assert not (tmp_path / "wiki" / "a" / "rec.json").exists()
