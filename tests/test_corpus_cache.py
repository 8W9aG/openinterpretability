import os
from functools import partial

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.corpus import first_documents, sample_documents
from openinterp.corpus_cache import ENV, ParquetCache, read_row_group, row_groups

COMMIT = "a" * 40


class CountingFS:
    """A local file system that counts how many times code opens a file."""

    def __init__(self):
        self.opens = 0
        self.local = fsspec.filesystem("file")

    def open(self, path, mode):
        self.opens += 1
        return self.local.open(path, mode)


@pytest.fixture
def pinned(tmp_path):
    path = tmp_path / f"org@{COMMIT}" / "part.parquet"
    path.parent.mkdir()
    rows = {"q": [f"q{i}" for i in range(6)], "a": [f"a{i}" for i in range(6)]}
    pq.write_table(pa.table(rows), path, row_group_size=2)
    return str(path)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(ENV, str(root))
    return root


def test_a_pinned_read_opens_the_file_once_and_returns_the_same_data(pinned, cache_dir):
    fs = CountingFS()
    cold = read_row_group(fs, pinned, 1, ["a", "q"])
    assert read_row_group(fs, pinned, 1, ["a", "q"]).equals(cold)
    assert fs.opens == 1 and cold.column_names == ["a", "q"]
    assert cold.column("q").to_pylist() == ["q2", "q3"]
    assert row_groups(fs, pinned) == row_groups(fs, pinned) == 3
    assert fs.opens == 2


def test_other_columns_and_row_groups_are_other_entries(pinned, cache_dir):
    fs = CountingFS()
    assert read_row_group(fs, pinned, 0, ["q"]).column_names == ["q"]
    assert read_row_group(fs, pinned, 0, ["a"]).column_names == ["a"]
    assert read_row_group(fs, pinned, 1, ["q"]).column("q").to_pylist() == ["q2", "q3"]
    assert fs.opens == 3


def test_a_path_without_a_full_commit_is_never_cached(tmp_path, cache_dir):
    path = tmp_path / "org@main" / "part.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"q": ["x"]}), path)
    fs = CountingFS()
    read_row_group(fs, str(path), 0, ["q"])
    read_row_group(fs, str(path), 0, ["q"])
    assert fs.opens == 2 and not cache_dir.exists()


def test_turning_the_cache_off_reads_every_time(pinned, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "off")
    fs = CountingFS()
    read_row_group(fs, pinned, 0, ["q"])
    read_row_group(fs, pinned, 0, ["q"])
    assert fs.opens == 2


def test_a_damaged_entry_is_read_again(pinned, cache_dir):
    fs = CountingFS()
    good = read_row_group(fs, pinned, 0, ["q"])
    entry = next(cache_dir.glob("*.parquet"))
    entry.write_bytes(b"not parquet")
    assert read_row_group(fs, pinned, 0, ["q"]).equals(good)
    assert fs.opens == 2
    assert pq.read_table(entry).equals(good)


def test_a_cache_that_cannot_write_does_not_fail_the_read(pinned, tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(ENV, str(blocker / "cache"))
    assert read_row_group(CountingFS(), pinned, 0, ["q"]).column("q").to_pylist() == ["q0", "q1"]


def test_eviction_removes_the_least_recently_used_and_keeps_the_new_entry(tmp_path):
    cache = ParquetCache(tmp_path, max_bytes=25)
    old, recent, new = tmp_path / "old", tmp_path / "recent", tmp_path / "new"
    for age, path in enumerate((new, recent, old)):
        path.write_bytes(b"x" * 10)
        os.utime(path, (1000 - age, 1000 - age))
    cache.evict(keep=new)
    assert sorted(p.name for p in cache.entries()) == ["new", "recent"]
    assert cache.clear() == 2 and cache.entries() == []


@pytest.mark.parametrize("seeded", [True, False])
def test_a_build_gives_the_same_documents_with_a_cold_and_warm_cache(pinned, cache_dir, seeded):
    fs = CountingFS()
    args = dict(text_column="q,a", documents=4, max_chars=99)
    seed = dict(seed=3, docs_per_row_group=2)
    read = (
        partial(sample_documents, fs, [pinned], **seed, **args)
        if seeded
        else partial(first_documents, fs, [pinned], **args)
    )
    cold = read()
    opens = fs.opens
    assert read() == cold and fs.opens == opens


def test_cli_reports_and_clears_the_cache(pinned, cache_dir):
    read_row_group(CountingFS(), pinned, 0, ["q"])
    shown = CliRunner().invoke(cli, ["corpus-cache"])
    assert shown.exit_code == 0 and "1 entries" in shown.output
    assert "Deleted 1 entries" in CliRunner().invoke(cli, ["corpus-cache", "--clear"]).output


def test_cli_says_when_the_cache_is_off(monkeypatch):
    monkeypatch.setenv(ENV, "off")
    assert "is off" in CliRunner().invoke(cli, ["corpus-cache"]).output
