"""Corpora sampled from Parquet datasets, with a source sidecar (offline: local files)."""

import json
from types import SimpleNamespace

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.cli_evidence import read_corpus
from openinterp.corpus import (
    build_from_hub,
    load_source,
    sample_documents,
    source_path,
    write_corpus,
)


@pytest.fixture
def shards(tmp_path):
    """Two Parquet files, 4 row groups of 10 rows each, with some empty texts."""
    paths = []
    for f in range(2):
        texts = [f"file{f} doc{i} " + "x" * i if i % 7 else "" for i in range(40)]
        path = tmp_path / "data" / f"part-{f}.parquet"
        path.parent.mkdir(exist_ok=True)
        pq.write_table(pa.table({"text": texts}), path, row_group_size=10)
        paths.append(str(path))
    return paths


def sample(shards, **kw):
    args = dict(text_column="text", documents=12, seed=0, max_chars=15, docs_per_row_group=3)
    return sample_documents(fsspec.filesystem("file"), shards, **(args | kw))


def test_sampling_is_seeded_spread_and_clean(shards):
    docs = sample(shards)
    assert docs == sample(shards) and docs != sample(shards, seed=1)
    assert len(docs) == 12 and all(docs) and max(map(len, docs)) == 15
    assert {d.split()[0] for d in docs} == {"file0", "file1"}  # both files, not one


def test_small_datasets_still_fill_the_sample(shards):
    docs = sample(shards[:1], documents=30, docs_per_row_group=1)  # 4 row groups only
    assert len(docs) == 30


def test_source_sidecar_pins_the_exact_corpus(tmp_path):
    out = tmp_path / "corpus.jsonl"
    source = write_corpus(
        ["a", "b"], out, dataset="d/x", revision="abc123", files="*.parquet", seed=0,
        max_chars=10, docs_per_row_group=1, license="odc-by",
    )  # fmt: skip
    assert source_path(out).name == "corpus.source.json" and source.documents == 2
    texts, description, digest, loaded = read_corpus(out)
    assert texts == ["a", "b"] and loaded == source
    assert description == "d/x@abc123 (*.parquet, 2 docs, seed 0)"

    out.write_text(json.dumps({"text": "edited"}) + "\n")  # no longer the sampled corpus
    assert load_source(out, read_corpus(out)[2]) is None
    assert read_corpus(out)[1] == "corpus.jsonl"


def test_build_from_hub_pins_revision_and_licence(tmp_path, shards, monkeypatch):
    local = fsspec.filesystem("file")

    class FakeFS:
        def glob(self, pattern):
            assert pattern == "datasets/org/data@sha1/data/*.parquet"
            return shards

        def open(self, path, mode):
            return local.open(path, mode)

    info = SimpleNamespace(
        sha="sha1", card_data=SimpleNamespace(to_dict=lambda: {"license": "mit"})
    )
    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=lambda d, revision: info)
    )
    source = build_from_hub("org/data", tmp_path / "c.jsonl", files="data/*.parquet", documents=5)
    assert (source.revision, source.license, source.documents) == ("sha1", "mit", 5)


def test_card_less_revisions_take_the_default_branchs_licence(tmp_path, shards, monkeypatch):
    local = fsspec.filesystem("file")

    class FakeFS:
        def glob(self, pattern):
            return shards

        def open(self, path, mode):
            return local.open(path, mode)

    main = SimpleNamespace(
        sha="main1", card_data=SimpleNamespace(to_dict=lambda: {"license": ["odc-by"]})
    )
    converted = SimpleNamespace(sha="conv1", card_data=None)  # refs/convert/parquet: no README
    infos = {None: main, "refs/convert/parquet": converted}
    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        lambda: SimpleNamespace(dataset_info=lambda d, revision=None: infos[revision]),
    )
    source = build_from_hub(
        "org/data", tmp_path / "c.jsonl", files="*.parquet", revision="refs/convert/parquet",
        documents=5,
    )  # fmt: skip
    assert (source.revision, source.license) == ("conv1", "odc-by")


def test_cli_uses_the_presets_revision(monkeypatch, tmp_path):
    calls = {}

    def fake_build(dataset, out, **kw):
        calls.update(kw, dataset=dataset)
        return SimpleNamespace(documents=kw["documents"], license="odc-by")

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    args = ["corpus", "--dataset", "c4", "--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert calls["dataset"] == "allenai/c4" and calls["revision"] == "refs/convert/parquet"
    assert calls["files"] == "en/partial-validation/*.parquet"
    CliRunner().invoke(cli, [*args, "--revision", "abc123"])
    assert calls["revision"] == "abc123"


def test_cli_needs_files_for_non_presets():
    result = CliRunner().invoke(cli, ["corpus", "--dataset", "someone/dataset"])
    assert result.exit_code != 0 and "--files is required" in result.output


def test_experiment_pages_show_the_corpus_licence(tmp_path, record):
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.wiki import write_experiment_page
    from openinterp.core.schemas import AuditRecord, CorpusSource, ExperimentProvenance

    source = CorpusSource(
        dataset="HuggingFaceFW/fineweb-edu", revision="87f09149", files="sample/10BT/*.parquet",
        documents=500, seed=0, max_chars=3000, docs_per_row_group=20, license="odc-by",
        sha256="0" * 64,
    )  # fmt: skip
    provenance = ExperimentProvenance(
        dtype="float32", corpus=str(source), n_snippets=500, tokens_scanned=1000,
        selection="random", harness_version="test", corpus_source=source,
    )  # fmt: skip
    sub = record.submission.model_copy(update={"provenance": provenance})
    rec = AuditRecord(submission=sub, verdict=record.verdict)
    compile_site([rec], tmp_path / "site")
    page = next((tmp_path / "site" / "models").rglob("N*.html")).read_text()
    assert "fineweb-edu@87f09149" in page and "licence odc-by" in page
    assert "licence odc-by" in write_experiment_page(rec, tmp_path / "wiki").read_text()
