"""Corpora sampled from Parquet datasets, with a source sidecar (offline: local files)."""

import hashlib
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
    first_documents,
    jsonl_bytes,
    load_source,
    load_sources,
    sample_documents,
    sample_tokens,
    source_path,
    wikitext_articles,
    write_corpus,
)
from tests.conftest import FakeTokenizer


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


WIKITEXT = [
    " as the war ended . \n",  # the end of an article cut by the row group's start
    "", " = Homarus gammarus = \n", "",
    " Homarus gammarus , known as the European lobster , is 60 cm ( 24 in ) long . \n",
    "", " = = Description = = \n", "",
    " It weighs 2 @.@ 2 kg , 1 @,@ 000 g are well @-@ known and the \" crusher \" ca n't . \n",
    " The lobsters ' claws and Apple Inc. ' s iPhone 's game ; 50 % of $ 5 [ sic ] . \n",
    "", " = Second = \n", "", " Its only paragraph . \n",
    "", " = Cut = \n", "", " An article cut by the row group's end . \n",
]  # fmt: skip


def test_wikitext_rows_become_whole_detokenised_articles():
    assert wikitext_articles(WIKITEXT) == [
        "Homarus gammarus\n\n"
        "Homarus gammarus, known as the European lobster, is 60 cm (24 in) long.\n\n"
        "Description\n\n"
        'It weighs 2.2 kg, 1,000 g are well-known and the "crusher" can\'t.\n\n'
        "The lobsters' claws and Apple Inc.'s iPhone's game; 50% of $5 [sic].",
        "Second\n\nIts only paragraph.",
    ]
    assert wikitext_articles(WIKITEXT[:10]) == []  # no article ends within the rows


def test_a_layout_joins_each_row_groups_rows_into_documents(tmp_path):
    path = tmp_path / "wikitext.parquet"
    pq.write_table(pa.table({"text": WIKITEXT * 3}), path, row_group_size=len(WIKITEXT))
    docs = sample([str(path)], documents=10, max_chars=1000, layout="wikitext")
    assert len(docs) == 6 and {d.split("\n")[0] for d in docs} == {"Homarus gammarus", "Second"}


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


def test_first_documents_are_the_leading_non_empty_rows_cut_but_not_stripped(tmp_path, shards):
    path = tmp_path / "padded.parquet"
    pq.write_table(pa.table({"text": ["  lead ", "", "second doc\n", "third"]}), path)
    fs = fsspec.filesystem("file")
    docs = first_documents(fs, [str(path)], text_column="text", documents=2, max_chars=6)
    assert docs == ["  lead", "second"]
    # Across files and row groups, in sorted file order.
    docs = first_documents(fs, shards[::-1], text_column="text", documents=12, max_chars=99)
    assert docs[0].startswith("file0 doc1 ") and docs[-1].startswith("file0 doc13 ")


def test_ascii_corpora_escape_non_ascii_and_say_so(tmp_path):
    out = tmp_path / "corpus.jsonl"
    source = write_corpus(["caf\u00e9 \u201cq\u201d"], out, ascii=True, first=True, dataset="d/x",
                          files="*", seed=0, max_chars=99, docs_per_row_group=1)  # fmt: skip
    assert out.read_bytes() == b'{"text": "caf\\u00e9 \\u201cq\\u201d"}\n'
    assert source.ascii and source.sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert read_corpus(out)[:2] == (["caf\u00e9 \u201cq\u201d"], "d/x (*, first 1 docs)")


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


def test_several_globs_each_give_an_equal_share_in_order(tmp_path, shards, monkeypatch):
    local = fsspec.filesystem("file")
    by_glob = {"datasets/org/data@sha1/a/*.parquet": shards[:1],
               "datasets/org/data@sha1/b/*.parquet": shards[1:]}  # fmt: skip

    class FakeFS:
        def glob(self, pattern):
            return by_glob[pattern]

        def open(self, path, mode):
            return local.open(path, mode)

    info = SimpleNamespace(sha="sha1", card_data=None)
    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=lambda d, revision: info)
    )
    out = tmp_path / "c.jsonl"
    for first in (False, True):
        source = build_from_hub(
            "org/data", out, files="a/*.parquet,b/*.parquet", documents=7, first=first
        )
        texts = [json.loads(line)["text"] for line in out.read_text().splitlines()]
        assert [t.split()[0] for t in texts] == ["file0"] * 4 + ["file1"] * 3
        assert (source.files, source.documents) == ("a/*.parquet,b/*.parquet", 7)


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


@pytest.mark.parametrize(
    "preset, dataset, files, revision, layout, column, per_group",
    [
        ("c4", "allenai/c4", "en/partial-validation/*.parquet", "refs/convert/parquet", None,
         "text", 20),
        ("pile-uncopyrighted", "monology/pile-uncopyrighted", "default/partial-train/*.parquet",
         "refs/convert/parquet", None, "text", 20),
        ("wikitext", "Salesforce/wikitext", "wikitext-103-raw-v1/train-*.parquet", None,
         "wikitext", "text", 20),
        ("tinystories", "roneneldan/TinyStories", "data/train-*.parquet", None, None, "text", 20),
        ("github-code", "codeparrot/github-code-clean", "data/train-*.parquet", None, None,
         "code", 100),
        ("wikipedia-multilingual", "wikimedia/wikipedia",
         "20231101.en/train-*.parquet,20231101.de/train-*.parquet,20231101.fr/train-*.parquet,"
         "20231101.ru/train-*.parquet,20231101.el/train-*.parquet,20231101.ar/train-*.parquet,"
         "20231101.hi/train-*.parquet,20231101.zh/train-*.parquet,20231101.ja/train-*.parquet,"
         "20231101.ko/train-*.parquet", None, None, "text", 50),
    ],
)  # fmt: skip
def test_cli_uses_the_presets_settings(
    monkeypatch, tmp_path, preset, dataset, files, revision, layout, column, per_group
):
    calls = {}

    def fake_build(dataset, out, **kw):
        calls.update(kw, dataset=dataset)
        return SimpleNamespace(documents=kw["documents"], license="odc-by", lang_detector=None)

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    args = ["corpus", "--dataset", preset, "--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert calls["dataset"] == dataset and calls["revision"] == revision
    assert (calls["files"], calls["layout"]) == (files, layout)
    assert (calls["text_column"], calls["docs_per_row_group"]) == (column, per_group)
    CliRunner().invoke(cli, [*args, "--revision", "abc123", "--docs-per-row-group", "7"])
    assert (calls["revision"], calls["docs_per_row_group"]) == ("abc123", 7)
    assert (calls["first"], calls["ascii"]) == (False, False)
    CliRunner().invoke(cli, [*args, "--first", "--ascii"])
    assert (calls["first"], calls["ascii"]) == (True, True)


def test_cli_detect_lang_replaces_a_presets_exact_languages(monkeypatch, tmp_path):
    calls = {}

    def fake_build(dataset, out, **kw):
        calls.update(kw)
        return SimpleNamespace(documents=3, license=None, lang_detector="d")

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    out = ["--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    args = ["corpus", "--dataset", "wikipedia-multilingual", *out, "--detect-lang"]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert (calls["detect_lang"], calls["lang_pattern"]) == (True, None)
    both = CliRunner().invoke(cli, [*args, "--lang-pattern", "x"])
    assert both.exit_code != 0 and "both set the languages" in both.output


def test_cli_needs_files_for_non_presets():
    result = CliRunner().invoke(cli, ["corpus", "--dataset", "someone/dataset"])
    assert result.exit_code != 0 and "--files is required" in result.output


DOCS = [f"doc {i} " + "x" * (i * 7 % 23) for i in range(40)]  # 6 to 28 tokens each


def test_token_sample_is_seeded_bounded_and_in_corpus_order():
    tok = FakeTokenizer()
    texts, sample = sample_tokens(tok, DOCS, max_tokens=200, seed=3)
    assert sample_tokens(tok, DOCS, 200, 3) == (texts, sample)  # re-draws the same
    assert sample_tokens(tok, DOCS, 200, 4)[0] != texts
    assert sample.tokens == sum(len(tok.encode(t)) for t in texts) <= 200
    assert sample.tokens > 200 - max(len(d) for d in DOCS)  # filled until nothing fits
    assert texts == [d for d in DOCS if d in texts]  # corpus order
    assert (sample.documents, sample.corpus_documents) == (len(texts), len(DOCS))
    assert sample.sha256 == hashlib.sha256(jsonl_bytes(texts)).hexdigest()


def test_token_sample_takes_everything_that_fits_and_at_least_one_document():
    tok = FakeTokenizer()
    texts, sample = sample_tokens(tok, DOCS, max_tokens=10**6, seed=0)
    assert texts == DOCS and sample.tokens == sum(map(len, DOCS))
    texts, sample = sample_tokens(tok, DOCS, max_tokens=1, seed=0)
    assert len(texts) == 1 and sample.tokens > sample.max_tokens
    with pytest.raises(ValueError, match="no documents"):
        sample_tokens(tok, [], 10, 0)


def test_full_sample_of_a_built_corpus_hashes_like_the_file(tmp_path):
    out = tmp_path / "corpus.jsonl"
    source = write_corpus(DOCS, out, dataset="d", files="*", seed=0, max_chars=99,
                          docs_per_row_group=1)  # fmt: skip
    _, sample = sample_tokens(FakeTokenizer(), read_corpus(out)[0], 10**6, 0)
    assert sample.sha256 == source.sha256


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
    _, sample = sample_tokens(FakeTokenizer(), DOCS, 200, 0)
    provenance = provenance.model_copy(update={"corpus_sample": sample})
    sub = record.submission.model_copy(update={"provenance": provenance})
    rec = AuditRecord(submission=sub, verdict=record.verdict)
    compile_site([rec], tmp_path / "site")
    page = next((tmp_path / "site" / "models").rglob("N*.html")).read_text()
    assert "fineweb-edu@87f09149" in page and "licence odc-by" in page
    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "licence odc-by" in wiki
    shown = f"seeded sample of {sample.documents}/40 docs (≤ 200 tokens, seed 0)"
    assert shown in page and shown in wiki


def test_build_infers_the_source_of_records_that_predate_their_corpus_sidecar(tmp_path, record):
    from openinterp.builder.compiler import compile_site
    from openinterp.core.schemas import AuditRecord, ExperimentProvenance

    out = tmp_path / "corpus.jsonl"
    source = write_corpus(DOCS, out, first=True, ascii=True, dataset="NeelNanda/pile-10k",
                          revision="127bfedc", files="data/*.parquet", seed=0, max_chars=3000,
                          docs_per_row_group=20, license="some-licence")  # fmt: skip
    assert load_sources([source_path(out), tmp_path / "missing.source.json"]) == {
        source.sha256: source
    }

    def built(corpus_source):
        provenance = ExperimentProvenance(
            dtype="float32", corpus="corpus.jsonl", corpus_sha256=source.sha256, n_snippets=40,
            tokens_scanned=1000, selection="random", harness_version="test",
            corpus_source=corpus_source,
        )  # fmt: skip
        sub = record.submission.model_copy(update={"provenance": provenance})
        site = tmp_path / f"site-{corpus_source is None}"
        report = compile_site([AuditRecord(submission=sub, verdict=record.verdict)], site,
                              corpus_sources={source.sha256: source})  # fmt: skip
        page = next((site / "models").rglob("N*.html")).read_text()
        return page, pq.read_table(report.parquet).to_pylist()[0]

    page, row = built(None)  # scanned before the sidecar existed
    assert "pile-10k@127bfedc (data/*.parquet, first 40 docs) (inferred)" in page
    assert "licence some-licence" in page
    assert (row["corpus_dataset"], row["corpus_license"]) == ("NeelNanda/pile-10k", "some-licence")
    assert row["corpus_source_inferred"] is True
    assert '"corpus_source":null' in row["record_json"]  # the record itself is unchanged

    page, row = built(source)  # recorded: shown as is
    assert "(inferred)" not in page and "licence some-licence" in page
    assert row["corpus_source_inferred"] is False
