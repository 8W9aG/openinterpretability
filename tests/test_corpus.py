"""Corpora sampled from Parquet datasets, with a source sidecar (offline: local files)."""

import hashlib
import json
from types import SimpleNamespace

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from openinterp.blends import (
    BLEND_ALIASES,
    BLENDS,
    NO_CODE_LANGUAGE,
    NO_DETECTION,
    NOT_BLENDED,
    blend_shares,
    build_blend,
)
from openinterp.cli import cli
from openinterp.cli_evidence import read_corpus
from openinterp.core.confounds import PROGRAMMING_LANGUAGE
from openinterp.core.corpora import CorpusSource
from openinterp.corpus import (
    FINEWEB2_LANGUAGES,
    PRESETS,
    WIKIPEDIA_LANGUAGES,
    Preset,
    _glob_lang,
    _license,
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
from openinterp.corpus_rebuild import rebuild, scanned_source
from openinterp.evidence import read_labels
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
        ("open-web-math", "open-web-math/open-web-math", "data/train-*.parquet", None, None,
         "text", 20),
        ("gsm8k", "openai/gsm8k", "main/train-*.parquet", None, None, "question,answer", 20),
        ("fineweb-2", "HuggingFaceFW/fineweb-2",
         ",".join(f"data/{f}/train/000_00000.parquet" for f in FINEWEB2_LANGUAGES), None, None,
         "text", 20),
        ("pes2o", "allenai/peS2o",
         "v2/partial-train/*.parquet,v2/partial-validation/000[1-3].parquet",
         "refs/convert/parquet", None, "text", 20),
    ],
)  # fmt: skip
def test_cli_uses_the_presets_settings(
    monkeypatch, tmp_path, preset, dataset, files, revision, layout, column, per_group
):
    calls = {}

    def fake_build(dataset, out, **kw):
        calls.update(kw, dataset=dataset)
        return SimpleNamespace(
            documents=kw["documents"], license="odc-by", lang_detector=None, code_detector=None
        )

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
        return SimpleNamespace(documents=3, license=None, lang_detector="d", code_detector=None)

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    out = ["--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    args = ["corpus", "--dataset", "wikipedia-multilingual", *out, "--detect-lang"]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert (calls["detect_lang"], calls["lang_pattern"]) == (True, None)
    both = CliRunner().invoke(cli, [*args, "--lang-pattern", "x"])
    assert both.exit_code != 0 and "both set the languages" in both.output


def test_cli_detect_code_reaches_the_builder_but_not_for_exact_languages(monkeypatch, tmp_path):
    calls = {}

    def fake_build(dataset, out, **kw):
        calls.update(kw)
        return SimpleNamespace(documents=3, license=None, lang_detector=None, code_detector="d")

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    out = ["--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    result = CliRunner().invoke(cli, ["corpus", "--dataset", "pile-10k", *out, "--detect-code"])
    assert result.exit_code == 0 and calls["detect_code"] is True
    assert "Programming languages detected by d" in result.output
    exact = CliRunner().invoke(cli, ["corpus", "--dataset", "github-code", *out, "--detect-code"])
    assert exact.exit_code != 0 and "exact programming languages" in exact.output


def test_blend_shares_follow_the_weights_and_sum_to_the_documents():
    assert blend_shares("mixed-v1", 500) == {
        "fineweb-edu": 100, "c4": 100, "pile-uncopyrighted": 100, "wikitext": 50,
        "github-code": 75, "wikipedia-multilingual": 75,
    }  # fmt: skip
    assert blend_shares("mixed-v3", 500) == {
        "fineweb-edu": 91, "c4": 91, "pile-uncopyrighted": 91, "wikitext": 46,
        "github-code": 68, "wikipedia-multilingual": 68, "open-web-math": 45,
    }  # fmt: skip
    assert blend_shares("mixed-v4", 500) == {
        "fineweb-edu": 84, "c4": 84, "pile-uncopyrighted": 84, "wikitext": 42,
        "github-code": 62, "wikipedia-multilingual": 62, "open-web-math": 41, "ultrachat": 41,
    }  # fmt: skip
    assert blend_shares("mixed-v5", 500) == {
        "fineweb-edu": 75, "c4": 75, "pile-uncopyrighted": 74, "wikitext": 37,
        "github-code": 55, "wikipedia-multilingual": 55, "open-web-math": 37, "ultrachat": 37,
        "fineweb-2": 55,
    }  # fmt: skip
    assert BLENDS["mixed-v7"] == BLENDS["mixed-v6"]
    assert blend_shares("mixed", 500) == blend_shares("mixed-v7", 500) == {
        "fineweb-edu": 69, "c4": 69, "pile-uncopyrighted": 69, "wikitext": 35,
        "github-code": 52, "wikipedia-multilingual": 52, "open-web-math": 35, "ultrachat": 34,
        "fineweb-2": 51, "pes2o": 34,
    }  # fmt: skip
    shares = blend_shares("mixed", 25)
    assert sum(shares.values()) == 25 and list(shares.values()) == [4, 4, 4, 2, 3, 3, 1, 1, 2, 1]
    assert all(set(weights) <= set(PRESETS) for weights in BLENDS.values())


def test_text_columns_join_into_one_document(tmp_path):
    path = tmp_path / "qa.parquet"
    table = pa.table({"q": ["q0", "q1", None, None], "a": ["a0", None, "a2", None]})
    pq.write_table(table, path)
    fs = fsspec.filesystem("file")
    args = dict(text_column="q,a", documents=9, max_chars=99)
    joined = ["q0\n\na0", "q1", "a2"]  # a row with only nulls is empty, so it isn't a document
    assert first_documents(fs, [str(path)], **args) == joined
    assert sorted(
        sample_documents(fs, [str(path)], seed=0, docs_per_row_group=9, **args)
    ) == sorted(joined)


@pytest.mark.parametrize(
    "card, licence",
    [
        ({"license": "mit"}, "mit"),
        ({"license": "mit", "dataset_info": {"license": "odc-by"}}, "mit"),
        ({"dataset_info": {"license": "odc-by"}}, "odc-by"),
        ({"dataset_info": [{"config_name": "main"}]}, None),
        ({}, None),
    ],
)
def test_the_licence_comes_from_the_card_or_its_dataset_info(card, licence):
    info = SimpleNamespace(card_data=SimpleNamespace(to_dict=lambda: card))
    assert _license(None, "org/data", info, None) == licence


def test_every_preset_is_in_the_latest_mixed_blend_or_says_why_not():
    latest = BLENDS[BLEND_ALIASES["mixed"]]
    assert set(latest) | set(NOT_BLENDED) == set(PRESETS), "a new preset needs a new mixed-vN"
    assert not set(latest) & set(NOT_BLENDED)
    assert BLEND_ALIASES["mixed"] == max(BLENDS, key=lambda name: int(name.split("-v")[1]))


def test_a_blend_concatenates_its_presets_samples_and_labels_their_source(
    tmp_path, shards, monkeypatch
):
    local = fsspec.filesystem("file")
    by_dataset = {"org/web": shards[:1], "org/wiki": shards[1:]}

    class FakeFS:
        def glob(self, pattern):
            repo = pattern.removeprefix("datasets/").split("@")[0]
            return by_dataset[repo]

        def open(self, path, mode):
            return local.open(path, mode)

    def info(dataset, revision=None):
        card = SimpleNamespace(to_dict=lambda: {"license": dataset.split("/")[1] + "-licence"})
        return SimpleNamespace(sha=dataset.split("/")[1] + "-sha", card_data=card)

    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=info))
    monkeypatch.setitem(PRESETS, "web", Preset("org/web", "data/*.parquet"))
    monkeypatch.setitem(
        PRESETS,
        "wiki",
        Preset("org/wiki", "1.en/*.parquet,1.de/*.parquet", lang_pattern=r"^1\.(\w+)/"),
    )
    monkeypatch.setitem(BLENDS, "test-v1", {"web": 2, "wiki": 1})
    monkeypatch.setattr("openinterp.blends.NO_DETECTION", {"test-v1"})

    out = tmp_path / "mixed.jsonl"
    source = build_blend("test-v1", out, documents=9, seed=3, max_chars=99)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["source"] for r in rows] == ["web"] * 6 + ["wiki"] * 3
    assert [r["lang"] for r in rows] == [None] * 6 + ["en", "en", "de"]
    assert read_labels(out) == {"language": [r["lang"] for r in rows],
                                "source": [r["source"] for r in rows]}  # fmt: skip
    assert (source.blend, source.dataset, source.documents) == ("test-v1", "test-v1", 9)
    assert source.license == "org/web: web-licence; org/wiki: wiki-licence"
    assert str(source) == "test-v1 (2 presets, 9 docs, seed 3)"
    assert read_corpus(out)[3] == source  # the sidecar round-trips

    # Each component is exactly what building its preset alone writes.
    for part in source.components:
        name = next(k for k, v in PRESETS.items() if v.dataset == part.dataset)
        preset, alone = PRESETS[name], tmp_path / f"{name}.jsonl"
        built = build_from_hub(
            preset.dataset, alone, files=preset.files, documents=part.documents, seed=3,
            max_chars=99, lang_pattern=preset.lang_pattern,
        )  # fmt: skip
        assert built == part


def test_fineweb_2_folders_are_labelled_with_the_codes_wikipedia_uses():
    pattern = PRESETS["fineweb-2"].lang_pattern
    globs = PRESETS["fineweb-2"].files.split(",")
    langs = [_glob_lang(pattern, g) for g in globs]
    assert langs == list(FINEWEB2_LANGUAGES.values()) and len(set(langs)) == len(langs)
    assert set(WIKIPEDIA_LANGUAGES) - {"en", "de", "fr"} <= set(langs)  # shared codes match
    assert _glob_lang(r"^1\.(\w+)/", "1.en/x.parquet") == "en"  # other codes are kept


def _source(**kw):
    fields = dict(dataset="org/a", revision="r1", files="*.parquet", documents=2, seed=7,
                  max_chars=9, docs_per_row_group=20, sha256="s" * 64)  # fmt: skip
    return CorpusSource(**(fields | kw))


def _fake_build(monkeypatch, name, sha="s" * 64):
    calls = {}

    def build(first_arg, out, **kw):
        calls.update(kw, first_arg=first_arg)
        out.write_text("x")
        source_path(out).write_text("{}")
        return SimpleNamespace(sha256=sha)

    monkeypatch.setattr(f"openinterp.corpus_rebuild.{name}", build)
    return calls


def test_rebuild_pins_a_datasets_source(monkeypatch, tmp_path):
    calls = _fake_build(monkeypatch, "build_from_hub")
    source = _source(text_column="q,a", lang_detector="d", first=True, ascii=True)
    assert rebuild(source, tmp_path / "c.jsonl").sha256 == source.sha256
    assert (calls["first_arg"], calls["revision"], calls["text_column"]) == ("org/a", "r1", "q,a")
    assert (calls["seed"], calls["detect_lang"], calls["first"]) == (7, True, True)
    assert calls["detect_code"] is False
    rebuild(_source(code_detector="magika"), tmp_path / "c.jsonl")
    assert calls["detect_code"] is True


def test_rebuild_pins_each_part_of_a_blend(monkeypatch, tmp_path):
    calls = _fake_build(monkeypatch, "build_blend")
    source = _source(
        blend="mixed-v3", components=[_source(), _source(dataset="org/b", revision="r2")]
    )
    rebuild(source, tmp_path / "c.jsonl")
    assert calls["first_arg"] == "mixed-v3" and calls["seed"] == 7
    assert calls["revisions"] == {"org/a": "r1", "org/b": "r2"}


def test_rebuild_leaves_no_corpus_if_the_bytes_differ(monkeypatch, tmp_path):
    _fake_build(monkeypatch, "build_from_hub", sha="d" * 64)
    out = tmp_path / "c.jsonl"
    with pytest.raises(ValueError, match="d{64}"):
        rebuild(_source(), out)
    assert not out.exists() and not source_path(out).exists()


def test_a_blends_revisions_override_its_presets(monkeypatch, tmp_path):
    seen = {}

    def sample(dataset, *, revision, documents, **kw):
        seen[dataset] = revision
        source = dict(dataset=dataset, revision=revision, files="f", seed=0, max_chars=9)
        return ["text"] * documents, None, source

    monkeypatch.setattr("openinterp.blends._sample_hub", sample)
    monkeypatch.setitem(PRESETS, "a", Preset("org/a", "*.parquet", revision="branch"))
    monkeypatch.setitem(PRESETS, "b", Preset("org/b", "*.parquet"))
    monkeypatch.setitem(BLENDS, "test-v1", {"a": 1, "b": 1})
    build_blend("test-v1", tmp_path / "c.jsonl", documents=2, revisions={"org/a": "commit"})
    assert seen == {"org/a": "commit", "org/b": None}


def test_a_blend_writes_chat_with_its_presets_template_at_the_pinned_commit(monkeypatch, tmp_path):
    seen = {}

    def sample(dataset, *, revision, documents, chat_template, chat_template_revision, **kw):
        seen[dataset] = (chat_template, chat_template_revision)
        source = dict(dataset=dataset, revision=revision, files="f", seed=0, max_chars=9)
        return ["text"] * documents, None, source

    monkeypatch.setattr("openinterp.blends._sample_hub", sample)
    monkeypatch.setitem(PRESETS, "a", Preset("org/a", "*.parquet"))
    monkeypatch.setitem(PRESETS, "chat", Preset("org/chat", "*.parquet", chat_template="org/m"))
    monkeypatch.setitem(BLENDS, "test-v1", {"a": 1, "chat": 1})
    build_blend("test-v1", tmp_path / "c.jsonl", documents=2, revisions={"org/m": "m1"})
    assert seen == {"org/a": (None, None), "org/chat": ("org/m", "m1")}


def test_rebuild_pins_a_blends_chat_template_commit(monkeypatch, tmp_path):
    calls = _fake_build(monkeypatch, "build_blend")
    chat = _source(dataset="org/chat", chat_template="org/m", chat_template_revision="m1")
    rebuild(_source(blend="mixed-v4", components=[_source(), chat]), tmp_path / "c.jsonl")
    assert calls["revisions"] == {"org/a": "r1", "org/chat": "r1", "org/m": "m1"}


def test_scanned_source_prefers_the_record_then_a_committed_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    own = _source()
    prov = SimpleNamespace(corpus="c", corpus_sha256="e" * 64, corpus_source=own)
    assert scanned_source(SimpleNamespace(provenance=prov)) == own
    legacy = SimpleNamespace(corpus="corpus.jsonl", corpus_sha256="e" * 64, corpus_source=None)
    with pytest.raises(ValueError, match="how to rebuild"):
        scanned_source(SimpleNamespace(provenance=legacy))
    with pytest.raises(ValueError, match="no provenance"):
        scanned_source(SimpleNamespace(provenance=None))
    sidecar = _source(sha256="e" * 64)
    (tmp_path / "corpus.source.json").write_text(sidecar.model_dump_json())
    assert scanned_source(SimpleNamespace(provenance=legacy)) == sidecar


def test_cli_same_as_takes_only_out(monkeypatch, tmp_path):
    record = tmp_path / "record.json"
    record.write_text("{}")
    args = ["corpus", "--same-as", str(record), "--out", str(tmp_path / "c.jsonl")]
    refused = CliRunner().invoke(cli, [*args, "--seed", "3"])
    assert refused.exit_code != 0 and "not with: seed" in refused.output


def test_cli_builds_the_mixed_blend_by_default(monkeypatch, tmp_path):
    calls = {}

    def fake_blend(name, out, **kw):
        calls.update(kw, name=name)
        return SimpleNamespace(documents=kw["documents"], license=None, lang_detector=None,
                               code_detector=None,
                               components=[])  # fmt: skip

    monkeypatch.setattr("openinterp.blends.build_blend", fake_blend)
    args = ["corpus", "--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert calls == {"name": "mixed", "documents": 3, "seed": 0, "max_chars": 3000,
                     "ascii": False}  # fmt: skip
    wrong = CliRunner().invoke(cli, [*args, "--first", "--revision", "abc"])
    assert wrong.exit_code != 0 and "--revision, --first: set per preset" in wrong.output


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


@pytest.fixture
def code_shards(tmp_path):
    """A Parquet file whose rows record their language, with some empty texts."""
    texts = [f"doc{i}" if i % 3 else "" for i in range(24)]
    langs = [f"lang{i}" for i in range(24)]
    path = tmp_path / "code.parquet"
    pq.write_table(pa.table({"text": texts, "language": langs}), path, row_group_size=8)
    return [str(path)]


def test_sampled_documents_keep_their_own_label(code_shards):
    fs, labels = fsspec.filesystem("file"), []
    docs = sample_documents(
        fs, code_shards, text_column="text", documents=9, seed=0, max_chars=9,
        docs_per_row_group=3, label_column="language", labels=labels,
    )  # fmt: skip
    assert len(docs) == len(labels) == 9
    assert [d.removeprefix("doc") for d in docs] == [x.removeprefix("lang") for x in labels]


def test_first_documents_keep_their_own_label(code_shards):
    fs, labels = fsspec.filesystem("file"), []
    docs = first_documents(
        fs, code_shards, text_column="text", documents=4, max_chars=9,
        label_column="language", labels=labels,
    )  # fmt: skip
    assert docs == ["doc1", "doc2", "doc4", "doc5"]
    assert labels == ["lang1", "lang2", "lang4", "lang5"]


@pytest.fixture
def code_blend(monkeypatch, code_shards):
    """Two blend versions over a code preset and a preset of prose, on a fake Hub."""
    local = fsspec.filesystem("file")

    class FakeFS:
        def glob(self, pattern):
            return code_shards

        def open(self, path, mode):
            return local.open(path, mode)

    def info(dataset, revision=None):
        return SimpleNamespace(sha="sha", card_data=None)

    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=info))
    monkeypatch.setitem(
        PRESETS, "code", Preset("org/code", "*.parquet", code_language_column="language")
    )
    monkeypatch.setitem(PRESETS, "prose", Preset("org/prose", "*.parquet"))
    for name in ("test-v1", "test-v2"):
        monkeypatch.setitem(BLENDS, name, {"code": 1, "prose": 1})
    monkeypatch.setattr("openinterp.blends.NO_CODE_LANGUAGE", {"test-v1"})
    monkeypatch.setattr("openinterp.blends.NO_DETECTION", {"test-v1", "test-v2"})


def test_a_blend_records_code_languages_and_its_older_version_stays_without(code_blend, tmp_path):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    build_blend("test-v1", old, documents=4, seed=1, max_chars=9)
    source = build_blend("test-v2", new, documents=4, seed=1, max_chars=9)

    rows = [json.loads(line) for line in new.read_text().splitlines()]
    assert all(r["code_language"].startswith("lang") for r in rows[:2])
    assert [r["code_language"] for r in rows[2:]] == [None, None]
    assert read_labels(new)[PROGRAMMING_LANGUAGE] == [r["code_language"] for r in rows]
    assert source.components[0].code_language_column == "language"
    assert source.components[1].code_language_column is None

    assert PROGRAMMING_LANGUAGE not in read_labels(old)
    assert "code_language" not in old.read_text()
    kept = [{k: v for k, v in r.items() if k != "code_language"} for r in rows]
    assert [json.loads(line) for line in old.read_text().splitlines()] == kept


def test_a_blend_detects_only_the_labels_a_part_lacks(code_blend, tmp_path, monkeypatch):
    from openinterp import codeid, langid

    monkeypatch.setattr(langid, "detect", lambda texts: ["en"] * len(texts))
    monkeypatch.setattr(langid, "DETECTOR", "fake lid")
    monkeypatch.setattr(codeid, "detect", lambda texts: ["Python"] * len(texts))
    monkeypatch.setattr(codeid, "detector", lambda: "fake magika")
    monkeypatch.setitem(BLENDS, "test-v3", {"code": 1, "prose": 1})
    plain, detected = tmp_path / "v2.jsonl", tmp_path / "v3.jsonl"
    build_blend("test-v2", plain, documents=4, seed=1, max_chars=9)
    source = build_blend("test-v3", detected, documents=4, seed=1, max_chars=9)

    rows = [json.loads(line) for line in detected.read_text().splitlines()]
    assert all(r["code_language"].startswith("lang") for r in rows[:2])  # exact, kept
    assert [r["code_language"] for r in rows[2:]] == ["Python", "Python"]
    assert [r["lang"] for r in rows] == ["en"] * 4
    code, prose = source.components
    assert (code.code_detector, code.lang_detector) == (None, "fake lid")
    assert (prose.code_detector, prose.lang_detector) == ("fake magika", "fake lid")
    assert "Python" not in plain.read_text() and '"en"' not in plain.read_text()


def test_released_blends_keep_their_bytes_without_detection():
    assert {"mixed-v1", "mixed-v6"} <= NO_DETECTION and "mixed-v7" not in NO_DETECTION


def test_released_mixed_v1_keeps_its_bytes_and_mixed_v2_its_mix():
    assert "mixed-v1" in NO_CODE_LANGUAGE
    assert BLENDS["mixed-v2"] == BLENDS["mixed-v1"]


def test_a_layout_and_a_label_column_are_refused(code_blend, tmp_path):
    with pytest.raises(ValueError, match="layout"):
        build_from_hub(
            "org/code", tmp_path / "c.jsonl", files="*.parquet", layout="wikitext",
            code_language_column="language",
        )  # fmt: skip


def _chat(messages):
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def test_a_chat_template_writes_each_conversation_as_one_document(tmp_path):
    path = tmp_path / "chat.parquet"
    talks = [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        [],  # an empty conversation is not a document
        [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
    ]
    pq.write_table(pa.table({"messages": talks}), path)
    fs = fsspec.filesystem("file")
    args = dict(text_column="messages", documents=9, max_chars=99, chat=_chat)
    written = [
        "<user>hi</user><assistant>hello</assistant>",
        "<user>2+2?</user><assistant>4</assistant>",
    ]
    assert first_documents(fs, [str(path)], **args) == written
    assert sample_documents(fs, [str(path)], seed=0, docs_per_row_group=9, **args) == written
    with pytest.raises(ValueError, match="one column"):
        first_documents(fs, [str(path)], **(args | {"text_column": "messages,other"}))


def test_build_from_hub_pins_the_chat_templates_model(tmp_path, monkeypatch):
    path = tmp_path / "data" / "chat.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"messages": [[{"role": "user", "content": "hi"}]]}), path)
    local = fsspec.filesystem("file")
    fs = SimpleNamespace(glob=lambda pattern: [str(path)], open=local.open)
    info = SimpleNamespace(sha="sha1", card_data=None)
    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda: fs)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=lambda d, revision=None: info)
    )
    asked = []
    monkeypatch.setattr(
        "openinterp.corpus.chat_formatter",
        lambda model, revision: asked.append((model, revision)) or (_chat, "m1"),
    )
    out = tmp_path / "c.jsonl"
    source = build_from_hub(
        "org/chat", out, files="data/*.parquet", text_column="messages", chat_template="org/m"
    )
    assert asked == [("org/m", None)]
    assert (source.chat_template, source.chat_template_revision) == ("org/m", "m1")
    assert json.loads(out.read_text())["text"] == "<user>hi</user>"
    assert "org/m's chat template" in str(source)
    with pytest.raises(ValueError, match="layout"):
        build_from_hub("org/chat", out, files="x", chat_template="org/m", layout="wikitext")


def test_rebuild_pins_the_chat_templates_commit(monkeypatch, tmp_path):
    calls = _fake_build(monkeypatch, "build_from_hub")
    rebuild(_source(chat_template="org/m", chat_template_revision="m1"), tmp_path / "c.jsonl")
    assert (calls["chat_template"], calls["chat_template_revision"]) == ("org/m", "m1")


def test_cli_chat_presets_take_their_template_unless_told_another(monkeypatch, tmp_path):
    calls = {}

    def fake_build(dataset, out, **kw):
        calls.update(kw, dataset=dataset)
        return SimpleNamespace(
            documents=kw["documents"], license="mit", lang_detector=None, code_detector=None
        )

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    args = ["corpus", "--dataset", "ultrachat", "--docs", "3", "--out", str(tmp_path / "c.jsonl")]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert (calls["dataset"], calls["text_column"]) == ("HuggingFaceH4/ultrachat_200k", "messages")
    assert (calls["files"], calls["chat_template"]) == (
        "data/train_sft-*.parquet",
        "Qwen/Qwen2.5-0.5B",
    )
    assert CliRunner().invoke(cli, [*args, "--chat-template", "org/instruct"]).exit_code == 0
    assert calls["chat_template"] == "org/instruct"
    CliRunner().invoke(cli, ["corpus", "--dataset", "c4", "--out", str(tmp_path / "c.jsonl")])
    assert calls["chat_template"] is None
    blend = CliRunner().invoke(cli, ["corpus", "--chat-template", "Qwen/Qwen2.5-0.5B"])
    assert blend.exit_code != 0 and "--chat-template" in blend.output


def test_cli_reports_a_model_without_a_chat_template(monkeypatch, tmp_path):
    def fake_build(dataset, out, **kw):
        raise ValueError("openai-community/gpt2's tokenizer has no chat template")

    monkeypatch.setattr("openinterp.corpus.build_from_hub", fake_build)
    args = ["corpus", "--dataset", "ultrachat", "--chat-template", "openai-community/gpt2"]
    result = CliRunner().invoke(cli, [*args, "--out", str(tmp_path / "c.jsonl")])
    assert result.exit_code == 1 and "has no chat template" in result.output
