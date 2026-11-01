import json
from types import SimpleNamespace

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.concepts import (
    Sentence,
    build_concept,
    group_sentences,
    join_words,
    locate_words,
    sample_concept,
)
from openinterp.core.probing import ConceptSource
from openinterp.harness.annotations import Span
from openinterp.harness.probing import read_labels

NER = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]


def test_join_words_attaches_punctuation_and_clitics():
    text, offsets = join_words(
        ["(", "AP", ")", "Smith", "does", "n't", "agree", ",", "she", "said", "."]
    )
    assert text == "(AP) Smith doesn't agree, she said."
    assert [text[a:b] for a, b in offsets][:7] == ["(", "AP", ")", "Smith", "does", "n't", "agree"]


def test_locate_words_gives_fused_words_their_surface_form():
    text = "Il parle du film."
    words = ["Il", "parle", "de", "le", "film", "."]
    offsets = locate_words(text, words, [{"id": "3-4", "form": "du"}])
    assert [text[a:b] for a, b in offsets] == ["Il", "parle", "du", "du", "film", "."]
    assert locate_words("Il parle du film.", words) is None  # "de" isn't where it should be


def test_group_sentences_starts_documents_and_cuts_long_ones():
    a = Sentence("Bob ran.", [Span(0, 3, "PER")], newdoc=True)
    b = Sentence("Al sat.", [Span(0, 2, "PER")], newpar=False)
    c = Sentence("Next doc.", [], newdoc=True)
    docs = group_sentences([a, b, c], max_chars=100)
    assert [t for t, _ in docs] == ["Bob ran. Al sat.", "Next doc."]
    assert [docs[0][0][s.start : s.end] for s in docs[0][1]] == ["Bob", "Al"]
    assert [t for t, _ in group_sentences([a, b], max_chars=10)] == ["Bob ran.", "Al sat."]
    cut = group_sentences([Sentence("Alexander", [Span(0, 9, "PER")])], max_chars=4)
    assert cut == [("Alex", [Span(0, 4, "PER")])]


def test_sample_concept_balances_document_classes():
    docs = [(f"d{i}", [Span(0, 2, "fr" if i < 3 else "en")]) for i in range(10)]
    rows = sample_concept(docs, "fr", 4, seed=0, balance=True)
    assert sum(bool(spans) for _, spans in rows) == 2 and len(rows) == 4
    assert [t for t, _ in rows] == sorted(t for t, _ in rows)  # split order
    sub = sample_concept([("x", [Span(0, 1, "nmod:poss")])], "nmod", 5, 0, balance=False)
    assert sub == [("x", [(0, 1)])]


def _parquet(path, table: pa.Table, features: dict):
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    pq.write_table(table.replace_schema_metadata(meta), path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub serving three presets' layouts from local Parquet files."""
    root = tmp_path / "hub"
    (root / "conll").mkdir(parents=True)
    (root / "ud").mkdir()
    (root / "news").mkdir()
    _parquet(
        root / "conll" / "0000.parquet",
        pa.table({
            "tokens": [["John", "Smith", "visited", "IBM", "."], ["LONDON", "1996-08-30"]],
            "ner_tags": [[1, 2, 0, 3, 0], [0, 0]],
        }),
        {"ner_tags": {"feature": {"names": NER, "_type": "ClassLabel"}, "_type": "Sequence"}},
    )  # fmt: skip
    _parquet(
        root / "ud" / "dev.parquet",
        pa.table({
            "text": ["Il parle du film.", "Bonjour."],
            "tokens": [["Il", "parle", "de", "le", "film", "."], ["Bonjour", "."]],
            "mwt": [[{"id": "3-4", "form": "du"}], []],
            "comments": [["newdoc id = a"], ["newdoc id = b"]],
            "upos": [["PRON", "VERB", "ADP", "DET", "NOUN", "PUNCT"], ["INTJ", "PUNCT"]],
        }),
        {},
    )  # fmt: skip
    _parquet(
        root / "news" / "test-0.parquet",
        pa.table({"text": ["Goal!", "Stocks fell.", "Win.", "Rates rose."], "label": [1, 2, 1, 2]}),
        {"label": {"names": ["World", "Sports", "Business", "Sci/Tech"], "_type": "ClassLabel"}},
    )
    dirs = {"eriktks/conll2003": "conll", "universal-dependencies/universal_dependencies": "ud",
            "fancyzhx/ag_news": "news"}  # fmt: skip
    local = fsspec.filesystem("file")

    class FakeFS:
        def glob(self, pattern):
            dataset = pattern.removeprefix("datasets/").split("@")[0]
            return sorted(str(p) for p in (root / dirs[dataset]).glob("*.parquet"))

        def open(self, path, mode):
            return local.open(path, mode)

        def read_text(self, path):
            assert path.endswith("metadata.json")
            return json.dumps({"fr_gsd": {"license": "CC BY-SA 4.0"}})

    card = SimpleNamespace(to_dict=lambda: {"license": ["other"]})
    info = SimpleNamespace(sha="sha1", card_data=card)
    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=lambda d, revision=None: info)
    )


def test_word_tags_become_spans_probe_compare_reads(tmp_path, hub):
    out = tmp_path / "per.jsonl"
    source = build_concept("conll2003-ner", "PER", out)
    [(text, spans)] = read_labels(out)
    assert text == "John Smith visited IBM.\nLONDON 1996-08-30"
    assert [text[s:e] for s, e in spans] == ["John", "Smith"]
    assert ConceptSource.model_validate_json((tmp_path / "per.source.json").read_text()) == source
    assert (source.revision, source.license, source.level, source.spans) == (
        "sha1",
        "other",
        "token",
        2,
    )
    with pytest.raises(ValueError, match="it has: PER, ORG"):
        build_concept("conll2003-ner", "LOC", out)


def test_treebanks_keep_documents_and_their_own_licence(tmp_path, hub):
    out = tmp_path / "det.jsonl"
    source = build_concept("ud-upos", "DET", out, subset="fr_gsd")
    rows = read_labels(out)
    assert [t for t, _ in rows] == ["Il parle du film.", "Bonjour."]
    assert [rows[0][0][s:e] for s, e in rows[0][1]] == ["du"]
    assert source.license == "CC BY-SA 4.0" and source.files == "parquet/fr_gsd/dev.parquet"


def test_document_classes_cover_the_whole_text(tmp_path, hub):
    out = tmp_path / "sports.jsonl"
    source = build_concept("ag_news", "Sports", out, documents=2, seed=1)
    rows = read_labels(out)
    assert sorted(bool(spans) for _, spans in rows) == [False, True]
    assert all(spans == [(0, len(t))] for t, spans in rows if spans)
    assert (source.level, source.documents, source.positive_documents) == ("document", 2, 1)


def test_cli_lists_labels_without_one(hub):
    result = CliRunner().invoke(cli, ["concepts", "--dataset", "ag_news"])
    assert result.exit_code == 0, result.output
    assert "2  Sports" in result.output and "2  Business" in result.output
