"""Per-token labels for the scan (`harness/annotations.py`) and the confound they give."""

import json
import math
from types import SimpleNamespace

import fsspec
import torch

from openinterp.agents.proposer import format_confounds
from openinterp.corpus import build_from_hub, jsonl_bytes, sample_indices, sample_tokens
from openinterp.evidence import read_corpus, read_labels
from openinterp.harness.annotations import (
    UNLABELLED,
    DocumentLabels,
    LabelAccumulator,
    Span,
    TokenLabels,
    activation_bins,
    align,
    mutual_information,
)
from openinterp.harness.scan import Piece, cut_pieces, scan_neuron
from tests.conftest import FakeTokenizer
from tests.test_corpus import shards  # noqa: F401  (fixture)


class CharTokenizer:
    """One token per character, with offsets, and a BOS special token (id 0)."""

    all_special_ids = [0]

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"offset_mapping": [(0, 0)] * add_special_tokens + offsets}

    def encode(self, text, add_special_tokens=True):
        return [0] * add_special_tokens + [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)


def test_align_takes_the_span_covering_most_of_each_token():
    spans = [Span(0, 3, "A"), Span(3, 10, "B")]
    assert align([(0, 0), (0, 2), (2, 5), (8, 12), (12, 14)], spans) == [
        None, "A", "B", "B", None
    ]  # fmt: skip


def test_document_labels_cover_every_token_but_special_ones():
    labels = TokenLabels(CharTokenizer(), ["ab", "cd"], [DocumentLabels("language", ["en", None])])
    first = labels.of(Piece(0, 0, 1, "ab"), torch.tensor([0, 97, 98]))["language"]
    second = labels.of(Piece(1, 0, 1, "cd"), torch.tensor([99, 100]))["language"]
    assert first.tolist() == [UNLABELLED, 0, 0] and second.tolist() == [UNLABELLED] * 2
    assert labels.names == {"language": ["en"]}


class WordTagger:
    name = "word"

    def spans(self, doc, text):
        start, out = 0, []
        for word in text.split(" "):
            out.append(Span(start, start + len(word), word.upper()))
            start += len(word) + 1
        return out


def test_word_spans_align_inside_windows():
    tok, text = CharTokenizer(), "ab cd ef"
    labels = TokenLabels(tok, [text], [WordTagger()])
    for piece in cut_pieces(tok, [text], window_tokens=4, stride=2):
        ids = torch.tensor(tok.encode(piece.text))
        got = [labels.names["word"][i] if i >= 0 else None for i in labels.of(piece, ids)["word"]]
        want = [None] + [{"a": "AB", "b": "AB", "c": "CD", "d": "CD", "e": "EF", "f": "EF"}
                         .get(c) for c in piece.text]  # fmt: skip
        assert got == want


def test_pieces_without_offsets_or_matching_text_are_unlabelled():
    labels = TokenLabels(FakeTokenizer(), ["ab cd"], [WordTagger()])  # no offset mapping
    assert labels.of(Piece(0, 0, 1, "ab cd"), torch.tensor([1] * 5))["word"].tolist() == [-1] * 5
    labels = TokenLabels(CharTokenizer(), ["ab cd"], [WordTagger()])
    assert labels.of(Piece(0, 2, 4, "zz"), torch.tensor([0, 1, 2]))["word"].tolist() == [-1] * 3


def test_zero_is_its_own_bin():
    bins = activation_bins(torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]), bins=2)
    assert bins.tolist() == [0, 0, 1, 1, 2, 2]


def test_mutual_information_is_near_zero_when_independent_and_one_bit_when_determined():
    gen = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 2, (20_000,), generator=gen)
    noise = torch.rand(20_000, generator=gen, dtype=torch.float64).clamp(min=0.5) - 0.5
    assert mutual_information(noise, labels, 2) < 0.002
    determined = labels.double() * (1 + torch.rand(20_000, generator=gen, dtype=torch.float64))
    assert math.isclose(mutual_information(determined, labels, 2), 1.0, abs_tol=0.01)


def test_label_confound_counts_r2_and_active_tokens():
    acc = LabelAccumulator()
    acc.update(torch.tensor([0, 0, 1, 1, UNLABELLED]), torch.tensor([0.0, 0.0, 2.0, 2.0, 9.0]))
    confound = acc.finalize(["en", "de"], line=1.0)
    assert confound.r2 == 1.0 and (confound.labelled_tokens, confound.unlabelled_tokens) == (4, 1)
    assert [(s.label, s.count, s.mean_strength, s.active) for s in confound.levels] == [
        ("en", 2, 0.0, 0), ("de", 2, 2.0, 2)
    ]  # fmt: skip
    assert confound.mutual_information_bits > 0.5


TEXTS = ["the cat sat", "le chat noir", "the dog ran far", "un chien court"]


def test_the_scan_records_label_confounds_without_changing_anything_else(toy_runner):
    plain = scan_neuron(toy_runner, TEXTS, 1, 2, "positive")
    tagged = scan_neuron(
        toy_runner, TEXTS, 1, 2, "positive",
        taggers=[DocumentLabels("language", ["en", "fr", "en", "fr"])],
    )  # fmt: skip
    assert plain.confounds.labels is None
    assert tagged.examples == plain.examples and tagged.values == plain.values
    assert tagged.confounds.model_copy(update={"labels": None}) == plain.confounds
    language = tagged.confounds.labels["language"]
    assert language.labelled_tokens == sum(map(len, TEXTS)) and language.unlabelled_tokens == 0
    assert {s.label: s.count for s in language.levels} == {"en": 26, "fr": 26}


def test_windowed_scans_label_each_token_once(toy_runner):
    tagged = scan_neuron(
        toy_runner, TEXTS, 1, 2, "positive", window_tokens=6, stride=3,
        taggers=[DocumentLabels("language", ["en", "fr", "en", "fr"])],
    )  # fmt: skip
    assert tagged.confounds.labels["language"].labelled_tokens == sum(map(len, TEXTS))


def test_the_proposer_sees_label_confounds(toy_runner):
    scan = scan_neuron(
        toy_runner, TEXTS, 1, 2, "positive",
        taggers=[DocumentLabels("language", ["en", "fr", "en", "fr"])],
    )  # fmt: skip
    text = format_confounds(scan.confounds)
    assert "Language explains" in text and "active tokens by language: " in text
    assert "Language" not in format_confounds(scan.confounds.model_copy(update={"labels": None}))


def test_corpora_record_and_read_back_document_languages(tmp_path):
    out = tmp_path / "c.jsonl"
    out.write_bytes(jsonl_bytes(TEXTS, labels={"lang": ["en", "fr", "en", "fr"]}))
    assert read_corpus(out)[0] == TEXTS
    assert read_labels(out) == {"language": ["en", "fr", "en", "fr"]}
    assert jsonl_bytes(TEXTS) == jsonl_bytes(TEXTS, labels={})  # unlabelled corpora unchanged
    plain = tmp_path / "plain.jsonl"
    plain.write_bytes(jsonl_bytes(TEXTS))
    assert read_labels(plain) == {} and read_labels(None) == {}


def test_sample_indices_match_the_sample():
    tok = FakeTokenizer()
    picked, sample = sample_indices(tok, TEXTS * 5, 60, 1)
    assert ([(TEXTS * 5)[i] for i in picked], sample) == sample_tokens(tok, TEXTS * 5, 60, 1)


def test_build_from_hub_records_each_globs_language(tmp_path, shards, monkeypatch):  # noqa: F811
    local = fsspec.filesystem("file")
    by_glob = {"datasets/org/wiki@sha1/20231101.en/*.parquet": shards[:1],
               "datasets/org/wiki@sha1/20231101.de/*.parquet": shards[1:]}  # fmt: skip

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
    files = "20231101.en/*.parquet,20231101.de/*.parquet"
    source = build_from_hub("org/wiki", out, files=files, documents=5, lang_pattern=r"^\d+\.(\w+)/")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["lang"] for r in rows] == ["en"] * 3 + ["de"] * 2
    assert [r["text"].split()[0] for r in rows] == ["file0"] * 3 + ["file1"] * 2
    assert source.lang_pattern == r"^\d+\.(\w+)/"
