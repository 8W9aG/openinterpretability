"""Detecting a document's programming language (`openinterp/codeid.py`) for `openinterp corpus`."""

import json
from types import SimpleNamespace

import fsspec
import pytest

from openinterp import codeid
from openinterp.corpus import build_from_hub
from openinterp.corpus_presets import GITHUB_CODE_LANGUAGES
from tests.test_corpus import shards  # noqa: F401  (fixture)


class FakeMagika:
    """Labels text by its first word, sure of it unless the word is 'maybe'.

    The test shards' documents start "fileF docN": it labels those with an odd N as Python.
    """

    def identify_bytes(self, data):
        words = data.decode().split() or ["txt"]
        word = "python" if odd_doc(data.decode()) else words[0]
        return SimpleNamespace(
            prediction=SimpleNamespace(
                dl=SimpleNamespace(label=word), score=0.5 if word == "maybe" else 0.99
            )
        )

    def get_model_name(self):
        return "fake_v1"


def odd_doc(text: str) -> bool:
    """Is ``text`` a test shard's document with an odd number?"""
    words = text.split()
    return len(words) > 1 and words[1].startswith("doc") and int(words[1][3:]) % 2 == 1


@pytest.fixture
def magika(monkeypatch):
    monkeypatch.setattr(codeid, "_magika", FakeMagika)
    codeid.detector.cache_clear()
    yield
    codeid.detector.cache_clear()


def test_every_github_code_language_has_one_magika_label():
    assert sorted(codeid.MAGIKA_LABELS.values()) == sorted(GITHUB_CODE_LANGUAGES)


def test_detect_names_only_github_code_languages_it_is_sure_of(magika):
    texts = ["python def f():", "cpp int x;", "json {}", "txt Some prose.", "maybe x"]
    assert codeid.detect(texts) == ["Python", "C++", None, None, None]


def test_the_detector_records_magikas_version_model_and_line(magika):
    assert "model fake_v1" in codeid.detector()
    assert f">= {codeid.MIN_CONFIDENCE}" in codeid.detector()


def test_build_from_hub_records_detected_programming_languages(
    tmp_path,
    shards,  # noqa: F811
    monkeypatch,
    magika,
):
    local = fsspec.filesystem("file")

    class FakeFS:
        def glob(self, pattern):
            return shards

        def open(self, path, mode):
            return local.open(path, mode)

    info = SimpleNamespace(sha="sha1", card_data=None)
    monkeypatch.setattr("huggingface_hub.HfFileSystem", FakeFS)
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda: SimpleNamespace(dataset_info=lambda d, revision: info)
    )
    out = tmp_path / "c.jsonl"
    source = build_from_hub("org/web", out, files="*.parquet", documents=8, detect_code=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    labels = [r.get("code_language") for r in rows]
    assert labels == ["Python" if odd_doc(r["text"]) else None for r in rows]
    assert "Python" in labels and None in labels
    assert source.code_detector == codeid.detector() and source.code_language_column is None
    with pytest.raises(ValueError, match="not both"):
        build_from_hub(
            "org/web", out, files="*.parquet", code_language_column="lang", detect_code=True
        )
