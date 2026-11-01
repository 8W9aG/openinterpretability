"""Detecting a document's language (`openinterp/langid.py`) for `openinterp corpus`."""

import hashlib
import json
from types import SimpleNamespace

import fsspec
import pytest

from openinterp import langid
from openinterp.corpus import build_from_hub
from tests.test_corpus import shards  # noqa: F401  (fixture)


class FakeModel:
    """Labels text by its first word, sure of it unless the word is 'maybe'."""

    def __init__(self):
        self.f = self
        self.seen = []

    def predict(self, text, k, threshold, on_unicode_error):
        self.seen.append(text)
        word = text.split()[0] if text.split() else "en"
        return [(0.5 if word == "maybe" else 0.99, f"__label__{word}")]


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(langid, "_model", lambda: fake)
    return fake


def test_detect_labels_only_documents_it_is_sure_of(model):
    assert langid.detect(["fr le chat", "maybe code()", "de\nKatze"]) == ["fr", None, "de"]
    assert model.seen[2] == "de Katze"  # one line: fastText reads up to a newline


def test_the_model_is_downloaded_once_and_checked(tmp_path, monkeypatch):
    data = b"model bytes"
    calls = []

    class Response:
        def __enter__(self):
            calls.append(1)
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return data

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(langid.urllib.request, "urlopen", lambda url: Response())
    with pytest.raises(ValueError, match="SHA-256"):
        langid.model_path()
    monkeypatch.setattr(langid, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    path = langid.model_path()
    assert path.read_bytes() == data and langid.model_path() == path and len(calls) == 2


def test_build_from_hub_records_detected_languages(tmp_path, shards, monkeypatch, model):  # noqa: F811
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
    source = build_from_hub("org/web", out, files="*.parquet", documents=4, detect_lang=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["lang"] for r in rows] == [r["text"].split()[0] for r in rows]
    assert source.lang_detector == langid.DETECTOR and source.lang_pattern is None
    with pytest.raises(ValueError, match="not both"):
        build_from_hub("org/web", out, files="*.parquet", lang_pattern="x", detect_lang=True)
