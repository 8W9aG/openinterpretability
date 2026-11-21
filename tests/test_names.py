"""The names model: its download, its unpacking and its part-of-speech pass (names.py)."""

import hashlib
import io
import zipfile
from types import SimpleNamespace

import pytest

from openinterpretability import names
from openinterpretability.names import pos_matches as real_pos_matches


def _wheel_bytes() -> bytes:
    """A small wheel that holds the model's folder, as the real wheel does."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"en_core_web_trf/{names.MODEL}/meta.json", '{"name": "trf"}')
    return buffer.getvalue()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Serve a small wheel at the model's URL, with its SHA-256 pinned. Count the requests."""
    data = _wheel_bytes()
    asked = []

    def urlopen(url):
        asked.append(url)
        return io.BytesIO(data)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(names, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    return asked


def test_a_wheel_that_matches_its_checksum_is_kept(tmp_path, served):
    wheel = tmp_path / "m.whl"
    names._download(wheel)
    assert wheel.read_bytes() == _wheel_bytes()
    assert not wheel.with_suffix(".part").exists()


def test_the_model_is_downloaded_and_unpacked_once(tmp_path, served):
    path = names.model_path()
    root = tmp_path / "openinterpretability" / names.MODEL
    assert path == root / "en_core_web_trf" / names.MODEL
    assert (path / "meta.json").read_text() == '{"name": "trf"}'
    assert not root.with_suffix(".whl").exists()  # the wheel is removed after unpacking
    assert not root.with_suffix(".unpacking").exists()
    assert names.model_path() == path
    assert served == [names.MODEL_URL]  # the second call reads the disk


def test_the_model_loads_only_the_pipes_the_labels_need(tmp_path, served, monkeypatch):
    import spacy

    loads = []

    def load(path, exclude=()):
        loads.append((path, list(exclude)))
        return SimpleNamespace(pipe_names=["transformer", "tagger", "parser", "lemmatizer"])

    monkeypatch.setattr(spacy, "load", load)
    names._model.cache_clear()
    try:
        names._model()
    finally:
        names._model.cache_clear()
    path = names.model_path()
    assert loads == [(path, []), (path, ["parser", "lemmatizer"])]


def test_tags_alone_run_the_model_and_cache_the_names_too(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    class Doc(list):
        ents = [SimpleNamespace(start_char=0, end_char=3, label_="PERSON")]

    def model(text):
        calls.append(text)
        word = SimpleNamespace(idx=0, text="Ada", pos_="PROPN", is_space=False)
        return Doc([word])

    monkeypatch.setattr(names, "_model", lambda: model)
    assert real_pos_matches("Ada") == ((0, 3, "PROPN"),)
    assert names._cached("names", "Ada").exists()
    assert real_pos_matches("Ada") == ((0, 3, "PROPN"),)
    assert calls == ["Ada"]
