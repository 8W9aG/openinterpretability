"""The default corpus of `explain` (openinterpretability/explain_corpus.py, #5829)."""

import json
from pathlib import Path

import pytest

from openinterpretability.core.corpora import CorpusSource
from openinterpretability.explain_corpus import EXPLAIN_CORPUS, default_corpus
from tests.test_explain import TEXTS

ROOT = Path(__file__).resolve().parents[1]
CORPUS = "".join(json.dumps({"text": t}) + "\n" for t in TEXTS)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A rebuilt-corpora cache in ``tmp_path``, and a fake Hub build that records each call."""
    built = []

    def rebuild(source, out):
        built.append(out)
        out.write_text(CORPUS)
        return source

    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", str(tmp_path / "cache"))
    monkeypatch.setattr("openinterpretability.corpus_rebuild.rebuild", rebuild)
    return built


def test_the_spec_is_the_committed_scan_corpus():
    committed = (ROOT / "corpus.source.json").read_text(encoding="utf-8")
    assert CorpusSource.model_validate_json(committed) == EXPLAIN_CORPUS


def test_a_fresh_directory_builds_the_corpus_once_and_then_reuses_it(hub, tmp_path):
    said = []
    first = default_corpus(said.append, local=tmp_path / "corpus.jsonl")
    second = default_corpus(said.append, local=tmp_path / "corpus.jsonl")
    assert first == second == tmp_path / "cache" / f"{EXPLAIN_CORPUS.sha256}.jsonl"
    assert hub == [first]  # built once
    assert said == ["Building the explain corpus (NeelNanda/pile-10k, 500 documents), once. "
                    "Later runs reuse it."]  # fmt: skip


def test_a_local_corpus_with_other_bytes_is_not_used(hub, tmp_path):
    local = tmp_path / "corpus.jsonl"
    local.write_text('{"text": "my own corpus"}\n')
    assert default_corpus(lambda _: None, local=local) != local
    assert len(hub) == 1


def test_with_the_cache_off_the_error_names_the_option(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", "off")
    with pytest.raises(ValueError, match="Pass --corpus with a corpus file"):
        default_corpus(lambda _: None, local=tmp_path / "corpus.jsonl")


def test_a_corpus_hash_that_is_not_a_sha256_never_names_a_cache_file(monkeypatch, tmp_path):
    from openinterpretability.corpus_rebuild import cache_path

    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", str(tmp_path))
    assert cache_path(EXPLAIN_CORPUS).parent == tmp_path
    for bad in ("../../outside", EXPLAIN_CORPUS.sha256.upper(), ""):
        with pytest.raises(ValueError, match="not a SHA-256"):
            cache_path(EXPLAIN_CORPUS.model_copy(update={"sha256": bad}))
